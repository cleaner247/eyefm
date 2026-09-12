"""Subject-level multi-instance classifier built on EyeVQ-BERT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eyemae.eyevq.pretrain.model import EyeVQBERT


class EyeVQSubjectMIL(nn.Module):
    """Classify subject trial bags with explicit trial- and task-level pooling."""

    def __init__(
        self,
        bert: EyeVQBERT,
        *,
        num_tasks: int = 4,
        num_classes: int = 2,
        output_dim: int | None = None,
        classifier_hidden: int = 32,
        task_bottleneck_dim: int = 16,
        residual_hidden: int = 16,
        residual_include_task_mask: bool = True,
        dropout: float = 0.3,
        task_pooling: str = "concat_task_cls",
        trial_pooling: str = "feature_mean",
        freeze_embedding: bool = True,
        freeze_bottom_layers: int = 4,
        demographic_dim: int = 0,
        demographic_projection_dim: int = 128,
        demographic_fusion: str = "additive_logit",
        demographic_max_alpha: float = 0.3,
        task_residual_logit_scale: float = 0.2,
        classifier_head: str = "mlp",
        cartesian_trials_per_task: int = 4,
        missing_task_embedding: str = "none",
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= freeze_bottom_layers <= len(bert.transformer):
            raise ValueError("freeze_bottom_layers is outside the BERT layer range")
        self.bert = bert
        self.num_tasks = int(num_tasks)
        self.num_classes = int(num_classes)
        default_output_dim = 1 if self.num_classes == 2 else self.num_classes
        self.output_dim = default_output_dim if output_dim is None else int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.task_pooling = str(task_pooling)
        self.trial_pooling = str(trial_pooling)
        self.demographic_dim = int(demographic_dim)
        self.demographic_projection_dim = int(demographic_projection_dim)
        self.demographic_fusion = str(demographic_fusion)
        self.demographic_max_alpha = float(demographic_max_alpha)
        self.task_residual_logit_scale = float(task_residual_logit_scale)
        self.classifier_head = str(classifier_head)
        self.residual_include_task_mask = bool(residual_include_task_mask)
        self.cartesian_trials_per_task = int(cartesian_trials_per_task)
        self.missing_task_embedding = str(missing_task_embedding)
        if self.demographic_dim < 0:
            raise ValueError("demographic_dim must be non-negative")
        if self.demographic_projection_dim < 0:
            raise ValueError("demographic_projection_dim must be non-negative")
        if self.demographic_fusion not in {
            "additive_logit", "bounded_additive_logit", "concat_task_cls", "late_residual"
        }:
            raise ValueError(
                "demographic_fusion must be additive_logit, bounded_additive_logit, "
                "concat_task_cls, or late_residual"
            )
        if not 0.0 <= self.demographic_max_alpha <= 1.0:
            raise ValueError("demographic_max_alpha must be in [0,1]")
        if not 0.0 < self.task_residual_logit_scale <= 0.2:
            raise ValueError("task_residual_logit_scale must be in (0,0.2]")
        if self.classifier_head not in {"mlp", "linear"}:
            raise ValueError("classifier_head must be mlp or linear")
        if self.cartesian_trials_per_task <= 0:
            raise ValueError("cartesian_trials_per_task must be positive")
        if self.missing_task_embedding not in {"none", "learned_per_task"}:
            raise ValueError(
                "missing_task_embedding must be none or learned_per_task"
            )
        if self.trial_pooling not in {
            "feature_mean", "logit_mean", "cartesian_logit_mean"
        }:
            raise ValueError(
                "trial_pooling must be feature_mean, logit_mean, or "
                "cartesian_logit_mean"
            )
        if self.trial_pooling == "logit_mean" and self.task_pooling not in {
            "shared_head_softmax", "shared_head_mean", "shared_head_constrained",
            "shared_head_residual"
        }:
            raise ValueError(
                "logit_mean trial pooling requires a shared task head"
            )
        if (
            self.trial_pooling == "cartesian_logit_mean"
            and self.task_pooling != "cartesian_task_cls"
        ):
            raise ValueError(
                "cartesian_logit_mean requires cartesian_task_cls pooling"
            )
        if (
            self.task_pooling == "cartesian_task_cls"
            and self.trial_pooling != "cartesian_logit_mean"
        ):
            raise ValueError(
                "cartesian_task_cls requires cartesian_logit_mean trial pooling"
            )
        if (
            self.missing_task_embedding != "none"
            and self.task_pooling != "cartesian_task_cls"
        ):
            raise ValueError(
                "missing_task_embedding is supported only by cartesian_task_cls"
            )
        if self.demographic_fusion == "concat_task_cls" and (
            self.demographic_dim <= 0
            or self.task_pooling not in {
                "shared_head_softmax", "shared_head_mean", "cartesian_task_cls"
            }
        ):
            raise ValueError(
                "concat_task_cls requires demographics and shared-head pooling"
            )
        if self.demographic_fusion == "late_residual" and (
            self.demographic_dim <= 0
            or self.task_pooling != "shared_head_residual"
        ):
            raise ValueError(
                "late_residual requires demographics and shared_head_residual pooling"
            )
        if (
            self.task_pooling == "shared_head_residual"
            and self.demographic_fusion != "late_residual"
        ):
            raise ValueError(
                "shared_head_residual requires demographic_fusion=late_residual"
            )
        if (
            self.task_pooling == "shared_head_residual"
            and self.trial_pooling != "logit_mean"
        ):
            raise ValueError(
                "shared_head_residual requires trial_pooling=logit_mean"
            )

        # The mask prediction head is pretraining-only.  Embeddings and the
        # lower transformer are intentionally frozen for the small MCI cohort.
        self.bert.pred_head.requires_grad_(False)
        if self.bert.manual_feat_head is not None:
            self.bert.manual_feat_head.requires_grad_(False)
        if self.bert.span_length_embed is not None:
            self.bert.span_length_embed.requires_grad_(False)
        if freeze_embedding:
            self.bert.embed.requires_grad_(False)
        else:
            self.bert.embed.mask_token.requires_grad_(False)
        for block in self.bert.transformer[: int(freeze_bottom_layers)]:
            block.requires_grad_(False)
        # ``out_norm`` is part of the encoder representation, but it lives
        # outside ``bert.transformer``.  A full linear probe must freeze it as
        # well; otherwise ``freeze_bottom_layers == n_layers`` silently leaves
        # its scale vector trainable and the experiment is not encoder-frozen.
        if int(freeze_bottom_layers) == len(self.bert.transformer):
            self.bert.out_norm.requires_grad_(False)

        hidden = int(classifier_hidden)
        if hidden <= 0:
            raise ValueError("classifier_hidden must be positive")
        if self.task_pooling == "cartesian_task_cls":
            if self.classifier_head != "mlp":
                raise ValueError("cartesian_task_cls requires classifier_head=mlp")
            # Each Cartesian-product item has one fixed slot per task. Normalize
            # the pretrained representations independently, concatenate the
            # four slots, append demographics once, then apply a compact MLP.
            # The first affine layer is evaluated in a factorized form below so
            # no [B,K^T,T*D] concatenation tensor needs to be materialized.
            self.cartesian_feature_norm = nn.LayerNorm(bert.d_model)
            cartesian_input_dim = self.num_tasks * bert.d_model + self.demographic_dim
            self.cartesian_hidden = nn.Linear(cartesian_input_dim, hidden)
            self.cartesian_activation = nn.GELU()
            self.cartesian_dropout = nn.Dropout(float(dropout))
            self.cartesian_output = nn.Linear(hidden, self.output_dim)
            if self.missing_task_embedding == "learned_per_task":
                # A separate learned CLS placeholder preserves task identity.
                # It is normalized by the same LayerNorm as real BERT CLSs.
                self.cartesian_mask_cls_embeddings = nn.Parameter(
                    torch.empty(self.num_tasks, bert.d_model)
                )
                nn.init.normal_(self.cartesian_mask_cls_embeddings, std=0.02)
            else:
                self.register_parameter("cartesian_mask_cls_embeddings", None)
        elif self.task_pooling in {
            "concat_task_cls", "concat_task_cls_aux", "concat_task_mean_std"
        }:
            feature_multiplier = 2 if self.task_pooling == "concat_task_mean_std" else 1
            concatenated_dim = self.num_tasks * bert.d_model * feature_multiplier
            self.subject_head = nn.Sequential(
                nn.LayerNorm(concatenated_dim),
                nn.Linear(concatenated_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.output_dim),
            )
            if self.task_pooling == "concat_task_cls_aux":
                self.task_head = nn.Sequential(
                    nn.LayerNorm(bert.d_model),
                    nn.Linear(bert.d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, self.output_dim),
                )
        elif self.task_pooling in {
            "shared_head_softmax", "shared_head_mean", "shared_head_constrained",
            "shared_head_residual"
        }:
            if self.demographic_fusion == "concat_task_cls":
                # Normalize the pretrained CLS representation on its own, then
                # append the compact demographic projection.  Joint
                # normalization would let the low-dimensional demographic
                # branch change the scale of every pretrained CLS coordinate.
                if self.demographic_projection_dim == 0:
                    # Explicit direct-concatenation mode: keep the fold-safe
                    # 16-D encoding unchanged and add no learned projection.
                    self.demographic_projection = nn.Identity()
                    self.demographic_activation = nn.Identity()
                    appended_demographic_dim = self.demographic_dim
                else:
                    self.demographic_projection = nn.Linear(
                        self.demographic_dim, self.demographic_projection_dim
                    )
                    self.demographic_activation = nn.GELU()
                    appended_demographic_dim = self.demographic_projection_dim
                task_input_dim = bert.d_model + appended_demographic_dim
                self.task_feature_norm = nn.LayerNorm(bert.d_model)
                if self.classifier_head == "linear":
                    self.task_head = nn.Linear(task_input_dim, self.output_dim)
                else:
                    self.task_head = nn.Sequential(
                        nn.Linear(task_input_dim, hidden),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(hidden, self.output_dim),
                    )
            elif self.classifier_head == "linear":
                self.task_head = nn.Sequential(
                    nn.LayerNorm(bert.d_model),
                    nn.Linear(bert.d_model, self.output_dim),
                )
            else:
                self.task_head = nn.Sequential(
                    nn.LayerNorm(bert.d_model),
                    nn.Linear(bert.d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, self.output_dim),
                )
            if self.task_pooling == "shared_head_softmax":
                # Optional legacy mode: learn one global task-weight vector.
                self.task_weight_logits = nn.Parameter(torch.zeros(self.num_tasks))
            if self.task_pooling == "shared_head_constrained":
                self.task_weight_residual = nn.Parameter(torch.zeros(self.num_tasks))
            if self.task_pooling == "shared_head_residual":
                residual_width = int(residual_hidden)
                if residual_width <= 0:
                    raise ValueError("residual_hidden must be positive")
                # The residual sees subject-level evidence only: one shared-head
                # logit vector per task, explicit task-presence bits, and the
                # fold-safe demographic encoding. Its zero-initialized output
                # makes the initial model exactly the uniform task-logit mean.
                residual_input_dim = (
                    self.num_tasks * self.output_dim
                    + (
                        self.num_tasks
                        if self.residual_include_task_mask
                        else 0
                    )
                    + self.demographic_dim
                )
                self.cross_task_residual_head = nn.Sequential(
                    nn.Linear(residual_input_dim, residual_width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(residual_width, self.output_dim),
                )
                nn.init.zeros_(self.cross_task_residual_head[-1].weight)
                nn.init.zeros_(self.cross_task_residual_head[-1].bias)
        elif self.task_pooling == "task_bottleneck_concat":
            bottleneck = int(task_bottleneck_dim)
            if bottleneck <= 0:
                raise ValueError("task_bottleneck_dim must be positive")
            # Apply the same compact projection to every task. The subsequent
            # subject head retains cross-task interactions without exposing a
            # 4*d_model input to the small downstream cohort.
            self.task_projector = nn.Sequential(
                nn.LayerNorm(bert.d_model),
                nn.Linear(bert.d_model, bottleneck),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            compact_dim = self.num_tasks * bottleneck
            self.subject_head = nn.Sequential(
                nn.LayerNorm(compact_dim),
                nn.Linear(compact_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.output_dim),
            )
        else:
            raise ValueError(
                "task_pooling must be concat_task_cls, concat_task_cls_aux, "
                "concat_task_mean_std, shared_head_softmax, shared_head_mean, "
                "shared_head_constrained, shared_head_residual, task_bottleneck_concat, or "
                "cartesian_task_cls"
            )
        # A nested additive linear branch gives a fair comparison: setting its
        # weights to zero exactly recovers the eye-only classifier.
        self.demographic_head = (
            nn.Linear(self.demographic_dim, self.output_dim)
            if self.demographic_dim > 0
            and self.demographic_fusion in {"additive_logit", "bounded_additive_logit"}
            else None
        )
        if self.demographic_head is not None:
            nn.init.zeros_(self.demographic_head.weight)
            nn.init.zeros_(self.demographic_head.bias)
        if self.demographic_fusion == "bounded_additive_logit":
            self.demographic_alpha_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.register_parameter("demographic_alpha_logit", None)

    def _format_head_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Keep binary logits scalar and multiclass logits class-indexed."""
        return logits.squeeze(-1) if self.output_dim == 1 else logits

    def _fuse_cls_with_demographics(
        self,
        cls_features: torch.Tensor,
        demographic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize CLS alone, then append projected demographic features."""
        if cls_features.shape[:-1] != demographic_features.shape[:-1]:
            raise ValueError(
                "CLS and demographic features must share their leading dimensions"
            )
        projected_demographics = self.demographic_activation(
            self.demographic_projection(demographic_features)
        )
        normalized_cls = self.task_feature_norm(cls_features)
        return torch.cat((normalized_cls, projected_demographics), dim=-1)

    def _add_demographics(
        self,
        result: dict[str, torch.Tensor],
        demographic_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.demographic_fusion in {"concat_task_cls", "late_residual"}:
            return result
        if self.demographic_head is None:
            if demographic_features is not None:
                raise ValueError("Demographic features were provided to an eye-only model")
            return result
        if demographic_features is None:
            raise ValueError("Demographic features are required by this model")
        if demographic_features.ndim != 2 or demographic_features.shape != (
            result["subject_logits"].shape[0], self.demographic_dim
        ):
            raise ValueError(
                "demographic_features must have shape "
                f"[B,{self.demographic_dim}]"
            )
        raw_demographic_logits = self._format_head_logits(
            self.demographic_head(demographic_features)
        )
        if self.demographic_fusion == "bounded_additive_logit":
            demographic_alpha = self.demographic_max_alpha * torch.sigmoid(
                self.demographic_alpha_logit
            )
            demographic_logits = raw_demographic_logits * demographic_alpha
            result["demographic_alpha"] = demographic_alpha
            result["raw_demographic_logits"] = raw_demographic_logits
        else:
            demographic_logits = raw_demographic_logits
        result["eye_subject_logits"] = result["subject_logits"]
        result["demographic_logits"] = demographic_logits
        result["subject_logits"] = result["subject_logits"] + demographic_logits
        return result

    def encode_trials(
        self,
        stim_patches: torch.Tensor,
        eye_patches: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.bert.encode(
            stim_patches, eye_patches, pad_mask, eye_nonmissing_frac, task_ids
        )
        return hidden[:, 0, :]

    def classify_task_features(
        self,
        task_features: torch.Tensor,
        task_present_mask: torch.Tensor,
        demographic_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if task_features.ndim != 3 or task_features.shape[1] != self.num_tasks:
            raise ValueError(
                f"task_features must have shape [B,{self.num_tasks},D]"
            )
        if task_present_mask.shape != task_features.shape[:2]:
            raise ValueError(
                f"task_present_mask must have shape {tuple(task_features.shape[:2])}"
            )
        masked_features = task_features.masked_fill(
            ~task_present_mask.unsqueeze(-1), 0.0
        )
        if self.task_pooling in {
            "concat_task_cls", "concat_task_cls_aux", "concat_task_mean_std"
        }:
            # A missing task keeps a fixed zero-filled slot.
            concatenated_features = masked_features.flatten(start_dim=1)
            subject_logits = self._format_head_logits(
                self.subject_head(concatenated_features)
            )
            result = {
                "subject_logits": subject_logits,
                "task_features": task_features,
                "concatenated_features": concatenated_features,
            }
            if self.task_pooling == "concat_task_cls_aux":
                result["task_logits"] = self._format_head_logits(
                    self.task_head(masked_features)
                )
            return self._add_demographics(result, demographic_features)

        if self.task_pooling == "task_bottleneck_concat":
            compact_task_features = self.task_projector(masked_features)
            compact_task_features = compact_task_features.masked_fill(
                ~task_present_mask.unsqueeze(-1), 0.0
            )
            compact_features = compact_task_features.flatten(start_dim=1)
            subject_logits = self._format_head_logits(
                self.subject_head(compact_features)
            )
            return self._add_demographics({
                "subject_logits": subject_logits,
                "task_features": task_features,
                "compact_task_features": compact_task_features,
                "compact_features": compact_features,
            }, demographic_features)

        if not torch.all(task_present_mask.any(dim=1)):
            raise ValueError("Every subject must contain at least one task")
        task_head_inputs = masked_features
        if self.demographic_fusion == "concat_task_cls":
            if demographic_features is None:
                raise ValueError("CLS concatenation requires demographic features")
            if demographic_features.ndim != 2 or demographic_features.shape != (
                task_features.shape[0], self.demographic_dim
            ):
                raise ValueError(
                    "demographic_features must have shape "
                    f"[B,{self.demographic_dim}]"
                )
            repeated_demographics = demographic_features.unsqueeze(1).expand(
                -1, self.num_tasks, -1
            )
            task_head_inputs = self._fuse_cls_with_demographics(
                masked_features, repeated_demographics
            )
            task_logits = self._format_head_logits(
                self.task_head(task_head_inputs)
            )
        else:
            task_logits = self._format_head_logits(
                self.task_head(task_head_inputs)
            )
        result = self.aggregate_task_logits(
            task_logits, task_present_mask, demographic_features
        )
        result.update({
            "task_features": task_features,
            "task_head_inputs": task_head_inputs,
        })
        return result

    def classify_individual_trial_features(
        self,
        trial_features: torch.Tensor,
        demographic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the shared task head independently to flattened trial CLSs."""
        if self.task_pooling not in {
            "shared_head_softmax", "shared_head_mean", "shared_head_constrained",
            "shared_head_residual"
        }:
            raise ValueError("Individual-trial classification requires a shared task head")
        if trial_features.ndim != 2 or trial_features.shape[-1] != self.bert.d_model:
            raise ValueError(
                f"trial_features must have shape [N,{self.bert.d_model}]"
            )
        head_inputs = trial_features
        if self.demographic_fusion == "concat_task_cls":
            if demographic_features is None:
                raise ValueError("CLS concatenation requires demographic features")
            if demographic_features.shape != (
                trial_features.shape[0], self.demographic_dim
            ):
                raise ValueError(
                    "demographic_features must match the flattened trial axis"
                )
            head_inputs = self._fuse_cls_with_demographics(
                trial_features, demographic_features
            )
        return self._format_head_logits(self.task_head(head_inputs))

    def classify_cartesian_trial_features(
        self,
        trial_features: torch.Tensor,
        task_present_mask: torch.Tensor,
        demographic_features: torch.Tensor | None = None,
        trial_slot_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Average logits over every one-trial-per-task Cartesian product.

        For four tasks with K trials this evaluates K^4 task-ordered tuples.
        The task-wise affine contributions are computed before broadcasting,
        which is exactly equivalent to concatenating the four normalized CLS
        vectors and demographics for every tuple but avoids materializing the
        much wider concatenation tensor.
        """
        if self.task_pooling != "cartesian_task_cls":
            raise ValueError(
                "Cartesian trial classification requires cartesian_task_cls"
            )
        if (
            trial_features.ndim != 4
            or trial_features.shape[1] != self.num_tasks
            or trial_features.shape[-1] != self.bert.d_model
        ):
            raise ValueError(
                "trial_features must have shape "
                f"[B,{self.num_tasks},K,{self.bert.d_model}]"
            )
        batch_size, _, trials_per_task, feature_dim = trial_features.shape
        if trials_per_task <= 0:
            raise ValueError("Every task must contain at least one trial")
        if task_present_mask.shape != (batch_size, self.num_tasks):
            raise ValueError(
                f"task_present_mask must have shape [B,{self.num_tasks}]"
            )
        if self.demographic_dim > 0:
            if demographic_features is None or demographic_features.shape != (
                batch_size,
                self.demographic_dim,
            ):
                raise ValueError(
                    "demographic_features must have shape "
                    f"[B,{self.demographic_dim}]"
                )
        elif demographic_features is not None:
            raise ValueError("Demographics were provided to an eye-only model")
        if trial_slot_mask is None:
            if not torch.all(task_present_mask):
                raise ValueError(
                    "Cartesian task-CLS pooling requires every task to be present"
                )
            trial_slot_mask = torch.ones(
                batch_size,
                self.num_tasks,
                trials_per_task,
                dtype=torch.bool,
                device=trial_features.device,
            )
        elif trial_slot_mask.shape != (
            batch_size,
            self.num_tasks,
            trials_per_task,
        ):
            raise ValueError(
                "trial_slot_mask must match the [B,T,K] trial axes"
            )
        else:
            trial_slot_mask = trial_slot_mask.to(
                device=trial_features.device, dtype=torch.bool
            )
        if not torch.equal(trial_slot_mask.any(dim=2), task_present_mask):
            raise ValueError(
                "task_present_mask must equal trial_slot_mask.any(dim=2)"
            )
        has_incomplete_task_slots = not bool(torch.all(trial_slot_mask).item())
        if has_incomplete_task_slots:
            if self.cartesian_mask_cls_embeddings is None:
                raise ValueError(
                    "Cartesian task-CLS pooling requires every task to be present "
                    "unless learned missing-task CLS embeddings are enabled"
                )
            subject_outputs = []
            combination_counts = []
            for subject_index in range(batch_size):
                per_task_features = [
                    trial_features[subject_index, task_position][
                        trial_slot_mask[subject_index, task_position]
                    ]
                    for task_position in range(self.num_tasks)
                ]
                subject_demographics = (
                    demographic_features[subject_index : subject_index + 1]
                    if demographic_features is not None
                    else None
                )
                output = self.classify_all_cartesian_task_features(
                    per_task_features, subject_demographics
                )
                subject_outputs.append(output["subject_logits"])
                combination_counts.append(int(output["combination_count"]))
            subject_logits = torch.cat(subject_outputs, dim=0)
            # Keep the complete parameter in every rank's graph even when a
            # particular local batch contains no fully missing task. This is a
            # zero-valued dependency, but prevents DDP unused-parameter stalls.
            subject_logits = subject_logits + (
                0.0 * self.cartesian_mask_cls_embeddings.sum()
            )
            return {
                "subject_logits": subject_logits,
                "combination_counts": torch.tensor(
                    combination_counts,
                    dtype=torch.long,
                    device=trial_features.device,
                ),
                "trial_features": trial_features,
            }
        normalized = self.cartesian_feature_norm(trial_features)
        hidden_width = self.cartesian_hidden.out_features
        product_axes = [1] * self.num_tasks
        preactivation = self.cartesian_hidden.bias.reshape(
            1, *product_axes, hidden_width
        )
        weight = self.cartesian_hidden.weight
        for task_position in range(self.num_tasks):
            start = task_position * feature_dim
            stop = start + feature_dim
            contribution = F.linear(
                normalized[:, task_position], weight[:, start:stop]
            )
            contribution_shape = (
                [batch_size]
                + [1] * task_position
                + [trials_per_task]
                + [1] * (self.num_tasks - task_position - 1)
                + [hidden_width]
            )
            preactivation = preactivation + contribution.reshape(
                *contribution_shape
            )
        if self.demographic_dim > 0:
            demographic_weight = weight[:, self.num_tasks * feature_dim :]
            demographic_contribution = F.linear(
                demographic_features, demographic_weight
            ).reshape(batch_size, *product_axes, hidden_width)
            preactivation = preactivation + demographic_contribution

        hidden = self.cartesian_activation(preactivation)
        hidden = self.cartesian_dropout(hidden)
        combination_logits = self._format_head_logits(
            self.cartesian_output(hidden)
        )
        combination_axes = tuple(range(1, self.num_tasks + 1))
        subject_logits = combination_logits.mean(dim=combination_axes)
        if self.cartesian_mask_cls_embeddings is not None:
            subject_logits = subject_logits + (
                0.0 * self.cartesian_mask_cls_embeddings.sum()
            )
        return {
            "subject_logits": subject_logits,
            "combination_logits": combination_logits,
            "trial_features": trial_features,
        }

    def classify_all_cartesian_task_features(
        self,
        task_features: list[torch.Tensor],
        demographic_features: torch.Tensor | None = None,
        *,
        pair_chunk_size: int = 64,
    ) -> dict[str, torch.Tensor | int]:
        """Exactly average all Cartesian logits for one variable-size subject.

        Evaluation can contain tens of trials per task, making a dense
        [N0,N1,N2,N3,H] tensor unnecessarily large. Pairwise task contributions
        are therefore combined in bounded chunks while retaining the exact
        full-product mean.
        """
        if self.task_pooling != "cartesian_task_cls" or self.num_tasks != 4:
            raise ValueError(
                "All-trial Cartesian evaluation requires four cartesian task slots"
            )
        if len(task_features) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task feature tensors")
        if pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        resolved_features = []
        for task_position, features in enumerate(task_features):
            if (
                features.ndim != 2
                or features.shape[1] != self.bert.d_model
            ):
                raise ValueError(
                    "Every task feature tensor must have shape [N,D] with N>=0"
                )
            if features.shape[0] == 0:
                if self.cartesian_mask_cls_embeddings is None:
                    raise ValueError(
                        "A Cartesian task is missing and no learned mask-CLS "
                        "embedding is enabled"
                    )
                features = self.cartesian_mask_cls_embeddings[
                    task_position : task_position + 1
                ]
            resolved_features.append(features)
        if self.demographic_dim > 0:
            if demographic_features is None or demographic_features.shape != (
                1,
                self.demographic_dim,
            ):
                raise ValueError(
                    "demographic_features must have shape "
                    f"[1,{self.demographic_dim}]"
                )
        elif demographic_features is not None:
            raise ValueError("Demographics were provided to an eye-only model")

        normalized = [
            self.cartesian_feature_norm(features)
            for features in resolved_features
        ]
        feature_dim = self.bert.d_model
        weight = self.cartesian_hidden.weight
        contributions = []
        for task_position, features in enumerate(normalized):
            start = task_position * feature_dim
            stop = start + feature_dim
            contributions.append(F.linear(features, weight[:, start:stop]))
        left_pairs = (
            contributions[0][:, None, :] + contributions[1][None, :, :]
        ).reshape(-1, self.cartesian_hidden.out_features)
        right_pairs = (
            contributions[2][:, None, :] + contributions[3][None, :, :]
        ).reshape(-1, self.cartesian_hidden.out_features)
        base = self.cartesian_hidden.bias
        if self.demographic_dim > 0:
            demographic_weight = weight[:, self.num_tasks * feature_dim :]
            base = base + F.linear(
                demographic_features, demographic_weight
            ).squeeze(0)

        logit_sum = None
        for start in range(0, left_pairs.shape[0], pair_chunk_size):
            left = left_pairs[start : start + pair_chunk_size]
            preactivation = (
                base[None, None, :]
                + left[:, None, :]
                + right_pairs[None, :, :]
            )
            hidden = self.cartesian_activation(preactivation)
            hidden = self.cartesian_dropout(hidden)
            logits = self._format_head_logits(self.cartesian_output(hidden))
            # Accumulate millions of evaluation logits in FP32 even when the
            # head itself runs under BF16 autocast.
            chunk_sum = logits.float().sum(dim=(0, 1))
            logit_sum = chunk_sum if logit_sum is None else logit_sum + chunk_sum
        combination_count = int(left_pairs.shape[0] * right_pairs.shape[0])
        if logit_sum is None:
            raise RuntimeError("Cartesian evaluation produced no combinations")
        subject_logits = (logit_sum / combination_count).unsqueeze(0)
        return {
            "subject_logits": subject_logits,
            "combination_count": combination_count,
        }

    def aggregate_task_logits(
        self,
        task_logits: torch.Tensor,
        task_present_mask: torch.Tensor,
        demographic_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Combine one binary scalar or class-logit vector per task."""
        expected_ndim = 2 if self.output_dim == 1 else 3
        if (
            task_logits.ndim != expected_ndim
            or task_logits.shape[1] != self.num_tasks
            or (self.output_dim > 1 and task_logits.shape[2] != self.output_dim)
        ):
            suffix = "" if self.output_dim == 1 else f",{self.output_dim}"
            raise ValueError(
                f"task_logits must have shape [B,{self.num_tasks}{suffix}]"
            )
        if task_present_mask.shape != task_logits.shape[:2]:
            raise ValueError("task_present_mask must match task-logit batch/task axes")
        if not torch.all(task_present_mask.any(dim=1)):
            raise ValueError("Every subject must contain at least one task")
        if self.task_pooling in {"shared_head_mean", "shared_head_residual"}:
            task_weights = task_present_mask.to(task_logits.dtype)
            task_weights = task_weights / task_weights.sum(dim=1, keepdim=True)
        elif self.task_pooling == "shared_head_constrained":
            bounded_logits = self.task_residual_logit_scale * torch.tanh(
                self.task_weight_residual
            )
            weight_logits = bounded_logits.unsqueeze(0).expand(
                task_logits.shape[0], -1
            )
            weight_logits = weight_logits.masked_fill(
                ~task_present_mask, torch.finfo(weight_logits.dtype).min
            )
            task_weights = torch.softmax(weight_logits, dim=1)
        elif self.task_pooling == "shared_head_softmax":
            weight_logits = self.task_weight_logits.unsqueeze(0).expand(
                task_logits.shape[0], -1
            )
            weight_logits = weight_logits.masked_fill(
                ~task_present_mask, torch.finfo(weight_logits.dtype).min
            )
            task_weights = torch.softmax(weight_logits, dim=1)
        else:
            raise ValueError("Task-logit aggregation requires a shared task head")
        weighted = (
            task_logits * task_weights
            if self.output_dim == 1
            else task_logits * task_weights.unsqueeze(-1)
        )
        base_subject_logits = weighted.sum(dim=1)
        result = {
            "subject_logits": base_subject_logits,
            "task_logits": task_logits,
            "task_weights": task_weights,
        }
        if self.task_pooling != "shared_head_residual":
            return self._add_demographics(result, demographic_features)

        if demographic_features is None:
            raise ValueError("Late residual fusion requires demographic features")
        if demographic_features.ndim != 2 or demographic_features.shape != (
            task_logits.shape[0], self.demographic_dim
        ):
            raise ValueError(
                "demographic_features must have shape "
                f"[B,{self.demographic_dim}]"
            )
        if (
            not self.residual_include_task_mask
            and not torch.all(task_present_mask)
        ):
            raise ValueError(
                "Residual fusion without task-mask inputs requires every task "
                "to be present"
            )
        task_mask = task_present_mask.to(task_logits.dtype)
        masked_task_logits = task_logits.masked_fill(
            ~task_present_mask.unsqueeze(-1)
            if self.output_dim > 1
            else ~task_present_mask,
            0.0,
        )
        residual_input_parts = [
            masked_task_logits.reshape(task_logits.shape[0], -1)
        ]
        if self.residual_include_task_mask:
            residual_input_parts.append(task_mask)
        residual_input_parts.append(
            demographic_features.to(
                device=task_logits.device, dtype=task_logits.dtype
            )
        )
        residual_inputs = torch.cat(residual_input_parts, dim=1)
        residual_logits = self._format_head_logits(
            self.cross_task_residual_head(residual_inputs)
        )
        if self.num_tasks == 1:
            coverage_gate = torch.zeros_like(task_weights[:, 0])
        else:
            coverage_gate = (
                task_mask.sum(dim=1) - 1.0
            ) / float(self.num_tasks - 1)
            coverage_gate = coverage_gate.clamp_(0.0, 1.0)
        gated_residual_logits = (
            residual_logits * coverage_gate
            if self.output_dim == 1
            else residual_logits * coverage_gate.unsqueeze(-1)
        )
        result.update({
            "subject_logits": base_subject_logits + gated_residual_logits,
            "base_subject_logits": base_subject_logits,
            "residual_logits": residual_logits,
            "residual_coverage_gate": coverage_gate,
            "gated_residual_logits": gated_residual_logits,
            "residual_inputs": residual_inputs,
        })
        return result

    def classify_trial_features(
        self,
        trial_features: torch.Tensor,
        task_present_mask: torch.Tensor,
        demographic_features: torch.Tensor | None = None,
        trial_slot_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Classify each trial, then mean-pool logits within every task."""
        if trial_features.ndim != 4 or trial_features.shape[1] != self.num_tasks:
            raise ValueError(
                f"trial_features must have shape [B,{self.num_tasks},K,D]"
            )
        if trial_features.shape[-1] != self.bert.d_model:
            raise ValueError("trial feature width does not match BERT d_model")
        batch_size, _, trials_per_task, feature_dim = trial_features.shape
        if trials_per_task <= 0:
            raise ValueError("Every task must contain at least one trial")
        if task_present_mask.shape != (batch_size, self.num_tasks):
            raise ValueError(
                f"task_present_mask must have shape [B,{self.num_tasks}]"
            )
        if trial_slot_mask is None:
            trial_slot_mask = task_present_mask.unsqueeze(-1).expand(
                -1, -1, trials_per_task
            )
        elif trial_slot_mask.shape != (
            batch_size, self.num_tasks, trials_per_task
        ):
            raise ValueError("trial_slot_mask must match the [B,T,K] trial axes")
        else:
            trial_slot_mask = trial_slot_mask.to(
                device=trial_features.device, dtype=torch.bool
            )
        if not torch.equal(trial_slot_mask.any(dim=2), task_present_mask):
            raise ValueError(
                "task_present_mask must equal trial_slot_mask.any(dim=2)"
            )
        flattened = trial_features.reshape(-1, feature_dim)
        repeated_demographics = None
        if self.demographic_fusion == "concat_task_cls":
            if demographic_features is None:
                raise ValueError("CLS concatenation requires demographic features")
            if demographic_features.shape != (batch_size, self.demographic_dim):
                raise ValueError(
                    f"demographic_features must have shape [B,{self.demographic_dim}]"
                )
            repeated_demographics = (
                demographic_features[:, None, None, :]
                .expand(-1, self.num_tasks, trials_per_task, -1)
                .reshape(-1, self.demographic_dim)
            )
        flat_trial_logits = self.classify_individual_trial_features(
            flattened, repeated_demographics
        )
        if self.output_dim == 1:
            trial_logits = flat_trial_logits.reshape(
                batch_size, self.num_tasks, trials_per_task
            )
        else:
            trial_logits = flat_trial_logits.reshape(
                batch_size, self.num_tasks, trials_per_task, self.output_dim
            )
        slot_weights = trial_slot_mask.to(trial_logits.dtype)
        denominator = slot_weights.sum(dim=2).clamp_min(1.0)
        if self.output_dim == 1:
            task_logits = (trial_logits * slot_weights).sum(dim=2) / denominator
        else:
            task_logits = (
                trial_logits * slot_weights.unsqueeze(-1)
            ).sum(dim=2) / denominator.unsqueeze(-1)
        task_logits = task_logits.masked_fill(
            ~task_present_mask.unsqueeze(-1)
            if self.output_dim > 1
            else ~task_present_mask,
            0.0,
        )
        result = self.aggregate_task_logits(
            task_logits, task_present_mask, demographic_features
        )
        result.update({
            "trial_logits": trial_logits,
            "trial_features": trial_features,
            "trial_slot_mask": trial_slot_mask,
        })
        return result

    def group_trial_features(
        self,
        cls_features: torch.Tensor,
        *,
        num_subjects: int,
        trials_per_task: int,
    ) -> torch.Tensor:
        """Restore the canonical flattened [subject, task, trial] axes."""
        expected = int(num_subjects) * self.num_tasks * int(trials_per_task)
        if cls_features.ndim != 2 or cls_features.shape[0] != expected:
            raise ValueError(
                f"Expected [{expected},D] flattened trial features, got "
                f"{tuple(cls_features.shape)}"
            )
        return cls_features.reshape(
            int(num_subjects), self.num_tasks, int(trials_per_task), -1
        )

    def pool_trial_features(
        self,
        cls_features: torch.Tensor,
        *,
        num_subjects: int,
        trials_per_task: int,
    ) -> torch.Tensor:
        """Mean-pool the fixed trial axis independently within every task."""
        grouped = self.group_trial_features(
            cls_features,
            num_subjects=int(num_subjects),
            trials_per_task=int(trials_per_task),
        )
        mean = grouped.mean(dim=2)
        if self.task_pooling != "concat_task_mean_std":
            return mean
        # Population standard deviation matches evaluation's all-trial
        # sufficient-statistic aggregation and captures within-task instability.
        variance = (grouped.square().mean(dim=2) - mean.square()).clamp_min(0.0)
        return torch.cat((mean, variance.sqrt()), dim=-1)

    def forward(
        self,
        *,
        stim_patches: torch.Tensor,
        eye_patches: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor,
        task_ids: torch.Tensor,
        num_subjects: int,
        trials_per_task: int,
        task_present_mask: torch.Tensor,
        trial_slot_mask: torch.Tensor | None = None,
        subject_mixup_permutation: torch.Tensor | None = None,
        subject_mixup_lambda: float | None = None,
        demographic_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_features = self.encode_trials(
            stim_patches, eye_patches, pad_mask, eye_nonmissing_frac, task_ids
        )
        if self.trial_pooling == "logit_mean":
            if subject_mixup_permutation is not None:
                raise ValueError("Feature mixup is incompatible with logit_mean pooling")
            grouped_trials = self.group_trial_features(
                cls_features,
                num_subjects=int(num_subjects),
                trials_per_task=int(trials_per_task),
            )
            return self.classify_trial_features(
                grouped_trials,
                task_present_mask,
                demographic_features,
                trial_slot_mask=trial_slot_mask,
            )
        if self.trial_pooling == "cartesian_logit_mean":
            if subject_mixup_permutation is not None:
                raise ValueError(
                    "Feature mixup is incompatible with Cartesian pooling"
                )
            grouped_trials = self.group_trial_features(
                cls_features,
                num_subjects=int(num_subjects),
                trials_per_task=int(trials_per_task),
            )
            return self.classify_cartesian_trial_features(
                grouped_trials,
                task_present_mask,
                demographic_features,
                trial_slot_mask=trial_slot_mask,
            )
        task_features = self.pool_trial_features(
            cls_features,
            num_subjects=int(num_subjects),
            trials_per_task=int(trials_per_task),
        )
        if subject_mixup_permutation is None:
            return self.classify_task_features(
                task_features, task_present_mask, demographic_features
            )
        if subject_mixup_lambda is None or not 0.0 <= subject_mixup_lambda <= 1.0:
            raise ValueError("subject_mixup_lambda must be in [0,1]")
        if subject_mixup_permutation.shape != (int(num_subjects),):
            raise ValueError("subject_mixup_permutation has an invalid shape")
        if not torch.all(task_present_mask):
            raise ValueError("Subject feature mixup requires every task to be present")
        permutation = subject_mixup_permutation.to(task_features.device)
        lam = float(subject_mixup_lambda)
        mixed_features = lam * task_features + (1.0 - lam) * task_features[permutation]
        mixed_demographics = None
        if demographic_features is not None:
            mixed_demographics = (
                lam * demographic_features
                + (1.0 - lam) * demographic_features[permutation]
            )
        return self.classify_task_features(
            mixed_features, task_present_mask, mixed_demographics
        )
