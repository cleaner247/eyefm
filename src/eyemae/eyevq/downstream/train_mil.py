#!/usr/bin/env python3
"""Train EyeVQ-BERT with subject-level multi-instance supervision.

Training uses fixed-shape subject bags and can either require all tasks or mask
missing tasks. A subject is owned by exactly one rank and appears at most once
per epoch. Validation/test apply the configured feature, trial-logit, or
Cartesian task-tuple aggregation rule without treating within-subject instances
as independent samples.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from eyemae.data import load_area_stats, read_packed_index
from eyemae.downstream_data import (
    PackedDownstreamDataset,
    collate_downstream_trials,
    packed_row_has_usable_eye,
)
from eyemae.downstream_metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    sigmoid,
    softmax,
    write_prediction_csv,
)
from eyemae.eyevq.config import CONFIG_VERSION, load_bert_checkpoint
from eyemae.eyevq.artifacts import build_run_identity
from eyemae.eyevq.downstream.mil_data import (
    TASK_IDS,
    TASK_NAMES,
    SubjectBagDataset,
    collate_subject_bags,
)
from eyemae.eyevq.downstream.demographics import (
    AGE_ENCODINGS,
    EDUCATION_CATEGORIES,
    SEX_CATEGORIES,
    encode_subject_demographics,
    fit_demographic_spec,
    read_demographic_index,
)
from eyemae.eyevq.downstream.mil_model import EyeVQSubjectMIL
from eyemae.eyevq.downstream.mil_sampler import DistributedSubjectEpochSampler
from eyemae.eyevq.downstream.split_audit import audit_split_rows, assert_clean_splits
from eyemae.eyevq.downstream.train import (
    get_encoder_head_lrs,
    register_cleanup,
    setup_distributed,
    setup_logging,
)
from eyemae.utils import atomic_torch_save, set_seed, write_json


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_contiguous_trial_views(
    tensor: torch.Tensor,
    *,
    num_subjects: int,
    num_tasks: int,
    trials_per_task: int,
    num_views: int,
) -> torch.Tensor:
    """Split each task bag into disjoint views and return view-major batches."""
    if num_views <= 0 or trials_per_task % num_views:
        raise ValueError("trials_per_task must be divisible by num_trial_views")
    expected = num_subjects * num_tasks * trials_per_task
    if tensor.shape[0] != expected:
        raise ValueError(f"Expected {expected} trials, got {tensor.shape[0]}")
    per_view = trials_per_task // num_views
    shaped = tensor.reshape(
        num_subjects, num_tasks, num_views, per_view, *tensor.shape[1:]
    )
    order = (2, 0, 1, 3, *range(4, shaped.ndim))
    return shaped.permute(order).reshape(expected, *tensor.shape[1:])


def aggregate_trial_view_task_logits(
    task_logits: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Aggregate disjoint K-trial views within each subject and task."""
    if task_logits.ndim < 3:
        raise ValueError("task_logits must have [view,subject,task,...] axes")
    if mode == "task_mean":
        return task_logits.mean(dim=0)
    if mode == "task_trimmed_mean":
        if task_logits.shape[0] < 4:
            raise ValueError("task_trimmed_mean requires at least four views")
        ordered = task_logits.sort(dim=0).values
        return ordered[1:-1].mean(dim=0)
    raise ValueError("trial-view aggregation must be task_mean or task_trimmed_mean")


def deranged_permutation(size: int, device: torch.device) -> torch.Tensor:
    """Return a random permutation with no subject paired with itself."""
    if size < 2:
        raise ValueError("Subject feature mixup requires at least two subjects per GPU")
    order = torch.randperm(size, device=device)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    if torch.any(permutation == torch.arange(size, device=device)):
        raise RuntimeError("Failed to construct a deranged permutation")
    return permutation


def task_coverage_weights(
    task_present_mask: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Return one reliability weight per subject from its available tasks."""
    if task_present_mask.ndim != 2:
        raise ValueError("task_present_mask must have shape [B,T]")
    present_count = task_present_mask.sum(dim=1)
    if torch.any(present_count <= 0):
        raise ValueError("Every weighted subject must contain at least one task")
    if mode == "none":
        return torch.ones(
            task_present_mask.shape[0],
            dtype=torch.float32,
            device=task_present_mask.device,
        )
    if mode == "present_fraction":
        return present_count.to(torch.float32) / float(task_present_mask.shape[1])
    raise ValueError(
        "train.task_coverage_loss_weighting must be none or present_fraction"
    )


def ddp_global_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute a DDP-correct weighted mean without reducing autograd tensors.

    DDP averages rank gradients. Scaling each local numerator by
    ``world_size/global_denominator`` therefore produces exactly the gradient
    of the global weighted mean even when task coverage or class composition
    differs between ranks.
    """
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    local_denominator = weights.detach().sum()
    global_denominator = local_denominator.clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if float(global_denominator.item()) <= 0:
        raise ValueError("Weighted loss denominator must be positive")
    return (values * weights).sum() * (
        float(world_size) / global_denominator
    )


def task_coverage_weighted_supervised_loss(
    view_logits: torch.Tensor,
    labels: torch.Tensor,
    task_present_mask: torch.Tensor,
    *,
    label_type: str,
    coverage_mode: str,
    pos_weight: torch.Tensor | None,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Apply class and task-coverage weights once per subject."""
    if view_logits.shape[1] != labels.shape[0]:
        raise ValueError("view logits and labels disagree on subject count")
    coverage = task_coverage_weights(task_present_mask, mode=coverage_mode)
    if label_type == "binary":
        per_view = F.binary_cross_entropy_with_logits(
            view_logits,
            labels.unsqueeze(0).expand_as(view_logits),
            pos_weight=pos_weight,
            reduction="none",
        )
        per_subject = per_view.mean(dim=0)
        effective_weights = coverage.to(per_subject.dtype)
    elif label_type == "multiclass":
        if class_weights is None:
            raise ValueError("Multiclass loss requires class_weights")
        num_views, _, num_classes = view_logits.shape
        per_subject = F.cross_entropy(
            view_logits.reshape(-1, num_classes),
            labels.unsqueeze(0).expand(num_views, -1).reshape(-1),
            reduction="none",
        ).reshape(num_views, -1).mean(dim=0)
        effective_weights = coverage.to(per_subject.dtype) * class_weights.index_select(
            0, labels
        ).to(per_subject.dtype)
    else:
        raise ValueError("label_type must be binary or multiclass")
    return ddp_global_weighted_mean(per_subject, effective_weights)


def masked_auxiliary_task_loss(
    task_logits: torch.Tensor,
    labels: torch.Tensor,
    task_present_mask: torch.Tensor,
    *,
    label_type: str,
    pos_weight: torch.Tensor | None,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Compute optional per-task supervision without scoring missing tasks."""
    num_views, num_subjects, num_tasks = task_logits.shape[:3]
    if task_present_mask.shape != (num_subjects, num_tasks):
        raise ValueError("task_present_mask does not match auxiliary task logits")
    present = task_present_mask.unsqueeze(0).expand(num_views, -1, -1)
    if label_type == "binary":
        expanded_labels = labels.view(1, -1, 1).expand_as(task_logits)
        values = F.binary_cross_entropy_with_logits(
            task_logits,
            expanded_labels,
            pos_weight=pos_weight,
            reduction="none",
        )
        weights = present.to(values.dtype)
    elif label_type == "multiclass":
        if class_weights is None:
            raise ValueError("Multiclass loss requires class_weights")
        num_classes = task_logits.shape[-1]
        expanded_labels = labels.view(1, -1, 1).expand(
            num_views, -1, num_tasks
        )
        values = F.cross_entropy(
            task_logits.reshape(-1, num_classes),
            expanded_labels.reshape(-1),
            reduction="none",
        ).reshape(num_views, num_subjects, num_tasks)
        weights = present.to(values.dtype) * class_weights.index_select(
            0, labels
        ).view(1, -1, 1).to(values.dtype)
    else:
        raise ValueError("label_type must be binary or multiclass")
    return ddp_global_weighted_mean(values, weights)


def _downstream_cfg(
    data_cfg: dict[str, Any],
    bert_cfg: dict[str, Any],
    label_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_label_cfg = dict(
        label_cfg or {"type": "binary", "num_classes": 2}
    )
    resolved_label_cfg.setdefault("missing_value", 2)
    resolved_label_cfg.setdefault("blink_value", 1)
    resolved_label_cfg.setdefault("nonblink_value", 0)
    return {
        "data": {
            "data_dir": data_cfg["data_dir"],
            "format": "packed_mmap",
            "require_any_eye_keep": bool(
                data_cfg.get("require_any_eye_keep", False)
            ),
            "max_open_shards_per_worker": int(data_cfg.get("max_open_shards_per_worker", 44)),
            "validate_offsets": bool(data_cfg.get("validate_offsets", False)),
        },
        "patch": dict(bert_cfg["patch"]),
        "area": {
            "stats_path": data_cfg["area_stats_path"],
            "use_log1p": bool(bert_cfg.get("area", {}).get("use_log1p", True)),
            "per_eye": bool(bert_cfg.get("area", {}).get("per_eye", False)),
            "clip": float(bert_cfg.get("area", {}).get("clip", 5.0)),
            "eps": float(bert_cfg.get("area", {}).get("eps", 1e-6)),
            "mad_scale": float(bert_cfg.get("area", {}).get("mad_scale", 1.4826)),
            "mad_floor": float(bert_cfg.get("area", {}).get("mad_floor", 0.02)),
            "min_subject_valid_frames": int(
                bert_cfg.get("area", {}).get("min_subject_valid_frames", 10_000)
            ),
        },
        "normalization": {"x_clip_deg": 30.0, "y_clip_deg": 20.0},
        "label": resolved_label_cfg,
        "attention": {
            "min_nonmissing_frac_for_eye_token": float(
                bert_cfg.get("vq", {}).get("min_nonmissing_frac", 0.50)
            )
        },
    }


def validate_mci_area_stats(
    area_stats: dict[str, Any],
    *,
    mode: str = "bert_pretraining_global",
    area_stats_path: str | Path | None = None,
    bert_area_stats_path: str | Path | None = None,
) -> None:
    """Require normalization that exactly matches the BERT training regime."""
    if mode == "per_subject_transductive":
        problems = []
        subjects = area_stats.get("subjects", {})
        if not subjects:
            problems.append("per-subject mode requires non-empty subject statistics")
        if area_stats_path is None or bert_area_stats_path is None:
            problems.append("per-subject mode requires both downstream and BERT stats paths")
        elif Path(area_stats_path).resolve() != Path(bert_area_stats_path).resolve():
            problems.append("downstream area-stat path does not match the BERT checkpoint")
        source = area_stats.get("source", {})
        if source.get("scope") != "all_pretraining_subjects_unlabeled":
            problems.append("source.scope must be all_pretraining_subjects_unlabeled")
        if source.get("normalization_scope") != "per_subject_eye_median_mad":
            problems.append("source.normalization_scope must be per_subject_eye_median_mad")
        normalization = source.get("normalization", {})
        if normalization.get("transform") != "identity":
            problems.append("source.normalization.transform must be identity")
        if normalization.get("center") != "per_subject_eye_median":
            problems.append("source.normalization.center must be per_subject_eye_median")
        if normalization.get("scale") != "per_subject_eye_raw_mad":
            problems.append("source.normalization.scale must be per_subject_eye_raw_mad")
        if not {"left", "right"} <= set(area_stats.get("global_by_eye", {})):
            problems.append("global_by_eye must contain left/right")
        if float(normalization.get("mad_to_sigma", 0.0)) != 1.4826:
            problems.append("source.normalization.mad_to_sigma must be 1.4826")
        if source.get("invalid_eye_policy") != "trial_final_keep_then_frame_qc":
            problems.append("source.invalid_eye_policy must enforce trial final_keep")
        if problems:
            raise ValueError("Invalid per-subject MCI area statistics: " + "; ".join(problems))
        return
    if mode != "bert_pretraining_global":
        raise ValueError(f"Unsupported data.area_stats_mode: {mode}")

    # Default production mode: require the frozen global normalization used by
    # the original BERT encoder.
    source = area_stats.get("source", {})
    problems = []
    if source.get("purpose") != "mci_downstream_frozen_copy":
        problems.append("source.purpose must be mci_downstream_frozen_copy")
    if source.get("normalization_reference") != "bert_pretraining":
        problems.append("source.normalization_reference must be bert_pretraining")
    if source.get("recomputed_on_mci") is not False:
        problems.append("source.recomputed_on_mci must be false")
    expected_sha256 = "71ee08a126ff2e18dd84a69ac1f37cc6387cfdac9dc2665525740d630f524807"
    if source.get("upstream_sha256") != expected_sha256:
        problems.append("upstream_sha256 does not match the BERT training area-stat")
    if area_stats.get("subjects"):
        problems.append("unexpected subject-specific statistics")
    global_stats = area_stats.get("global", {})
    if float(global_stats.get("mad", 0.0)) <= 0:
        problems.append("global.mad must be positive")
    if problems:
        raise ValueError("Invalid MCI downstream area statistics: " + "; ".join(problems))


def subjects_below_task_minimum(
    rows: list[dict[str, str]],
    *,
    task_ids: tuple[int, ...],
    min_trials_per_task: int,
) -> tuple[str, ...]:
    """Return subjects for which any required task has fewer than K trials."""
    if min_trials_per_task <= 0:
        raise ValueError("min_trials_per_task must be positive")
    counts = Counter(
        (str(row["ml_subject_id"]), int(row["task_id"])) for row in rows
    )
    subjects = sorted({str(row["ml_subject_id"]) for row in rows})
    return tuple(
        subject
        for subject in subjects
        if any(
            counts[(subject, int(task_id))] < min_trials_per_task
            for task_id in task_ids
        )
    )


def filter_subjects_below_task_minimum(
    dataset: PackedDownstreamDataset,
    *,
    task_ids: tuple[int, ...],
    min_trials_per_task: int,
) -> tuple[str, ...]:
    """Remove an entire subject if any required task has fewer than K trials."""
    excluded = subjects_below_task_minimum(
        dataset.rows,
        task_ids=task_ids,
        min_trials_per_task=min_trials_per_task,
    )
    excluded_set = set(excluded)
    dataset.num_rows_before_subject_task_filter = len(dataset.rows)
    dataset.rows_excluded_subject_task_minimum = sum(
        str(row["ml_subject_id"]) in excluded_set for row in dataset.rows
    )
    if excluded_set:
        keep = [
            index
            for index, row in enumerate(dataset.rows)
            if str(row["ml_subject_id"]) not in excluded_set
        ]
        dataset.rows = [dataset.rows[index] for index in keep]
        dataset.labels = [dataset.labels[index] for index in keep]
    dataset.excluded_subjects_below_task_minimum = excluded
    dataset.subjects_with_missing_tasks = ()
    dataset.masked_subject_task_groups = ()
    if not dataset.rows:
        raise ValueError("No downstream subjects satisfy the per-task trial minimum")
    return excluded


def mask_tasks_below_task_minimum(
    dataset: PackedDownstreamDataset,
    *,
    task_ids: tuple[int, ...],
    min_trials_per_task: int,
) -> tuple[str, ...]:
    """Keep partial subjects, masking only task groups that cannot provide K trials.

    A subject is removed only when none of the configured tasks reaches the
    minimum.  Rows belonging to a present task are retained in full so final
    evaluation can continue to use every valid trial from that task.
    """
    if min_trials_per_task <= 0:
        raise ValueError("min_trials_per_task must be positive")
    canonical_tasks = tuple(int(task_id) for task_id in task_ids)
    counts = Counter(
        (str(row["ml_subject_id"]), int(row["task_id"]))
        for row in dataset.rows
        if int(row["task_id"]) in canonical_tasks
    )
    subjects = sorted({str(row["ml_subject_id"]) for row in dataset.rows})
    present = {
        subject: tuple(
            counts[(subject, task_id)] >= min_trials_per_task
            for task_id in canonical_tasks
        )
        for subject in subjects
    }
    excluded = tuple(
        subject for subject in subjects if not any(present[subject])
    )
    excluded_set = set(excluded)
    partial = tuple(
        subject
        for subject in subjects
        if subject not in excluded_set and not all(present[subject])
    )
    masked_groups = tuple(
        {
            "subject_key": subject,
            "task_id": task_id,
            "valid_trials": int(counts[(subject, task_id)]),
        }
        for subject in partial
        for task_id, is_present in zip(canonical_tasks, present[subject])
        if not is_present
    )

    dataset.num_rows_before_subject_task_filter = len(dataset.rows)
    keep = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row["ml_subject_id"]) not in excluded_set
        and counts[(str(row["ml_subject_id"]), int(row["task_id"]))]
        >= min_trials_per_task
    ]
    dataset.rows_excluded_subject_task_minimum = len(dataset.rows) - len(keep)
    dataset.rows = [dataset.rows[index] for index in keep]
    dataset.labels = [dataset.labels[index] for index in keep]
    dataset.excluded_subjects_below_task_minimum = excluded
    dataset.subjects_with_missing_tasks = partial
    dataset.masked_subject_task_groups = masked_groups
    if not dataset.rows:
        raise ValueError("No downstream subject contains any usable task")
    return excluded


def retain_all_available_partial_tasks(
    dataset: PackedDownstreamDataset,
    *,
    task_ids: tuple[int, ...],
    target_trials_per_task: int,
) -> tuple[str, ...]:
    """Retain every available trial and mask only genuinely absent tasks.

    Tasks containing between one and K-1 valid trials remain present and use
    all of those trials. A subject is removed only when none of the configured
    tasks contains a valid trial.
    """
    if target_trials_per_task <= 0:
        raise ValueError("target_trials_per_task must be positive")
    canonical_tasks = tuple(int(task_id) for task_id in task_ids)
    canonical_task_set = set(canonical_tasks)
    counts = Counter(
        (str(row["ml_subject_id"]), int(row["task_id"]))
        for row in dataset.rows
        if int(row["task_id"]) in canonical_task_set
    )
    subjects = sorted({
        str(row["ml_subject_id"])
        for row in dataset.rows
        if int(row["task_id"]) in canonical_task_set
    })
    present = {
        subject: tuple(
            counts[(subject, task_id)] > 0 for task_id in canonical_tasks
        )
        for subject in subjects
    }
    excluded = tuple(subject for subject in subjects if not any(present[subject]))
    excluded_set = set(excluded)
    partial = tuple(
        subject
        for subject in subjects
        if subject not in excluded_set and not all(present[subject])
    )
    masked_groups = tuple(
        {
            "subject_key": subject,
            "task_id": task_id,
            "valid_trials": 0,
        }
        for subject in partial
        for task_id, is_present in zip(canonical_tasks, present[subject])
        if not is_present
    )
    below_k_groups = tuple(
        {
            "subject_key": subject,
            "task_id": task_id,
            "valid_trials": int(counts[(subject, task_id)]),
        }
        for subject in subjects
        if subject not in excluded_set
        for task_id in canonical_tasks
        if 0 < counts[(subject, task_id)] < target_trials_per_task
    )

    dataset.num_rows_before_subject_task_filter = len(dataset.rows)
    keep = [
        index
        for index, row in enumerate(dataset.rows)
        if str(row["ml_subject_id"]) not in excluded_set
        and int(row["task_id"]) in canonical_task_set
    ]
    dataset.rows_excluded_subject_task_minimum = len(dataset.rows) - len(keep)
    dataset.rows = [dataset.rows[index] for index in keep]
    dataset.labels = [dataset.labels[index] for index in keep]
    dataset.excluded_subjects_below_task_minimum = excluded
    dataset.subjects_with_missing_tasks = partial
    dataset.masked_subject_task_groups = masked_groups
    dataset.subject_task_groups_below_k_used_all = below_k_groups
    if not dataset.rows:
        raise ValueError("No downstream subject contains any usable task")
    return excluded


def apply_task_availability_policy(
    dataset: PackedDownstreamDataset,
    *,
    task_ids: tuple[int, ...],
    min_trials_per_task: int,
    require_all_tasks: bool,
    sample_all_available_below_k: bool = False,
) -> tuple[str, ...]:
    """Apply either strict complete-task filtering or partial-task masking."""
    if require_all_tasks:
        if sample_all_available_below_k:
            raise ValueError(
                "sample_all_available_below_k requires partial-task training"
            )
        return filter_subjects_below_task_minimum(
            dataset,
            task_ids=task_ids,
            min_trials_per_task=min_trials_per_task,
        )
    if sample_all_available_below_k:
        return retain_all_available_partial_tasks(
            dataset,
            task_ids=task_ids,
            target_trials_per_task=min_trials_per_task,
        )
    return mask_tasks_below_task_minimum(
        dataset,
        task_ids=task_ids,
        min_trials_per_task=min_trials_per_task,
    )


def _trial_loader(dataset, train_cfg: dict[str, Any]) -> DataLoader:
    workers = int(train_cfg.get("num_workers", 4)) if torch.cuda.is_available() else 0
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("eval_trial_batch_size", 16)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_downstream_trials,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 4)) if workers > 0 else None,
    )


def metrics_from_subject_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    threshold: float = 0.5,
    num_classes: int = 2,
) -> dict[str, float]:
    """Recompute subject metrics without another BERT evaluation."""
    labels = [int(row["label"]) for row in rows]
    if num_classes == 2:
        logits = [float(row["logit"]) for row in rows]
        return compute_binary_metrics(
            labels, logits, threshold=threshold, prefix=f"{split_name}/subject"
        )
    logits = [
        [float(row[f"logit_{class_id}"]) for class_id in range(num_classes)]
        for row in rows
    ]
    return compute_multiclass_metrics(
        labels,
        logits,
        num_classes=num_classes,
        prefix=f"{split_name}/subject",
    )


@torch.no_grad()
def evaluate_subjects(
    model: EyeVQSubjectMIL,
    loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
    threshold: float = 0.5,
    bf16: bool = False,
    subject_demographics: dict[str, torch.Tensor] | None = None,
    task_coverage_loss_weighting: str = "none",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate one deterministic subject prediction per retained subject."""
    model.eval()
    subjects: dict[str, dict[str, Any]] = {}

    for batch in loader:
        content = batch["content"].to(device, non_blocking=True).transpose(-1, -2).contiguous()
        stim = batch["stim"].to(device, non_blocking=True).transpose(-1, -2).contiguous()
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        nonmissing = batch["eye_nonmissing_frac"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        with _autocast(device, bf16):
            cls_features = model.encode_trials(stim, content, pad_mask, nonmissing, task_ids)
        cls_features = cls_features.float().cpu()

        for index, subject_key in enumerate(batch["subject_key"]):
            subject_key = str(subject_key)
            label = int(batch["label"][index].item())
            task_id = int(batch["task_id"][index].item())
            if task_id not in TASK_IDS:
                continue
            entry = subjects.setdefault(
                subject_key,
                {
                    "label": label,
                    "task_sums": [None for _ in TASK_IDS],
                    "task_square_sums": [None for _ in TASK_IDS],
                    "task_features": [[] for _ in TASK_IDS],
                    "task_counts": [0 for _ in TASK_IDS],
                },
            )
            if entry["label"] != label:
                raise ValueError(f"Subject has inconsistent labels: {subject_key}")
            feature = cls_features[index]
            if model.trial_pooling in {"logit_mean", "cartesian_logit_mean"}:
                entry["task_features"][task_id].append(feature.clone())
            else:
                previous = entry["task_sums"][task_id]
                entry["task_sums"][task_id] = (
                    feature.clone() if previous is None else previous + feature
                )
                square = feature.square()
                previous_square = entry["task_square_sums"][task_id]
                entry["task_square_sums"][task_id] = (
                    square.clone()
                    if previous_square is None
                    else previous_square + square
                )
            entry["task_counts"][task_id] += 1

    subject_keys = sorted(subjects)
    if not subject_keys:
        raise ValueError(f"No subjects were encoded for split={split_name}")
    present = torch.zeros(len(subject_keys), len(TASK_IDS), dtype=torch.bool)
    labels = torch.tensor(
        [subjects[key]["label"] for key in subject_keys],
        dtype=torch.float32 if model.num_classes == 2 else torch.long,
    )
    demographic_features = None
    if model.demographic_dim > 0:
        if subject_demographics is None:
            raise ValueError("Demographic model evaluation requires subject features")
        missing = [key for key in subject_keys if key not in subject_demographics]
        if missing:
            raise KeyError(f"Missing demographic features for subjects: {missing[:10]}")
        demographic_features = torch.stack(
            [subject_demographics[key] for key in subject_keys]
        ).to(device)
    if model.trial_pooling == "logit_mean":
        flat_features: list[torch.Tensor] = []
        flat_subject_indices: list[int] = []
        flat_task_indices: list[int] = []
        for subject_index, key in enumerate(subject_keys):
            entry = subjects[key]
            for task_id in TASK_IDS:
                trials = entry["task_features"][task_id]
                if trials:
                    present[subject_index, task_id] = True
                    flat_features.extend(trials)
                    flat_subject_indices.extend([subject_index] * len(trials))
                    flat_task_indices.extend([task_id] * len(trials))
        if not flat_features:
            raise ValueError(f"No trial features were collected for split={split_name}")
        subject_index_tensor = torch.tensor(flat_subject_indices, dtype=torch.long)
        task_index_tensor = torch.tensor(flat_task_indices, dtype=torch.long)
        flat_demographics = (
            demographic_features.index_select(
                0, subject_index_tensor.to(device)
            )
            if demographic_features is not None
            and model.demographic_fusion == "concat_task_cls"
            else None
        )
        with _autocast(device, bf16):
            flat_trial_logits = model.classify_individual_trial_features(
                torch.stack(flat_features).to(device), flat_demographics
            )
        flat_trial_logits = flat_trial_logits.float().cpu()
        flat_group_indices = subject_index_tensor * len(TASK_IDS) + task_index_tensor
        task_logit_sums = torch.zeros(
            len(subject_keys) * len(TASK_IDS),
            *(() if model.output_dim == 1 else (model.output_dim,)),
            dtype=torch.float32,
        )
        task_logit_sums.index_add_(0, flat_group_indices, flat_trial_logits)
        task_counts = torch.bincount(
            flat_group_indices, minlength=len(subject_keys) * len(TASK_IDS)
        ).to(torch.float32)
        count_denominator = task_counts.clamp_min(1.0)
        if model.output_dim > 1:
            count_denominator = count_denominator.unsqueeze(-1)
        task_logits_for_pooling = task_logit_sums / count_denominator
        task_logits_for_pooling = task_logits_for_pooling.reshape(
            len(subject_keys),
            len(TASK_IDS),
            *(() if model.output_dim == 1 else (model.output_dim,)),
        )
        with _autocast(device, bf16):
            output = model.aggregate_task_logits(
                task_logits_for_pooling.to(device),
                present.to(device),
                demographic_features,
            )
    elif model.trial_pooling == "cartesian_logit_mean":
        subject_logits = []
        combination_counts = []
        for subject_index, key in enumerate(subject_keys):
            entry = subjects[key]
            all_task_features = []
            for task_id in TASK_IDS:
                trials = entry["task_features"][task_id]
                if trials:
                    all_task_features.append(torch.stack(trials).to(device))
                    present[subject_index, task_id] = True
                else:
                    all_task_features.append(torch.empty(
                        0,
                        model.bert.d_model,
                        dtype=torch.float32,
                        device=device,
                    ))
            subject_demographic = (
                demographic_features[subject_index : subject_index + 1]
                if demographic_features is not None
                else None
            )
            with _autocast(device, bf16):
                subject_output = model.classify_all_cartesian_task_features(
                    all_task_features, subject_demographic
                )
            subject_logits.append(subject_output["subject_logits"])
            combination_counts.append(int(subject_output["combination_count"]))
        output = {
            "subject_logits": torch.cat(subject_logits, dim=0),
            "combination_counts": torch.tensor(combination_counts),
        }
    else:
        d_model = model.bert.d_model
        feature_dim = d_model * (
            2 if model.task_pooling == "concat_task_mean_std" else 1
        )
        features = torch.zeros(
            len(subject_keys), len(TASK_IDS), feature_dim, dtype=torch.float32
        )
        for subject_index, key in enumerate(subject_keys):
            entry = subjects[key]
            for task_id in TASK_IDS:
                count = int(entry["task_counts"][task_id])
                if count > 0:
                    mean = entry["task_sums"][task_id] / count
                    if model.task_pooling == "concat_task_mean_std":
                        mean_square = entry["task_square_sums"][task_id] / count
                        std = (mean_square - mean.square()).clamp_min(0.0).sqrt()
                        features[subject_index, task_id] = torch.cat((mean, std))
                    else:
                        features[subject_index, task_id] = mean
                    present[subject_index, task_id] = True
        with _autocast(device, bf16):
            output = model.classify_task_features(
                features.to(device), present.to(device), demographic_features
            )
    subject_logits = output["subject_logits"].float().cpu()
    task_logits = output.get("task_logits")
    task_weights = output.get("task_weights")
    eye_subject_logits = output.get("eye_subject_logits")
    demographic_logits = output.get("demographic_logits")
    base_subject_logits = output.get("base_subject_logits")
    residual_logits = output.get("residual_logits")
    residual_coverage_gate = output.get("residual_coverage_gate")
    gated_residual_logits = output.get("gated_residual_logits")
    combination_counts = output.get("combination_counts")
    if task_logits is not None:
        task_logits = task_logits.float().cpu()
    if task_weights is not None:
        task_weights = task_weights.float().cpu()
    if eye_subject_logits is not None:
        eye_subject_logits = eye_subject_logits.float().cpu()
    if demographic_logits is not None:
        demographic_logits = demographic_logits.float().cpu()
    if base_subject_logits is not None:
        base_subject_logits = base_subject_logits.float().cpu()
    if residual_logits is not None:
        residual_logits = residual_logits.float().cpu()
    if residual_coverage_gate is not None:
        residual_coverage_gate = residual_coverage_gate.float().cpu()
    if gated_residual_logits is not None:
        gated_residual_logits = gated_residual_logits.float().cpu()

    rows: list[dict[str, Any]] = []
    for index, key in enumerate(subject_keys):
        entry = subjects[key]
        row: dict[str, Any] = {
            "split": split_name,
            "subject_key": key,
            "label": int(labels[index].item()),
            "num_trials": int(sum(entry["task_counts"])),
            "num_present_tasks": int(present[index].sum().item()),
            "task_coverage_weight": float(
                task_coverage_weights(
                    present[index : index + 1],
                    mode=task_coverage_loss_weighting,
                )[0].item()
            ),
        }
        if combination_counts is not None:
            row["cartesian_combination_count"] = int(
                combination_counts[index].item()
            )
        if model.num_classes == 2:
            row["logit"] = float(subject_logits[index].item())
            row["prob"] = sigmoid(float(subject_logits[index].item()))
        else:
            logits = [
                float(subject_logits[index, class_id].item())
                for class_id in range(model.num_classes)
            ]
            probabilities = softmax(logits)
            row["pred"] = int(max(range(model.num_classes), key=probabilities.__getitem__))
            for class_id, (logit, probability) in enumerate(
                zip(logits, probabilities)
            ):
                row[f"logit_{class_id}"] = logit
                row[f"prob_{class_id}"] = probability
        for task_id, task_name in enumerate(TASK_NAMES):
            row[f"{task_name}_count"] = int(entry["task_counts"][task_id])
            if task_logits is not None and task_weights is not None:
                if model.num_classes == 2:
                    row[f"{task_name}_logit"] = float(
                        task_logits[index, task_id].item()
                    )
                else:
                    for class_id in range(model.num_classes):
                        row[f"{task_name}_logit_{class_id}"] = float(
                            task_logits[index, task_id, class_id].item()
                        )
                row[f"{task_name}_weight"] = float(task_weights[index, task_id].item())
        if eye_subject_logits is not None and demographic_logits is not None:
            if model.num_classes == 2:
                row["eye_logit"] = float(eye_subject_logits[index].item())
                row["demographic_logit"] = float(
                    demographic_logits[index].item()
                )
            else:
                for class_id in range(model.num_classes):
                    row[f"eye_logit_{class_id}"] = float(
                        eye_subject_logits[index, class_id].item()
                    )
                    row[f"demographic_logit_{class_id}"] = float(
                        demographic_logits[index, class_id].item()
                    )
        if (
            base_subject_logits is not None
            and residual_logits is not None
            and residual_coverage_gate is not None
            and gated_residual_logits is not None
        ):
            row["residual_coverage_gate"] = float(
                residual_coverage_gate[index].item()
            )
            if model.num_classes == 2:
                row["base_logit"] = float(base_subject_logits[index].item())
                row["residual_logit"] = float(residual_logits[index].item())
                row["gated_residual_logit"] = float(
                    gated_residual_logits[index].item()
                )
            else:
                for class_id in range(model.num_classes):
                    row[f"base_logit_{class_id}"] = float(
                        base_subject_logits[index, class_id].item()
                    )
                    row[f"residual_logit_{class_id}"] = float(
                        residual_logits[index, class_id].item()
                    )
                    row[f"gated_residual_logit_{class_id}"] = float(
                        gated_residual_logits[index, class_id].item()
                    )
        rows.append(row)

    metrics = metrics_from_subject_rows(
        rows,
        split_name=split_name,
        threshold=threshold,
        num_classes=model.num_classes,
    )
    per_subject_loss = (
        F.binary_cross_entropy_with_logits(
            subject_logits, labels, reduction="none"
        )
        if model.num_classes == 2
        else F.cross_entropy(subject_logits, labels, reduction="none")
    )
    coverage = task_coverage_weights(
        present, mode=task_coverage_loss_weighting
    ).cpu()
    loss = (per_subject_loss * coverage).sum() / coverage.sum()
    metrics[f"{split_name}/subject/loss"] = float(loss.item())
    metrics[f"{split_name}/subject/mean_task_coverage_weight"] = float(
        coverage.mean().item()
    )
    return metrics, rows


_TRANSFORMER_LAYER_PATTERN = re.compile(r"^bert\.transformer\.(\d+)\.")


def _uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    """Apply AdamW decay only to matrix-like non-normalization weights."""
    lowered = name.lower()
    if name == "cartesian_mask_cls_embeddings":
        return False
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return False
    if "norm" in lowered:
        return False
    return True


def build_optimizer_param_groups(
    model: EyeVQSubjectMIL,
    *,
    encoder_lr: float,
    head_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Create no-decay and layer-wise LR groups without duplicating parameters."""
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("layer_decay must be in (0, 1]")
    n_layers = len(model.bert.transformer)
    buckets: dict[tuple[str, float, float], list[torch.nn.Parameter]] = {}
    bucket_names: dict[tuple[str, float, float], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = (
            name.startswith("subject_head.")
            or name.startswith("task_head.")
            or name.startswith("task_projector.")
            or name.startswith("task_feature_norm.")
            or name.startswith("demographic_projection.")
            or name.startswith("cartesian_")
            or name.startswith("cross_task_residual_head.")
            or name.startswith("demographic_head.")
            or name == "task_weight_logits"
        )
        schedule = "head" if is_head else "encoder"
        if is_head:
            lr_scale = 1.0
        else:
            match = _TRANSFORMER_LAYER_PATTERN.match(name)
            if match:
                layer_index = int(match.group(1))
                lr_scale = layer_decay ** (n_layers - 1 - layer_index)
            elif name.startswith("bert.embed."):
                lr_scale = layer_decay ** n_layers
            else:
                # Final encoder norm and any downstream-used non-layer encoder
                # parameters follow the top transformer layer.
                lr_scale = 1.0
        decay = weight_decay if _uses_weight_decay(name, parameter) else 0.0
        key = (schedule, float(lr_scale), float(decay))
        buckets.setdefault(key, []).append(parameter)
        bucket_names.setdefault(key, []).append(name)

    groups: list[dict[str, Any]] = []
    for (schedule, lr_scale, decay), parameters in sorted(
        buckets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        base_lr = head_lr if schedule == "head" else encoder_lr
        groups.append({
            "params": parameters,
            "lr": base_lr * lr_scale,
            "weight_decay": decay,
            "schedule": schedule,
            "lr_scale": lr_scale,
            "parameter_names": bucket_names[(schedule, lr_scale, decay)],
        })
    if not groups:
        raise ValueError("No trainable parameters were found")
    return groups


def set_optimizer_step_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    encoder_lr: float,
    head_lr: float,
) -> None:
    for group in optimizer.param_groups:
        base_lr = head_lr if group["schedule"] == "head" else encoder_lr
        group["lr"] = base_lr * float(group["lr_scale"])


def update_early_stopping_counter(
    no_improve: int,
    *,
    val_improved: bool,
    global_step: int,
    min_steps: int,
) -> int:
    """Reset patience whenever raw validation AUROC sets a new best."""
    if global_step < min_steps:
        return int(no_improve)
    return 0 if val_improved else int(no_improve) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/eyevq/downstream_mci_subject_mil")
    parser.add_argument(
        "--bert-checkpoint",
        help="Override model.bert_checkpoint from the YAML configuration.",
    )
    parser.add_argument("--freeze-bottom-layers", type=int)
    parser.add_argument(
        "--unfreeze-embedding",
        action="store_true",
        help="Train all downstream-used BERT embedding parameters.",
    )
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--classifier-hidden", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--residual-hidden", type=int)
    parser.add_argument(
        "--classifier-head", choices=("mlp", "linear")
    )
    parser.add_argument("--trials-per-task", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--eval-trial-batch-size", type=int)
    parser.add_argument("--num-trial-views", type=int)
    parser.add_argument("--trial-view-consistency-weight", type=float)
    parser.add_argument(
        "--trial-pooling",
        choices=("feature_mean", "logit_mean", "cartesian_logit_mean"),
    )
    parser.add_argument("--train-index")
    parser.add_argument("--val-index")
    parser.add_argument("--test-index")
    parser.add_argument(
        "--enable-demographics",
        action="store_true",
        help="Fuse fold-safe age, education and sex features into the subject logit.",
    )
    parser.add_argument(
        "--demographic-fusion",
        choices=("additive_logit", "concat_task_cls", "late_residual"),
    )
    parser.add_argument(
        "--demographic-age-encoding", choices=AGE_ENCODINGS
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not load/evaluate the test split (required for hyperparameter search).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.bert_checkpoint is not None:
        cfg["model"]["bert_checkpoint"] = args.bert_checkpoint
    if args.freeze_bottom_layers is not None:
        cfg["model"]["freeze_bottom_layers"] = int(args.freeze_bottom_layers)
    if args.unfreeze_embedding:
        cfg["model"]["freeze_embedding"] = False
    if args.encoder_lr is not None:
        cfg["train"]["encoder_lr"] = float(args.encoder_lr)
        cfg["train"]["encoder_min_lr"] = float(args.encoder_lr) / 10.0
    if args.head_lr is not None:
        cfg["train"]["head_lr"] = float(args.head_lr)
        cfg["train"]["head_min_lr"] = float(args.head_lr) / 10.0
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.classifier_hidden is not None:
        cfg["model"]["classifier_hidden"] = int(args.classifier_hidden)
    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError("--dropout must be in [0, 1)")
        cfg["model"]["dropout"] = float(args.dropout)
    if args.residual_hidden is not None:
        cfg["model"]["residual_hidden"] = int(args.residual_hidden)
    if args.classifier_head is not None:
        cfg["model"]["classifier_head"] = args.classifier_head
    if args.trials_per_task is not None:
        cfg["mil"]["trials_per_task"] = int(args.trials_per_task)
    if args.early_stopping_patience is not None:
        cfg["train"]["early_stopping_patience_epochs"] = int(
            args.early_stopping_patience
        )
    if args.eval_trial_batch_size is not None:
        if args.eval_trial_batch_size <= 0:
            raise ValueError("--eval-trial-batch-size must be positive")
        cfg["train"]["eval_trial_batch_size"] = int(
            args.eval_trial_batch_size
        )
    if args.num_trial_views is not None:
        cfg["train"]["num_trial_views"] = int(args.num_trial_views)
    if args.trial_view_consistency_weight is not None:
        cfg["train"]["trial_view_consistency_weight"] = float(
            args.trial_view_consistency_weight
        )
    if args.trial_pooling is not None:
        cfg["mil"]["trial_pooling"] = args.trial_pooling
    if args.train_index is not None:
        cfg["data"]["train_index"] = args.train_index
    if args.val_index is not None:
        cfg["data"]["val_index"] = args.val_index
    if args.test_index is not None:
        cfg["data"]["test_index"] = args.test_index
    if args.enable_demographics:
        cfg.setdefault("demographics", {})["enabled"] = True
    if args.demographic_fusion is not None:
        cfg.setdefault("demographics", {})["fusion"] = args.demographic_fusion
    if args.demographic_age_encoding is not None:
        cfg.setdefault("demographics", {})["age_encoding"] = (
            args.demographic_age_encoding
        )
    output_dir = Path(args.output_dir)
    rank, world_size, _, device = setup_distributed()
    register_cleanup()
    logger = setup_logging(rank, output_dir)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    mil_cfg = cfg["mil"]
    label_cfg = cfg.setdefault("label", {"type": "binary", "num_classes": 2})
    label_type = str(label_cfg.get("type", "binary"))
    num_classes = int(label_cfg.get("num_classes", 2))
    if label_type == "binary":
        if num_classes != 2:
            raise ValueError("Binary Subject-MIL requires label.num_classes=2")
        selection_metric_name = "auroc"
    elif label_type == "multiclass":
        if num_classes < 3:
            raise ValueError("Multiclass Subject-MIL requires at least 3 classes")
        selection_metric_name = "macro_auroc_ovr"
    else:
        raise ValueError("label.type must be binary or multiclass")
    demographics_cfg = cfg.setdefault("demographics", {})
    demographics_enabled = bool(demographics_cfg.get("enabled", False))
    demographic_fusion = str(
        demographics_cfg.get("fusion", "additive_logit")
    )
    if demographics_enabled and demographic_fusion == "concat_task_cls":
        demographics_cfg.setdefault("projection_dim", 128)
    demographic_dim = (
        2 + len(SEX_CATEGORIES) + len(EDUCATION_CATEGORIES)
        if demographics_enabled
        else 0
    )
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed + rank)

    subjects_per_gpu = int(mil_cfg["subjects_per_gpu"])
    trials_per_task = int(mil_cfg["trials_per_task"])
    eligibility_min_trials = int(
        mil_cfg.get("eligibility_min_trials_per_task", trials_per_task)
    )
    evaluation_min_trials = int(
        mil_cfg.get("evaluation_min_trials_per_task", trials_per_task)
    )
    trial_pooling = str(mil_cfg.get("trial_pooling", "feature_mean"))
    num_trial_views = int(train_cfg.get("num_trial_views", 1))
    trial_view_aggregation = str(
        train_cfg.get("trial_view_aggregation", "independent_loss")
    )
    consistency_weight = float(train_cfg.get("trial_view_consistency_weight", 0.0))
    auxiliary_task_loss_weight = float(train_cfg.get("auxiliary_task_loss_weight", 0.0))
    feature_mixup_alpha = float(train_cfg.get("feature_mixup_alpha", 0.0))
    task_residual_l2 = float(train_cfg.get("task_residual_l2", 0.0))
    demographic_head_l2 = float(train_cfg.get("demographic_head_l2", 0.0))
    require_all_tasks = bool(mil_cfg.get("train_require_all_tasks", True))
    sample_all_available_below_k = bool(
        mil_cfg.get("sample_all_available_below_k", False)
    )
    missing_task_embedding = str(
        mil_cfg.get("missing_task_embedding", "none")
    )
    residual_include_task_mask = bool(
        model_cfg.get("residual_include_task_mask", True)
    )
    task_coverage_loss_weighting = str(
        train_cfg.get("task_coverage_loss_weighting", "none")
    )
    if subjects_per_gpu <= 0:
        raise ValueError("mil.subjects_per_gpu must be positive")
    if eligibility_min_trials < trials_per_task:
        raise ValueError(
            "mil.eligibility_min_trials_per_task cannot be smaller than trials_per_task"
        )
    if evaluation_min_trials <= 0 or evaluation_min_trials > trials_per_task:
        raise ValueError(
            "mil.evaluation_min_trials_per_task must be in [1,trials_per_task]"
        )
    if trial_pooling not in {
        "feature_mean", "logit_mean", "cartesian_logit_mean"
    }:
        raise ValueError(
            "mil.trial_pooling must be feature_mean, logit_mean, or "
            "cartesian_logit_mean"
        )
    if num_trial_views <= 0 or trials_per_task % num_trial_views:
        raise ValueError("mil.trials_per_task must be divisible by train.num_trial_views")
    if trial_view_aggregation not in {
        "independent_loss", "task_mean", "task_trimmed_mean"
    }:
        raise ValueError(
            "train.trial_view_aggregation must be independent_loss, task_mean, "
            "or task_trimmed_mean"
        )
    if trial_view_aggregation != "independent_loss" and num_trial_views <= 1:
        raise ValueError("Task-level trial-view aggregation requires multiple views")
    if trial_view_aggregation == "task_trimmed_mean" and num_trial_views < 4:
        raise ValueError("task_trimmed_mean requires at least four trial views")
    if consistency_weight < 0:
        raise ValueError("train.trial_view_consistency_weight must be non-negative")
    if auxiliary_task_loss_weight < 0:
        raise ValueError("train.auxiliary_task_loss_weight must be non-negative")
    if feature_mixup_alpha < 0:
        raise ValueError("train.feature_mixup_alpha must be non-negative")
    if task_residual_l2 < 0 or demographic_head_l2 < 0:
        raise ValueError("Constrained-head L2 penalties must be non-negative")
    if feature_mixup_alpha > 0 and num_trial_views != 1:
        raise ValueError("Feature mixup and multiple trial views cannot be combined")
    if feature_mixup_alpha > 0 and trial_pooling != "feature_mean":
        raise ValueError("Feature mixup requires feature_mean trial pooling")
    if feature_mixup_alpha > 0 and label_type != "binary":
        raise ValueError("Feature mixup is currently supported only for binary MIL")
    if feature_mixup_alpha > 0 and subjects_per_gpu < 2:
        raise ValueError("Feature mixup requires at least two subjects per GPU")
    if task_coverage_loss_weighting not in {"none", "present_fraction"}:
        raise ValueError(
            "train.task_coverage_loss_weighting must be none or present_fraction"
        )
    if not require_all_tasks and feature_mixup_alpha > 0:
        raise ValueError("Partial-task training is incompatible with feature mixup")
    if missing_task_embedding not in {"none", "learned_per_task"}:
        raise ValueError(
            "mil.missing_task_embedding must be none or learned_per_task"
        )
    if require_all_tasks and sample_all_available_below_k:
        raise ValueError(
            "mil.sample_all_available_below_k requires partial-task training"
        )
    if (
        mil_cfg.get("task_pooling") == "shared_head_residual"
        and not residual_include_task_mask
        and not require_all_tasks
    ):
        raise ValueError(
            "Residual fusion without task-mask inputs requires "
            "mil.train_require_all_tasks=true"
        )
    if sample_all_available_below_k and num_trial_views != 1:
        raise ValueError(
            "Retaining all short task bags requires train.num_trial_views=1"
        )
    if trial_pooling == "cartesian_logit_mean":
        if not require_all_tasks:
            if not sample_all_available_below_k:
                raise ValueError(
                    "Partial Cartesian training requires all available trials "
                    "below K to be retained"
                )
            if missing_task_embedding != "learned_per_task":
                raise ValueError(
                    "Partial Cartesian training requires one learned mask-CLS "
                    "embedding per task"
                )
        if num_trial_views != 1 or consistency_weight != 0:
            raise ValueError(
                "Cartesian task tuples require one trial view and zero "
                "view-consistency weight"
            )
        if auxiliary_task_loss_weight != 0:
            raise ValueError(
                "Cartesian task tuples do not expose per-task auxiliary logits"
            )
    elif missing_task_embedding != "none":
        raise ValueError(
            "mil.missing_task_embedding requires Cartesian task pooling"
        )
    if mil_cfg.get("class_sampling") != "epoch_shuffle_no_class_quota":
        raise ValueError(
            "mil.class_sampling must be epoch_shuffle_no_class_quota"
        )
    if tuple(mil_cfg.get("task_ids", TASK_IDS)) != TASK_IDS:
        raise ValueError(f"MIL task_ids must be {list(TASK_IDS)} in canonical order")
    required_true_flags = (
        "sample_without_replacement",
        "epoch_random_trial_sampling",
    )
    disabled = [name for name in required_true_flags if not bool(mil_cfg.get(name, False))]
    if disabled:
        raise ValueError(f"Subject MIL requires true flags: {disabled}")
    if not bool(mil_cfg.get("eval_use_all_trials", False)):
        raise ValueError(
            "Subject MIL evaluation requires mil.eval_use_all_trials=true"
        )
    if mil_cfg.get("task_pooling") not in {
        "concat_task_cls",
        "concat_task_cls_aux",
        "concat_task_mean_std",
        "shared_head_softmax",
        "shared_head_mean",
        "shared_head_constrained",
        "shared_head_residual",
        "task_bottleneck_concat",
        "cartesian_task_cls",
    }:
        raise ValueError(
            "mil.task_pooling must be concat_task_cls, concat_task_cls_aux, "
            "concat_task_mean_std, shared_head_softmax, shared_head_mean, "
            "shared_head_constrained, shared_head_residual, task_bottleneck_concat, or "
            "cartesian_task_cls"
        )
    if not bool(data_cfg.get("require_any_eye_keep", False)):
        raise ValueError(
            "Subject-MIL requires data.require_any_eye_keep=true"
        )

    if rank == 0:
        logger.info("Loading BERT from %s", model_cfg["bert_checkpoint"])
    bert, bert_cfg, bert_ckpt = load_bert_checkpoint(
        model_cfg["bert_checkpoint"], torch.device("cpu")
    )
    run_identity = (
        build_run_identity(
            "downstream",
            cfg,
            dependencies={"bert_checkpoint": model_cfg["bert_checkpoint"]},
        )
        if rank == 0 else None
    )
    model = EyeVQSubjectMIL(
        bert,
        num_tasks=len(TASK_IDS),
        num_classes=num_classes,
        classifier_hidden=int(model_cfg.get("classifier_hidden", 32)),
        task_bottleneck_dim=int(model_cfg.get("task_bottleneck_dim", 16)),
        residual_hidden=int(model_cfg.get("residual_hidden", 16)),
        residual_include_task_mask=residual_include_task_mask,
        dropout=float(model_cfg.get("dropout", 0.3)),
        task_pooling=str(mil_cfg.get("task_pooling", "concat_task_cls")),
        trial_pooling=trial_pooling,
        freeze_embedding=bool(model_cfg.get("freeze_embedding", True)),
        freeze_bottom_layers=int(model_cfg.get("freeze_bottom_layers", 4)),
        demographic_dim=demographic_dim,
        demographic_projection_dim=int(
            demographics_cfg.get("projection_dim", 128)
        ),
        demographic_fusion=demographic_fusion,
        demographic_max_alpha=float(
            demographics_cfg.get("max_alpha", 0.3)
        ),
        task_residual_logit_scale=float(
            mil_cfg.get("task_residual_logit_scale", 0.2)
        ),
        classifier_head=str(model_cfg.get("classifier_head", "mlp")),
        cartesian_trials_per_task=trials_per_task,
        missing_task_embedding=missing_task_embedding,
    ).to(device)

    downstream_cfg = _downstream_cfg(data_cfg, bert_cfg, label_cfg)
    area_stats_path = Path(data_cfg["area_stats_path"])
    if not area_stats_path.is_file():
        raise FileNotFoundError(
            f"EyeVQ MIL requires the configured area statistics file: {area_stats_path}"
        )
    data_dir = Path(data_cfg["data_dir"])
    area_stats = load_area_stats(area_stats_path)
    area_stats_mode = str(
        data_cfg.get("area_stats_mode", "bert_pretraining_global")
    )
    validate_mci_area_stats(
        area_stats,
        mode=area_stats_mode,
        area_stats_path=area_stats_path,
        bert_area_stats_path=bert_cfg.get("train", {}).get("area_stats_path"),
    )
    train_trials = PackedDownstreamDataset(
        data_dir, data_dir / data_cfg["train_index"], downstream_cfg, area_stats=area_stats
    )
    train_subjects_excluded = apply_task_availability_policy(
        train_trials,
        task_ids=TASK_IDS,
        min_trials_per_task=eligibility_min_trials,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
    )
    train_trial_counts = dict(Counter(row["ml_subject_id"] for row in train_trials.rows))
    val_trials = PackedDownstreamDataset(
        data_dir,
        data_dir / data_cfg["val_index"],
        downstream_cfg,
        area_stats=area_stats,
        train_subject_trial_counts=train_trial_counts,
    )
    val_subjects_excluded = apply_task_availability_policy(
        val_trials,
        task_ids=TASK_IDS,
        min_trials_per_task=evaluation_min_trials,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
    )
    test_trials = None
    if not args.skip_test:
        test_trials = PackedDownstreamDataset(
            data_dir,
            data_dir / data_cfg["test_index"],
            downstream_cfg,
            area_stats=area_stats,
            train_subject_trial_counts=train_trial_counts,
        )
        test_subjects_excluded = apply_task_availability_policy(
            test_trials,
            task_ids=TASK_IDS,
            min_trials_per_task=evaluation_min_trials,
            require_all_tasks=require_all_tasks,
            sample_all_available_below_k=sample_all_available_below_k,
        )
    else:
        test_subjects_excluded = ()
    # Always audit test identities, including validation-only searches where
    # loading test tensors is intentionally disabled.
    if test_trials is not None:
        test_audit_rows = test_trials.rows
    else:
        test_audit_rows = [
            row
            for row in read_packed_index(data_dir / data_cfg["test_index"])
            if packed_row_has_usable_eye(row)
        ]
        audit_dataset = SimpleNamespace(
            rows=test_audit_rows,
            labels=[0] * len(test_audit_rows),
        )
        test_subjects_excluded = apply_task_availability_policy(
            audit_dataset,
            task_ids=TASK_IDS,
            min_trials_per_task=evaluation_min_trials,
            require_all_tasks=require_all_tasks,
            sample_all_available_below_k=sample_all_available_below_k,
        )
        test_audit_rows = audit_dataset.rows
    split_audit = audit_split_rows({
        "train": train_trials.rows,
        "val": val_trials.rows,
        "test": test_audit_rows,
    })
    assert_clean_splits(split_audit)
    if area_stats_mode == "per_subject_transductive":
        stats_subjects = set(area_stats.get("subjects", {}))
        required_subjects = {
            str(row["ml_subject_id"])
            for rows in (train_trials.rows, val_trials.rows, test_audit_rows)
            for row in rows
        }
        missing_stats_subjects = sorted(required_subjects - stats_subjects)
        if missing_stats_subjects:
            raise ValueError(
                "Per-subject area statistics do not cover all audited MCI subjects: "
                + ", ".join(missing_stats_subjects[:20])
            )
    demographic_spec = None
    train_demographics = None
    val_demographics = None
    test_demographics = None
    if demographics_enabled:
        train_subject_keys = {
            str(row["ml_subject_id"]) for row in train_trials.rows
        }
        val_subject_keys = {str(row["ml_subject_id"]) for row in val_trials.rows}
        train_metadata_rows = read_demographic_index(
            data_dir / data_cfg["train_index"], subjects=train_subject_keys
        )
        val_metadata_rows = read_demographic_index(
            data_dir / data_cfg["val_index"], subjects=val_subject_keys
        )
        demographic_spec = fit_demographic_spec(
            train_metadata_rows,
            age_encoding=str(demographics_cfg.get("age_encoding", "zscore")),
        )
        if int(demographic_spec["feature_dim"]) != demographic_dim:
            raise RuntimeError("Configured demographic feature dimension is inconsistent")
        train_demographics = encode_subject_demographics(
            train_metadata_rows, demographic_spec
        )
        val_demographics = encode_subject_demographics(
            val_metadata_rows, demographic_spec
        )
        if test_trials is not None:
            test_subject_keys = {
                str(row["ml_subject_id"]) for row in test_trials.rows
            }
            test_metadata_rows = read_demographic_index(
                data_dir / data_cfg["test_index"], subjects=test_subject_keys
            )
            test_demographics = encode_subject_demographics(
                test_metadata_rows, demographic_spec
            )
        demographics_cfg.update({
            "enabled": True,
            "fusion": demographic_fusion,
            "age_encoding": demographic_spec["age_encoding"],
            "features": ["age", "education", "sex"],
            "fitted_spec": demographic_spec,
        })
    train_bags = SubjectBagDataset(
        train_trials,
        trials_per_task=trials_per_task,
        task_ids=TASK_IDS,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
        seed=seed,
        subject_features=train_demographics,
    )
    train_label_counts = Counter(int(label) for label in train_bags.labels)
    missing_train_classes = [
        class_id for class_id in range(num_classes)
        if train_label_counts[class_id] == 0
    ]
    if missing_train_classes:
        raise ValueError(
            f"Training subjects are missing classes: {missing_train_classes}"
        )
    class_weighting = str(train_cfg.get("class_weighting", "none"))
    if label_type == "binary":
        n_neg = train_label_counts[0]
        n_pos = train_label_counts[1]
        if class_weighting == "subject_pos_weight":
            positive_class_weight = float(n_neg) / float(n_pos)
        elif class_weighting == "none":
            positive_class_weight = 1.0
        else:
            raise ValueError(
                "Binary train.class_weighting must be none or subject_pos_weight"
            )
        pos_weight = torch.tensor(
            positive_class_weight, device=device, dtype=torch.float32
        )
        class_weights = None
    else:
        positive_class_weight = None
        if class_weighting == "subject_inverse_frequency":
            total_subjects = len(train_bags)
            class_weights = torch.tensor(
                [
                    total_subjects
                    / float(num_classes * train_label_counts[class_id])
                    for class_id in range(num_classes)
                ],
                device=device,
                dtype=torch.float32,
            )
        elif class_weighting == "none":
            class_weights = torch.ones(
                num_classes, device=device, dtype=torch.float32
            )
        else:
            raise ValueError(
                "Multiclass train.class_weighting must be none or "
                "subject_inverse_frequency"
            )
        pos_weight = None
    epochs = int(train_cfg["epochs"])
    sampler = DistributedSubjectEpochSampler(
        train_bags.labels,
        num_replicas=world_size,
        rank=rank,
        subjects_per_rank=subjects_per_gpu,
        epochs=epochs,
        seed=seed,
    )
    workers = int(train_cfg.get("num_workers", 4)) if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_bags,
        batch_sampler=sampler,
        collate_fn=collate_subject_bags,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 4)) if workers > 0 else None,
    )
    val_loader = _trial_loader(val_trials, train_cfg) if rank == 0 else None
    test_loader = (
        _trial_loader(test_trials, train_cfg)
        if rank == 0 and test_trials is not None else None
    )

    if rank == 0:
        label_counts_json = {
            str(class_id): int(train_label_counts[class_id])
            for class_id in range(num_classes)
        }
        class_weights_json = (
            {
                "0": 1.0,
                "1": float(positive_class_weight),
            }
            if label_type == "binary"
            else {
                str(class_id): float(class_weights[class_id].item())
                for class_id in range(num_classes)
            }
        )
        n_total = sum(parameter.numel() for parameter in model.parameters())
        n_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        first_audit = sampler.audit_epoch(0)
        exposure_audit = sampler.exposure_audit()
        train_task_coverage_counts = Counter(
            train_bags.task_count_by_subject.values()
        )
        logger.info(
            "BERT d=%d layers=%d pretrain_step=%s | params=%s trainable=%s",
            bert.d_model,
            len(bert.transformer),
            bert_ckpt.get("step", "?"),
            f"{n_total:,}",
            f"{n_trainable:,}",
        )
        logger.info(
            "Train=%d subjects (%d partial-task) labels=%s, task-policy "
            "excluded=%d | val=%d trials | "
            "test_audited=%d trials (tensors_loaded=%s)",
            len(train_bags),
            len(train_bags.subjects_with_missing_tasks),
            label_counts_json,
            len(train_subjects_excluded),
            len(val_trials), len(test_audit_rows), test_trials is not None,
        )
        logger.info(
            "Per GPU: %d subjects x %d tasks x %d fixed slots = %d encoded slots "
            "| global subjects=%d",
            subjects_per_gpu,
            len(TASK_IDS),
            trials_per_task,
            subjects_per_gpu * len(TASK_IDS) * trials_per_task,
            subjects_per_gpu * world_size,
        )
        logger.info("First-epoch audit: %s", first_audit)
        logger.info(
            "Subject %s class weighting: %s weights=%s",
            "BCE" if label_type == "binary" else "CE",
            class_weighting,
            class_weights_json,
        )
        write_json(output_dir / "class_weight.json", {
            "class_weight_for_label": class_weights_json,
            "source": class_weighting,
            "train_subject_label_counts": label_counts_json,
        })
        write_json(output_dir / "data_summary.json", {
            "train_trials_before_eye_filter": train_trials.num_rows_before_eye_filter,
            "train_trials_excluded_no_usable_eye": len(
                train_trials.excluded_no_usable_eye_rows
            ),
            "train_trials": len(train_trials),
            "train_trials_before_subject_task_filter": (
                train_trials.num_rows_before_subject_task_filter
            ),
            "train_trials_excluded_with_subject": (
                train_trials.rows_excluded_subject_task_minimum
            ),
            "train_subjects": len(train_bags),
            "train_complete_subjects": (
                len(train_bags) - len(train_bags.subjects_with_missing_tasks)
            ),
            "train_partial_task_subjects": len(
                train_bags.subjects_with_missing_tasks
            ),
            "train_subject_task_coverage_counts": {
                str(task_count): int(train_task_coverage_counts[task_count])
                for task_count in range(1, len(TASK_IDS) + 1)
            },
            "label_type": label_type,
            "num_classes": num_classes,
            "train_subject_label_counts": label_counts_json,
            "train_subjects_excluded_below_k_per_task": list(
                train_subjects_excluded
            ),
            "train_subjects_excluded_by_task_availability_policy": list(
                train_subjects_excluded
            ),
            "train_subjects_with_missing_tasks": list(
                train_trials.subjects_with_missing_tasks
            ),
            "train_masked_subject_task_groups": list(
                train_trials.masked_subject_task_groups
            ),
            "train_subject_task_groups_below_k_used_all": list(
                getattr(
                    train_trials,
                    "subject_task_groups_below_k_used_all",
                    (),
                )
            ),
            "subject_bag_unexpected_exclusions": list(train_bags.excluded_subjects),
            "val_trials_before_eye_filter": val_trials.num_rows_before_eye_filter,
            "val_trials_excluded_no_usable_eye": len(
                val_trials.excluded_no_usable_eye_rows
            ),
            "val_trials": len(val_trials),
            "val_trials_excluded_with_subject": (
                val_trials.rows_excluded_subject_task_minimum
            ),
            "val_subjects_excluded_below_k_per_task": list(val_subjects_excluded),
            "val_subjects_excluded_by_task_availability_policy": list(
                val_subjects_excluded
            ),
            "val_subjects_with_missing_tasks": list(
                val_trials.subjects_with_missing_tasks
            ),
            "val_masked_subject_task_groups": list(
                val_trials.masked_subject_task_groups
            ),
            "val_subject_task_groups_below_k_used_all": list(
                getattr(
                    val_trials,
                    "subject_task_groups_below_k_used_all",
                    (),
                )
            ),
            "test_trials_excluded_no_usable_eye": (
                len(test_trials.excluded_no_usable_eye_rows)
                if test_trials is not None
                else None
            ),
            "test_trials_before_eye_filter": (
                test_trials.num_rows_before_eye_filter
                if test_trials is not None
                else None
            ),
            "test_trials_before_subject_task_filter": (
                test_trials.num_rows_before_subject_task_filter
                if test_trials is not None
                else None
            ),
            "test_trials_excluded_with_subject": (
                test_trials.rows_excluded_subject_task_minimum
                if test_trials is not None
                else None
            ),
            "test_trials_audited": len(test_audit_rows),
            "test_subjects_excluded_below_k_per_task": list(
                test_subjects_excluded
            ),
            "test_subjects_excluded_by_task_availability_policy": list(
                test_subjects_excluded
            ),
            "test_subjects_with_missing_tasks": (
                list(test_trials.subjects_with_missing_tasks)
                if test_trials is not None
                else None
            ),
            "test_masked_subject_task_groups": (
                list(test_trials.masked_subject_task_groups)
                if test_trials is not None
                else None
            ),
            "test_subject_task_groups_below_k_used_all": (
                list(getattr(
                    test_trials,
                    "subject_task_groups_below_k_used_all",
                    (),
                ))
                if test_trials is not None
                else None
            ),
            "test_tensors_loaded": test_trials is not None,
            "test_evaluation_enabled": not args.skip_test,
            "area_stats_path": str(data_cfg["area_stats_path"]),
            "area_stats_mode": area_stats_mode,
            "area_stats_global": area_stats.get("global", {}),
            "area_stats_source": area_stats.get("source", {}),
            "area_stats_subject_count": len(area_stats.get("subjects", {})),
            "pretraining_overlap_policy": data_cfg.get(
                "pretraining_overlap_policy", "unspecified"
            ),
            "subjects_per_gpu": subjects_per_gpu,
            "trials_per_task": trials_per_task,
            "task_present_min_valid_trials": (
                1 if sample_all_available_below_k else trials_per_task
            ),
            "require_all_tasks": require_all_tasks,
            "sample_all_available_below_k": sample_all_available_below_k,
            "missing_task_embedding": missing_task_embedding,
            "task_availability_policy": (
                "require_all_tasks"
                if require_all_tasks
                else (
                    "retain_all_available_mask_only_absent_tasks"
                    if sample_all_available_below_k
                    else "mask_tasks_below_k_drop_only_if_all_missing"
                )
            ),
            "missing_task_aggregation": (
                "not_applicable_strict"
                if require_all_tasks
                else (
                    "learned_per_task_mask_cls"
                    if trial_pooling == "cartesian_logit_mean"
                    else "masked_uniform_mean"
                )
            ),
            "task_coverage_loss_weighting": task_coverage_loss_weighting,
            "num_trial_views": num_trial_views,
            "trials_per_task_per_view": trials_per_task // num_trial_views,
            "trial_view_consistency_weight": consistency_weight,
            "auxiliary_task_loss_weight": auxiliary_task_loss_weight,
            "feature_mixup_alpha": feature_mixup_alpha,
            "world_size": world_size,
            "class_sampling": "epoch_shuffle_no_class_quota",
            "task_pooling": str(mil_cfg["task_pooling"]),
            "trial_pooling": trial_pooling,
            "cross_task_residual_fusion": (
                {
                    "base": "uniform_mean_of_present_task_logits",
                    "inputs": (
                        "task_logits+task_presence+demographics"
                        if residual_include_task_mask
                        else "task_logits+demographics"
                    ),
                    "hidden": int(model_cfg.get("residual_hidden", 16)),
                    "include_task_mask": residual_include_task_mask,
                    "input_dim": (
                        len(TASK_IDS)
                        * (1 if label_type == "binary" else num_classes)
                        + (
                            len(TASK_IDS)
                            if residual_include_task_mask
                            else 0
                        )
                        + demographic_dim
                    ),
                    "output_initialization": "zero",
                    "coverage_gate": "(present_tasks-1)/(num_tasks-1)",
                }
                if mil_cfg["task_pooling"] == "shared_head_residual"
                else None
            ),
            "cartesian_combinations_per_subject": (
                trials_per_task ** len(TASK_IDS)
                if trial_pooling == "cartesian_logit_mean" and require_all_tasks
                else None
            ),
            "cartesian_max_combinations_per_subject": (
                trials_per_task ** len(TASK_IDS)
                if trial_pooling == "cartesian_logit_mean"
                else None
            ),
            "cartesian_subject_loss": (
                "bce_after_mean_combination_logits"
                if trial_pooling == "cartesian_logit_mean"
                else None
            ),
            "cartesian_evaluation": (
                (
                    "exact_all_valid_trial_product_missing_as_one_mask_cls"
                    if not require_all_tasks
                    else "exact_all_valid_trial_product"
                )
                if trial_pooling == "cartesian_logit_mean"
                else None
            ),
            "epochs": epochs,
            "steps_per_epoch": sampler.steps_per_epoch,
            "subjects_per_epoch": sampler.subjects_per_epoch,
            "dropped_subjects_per_epoch": sampler.dropped_per_epoch,
            "split_audit": split_audit,
            "demographics": demographics_cfg,
            "first_epoch_sampler_audit": first_audit,
            "sampler_exposure_audit": exposure_audit,
        })
        write_json(output_dir / "loss_config.json", {
            "loss": (
                "binary_cross_entropy_with_logits"
                if label_type == "binary"
                else "cross_entropy"
            ),
            "num_trial_views": num_trial_views,
            "trial_view_consistency": (
                "logit_mse_to_view_mean"
                if num_trial_views > 1 and consistency_weight > 0
                else "disabled"
            ),
            "trial_view_consistency_weight": consistency_weight,
            "auxiliary_task_loss": (
                "per_task_binary_cross_entropy_with_logits"
                if label_type == "binary"
                else "per_task_cross_entropy"
            ),
            "auxiliary_task_loss_weight": auxiliary_task_loss_weight,
            "feature_mixup": "subject_task_features" if feature_mixup_alpha > 0 else "none",
            "feature_mixup_alpha": feature_mixup_alpha,
            "sample_weighting": (
                "valid_task_fraction"
                if task_coverage_loss_weighting == "present_fraction"
                else "none"
            ),
            "task_coverage_loss_weighting": task_coverage_loss_weighting,
            "class_weighting": class_weighting,
            "positive_class_weight": positive_class_weight,
            "class_weights": class_weights_json,
            "sampling": "epoch_shuffle_without_class_quota",
            "trial_pooling": trial_pooling,
            "sample_all_available_below_k": sample_all_available_below_k,
            "missing_task_embedding": missing_task_embedding,
            "demographics": demographics_cfg,
        })

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model

    encoder_lr = float(train_cfg["encoder_lr"])
    head_lr = float(train_cfg["head_lr"])
    encoder_min_lr = float(train_cfg.get("encoder_min_lr", encoder_lr / 10))
    head_min_lr = float(train_cfg.get("head_min_lr", head_lr / 10))
    layer_decay = float(train_cfg.get("layer_decay", 1.0))
    weight_decay = float(train_cfg["weight_decay"])
    parameter_groups = build_optimizer_param_groups(
        raw_model,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        layer_decay=layer_decay,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
    )
    if rank == 0:
        write_json(output_dir / "optimizer_groups.json", {
            "layer_decay": layer_decay,
            "base_encoder_lr": encoder_lr,
            "base_head_lr": head_lr,
            "weight_decay": weight_decay,
            "groups": [
                {
                    "schedule": group["schedule"],
                    "lr_scale": float(group["lr_scale"]),
                    "weight_decay": float(group["weight_decay"]),
                    "num_parameters": int(sum(p.numel() for p in group["params"])),
                    "parameter_names": list(group["parameter_names"]),
                }
                for group in optimizer.param_groups
            ],
        })

    steps_per_epoch = len(sampler)
    total_steps = epochs * steps_per_epoch
    warmup_epochs = int(train_cfg.get("warmup_epochs", 4))
    if warmup_epochs <= 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be positive and smaller than epochs")
    warmup_steps = warmup_epochs * steps_per_epoch
    bf16 = bool(train_cfg.get("bf16", True))
    patience = int(train_cfg.get("early_stopping_patience_epochs", 10))
    early_stopping_min_epochs = int(train_cfg.get("early_stopping_min_epochs", 27))
    periodic_test_every_epochs = int(
        train_cfg.get("periodic_test_every_epochs", 0)
    )
    if patience <= 0:
        raise ValueError("early_stopping_patience_epochs must be positive")
    if periodic_test_every_epochs < 0:
        raise ValueError("periodic_test_every_epochs must be non-negative")
    if periodic_test_every_epochs > 0 and args.skip_test:
        raise ValueError(
            "periodic_test_every_epochs requires test evaluation to be enabled"
        )

    if rank == 0:
        logger.info(
            "Epoch training: epochs=%d steps_per_epoch=%d total_steps=%d "
            "warmup_epochs=%d | "
            "enc_top_lr=%.2e->%.2e layer_decay=%.3f "
            "head_lr=%.2e->%.2e",
            epochs, steps_per_epoch, total_steps, warmup_epochs,
            encoder_lr, encoder_min_lr, layer_decay, head_lr, head_min_lr,
        )
        logger.info(
            "Early stop: raw_val_%s min_epochs=%d patience_epochs=%d; "
            "every new best resets patience",
            selection_metric_name,
            early_stopping_min_epochs,
            patience,
        )

    best_raw_auroc = -math.inf
    best_raw_epoch = -1
    best_raw_step = -1
    no_improve = 0
    global_step = 0
    val_history: list[float] = []
    expected_task_ids = torch.tensor(TASK_IDS, dtype=torch.long).view(1, len(TASK_IDS), 1)
    training_stop_reason = "max_epochs"
    model.train()

    for epoch_index in range(epochs):
        sampler.set_epoch(epoch_index)
        epoch_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        epoch_loss_weight = torch.zeros((), dtype=torch.float64, device=device)
        epoch_correct = torch.zeros((), dtype=torch.float64, device=device)
        epoch_subjects = torch.zeros((), dtype=torch.float64, device=device)
        epoch_supervised_objective = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        epoch_consistency_objective = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        epoch_objective_steps = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        epoch_start = time.time()

        for batch in train_loader:
            if len(set(batch["subject_ids"])) != subjects_per_gpu:
                raise RuntimeError("A local MIL step contains a repeated subject")
            task_present_cpu = batch["task_present_mask"]
            trial_slot_cpu = batch["trial_slot_mask"]
            for subject_index, subject_tasks in enumerate(
                batch["sampled_trial_indices"]
            ):
                if len(subject_tasks) != len(TASK_IDS):
                    raise RuntimeError("A subject bag has an invalid task axis")
                for task_position, task_trials in enumerate(subject_tasks):
                    is_present = bool(
                        task_present_cpu[subject_index, task_position].item()
                    )
                    real_trial_count = len(task_trials)
                    slot_trial_count = int(
                        trial_slot_cpu[subject_index, task_position].sum().item()
                    )
                    if real_trial_count != slot_trial_count:
                        raise RuntimeError(
                            "A task bag's real trials disagree with its slot mask"
                        )
                    if is_present != (real_trial_count > 0):
                        raise RuntimeError(
                            "Task presence disagrees with the real-trial count"
                        )
                    if not 0 <= real_trial_count <= trials_per_task:
                        raise RuntimeError("A task bag exceeds the configured K")
                    if (
                        not sample_all_available_below_k
                        and is_present
                        and real_trial_count != trials_per_task
                    ):
                        raise RuntimeError(
                            "A present fixed-K task bag has an unexpected size"
                        )
                    if len(set(task_trials)) != real_trial_count:
                        raise RuntimeError("A task bag contains a repeated trial")
            observed_task_ids = batch["task_id"].reshape(
                subjects_per_gpu, len(TASK_IDS), trials_per_task
            )
            if not torch.equal(
                observed_task_ids, expected_task_ids.expand_as(observed_task_ids)
            ):
                raise RuntimeError(
                    "Subject bag trial order does not match canonical task order"
                )

            labels = batch["subject_label"].to(device, non_blocking=True)
            labels = labels.float() if label_type == "binary" else labels.long()
            task_present = batch["task_present_mask"].to(device, non_blocking=True)
            trial_slot_mask = batch["trial_slot_mask"].to(
                device, non_blocking=True
            )
            content = batch["content"].to(device, non_blocking=True).transpose(-1, -2).contiguous()
            stim = batch["stim"].to(device, non_blocking=True).transpose(-1, -2).contiguous()
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            nonmissing = batch["eye_nonmissing_frac"].to(device, non_blocking=True)
            task_ids = batch["task_id"].to(device, non_blocking=True)
            demographic_features = batch.get("subject_features")
            if demographic_features is not None:
                demographic_features = demographic_features.to(
                    device, non_blocking=True
                )

            encoder_step_lr, head_step_lr = get_encoder_head_lrs(
                global_step,
                warmup_steps,
                total_steps,
                encoder_lr,
                head_lr,
                encoder_min_lr,
                head_min_lr,
            )
            set_optimizer_step_lrs(
                optimizer,
                encoder_lr=encoder_step_lr,
                head_lr=head_step_lr,
            )
            optimizer.zero_grad(set_to_none=True)
            mixup_permutation = None
            mixup_lambda = None
            loss_labels = labels
            if feature_mixup_alpha > 0:
                mixup_permutation = deranged_permutation(subjects_per_gpu, device)
                sampled_lambda = random.betavariate(
                    feature_mixup_alpha, feature_mixup_alpha
                )
                mixup_lambda = max(sampled_lambda, 1.0 - sampled_lambda)
                loss_labels = (
                    mixup_lambda * labels
                    + (1.0 - mixup_lambda) * labels[mixup_permutation]
                )
            with _autocast(device, bf16):
                if num_trial_views > 1:
                    content = make_contiguous_trial_views(
                        content, num_subjects=subjects_per_gpu,
                        num_tasks=len(TASK_IDS), trials_per_task=trials_per_task,
                        num_views=num_trial_views,
                    )
                    stim = make_contiguous_trial_views(
                        stim, num_subjects=subjects_per_gpu,
                        num_tasks=len(TASK_IDS), trials_per_task=trials_per_task,
                        num_views=num_trial_views,
                    )
                    pad_mask = make_contiguous_trial_views(
                        pad_mask, num_subjects=subjects_per_gpu,
                        num_tasks=len(TASK_IDS), trials_per_task=trials_per_task,
                        num_views=num_trial_views,
                    )
                    nonmissing = make_contiguous_trial_views(
                        nonmissing, num_subjects=subjects_per_gpu,
                        num_tasks=len(TASK_IDS), trials_per_task=trials_per_task,
                        num_views=num_trial_views,
                    )
                    task_ids = make_contiguous_trial_views(
                        task_ids, num_subjects=subjects_per_gpu,
                        num_tasks=len(TASK_IDS), trials_per_task=trials_per_task,
                        num_views=num_trial_views,
                    )
                    forward_subjects = subjects_per_gpu * num_trial_views
                    forward_k = trials_per_task // num_trial_views
                    forward_task_present = task_present.unsqueeze(0).expand(
                        num_trial_views, -1, -1
                    ).reshape(forward_subjects, len(TASK_IDS))
                    forward_demographics = (
                        demographic_features.unsqueeze(0)
                        .expand(num_trial_views, -1, -1)
                        .reshape(forward_subjects, demographic_dim)
                        if demographic_features is not None
                        else None
                    )
                else:
                    forward_subjects = subjects_per_gpu
                    forward_k = trials_per_task
                    forward_task_present = task_present
                    forward_demographics = demographic_features
                output = model(
                    stim_patches=stim,
                    eye_patches=content,
                    pad_mask=pad_mask,
                    eye_nonmissing_frac=nonmissing,
                    task_ids=task_ids,
                    num_subjects=forward_subjects,
                    trials_per_task=forward_k,
                    task_present_mask=forward_task_present,
                    trial_slot_mask=(
                        trial_slot_mask
                        if trial_pooling == "cartesian_logit_mean"
                        or (
                            trial_pooling == "logit_mean"
                            and num_trial_views == 1
                        )
                        else None
                    ),
                    subject_mixup_permutation=mixup_permutation,
                    subject_mixup_lambda=mixup_lambda,
                    demographic_features=forward_demographics,
                )
                view_logits = output["subject_logits"].reshape(
                    num_trial_views,
                    subjects_per_gpu,
                    *(() if label_type == "binary" else (num_classes,)),
                )
                view_mean_logits = view_logits.mean(dim=0)
                task_logits = None
                if trial_view_aggregation == "independent_loss":
                    logits = view_mean_logits
                    supervised_logits = view_logits
                else:
                    if "task_logits" not in output:
                        raise RuntimeError(
                            "Task-level trial-view aggregation requires task logits"
                        )
                    task_logits = output["task_logits"].reshape(
                        num_trial_views,
                        subjects_per_gpu,
                        len(TASK_IDS),
                        *(() if label_type == "binary" else (num_classes,)),
                    )
                    pooled_task_logits = aggregate_trial_view_task_logits(
                        task_logits, mode=trial_view_aggregation
                    )
                    pooled_output = raw_model.aggregate_task_logits(
                        pooled_task_logits,
                        task_present,
                        demographic_features,
                    )
                    logits = pooled_output["subject_logits"]
                    supervised_logits = logits.unsqueeze(0)
                supervised = task_coverage_weighted_supervised_loss(
                    supervised_logits,
                    loss_labels if label_type == "binary" else labels,
                    task_present,
                    label_type=label_type,
                    coverage_mode=task_coverage_loss_weighting,
                    pos_weight=pos_weight,
                    class_weights=class_weights,
                )
                consistency = (
                    view_logits - view_mean_logits.unsqueeze(0)
                ).square().mean()
                auxiliary = logits.new_zeros(())
                if auxiliary_task_loss_weight > 0:
                    if "task_logits" not in output:
                        raise RuntimeError(
                            "auxiliary_task_loss_weight requires task logits"
                        )
                    if task_logits is None:
                        task_logits = output["task_logits"].reshape(
                            num_trial_views,
                            subjects_per_gpu,
                            len(TASK_IDS),
                            *(() if label_type == "binary" else (num_classes,)),
                        )
                    auxiliary = masked_auxiliary_task_loss(
                        task_logits,
                        labels,
                        task_present,
                        label_type=label_type,
                        pos_weight=pos_weight,
                        class_weights=class_weights,
                    )
                loss = (
                    supervised
                    + consistency_weight * consistency
                    + auxiliary_task_loss_weight * auxiliary
                )
                if getattr(raw_model, "task_weight_residual", None) is not None:
                    loss = loss + task_residual_l2 * raw_model.task_weight_residual.square().mean()
                if raw_model.demographic_head is not None and demographic_head_l2 > 0:
                    loss = loss + demographic_head_l2 * raw_model.demographic_head.weight.square().mean()
            loss.backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["grad_clip"])
                )
            optimizer.step()

            with torch.no_grad():
                if label_type == "binary":
                    per_subject = F.binary_cross_entropy_with_logits(
                        logits.float(), labels, reduction="none"
                    )
                    predictions = (logits > 0).long()
                    target_classes = (labels > 0.5).long()
                else:
                    per_subject = F.cross_entropy(
                        logits.float(), labels, reduction="none"
                    )
                    predictions = logits.argmax(dim=-1)
                    target_classes = labels
                coverage = task_coverage_weights(
                    task_present, mode=task_coverage_loss_weighting
                ).double()
                epoch_loss_sum += (per_subject.double() * coverage).sum()
                epoch_loss_weight += coverage.sum()
                epoch_correct += (predictions == target_classes).double().sum()
                epoch_subjects += labels.numel()
                epoch_supervised_objective += supervised.detach().double()
                epoch_consistency_objective += consistency.detach().double()
                epoch_objective_steps += 1
            global_step += 1

        totals = torch.stack([
            epoch_loss_sum,
            epoch_loss_weight,
            epoch_correct,
            epoch_subjects,
            epoch_supervised_objective,
            epoch_consistency_objective,
            epoch_objective_steps,
        ])
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = float((totals[0] / totals[1]).item())
        train_acc = float((totals[2] / totals[3]).item())
        train_mean_task_coverage = float((totals[1] / totals[3]).item())
        mean_supervised_objective = float((totals[4] / totals[6]).item())
        mean_consistency_objective = float((totals[5] / totals[6]).item())
        elapsed = time.time() - epoch_start
        current_epoch = epoch_index + 1
        audit = sampler.audit_epoch(epoch_index)
        should_stop = False

        if rank == 0:
            val_metrics, val_rows = evaluate_subjects(
                raw_model,
                val_loader,
                device,
                split_name="val",
                bf16=bf16,
                subject_demographics=val_demographics,
                task_coverage_loss_weighting=task_coverage_loss_weighting,
            )
            val_auroc = float(
                val_metrics[f"val/subject/{selection_metric_name}"]
            )
            val_history.append(val_auroc)
            encoder_group_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
                if group["schedule"] == "encoder"
            ]
            head_group_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
                if group["schedule"] == "head"
            ]
            logger.info(
                "E%d/%d S%d train_loss=%.4f train_acc=%.4f val_%s=%.4f "
                "val_bacc=%.4f obj_sup=%.4f obj_cons=%.4f "
                "obj_cons_weighted=%.4f lr_enc=%.2e..%.2e "
                "lr_head=%.2e time=%.0fs",
                current_epoch,
                epochs,
                global_step,
                train_loss,
                train_acc,
                selection_metric_name,
                val_auroc,
                val_metrics["val/subject/balanced_accuracy"],
                mean_supervised_objective,
                mean_consistency_objective,
                consistency_weight * mean_consistency_objective,
                min(encoder_group_lrs),
                max(encoder_group_lrs),
                max(head_group_lrs),
                elapsed,
            )
            checkpoint = {
                "epoch": current_epoch,
                "step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "bert_cfg": bert_cfg,
                "config_version": CONFIG_VERSION,
                "run_identity": run_identity,
                "val_metrics": val_metrics,
                "selection_metric": f"val/subject/{selection_metric_name}",
                "val_auroc_history": val_history,
                "sampler_audit": audit,
            }
            raw_improved = best_raw_epoch < 0 or val_auroc > best_raw_auroc
            checkpoint["best_raw_auroc"] = best_raw_auroc
            checkpoint["best_raw_epoch"] = best_raw_epoch
            checkpoint["best_raw_step"] = best_raw_step
            if raw_improved:
                best_raw_auroc = val_auroc
                best_raw_epoch = current_epoch
                best_raw_step = global_step
                checkpoint["best_raw_auroc"] = best_raw_auroc
                checkpoint["best_raw_epoch"] = best_raw_epoch
                checkpoint["best_raw_step"] = best_raw_step
                atomic_torch_save(checkpoint, output_dir / "ckpt_best.pt")
                write_prediction_csv(output_dir / "predictions_val_best.csv", val_rows)
                logger.info(
                    "New best val-%s checkpoint at epoch=%d step=%d "
                    "val_auc=%.4f; "
                    "early-stop patience reset",
                    selection_metric_name,
                    best_raw_epoch,
                    best_raw_step,
                    best_raw_auroc,
                )

            no_improve = update_early_stopping_counter(
                no_improve,
                val_improved=raw_improved,
                global_step=current_epoch,
                min_steps=early_stopping_min_epochs,
            )
            if current_epoch >= early_stopping_min_epochs:
                logger.info(
                    "Early-stop counter=%d/%d%s",
                    no_improve,
                    patience,
                    " (reset by val-AUROC new best)" if raw_improved else "",
                )
            atomic_torch_save(checkpoint, output_dir / "ckpt_last.pt")
            write_json(output_dir / "metrics_last.json", {
                "epoch": current_epoch,
                "step": global_step,
                "train/subject_loss": train_loss,
                "train/subject_accuracy": train_acc,
                "train/mean_task_coverage_weight": train_mean_task_coverage,
                "early_stopping_no_improve_epochs": no_improve,
                "sampler_audit": audit,
                **val_metrics,
            })
            if (
                periodic_test_every_epochs > 0
                and current_epoch % periodic_test_every_epochs == 0
            ):
                if test_loader is None:
                    raise RuntimeError(
                        "Periodic test evaluation requested without a test loader"
                    )
                periodic_test_metrics, periodic_test_rows = evaluate_subjects(
                    raw_model,
                    test_loader,
                    device,
                    split_name="test",
                    threshold=0.5,
                    bf16=bf16,
                    subject_demographics=test_demographics,
                    task_coverage_loss_weighting=task_coverage_loss_weighting,
                )
                periodic_dir = output_dir / "periodic_eval"
                atomic_torch_save(
                    checkpoint,
                    periodic_dir / f"ckpt_epoch{current_epoch:03d}.pt",
                )
                write_prediction_csv(
                    periodic_dir / f"predictions_val_epoch{current_epoch:03d}.csv",
                    val_rows,
                )
                write_prediction_csv(
                    periodic_dir / f"predictions_test_epoch{current_epoch:03d}.csv",
                    periodic_test_rows,
                )
                write_json(
                    periodic_dir / f"metrics_epoch{current_epoch:03d}.json",
                    {
                        "epoch": current_epoch,
                        "step": global_step,
                        "checkpoint": str(
                            periodic_dir / f"ckpt_epoch{current_epoch:03d}.pt"
                        ),
                        "selection_policy": (
                            "diagnostic_only; periodic test is never used for "
                            "early stopping or checkpoint selection"
                        ),
                        "val": val_metrics,
                        "test": periodic_test_metrics,
                        "cfg": cfg,
                        "run_identity": run_identity,
                    },
                )
                logger.info(
                    "Periodic diagnostic epoch=%d: val_%s=%.4f "
                    "test_%s=%.4f (not used for selection)",
                    current_epoch,
                    selection_metric_name,
                    val_auroc,
                    selection_metric_name,
                    periodic_test_metrics[
                        f"test/subject/{selection_metric_name}"
                    ],
                )
            if current_epoch >= early_stopping_min_epochs and no_improve >= patience:
                logger.info(
                    "Early stopping at epoch=%d step=%d: validation %s did "
                    "not set a new best for %d epochs",
                    current_epoch,
                    global_step,
                    selection_metric_name,
                    patience,
                )
                should_stop = True
                training_stop_reason = "early_stopping"

        stop_tensor = torch.tensor(int(should_stop), dtype=torch.int32, device=device)
        if world_size > 1:
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
        # Rank 0 evaluation switched only its raw model to eval mode.
        model.train()
        if bool(stop_tensor.item()):
            break

    if rank == 0:
        if best_raw_epoch < 0 or not (output_dir / "ckpt_best.pt").is_file():
            raise RuntimeError("Training ended without a selectable ckpt_best.pt")
        if not args.skip_test and test_loader is None:
            raise RuntimeError("Final test was requested but the test loader is unavailable")
        logger.info(
            "Training ended by %s; loading ckpt_best.pt for final validation%s",
            training_stop_reason,
            " and test" if not args.skip_test else "",
        )
        best_checkpoint = torch.load(
            output_dir / "ckpt_best.pt", map_location=device, weights_only=False
        )
        raw_model.load_state_dict(best_checkpoint["model_state_dict"])
        val_metrics, val_rows = evaluate_subjects(
            raw_model, val_loader, device, split_name="val", bf16=bf16,
            subject_demographics=val_demographics,
            task_coverage_loss_weighting=task_coverage_loss_weighting,
        )
        if label_type == "binary":
            # MCI uses a fixed, predeclared decision threshold.  Do not tune
            # this value on the small validation split: that made the final
            # decision rule unstable even though AUROC was unchanged.
            threshold = 0.5
            val_bacc = float(val_metrics["val/subject/balanced_accuracy"])
        else:
            threshold = None
            val_bacc = float(val_metrics["val/subject/balanced_accuracy"])
        write_prediction_csv(output_dir / "predictions_val_best.csv", val_rows)
        selected_step = int(best_checkpoint["step"])
        selected_epoch = int(best_checkpoint["epoch"])
        val_result = {
            "best_epoch": selected_epoch,
            "best_step": selected_step,
            "best_val_auroc": float(
                val_metrics[f"val/subject/{selection_metric_name}"]
            ),
            "checkpoint_selection_metric": (
                f"val/subject/{selection_metric_name}"
            ),
            "early_stopping_metric": f"val/subject/{selection_metric_name}",
            "label_type": label_type,
            "num_classes": num_classes,
            "val": val_metrics,
            "cfg": cfg,
            "run_identity": run_identity,
            "training_stop_reason": training_stop_reason,
            "selected_checkpoint": str(output_dir / "ckpt_best.pt"),
            "test_evaluated": False,
        }
        if label_type == "binary":
            val_result.update({
                "decision_rule": "fixed_threshold",
                "decision_threshold": threshold,
                "threshold_policy": "fixed_0.5",
                "val_balanced_accuracy_at_fixed_threshold": val_bacc,
                "val_default_05": val_metrics,
            })
        else:
            val_result.update({
                "decision_rule": "argmax",
                "val_balanced_accuracy": val_bacc,
            })
        write_json(output_dir / "metrics_val_best.json", val_result)
        if args.skip_test:
            logger.info(
                "Best epoch=%d step=%d val_auroc=%.4f | test skipped by request",
                selected_epoch,
                selected_step,
                val_result["best_val_auroc"],
            )
        else:
            test_metrics, test_rows = evaluate_subjects(
                raw_model,
                test_loader,
                device,
                split_name="test",
                threshold=threshold if threshold is not None else 0.5,
                bf16=bf16,
                subject_demographics=test_demographics,
                task_coverage_loss_weighting=task_coverage_loss_weighting,
            )
            write_prediction_csv(output_dir / "predictions_test.csv", test_rows)
            if label_type == "binary":
                test_payload = {
                    **val_result,
                    "test_evaluated": True,
                    "test": test_metrics,
                    "test_default_05": test_metrics,
                }
                logger.info(
                    "Best epoch=%d step=%d val_auroc=%.4f | "
                    "test_auroc=%.4f test_bacc=%.4f threshold=%.2f",
                    selected_epoch,
                    selected_step,
                    val_result["best_val_auroc"],
                    test_metrics["test/subject/auroc"],
                    test_metrics["test/subject/balanced_accuracy"],
                    threshold,
                )
            else:
                test_payload = {
                    **val_result,
                    "test_evaluated": True,
                    "test": test_metrics,
                }
                logger.info(
                    "Best epoch=%d step=%d val_macro_auroc=%.4f | "
                    "test_macro_auroc=%.4f test_bacc=%.4f",
                    selected_epoch,
                    selected_step,
                    val_result["best_val_auroc"],
                    test_metrics["test/subject/macro_auroc_ovr"],
                    test_metrics["test/subject/balanced_accuracy"],
                )
            write_json(output_dir / "metrics_test.json", test_payload)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
