"""
EyeVQ-BERT: Masked Code Prediction Model (ViT-based).

Architecture:
  Raw patches → BERT's own PatchEmbedding → Transformer → predict VQ code IDs.
  Transformer aligned with EyeMAE: RMSNorm + SwiGLU + MultiheadAttention.
  Tokenizer only used offline to generate VQ code IDs as labels.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from eyemae.eyevq.tokenizer.model import (
    PatchEmbedding, Q_BINS, TYPE_S, TYPE_L, TYPE_R,
    RMSNorm, SwiGLU, TransformerBlock, AxialCrossTransformerBlock,
    build_stim_isolated_attn_mask,
)
from eyemae.eyevq.tokenizer.losses import (
    compute_eye_recon_loss,
    compute_manual_feature_loss,
)

TARGET_FACTORIZED_CODE = "factorized_code"
TARGET_JOINT_CODE = "joint_code"
TARGET_RAW_PATCH = "raw_patch"
TARGET_DIRECT_RECONSTRUCTION = "direct_reconstruction"
TARGET_NORMALIZED_LATENT = "normalized_latent"
SUPPORTED_TARGET_TYPES = {
    TARGET_FACTORIZED_CODE,
    TARGET_JOINT_CODE,
    TARGET_RAW_PATCH,
    TARGET_DIRECT_RECONSTRUCTION,
    TARGET_NORMALIZED_LATENT,
}


# ──────────────────────────────────────────────
# Transformer Builder (EyeMAE-aligned)
# ──────────────────────────────────────────────

def build_transformer(
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    dim_feedforward: int = 1536,
    dropout: float = 0.0,
    attention_layout: str = "joint",
) -> nn.ModuleList:
    """Build EyeMAE-aligned transformer blocks."""
    if attention_layout not in {"joint", "axial_cross"}:
        raise ValueError(f"Unsupported attention_layout={attention_layout!r}")
    block_cls = (
        AxialCrossTransformerBlock
        if attention_layout == "axial_cross"
        else TransformerBlock
    )
    return nn.ModuleList([
        block_cls(d_model, n_heads, dim_feedforward, dropout)
        for _ in range(n_layers)
    ])


# ──────────────────────────────────────────────
# Prediction Head (single Linear)
# ──────────────────────────────────────────────

class PredictionHead(nn.Module):
    """Predict masked eye code IDs: single Linear layer."""

    def __init__(self, d_model: int = 512, K_e: int = 512):
        super().__init__()
        self.net = nn.Linear(d_model, K_e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [N_masked, d_model] → logits [N_masked, K_e]"""
        return self.net(x)


def code_ids_to_coords(
    code_ids: torch.Tensor,   # [..., 1] 联合 FSQ index
    base: list[int],          # [D] 编码步长 (FSQ.base, 如 [1,9,63,441,2205])
    L: list[int],             # [D] 每维 level 数
) -> torch.Tensor:
    """FSQ 联合 index → 各维坐标 [..., D] (用于 per-dim 准确率 / digit distance)。"""
    coords = torch.stack(
        [(code_ids // b) % li for b, li in zip(base, L)],
        dim=-1,
    )
    return coords


def mean_masked_loss_per_trial(
    per_token_loss: torch.Tensor,
    masked_trial_indices: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average masked tokens within each trial, then across supervised trials."""
    if per_token_loss.ndim != 1 or masked_trial_indices.shape != per_token_loss.shape:
        raise ValueError("per-token loss and trial indices must be aligned 1-D tensors")
    sums = per_token_loss.new_zeros(int(batch_size)).index_add(
        0, masked_trial_indices, per_token_loss
    )
    counts = per_token_loss.new_zeros(int(batch_size)).index_add(
        0, masked_trial_indices, torch.ones_like(per_token_loss)
    )
    supervised = counts > 0
    # Keep the denominator on device.  Converting it to a Python integer here
    # inserts a CUDA->CPU synchronization in every model forward.
    n_supervised = supervised.sum()
    per_trial = sums / counts.clamp_min(1)
    loss = per_trial.sum() / n_supervised.clamp_min(1)
    return loss, n_supervised


def factorized_fsq_cross_entropy(
    dim_logits: list[torch.Tensor],
    coords: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-dimension CE and the joint factorized NLL per token.

    The D FSQ coordinates jointly identify one code.  Their categorical
    negative log-likelihoods must therefore be *summed*, not averaged:
    ``-log prod_d p(c_d | h) = sum_d CE_d``.  Besides being the probabilistic
    objective, the sum keeps the random-prediction loss on the same scale as
    a joint ``prod(levels)``-class head.
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must have shape [tokens, dimensions], got {coords.shape}")
    if len(dim_logits) != coords.shape[1]:
        raise ValueError(
            "one logits tensor is required per FSQ dimension; "
            f"got {len(dim_logits)} heads for {coords.shape[1]} coordinates"
        )
    per_dim_ce = torch.stack(
        [
            F.cross_entropy(
                logits,
                coords[:, dimension],
                reduction="none",
                label_smoothing=label_smoothing,
            )
            for dimension, logits in enumerate(dim_logits)
        ],
        dim=-1,
    )
    return per_dim_ce, per_dim_ce.sum(dim=-1)


def raw_patch_reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing: torch.Tensor,
    *,
    blink_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizer-aligned per-token losses for raw eye patches.

    Missing frames never contribute.  Blink frames supervise the blink channel,
    but do not force the undefined x/y/area values toward their padding value.
    XY, area, blink, and velocity are returned separately so the exact tokenizer
    component weights can be reused by raw-patch BERT.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("raw patch prediction and target must have shape [tokens, 4, samples]")
    if missing.shape != target.shape[:1] + target.shape[2:]:
        raise ValueError("missing must have shape [tokens, samples]")
    valid = ~missing.bool()
    blink_target = target[:, 3, :].clamp(0, 1)
    if not math.isfinite(blink_pos_weight) or blink_pos_weight <= 0.0:
        raise ValueError("blink_pos_weight must be finite and positive")
    coord_valid = valid & (blink_target < 0.5)

    xy_error = F.smooth_l1_loss(
        prediction[:, :2, :], target[:, :2, :], reduction="none"
    )
    xy_loss = (xy_error * coord_valid.unsqueeze(1)).sum(dim=(1, 2)) / (
        coord_valid.sum(dim=1).mul(2).clamp_min(1)
    )

    area_error = F.smooth_l1_loss(
        prediction[:, 2, :], target[:, 2, :], reduction="none"
    )
    area_loss = (area_error * coord_valid).sum(dim=1) / (
        coord_valid.sum(dim=1).clamp_min(1)
    )

    blink_error = F.binary_cross_entropy_with_logits(
        prediction[:, 3, :],
        blink_target,
        pos_weight=prediction.new_tensor(blink_pos_weight),
        reduction="none",
    )
    blink_class_weight = torch.where(
        blink_target >= 0.5,
        blink_error.new_tensor(blink_pos_weight),
        blink_error.new_tensor(1.0),
    )
    blink_loss = (blink_error * valid).sum(dim=1) / (
        blink_class_weight * valid
    ).sum(dim=1).clamp_min(1e-8)

    velocity_prediction = prediction[:, :2, 1:] - prediction[:, :2, :-1]
    velocity_target = target[:, :2, 1:] - target[:, :2, :-1]
    velocity_valid = coord_valid[:, 1:] & coord_valid[:, :-1]
    velocity_error = F.smooth_l1_loss(
        velocity_prediction, velocity_target, reduction="none"
    )
    velocity_loss = (
        velocity_error * velocity_valid.unsqueeze(1)
    ).sum(dim=(1, 2)) / velocity_valid.sum(dim=1).mul(2).clamp_min(1)
    return xy_loss, area_loss, blink_loss, velocity_loss, valid


# ──────────────────────────────────────────────
# BERT Token Embedding (raw patches → embeddings)
# ──────────────────────────────────────────────

class BertTokenEmbedding(nn.Module):
    """Constructs BERT token embeddings from RAW patches.

    Has its own PatchEmbedding instances — same architecture as tokenizer's
    encoder, but completely independent weights.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_time: int = 512,
        patch_samples: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_samples = patch_samples

        # Own PatchEmbedding (same structure, NOT shared with tokenizer)
        self.stim_patch_embed = PatchEmbedding(d_model, patch_samples)
        self.eye_patch_embed = PatchEmbedding(d_model, patch_samples)   # L/R shared

        # BERT CLS
        self.bert_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bert_cls_token, std=0.02)

        # Auxiliary embeddings (independent)
        self.type_embed = nn.Embedding(4, d_model)            # CLS=0, S=1, L=2, R=3
        self.pos_embed = nn.Embedding(max_time, d_model)
        self.quality_embed = nn.Embedding(Q_BINS, d_model)    # L/R only

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Init
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.quality_embed.weight, std=0.02)

    def forward(
        self,
        stim: torch.Tensor,                    # [B, N, 4, patch_samples]
        eye_l: torch.Tensor,                   # [B, N, 4, patch_samples]
        eye_r: torch.Tensor,                   # [B, N, 4, patch_samples]
        eye_nonmissing_frac: torch.Tensor,     # [B, N, 2]
        task_ids: torch.Tensor,                # [B]
        bert_mask: torch.Tensor | None = None, # [B, 1+N*3]
    ) -> torch.Tensor:
        """Returns: seq [B, 1+N*3, d_model]"""
        B, N = stim.shape[:2]
        device = stim.device
        ps = self.patch_samples

        # ── Patch embedding ──
        s_emb = self.stim_patch_embed(stim.reshape(B * N, 4, ps)).reshape(B, N, self.d_model)
        l_emb = self.eye_patch_embed(eye_l.reshape(B * N, 4, ps)).reshape(B, N, self.d_model)
        r_emb = self.eye_patch_embed(eye_r.reshape(B * N, 4, ps)).reshape(B, N, self.d_model)

        # ── Type + pos + quality: 作为 aux 与 patch content 分离 ──
        # (关键修复: mask 只替换 patch content, 保留 position/type/quality,
        #  使模型能区分同一 trial 内不同时间点 / 左右眼 / 质量)
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)

        q_l = (eye_nonmissing_frac[:, :, 0] * Q_BINS).long().clamp(0, Q_BINS - 1)
        q_r = (eye_nonmissing_frac[:, :, 1] * Q_BINS).long().clamp(0, Q_BINS - 1)

        aux_s = self.type_embed.weight[TYPE_S] + pos_emb                        # [B,N,d]
        aux_l = self.type_embed.weight[TYPE_L] + pos_emb + self.quality_embed(q_l)
        aux_r = self.type_embed.weight[TYPE_R] + pos_emb + self.quality_embed(q_r)

        # ── Build sequence: content (patch) 与 aux (type+pos+quality) 分开 ──
        content_cls = self.bert_cls_token.expand(B, 1, -1)
        content_body = torch.stack([s_emb, l_emb, r_emb], dim=2).reshape(B, N * 3, self.d_model)
        content_all = torch.cat([content_cls, content_body], dim=1)             # [B, 1+N*3, d]
        aux_body = torch.stack([aux_s, aux_l, aux_r], dim=2).reshape(B, N * 3, self.d_model)
        aux_all = torch.cat([content_all.new_zeros(B, 1, self.d_model), aux_body], dim=1)

        # ── Apply mask: 只替换 patch content, 保留 position/type/quality ──
        if bert_mask is not None:
            mask_expanded = bert_mask.unsqueeze(-1)                             # [B, 1+N*3, 1]
            content_all = torch.where(mask_expanded, self.mask_token, content_all)

        seq = content_all + aux_all
        return seq


# ──────────────────────────────────────────────
# EyeVQ-BERT Model
# ──────────────────────────────────────────────

class EyeVQBERT(nn.Module):
    """BERT-style masked code prediction.

    Ingests raw patches directly. Tokenizer is ONLY used offline
    to generate VQ code IDs as supervision labels.
    """

    def __init__(
        self,
        *,
        K_e: int = 512,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        dim_ff: int = 1536,
        dropout: float = 0.0,
        max_time: int = 512,
        patch_samples: int = 20,
        label_smoothing: float = 0.1,
        stim_isolated_attn: bool = False,
        stim_attend_cls: bool = True,
        attention_layout: str = "joint",
        factorized_fsq: bool = False,              # True → one categorical head per FSQ dimension
        fsq_L: list[int] | None = None,            # 每维 level 数 (factorized 时必填, joint 时可算 digit distance)
        min_nonmissing_frac: float = 0.50,
        predictor_span_length_embedding: bool = False,
        max_mask_span_length: int = 0,
        target_type: str | None = None,
        latent_dim: int = 64,
        raw_xy_weight: float = 1.0,
        raw_area_weight: float = 1.0,
        raw_blink_weight: float = 1.0,
        raw_blink_pos_weight: float = 1.0,
        raw_velocity_weight: float = 0.0,
        manual_feature_dim: int = 38,
        direct_loss_cfg: dict | None = None,
    ):
        super().__init__()
        self.K_e = K_e
        self.d_model = d_model
        self.label_smoothing = label_smoothing
        self.stim_isolated_attn = stim_isolated_attn
        self.stim_attend_cls = bool(stim_attend_cls)
        self.attention_layout = attention_layout
        self.factorized_fsq = factorized_fsq
        self.target_type = target_type or (
            TARGET_FACTORIZED_CODE if factorized_fsq else TARGET_JOINT_CODE
        )
        if self.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(f"Unsupported BERT target_type={self.target_type!r}")
        if self.target_type == TARGET_FACTORIZED_CODE and not factorized_fsq:
            raise ValueError("factorized_code target requires factorized_fsq=true")
        if self.target_type != TARGET_FACTORIZED_CODE and factorized_fsq:
            raise ValueError(f"{self.target_type} target requires factorized_fsq=false")
        self.raw_xy_weight = float(raw_xy_weight)
        self.raw_area_weight = float(raw_area_weight)
        self.raw_blink_weight = float(raw_blink_weight)
        self.raw_blink_pos_weight = float(raw_blink_pos_weight)
        self.raw_velocity_weight = float(raw_velocity_weight)
        self.manual_feature_dim = int(manual_feature_dim)
        self.direct_loss_cfg = direct_loss_cfg
        self.patch_samples = int(patch_samples)
        self.fsq_L = list(fsq_L) if fsq_L is not None else None
        self.min_nonmissing_frac = float(min_nonmissing_frac)
        self.predictor_span_length_embedding = bool(predictor_span_length_embedding)
        self.max_mask_span_length = int(max_mask_span_length)
        if self.predictor_span_length_embedding and self.max_mask_span_length < 1:
            raise ValueError("predictor span-length embedding requires max_mask_span_length >= 1")
        self.fsq_base: list[int] = []
        if self.factorized_fsq and not self.fsq_L:
            raise ValueError("factorized_fsq requires a non-empty fsq_L")
        if self.fsq_L is not None:
            stride = 1
            for li in self.fsq_L:
                self.fsq_base.append(stride)
                stride *= li

        self.embed = BertTokenEmbedding(
            d_model=d_model, max_time=max_time, patch_samples=patch_samples,
        )
        self.transformer = build_transformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            attention_layout=attention_layout,
        )
        # Final RMSNorm (mirrors tokenizer encoder's out_norm). Needed when
        # initializing BERT from tokenizer weights — the pretrained transformer
        # outputs are normalized by out_norm before use; without it the hidden
        # scale explodes (std ~260) and pred_head produces huge logits.
        self.out_norm = RMSNorm(d_model)
        if self.predictor_span_length_embedding:
            # This metadata conditions only the disposable mask predictor.  It
            # is deliberately absent from ``encode`` and downstream features.
            self.span_length_embed = nn.Embedding(
                self.max_mask_span_length + 1,
                d_model,
                padding_idx=0,
            )
            nn.init.normal_(self.span_length_embed.weight, std=0.02)
            with torch.no_grad():
                self.span_length_embed.weight[0].zero_()
        else:
            self.span_length_embed = None
        if self.target_type in {TARGET_RAW_PATCH, TARGET_DIRECT_RECONSTRUCTION}:
            self.pred_head = PredictionHead(
                d_model=d_model, K_e=4 * self.patch_samples
            )
        elif self.target_type == TARGET_NORMALIZED_LATENT:
            self.pred_head = PredictionHead(d_model=d_model, K_e=int(latent_dim))
        elif self.factorized_fsq:
            # Factorized FSQ head: one Linear for each configured dimension.
            self.pred_head = nn.ModuleList(
                [nn.Linear(d_model, li) for li in self.fsq_L]
            )
        else:
            self.pred_head = PredictionHead(d_model=d_model, K_e=K_e)
        if self.target_type == TARGET_DIRECT_RECONSTRUCTION:
            if self.manual_feature_dim <= 0:
                raise ValueError("direct reconstruction requires manual_feature_dim > 0")
            if not isinstance(self.direct_loss_cfg, dict):
                raise ValueError("direct reconstruction requires a complete loss config")
            self.manual_feat_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.manual_feature_dim),
            )
        else:
            self.manual_feat_head = None

        # 缓存 stim-isolated 结构 mask (用 max_time 预构建, forward 只切片)
        self._stim_mask_base: torch.Tensor | None = None
        if stim_isolated_attn:
            total = 1 + 3 * int(max_time)
            self._stim_mask_base = build_stim_isolated_attn_mask(
                total,
                int(max_time),
                "cpu",
                with_cls=True,
                stim_attend_cls=self.stim_attend_cls,
            )

    def _get_stim_mask(self, n_time: int, device) -> torch.Tensor | None:
        if self._stim_mask_base is None:
            return None
        total = 1 + 3 * n_time
        return self._stim_mask_base[:total, :total].to(device)

    def encode(
        self,
        stim_patches: torch.Tensor,
        eye_patches: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor,
        task_ids: torch.Tensor,
        bert_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode raw patches with the exact pretraining attention graph."""
        B, N = stim_patches.shape[:2]
        seq = self.embed(
            stim=stim_patches,
            eye_l=eye_patches[:, :, 0, :, :],
            eye_r=eye_patches[:, :, 1, :, :],
            eye_nonmissing_frac=eye_nonmissing_frac,
            task_ids=task_ids,
            bert_mask=bert_mask,
        )
        cls_pad = pad_mask.new_zeros(B, 1)
        eye_invalid = eye_nonmissing_frac < self.min_nonmissing_frac
        body_pad = torch.stack(
            [pad_mask, pad_mask | eye_invalid[:, :, 0], pad_mask | eye_invalid[:, :, 1]],
            dim=2,
        ).reshape(B, N * 3)
        key_padding_mask = torch.cat([cls_pad, body_pad], dim=1)
        attn_mask = self._get_stim_mask(N, stim_patches.device) if self.stim_isolated_attn else None
        hidden = seq
        for block in self.transformer:
            if self.attention_layout == "axial_cross":
                hidden = block(hidden, key_padding_mask=key_padding_mask, n_time=N)
            else:
                hidden = block(
                    hidden, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
        return self.out_norm(hidden)

    def forward(
        self,
        stim_patches: torch.Tensor,          # [B, N, 4, 20]  raw
        eye_patches: torch.Tensor,           # [B, N, 2, 4, 20]
        quality: torch.Tensor,               # [B, N, 2, 20, 1]  (unused)
        pad_mask: torch.Tensor,              # [B, N]
        eye_nonmissing_frac: torch.Tensor,   # [B, N, 2]
        task_ids: torch.Tensor,              # [B]
        eye_code_ids: torch.Tensor | None,   # [B, N, 2], unused for raw_patch
        bert_mask: torch.Tensor,             # [B, 1+N*3]
        mask_span_lengths: torch.Tensor | None = None,  # [B,N], zero for visible patches
        eye_latent_targets: torch.Tensor | None = None, # [B,N,2,D], normalized AE targets
        manual_feature_targets: torch.Tensor | None = None,  # [B,F]
        manual_feature_loss_mask: torch.Tensor | None = None,  # [B,F]
        manual_feature_binary_mask: torch.Tensor | None = None,  # [F]
        manual_feature_count_mask: torch.Tensor | None = None,  # [F]
        manual_feature_pos_weight: torch.Tensor | None = None,  # [F]
        manual_feature_bce_scale: torch.Tensor | None = None,  # [F]
        manual_feature_weights: torch.Tensor | None = None,  # [B,F]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward. eye_code_ids are ground-truth labels, not inputs."""
        B, N = stim_patches.shape[:2]
        total_len = 1 + N * 3
        device = stim_patches.device

        hidden = self.encode(
            stim_patches, eye_patches, pad_mask, eye_nonmissing_frac, task_ids, bert_mask
        )
        # ── Masked positions (before pred_head to save ~60% head computation) ──
        # Positions: 2,5,8,... = L; 3,6,9,... = R  for t=0,1,...,N-1
        positions = torch.arange(total_len, device=device)
        is_l = (positions >= 2) & ((positions - 2) % 3 == 0)  # 2,5,8,...
        is_r = (positions >= 3) & ((positions - 3) % 3 == 0)  # 3,6,9,...

        l_masked = bert_mask & is_l.unsqueeze(0)
        r_masked = bert_mask & is_r.unsqueeze(0)
        eye_masked = (l_masked | r_masked)  # [B, total_len]

        # 只对 masked 位置算 logits (替代全序列 logits_all = pred_head(hidden))
        masked_hidden = hidden[eye_masked]             # [N_masked, d_model]
        masked_trial_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(B, total_len)[eye_masked]
        )

        if self.span_length_embed is not None:
            if mask_span_lengths is None or mask_span_lengths.shape != (B, N):
                raise ValueError(
                    "predictor span-length embedding requires mask_span_lengths with "
                    f"shape {(B, N)}"
                )
            flat_span_lengths = torch.zeros(
                B, total_len, dtype=torch.long, device=device
            )
            flat_span_lengths[:, 2::3] = mask_span_lengths
            flat_span_lengths[:, 3::3] = mask_span_lengths
            masked_span_lengths = flat_span_lengths[eye_masked]
            masked_hidden = masked_hidden + self.span_length_embed(masked_span_lengths)

        if self.target_type in {TARGET_RAW_PATCH, TARGET_DIRECT_RECONSTRUCTION}:
            flat_raw = eye_patches.new_zeros(B, total_len, 4, self.patch_samples)
            flat_raw[:, 2::3] = eye_patches[:, :, 0]
            flat_raw[:, 3::3] = eye_patches[:, :, 1]
            flat_missing = torch.ones(
                B, total_len, self.patch_samples, dtype=torch.bool, device=device
            )
            missing = quality.squeeze(-1) > 0.5
            flat_missing[:, 2::3] = missing[:, :, 0]
            flat_missing[:, 3::3] = missing[:, :, 1]
            masked_targets_raw = flat_raw[eye_masked]
            masked_missing = flat_missing[eye_masked]
            prediction = self.pred_head(masked_hidden).reshape(
                -1, 4, self.patch_samples
            )
            if self.target_type == TARGET_DIRECT_RECONSTRUCTION:
                l_time_mask = l_masked[:, 2::3]
                r_time_mask = r_masked[:, 3::3]
                if not torch.equal(l_time_mask, r_time_mask):
                    raise ValueError(
                        "direct reconstruction requires paired L/R masking"
                    )
                masked_eye = torch.stack([l_time_mask, r_time_mask], dim=2)
                prediction_full = prediction.new_zeros(
                    B, N, 2, 4, self.patch_samples
                )
                prediction_full[masked_eye] = prediction
                reconstruction_pad_mask = ~masked_eye.any(dim=2)
                eye_loss, eye_stats = compute_eye_recon_loss(
                    prediction_full,
                    eye_patches,
                    quality,
                    reconstruction_pad_mask,
                    {"loss": self.direct_loss_cfg},
                )
                trial_is_supervised = masked_eye.any(dim=(1, 2))
                gated_feature_mask = manual_feature_loss_mask
                if gated_feature_mask is not None:
                    gated_feature_mask = (
                        gated_feature_mask.bool()
                        & trial_is_supervised.unsqueeze(1)
                    )
                feature_prediction = self.manual_feat_head(hidden[:, 0, :])
                feature_loss, feature_stats = compute_manual_feature_loss(
                    feature_prediction,
                    manual_feature_targets,
                    gated_feature_mask,
                    manual_feature_binary_mask,
                    {"loss": self.direct_loss_cfg},
                    feature_weights=manual_feature_weights,
                    count_mask=manual_feature_count_mask,
                    binary_pos_weight=manual_feature_pos_weight,
                    binary_bce_scale=manual_feature_bce_scale,
                )
                eye_weighted = float(
                    self.direct_loss_cfg.get("eye_recon_group_weight", 1.0)
                ) * eye_loss
                feature_weighted = float(
                    self.direct_loss_cfg.get("manual_feature_group_weight", 0.0)
                ) * feature_loss
                loss = eye_weighted + feature_weighted
                return loss, {
                    "bert_loss": loss.detach(),
                    "n_masked": masked_eye.sum(),
                    "n_supervised_trials": trial_is_supervised.sum(),
                    "L_total": loss.detach(),
                    "L_eye_weighted": eye_weighted.detach(),
                    "L_feat_weighted": feature_weighted.detach(),
                    **eye_stats,
                    **feature_stats,
                }
            xy_loss, area_loss, blink_loss, velocity_loss, valid = (
                raw_patch_reconstruction_losses(
                    prediction,
                    masked_targets_raw,
                    masked_missing,
                    blink_pos_weight=self.raw_blink_pos_weight,
                )
            )
            per_token_loss = (
                self.raw_xy_weight * xy_loss
                + self.raw_area_weight * area_loss
                + self.raw_blink_weight * blink_loss
                + self.raw_velocity_weight * velocity_loss
            )
            loss, n_supervised_trials = mean_masked_loss_per_trial(
                per_token_loss, masked_trial_indices, B
            )
            with torch.no_grad():
                blink_prediction = prediction[:, 3, :] >= 0
                blink_target = masked_targets_raw[:, 3, :] >= 0.5
                accuracy = (
                    ((blink_prediction == blink_target) & valid).sum()
                    / valid.sum().clamp_min(1)
                )
            return loss, {
                "bert_loss": loss.detach(),
                "bert_acc": accuracy,
                "n_masked": eye_masked.sum(),
                "n_supervised_trials": n_supervised_trials,
                "per_token_loss": per_token_loss.detach(),
                "raw_xy_loss": xy_loss.mean().detach(),
                "raw_area_loss": area_loss.mean().detach(),
                "raw_blink_loss": blink_loss.mean().detach(),
                "raw_velocity_loss": velocity_loss.mean().detach(),
            }

        if self.target_type == TARGET_NORMALIZED_LATENT:
            if eye_latent_targets is None:
                raise ValueError("eye_latent_targets are required for normalized_latent")
            if eye_latent_targets.shape[:3] != (B, N, 2):
                raise ValueError(
                    "eye_latent_targets must have shape [B,N,2,D], got "
                    f"{tuple(eye_latent_targets.shape)}"
                )
            target_dim = eye_latent_targets.shape[-1]
            flat_targets = eye_latent_targets.new_zeros(B, total_len, target_dim)
            flat_targets[:, 2::3] = eye_latent_targets[:, :, 0]
            flat_targets[:, 3::3] = eye_latent_targets[:, :, 1]
            masked_targets = flat_targets[eye_masked].float()
            prediction = self.pred_head(masked_hidden).float()
            per_token_loss = F.mse_loss(
                prediction, masked_targets, reduction="none"
            ).mean(dim=-1)
            loss, n_supervised_trials = mean_masked_loss_per_trial(
                per_token_loss, masked_trial_indices, B
            )
            with torch.no_grad():
                cosine = F.cosine_similarity(prediction, masked_targets, dim=-1).mean()
            return loss, {
                "bert_loss": loss.detach(),
                # Kept under the generic metric key so existing distributed
                # aggregation remains compatible; for this target it is cosine.
                "bert_acc": cosine,
                "latent_cosine": cosine,
                "latent_mse": per_token_loss.mean().detach(),
                "n_masked": eye_masked.sum(),
                "n_supervised_trials": n_supervised_trials,
                "per_token_loss": per_token_loss.detach(),
            }

        if eye_code_ids is None:
            raise ValueError(f"eye_code_ids are required for target_type={self.target_type}")
        flat_targets = torch.zeros(B, total_len, dtype=torch.long, device=device)
        flat_targets[:, 2::3] = eye_code_ids[:, :, 0]
        flat_targets[:, 3::3] = eye_code_ids[:, :, 1]
        masked_targets = flat_targets[eye_masked]

        ce_kwargs = dict(label_smoothing=self.label_smoothing)

        if self.factorized_fsq:
            # ── Factorized FSQ head: 每维各自 CE (保留联合 exact accuracy) ──
            dim_logits = [h(masked_hidden) for h in self.pred_head]                 # list[D] of [M, Ld]
            coords = code_ids_to_coords(masked_targets, self.fsq_base, self.fsq_L)  # [M, D]
            per_dim_ce, per_token_loss = factorized_fsq_cross_entropy(
                dim_logits,
                coords,
                label_smoothing=self.label_smoothing,
            )
            loss, n_supervised_trials = mean_masked_loss_per_trial(
                per_token_loss, masked_trial_indices, B
            )
            per_dim_trial_ce = torch.stack([
                mean_masked_loss_per_trial(
                    per_dim_ce[:, dimension], masked_trial_indices, B
                )[0].detach()
                for dimension in range(per_dim_ce.shape[1])
            ])

            with torch.no_grad():
                pred_coords = torch.stack(
                    [dl.argmax(dim=-1) for dl in dim_logits], dim=-1)               # [M, D]
                pred_ids = (pred_coords * torch.tensor(self.fsq_base, device=device)).sum(-1)
                acc = (pred_ids == masked_targets).float().mean()
                per_dim_acc = (pred_coords == coords).float().mean(0)              # [D]
                dim_dist = (pred_coords - coords).abs().float().mean(0)             # [D]
            return loss, {
                "bert_loss": loss.detach(), "bert_acc": acc, "n_masked": eye_masked.sum(),
                "n_supervised_trials": n_supervised_trials,
                "per_token_loss": per_token_loss.detach(), "masked_targets": masked_targets.detach(),
                "per_dim_ce": per_dim_trial_ce,
                "per_dim_acc": per_dim_acc, "dim_dist": dim_dist,
            }

        # ── Joint 11025-way head ──
        masked_logits = self.pred_head(masked_hidden)                              # [N_masked, K_e]
        per_token_loss = F.cross_entropy(masked_logits, masked_targets, reduction="none", **ce_kwargs)
        loss, n_supervised_trials = mean_masked_loss_per_trial(
            per_token_loss, masked_trial_indices, B
        )

        with torch.no_grad():
            pred_ids = masked_logits.argmax(dim=-1)
            acc = (pred_ids == masked_targets).float().mean()
            top5 = masked_logits.topk(5, dim=-1).indices
            top10 = masked_logits.topk(10, dim=-1).indices
            acc_top5 = (top5 == masked_targets.unsqueeze(-1)).any(-1).float().mean()
            acc_top10 = (top10 == masked_targets.unsqueeze(-1)).any(-1).float().mean()
            stats = {
                "bert_loss": loss.detach(), "bert_acc": acc, "n_masked": eye_masked.sum(),
                "n_supervised_trials": n_supervised_trials,
                "per_token_loss": per_token_loss.detach(), "masked_targets": masked_targets.detach(),
                "acc_top5": acc_top5, "acc_top10": acc_top10,
            }
            # FSQ digit distance (需要 fsq_L 配置)
            if self.fsq_L is not None:
                pred_coords = code_ids_to_coords(pred_ids, self.fsq_base, self.fsq_L)
                tgt_coords = code_ids_to_coords(masked_targets, self.fsq_base, self.fsq_L)
                stats["dim_dist"] = (pred_coords - tgt_coords).abs().float().mean(0)
            return loss, stats
