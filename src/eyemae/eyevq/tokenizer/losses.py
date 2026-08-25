"""
Loss functions for EyeVQ tokenizer training (ViT-based).

Includes:
  - Eye patch reconstruction losses (xy/area/blink/velocity)
  - Manual feature prediction loss (38-dim trial-level)
  - VQ commitment loss (L/R eye only)
  - No stim reconstruction loss
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Weighted mean over valid positions."""
    mask_f = mask.to(x.dtype)
    num = (x * mask_f).sum()
    den = mask_f.sum().clamp(min=eps)
    return num / den


# ──────────────────────────────────────────────
# Eye Patch Reconstruction Loss
# ──────────────────────────────────────────────

def compute_eye_recon_loss(
    pred_eye: torch.Tensor,        # [B, N, 2, 4, 20]
    target_eye: torch.Tensor,      # [B, N, 2, 4, 20]
    quality: torch.Tensor,         # [B, N, 2, 20, 1]
    pad_mask: torch.Tensor,        # [B, N]
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-patch eye reconstruction losses.

    Loss gating:
      - missing frames: excluded from ALL losses
      - blink frames: excluded from xy/area, included in blink BCE
      - valid non-blink: all losses apply
    """
    loss_cfg = cfg["loss"]

    B, N = pred_eye.shape[:2]
    missing = quality[..., 0] > 0.5                          # [B, N, 2, 20]
    target_blink = target_eye[..., 3, :] > 0.5               # [B, N, 2, 20]
    nonpad = (~pad_mask.bool())[:, :, None, None]             # [B, N, 1, 1]

    # Base valid: non-pad, non-missing
    base_valid = nonpad & (~missing)

    # Coord valid: also exclude blink
    coord_valid = base_valid & (~target_blink)

    # Blink valid: base valid (includes blink frames)
    blink_valid = base_valid

    # ── Eye losses ──
    # xy
    xy_raw = F.smooth_l1_loss(
        pred_eye[..., 0:2, :], target_eye[..., 0:2, :], reduction="none"
    )  # [B, N, 2, 2, 20]
    xy_loss = masked_mean(xy_raw, coord_valid.unsqueeze(-2).expand_as(xy_raw))

    # area
    area_raw = F.smooth_l1_loss(
        pred_eye[..., 2, :], target_eye[..., 2, :], reduction="none"
    )
    area_loss = masked_mean(area_raw, coord_valid)

    # Blink.  ``eye_blink_pos_weight`` is optional and normalized by the
    # corresponding class-weight mass, so changing it changes positive versus
    # negative emphasis without silently changing the overall BCE scale.
    blink_pos_weight = float(loss_cfg.get("eye_blink_pos_weight", 1.0))
    if blink_pos_weight <= 0.0:
        raise ValueError("loss.eye_blink_pos_weight must be positive")
    blink_raw = F.binary_cross_entropy_with_logits(
        pred_eye[..., 3, :],
        target_eye[..., 3, :],
        pos_weight=pred_eye.new_tensor(blink_pos_weight),
        reduction="none",
    )
    blink_mask_f = blink_valid.to(blink_raw.dtype)
    blink_class_weight = torch.where(
        target_blink,
        blink_raw.new_tensor(blink_pos_weight),
        blink_raw.new_tensor(1.0),
    )
    blink_loss = (blink_raw * blink_mask_f).sum() / (
        blink_class_weight * blink_mask_f
    ).sum().clamp(min=1e-8)
    blink_positive_fraction = masked_mean(target_blink.float(), blink_valid)

    # velocity (within-patch first difference)
    v_pred = pred_eye[..., 0:2, 1:] - pred_eye[..., 0:2, :-1]  # [B,N,2,2,19]
    v_true = target_eye[..., 0:2, 1:] - target_eye[..., 0:2, :-1]
    v_valid = coord_valid[..., 1:] & coord_valid[..., :-1]
    v_raw = F.smooth_l1_loss(v_pred, v_true, reduction="none")
    vel_loss = masked_mean(v_raw, v_valid.unsqueeze(-2).expand_as(v_raw))

    # ── Weighted total ──
    w = loss_cfg
    xy_loss_weighted = float(w["eye_xy_weight"]) * xy_loss
    area_loss_weighted = float(w["eye_area_weight"]) * area_loss
    blink_loss_weighted = float(w["eye_blink_weight"]) * blink_loss
    vel_loss_weighted = float(w["eye_velocity_weight"]) * vel_loss
    L_eye = (
        xy_loss_weighted
        + area_loss_weighted
        + blink_loss_weighted
        + vel_loss_weighted
    )

    stats = {
        "xy_loss": xy_loss.detach(),
        "area_loss": area_loss.detach(),
        "blink_loss": blink_loss.detach(),
        "vel_loss": vel_loss.detach(),
        "xy_loss_weighted": xy_loss_weighted.detach(),
        "area_loss_weighted": area_loss_weighted.detach(),
        "blink_loss_weighted": blink_loss_weighted.detach(),
        "vel_loss_weighted": vel_loss_weighted.detach(),
        "blink_positive_fraction": blink_positive_fraction.detach(),
        "L_eye": L_eye.detach(),
    }

    return L_eye, stats


# ──────────────────────────────────────────────
# Manual Feature Prediction Loss
# ──────────────────────────────────────────────

def compute_manual_feature_loss(
    pred_features: torch.Tensor,     # [B, num_features]
    target_features: torch.Tensor | None,   # [B, num_features] or None
    loss_mask: torch.Tensor | None,         # [B, num_features] bool — per-trial per-feature gate
    binary_mask: torch.Tensor | None,       # [num_features] bool — binary features use BCE
    cfg: dict[str, Any],
    feature_weights: torch.Tensor | None = None,  # [B, num_features] per-trial per-feature weight
    feat_scales: torch.Tensor | None = None,      # [num_features] per-feature loss scale
    count_mask: torch.Tensor | None = None,       # [num_features] bool — excluded from loss
    binary_pos_weight: torch.Tensor | None = None,  # [num_features], train-only priors
    binary_bce_scale: torch.Tensor | None = None,   # [num_features], preserves BCE scale
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute manual feature loss, gated by loss_mask.

    - Binary features (binary_mask=True): optionally class-balanced BCE with
      logits, using training-split priors only.
    - Continuous features: SmoothL1, target robust z-scored.
    - Count features: excluded; raw integer counts are neither continuous
      normalized targets nor suitable regression targets for this head.
    - Aggregation = per-trial task-weighted average (NOT per-feature sum):
        L_feat = mean over valid trials of [ Σ_j w_ij·raw_ij / Σ_j w_ij ]
      Each trial's task-feature weights sum to 1.0, and every valid trial
      contributes equally → different task's trial-count imbalance does NOT
      bias the loss (previously per-feature sum over-amplified the dominant task).
    """
    if target_features is None or loss_mask is None:
        zero = torch.tensor(0.0, device=pred_features.device)
        return zero, {
            "L_feat": zero.detach(),
            "L_feat_binary": zero.detach(),
            "L_feat_continuous": zero.detach(),
        }

    n_b, n_feat = pred_features.shape
    if target_features.shape != pred_features.shape or loss_mask.shape != pred_features.shape:
        raise ValueError(
            "pred_features, target_features, and loss_mask must have identical [B,F] shapes"
        )

    # Per-trial per-feature raw loss [B, F]
    if binary_mask is not None:
        is_bin = binary_mask.to(device=pred_features.device, dtype=torch.bool)
    else:
        is_bin = torch.zeros(n_feat, dtype=torch.bool, device=pred_features.device)
    if is_bin.shape != (n_feat,):
        raise ValueError(f"binary_mask must have shape {(n_feat,)}, got {tuple(is_bin.shape)}")

    if count_mask is not None:
        is_count = count_mask.to(device=pred_features.device, dtype=torch.bool)
    else:
        is_count = torch.zeros(n_feat, dtype=torch.bool, device=pred_features.device)
    if is_count.shape != (n_feat,):
        raise ValueError(f"count_mask must have shape {(n_feat,)}, got {tuple(is_count.shape)}")
    if torch.any(is_bin & is_count):
        raise ValueError("A manual feature cannot be both binary and count")

    effective_mask = loss_mask.to(dtype=torch.bool) & (~is_count.unsqueeze(0))
    mask_f = effective_mask.to(pred_features.dtype)

    if binary_pos_weight is None:
        pos_weight = torch.ones(n_feat, dtype=pred_features.dtype, device=pred_features.device)
    else:
        pos_weight = binary_pos_weight.to(device=pred_features.device, dtype=pred_features.dtype)
    if pos_weight.shape != (n_feat,) or torch.any(pos_weight <= 0):
        raise ValueError("binary_pos_weight must be positive with shape [num_features]")

    if binary_bce_scale is None:
        bce_scale = torch.ones(n_feat, dtype=pred_features.dtype, device=pred_features.device)
    else:
        bce_scale = binary_bce_scale.to(device=pred_features.device, dtype=pred_features.dtype)
    if bce_scale.shape != (n_feat,) or torch.any(bce_scale <= 0):
        raise ValueError("binary_bce_scale must be positive with shape [num_features]")

    # Evaluate each loss only on its own feature type.  In particular, count
    # columns are not merely multiplied by a zero mask after the fact: no
    # regression loss is constructed for them at all.  Their 38D output rows
    # remain for checkpoint/interface compatibility and receive zero gradient.
    is_continuous = (~is_bin) & (~is_count)
    # Autocast deliberately evaluates BCE/SmoothL1 in FP32 even when the
    # prediction head emits BF16.  Accumulate the heterogeneous feature losses
    # in FP32 as well; assigning an FP32 BCE result into a BF16 temporary would
    # otherwise fail before backward.
    loss_dtype = (
        torch.float32
        if pred_features.dtype in (torch.float16, torch.bfloat16)
        else pred_features.dtype
    )
    raw = torch.zeros(
        pred_features.shape,
        dtype=loss_dtype,
        device=pred_features.device,
    )
    if torch.any(is_bin):
        raw[:, is_bin] = F.binary_cross_entropy_with_logits(
            pred_features[:, is_bin],
            target_features[:, is_bin],
            pos_weight=pos_weight[is_bin],
            reduction="none",
        ).to(loss_dtype) * bce_scale[is_bin].to(loss_dtype)
    if torch.any(is_continuous):
        raw[:, is_continuous] = F.smooth_l1_loss(
            pred_features[:, is_continuous],
            target_features[:, is_continuous],
            reduction="none",
        ).to(loss_dtype)

    # Binary targets are substantially easier for the trial head to memorize
    # than the robust-z continuous targets. Keep the two objective scales
    # independently configurable instead of weakening the whole feature loss.
    loss_cfg = cfg.get("loss", cfg)
    binary_loss_weight = float(loss_cfg.get("manual_feature_binary_weight", 1.0))
    continuous_loss_weight = float(
        loss_cfg.get("manual_feature_continuous_weight", 1.0)
    )
    if binary_loss_weight < 0.0 or continuous_loss_weight < 0.0:
        raise ValueError("Manual-feature type loss weights must be non-negative")
    raw[:, is_bin] *= binary_loss_weight
    raw[:, is_continuous] *= continuous_loss_weight

    # Per-feature scale (applied on raw, before per-trial aggregation)
    if feat_scales is not None:
        raw = raw / feat_scales.clamp(min=1e-8)  # feat_scales [F] broadcasts over B

    # Per-trial task-weighted average
    if feature_weights is not None:
        w = feature_weights.to(device=pred_features.device, dtype=pred_features.dtype)
        if w.shape != pred_features.shape:
            raise ValueError(
                f"feature_weights must have shape {tuple(pred_features.shape)}, got {tuple(w.shape)}"
            )
    else:
        w = torch.ones_like(pred_features)

    combined_weight = mask_f * w
    wsum = combined_weight.sum(dim=1)
    weighted = (raw * combined_weight).sum(dim=1)

    per_trial = weighted / wsum.clamp(min=1e-8)  # [B]
    valid = wsum > 0
    if valid.any():
        L_feat = per_trial[valid].mean()
        binary_numerator = (
            raw * combined_weight * is_bin.to(raw.dtype).unsqueeze(0)
        ).sum(dim=1)
        continuous_numerator = (
            raw
            * combined_weight
            * is_continuous.to(raw.dtype).unsqueeze(0)
        ).sum(dim=1)
        L_feat_binary = (binary_numerator / wsum.clamp(min=1e-8))[valid].mean()
        L_feat_continuous = (
            continuous_numerator / wsum.clamp(min=1e-8)
        )[valid].mean()
    else:
        L_feat = torch.tensor(0.0, device=pred_features.device)
        L_feat_binary = torch.tensor(0.0, device=pred_features.device)
        L_feat_continuous = torch.tensor(0.0, device=pred_features.device)

    stats = {
        "L_feat": L_feat.detach(),
        "L_feat_binary": L_feat_binary.detach(),
        "L_feat_continuous": L_feat_continuous.detach(),
    }
    return L_feat, stats


# ──────────────────────────────────────────────
# Total Loss
# ──────────────────────────────────────────────

def compute_total_loss(
    pred_eye: torch.Tensor,
    target_eye: torch.Tensor,
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
    quality: torch.Tensor,
    pad_mask: torch.Tensor,
    commit_eye: torch.Tensor,
    loss_mask: torch.Tensor | None = None,     # [B, F] per-trial per-feature gate
    binary_mask: torch.Tensor | None = None,   # [F] binary features use BCE
    count_mask: torch.Tensor | None = None,    # [F] count features are excluded
    binary_pos_weight: torch.Tensor | None = None,  # [F] train-prior BCE weights
    binary_bce_scale: torch.Tensor | None = None,   # [F] expected-scale correction
    feature_weights: torch.Tensor | None = None,  # [B, F] per-trial per-feature weights
    feat_scales: torch.Tensor | None = None,      # [F] per-feature loss scale (class-aligned)
    cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute VQ-VAE training loss: eye recon + manual features + VQ commitment."""
    w = cfg["loss"]

    # 1. Eye reconstruction
    L_eye, recon_stats = compute_eye_recon_loss(
        pred_eye, target_eye, quality, pad_mask, cfg,
    )

    # 2. Manual feature prediction (gated by loss_mask, BCE for binary)
    L_feat, feat_stats = compute_manual_feature_loss(
        pred_features, target_features, loss_mask, binary_mask, cfg,
        feature_weights=feature_weights,
        feat_scales=feat_scales,
        count_mask=count_mask,
        binary_pos_weight=binary_pos_weight,
        binary_bce_scale=binary_bce_scale,
    )

    # 3. VQ commitment
    L_commit = commit_eye if isinstance(commit_eye, torch.Tensor) else torch.tensor(0.0, device=L_eye.device)
    L_commit_w = float(w.get("eye_commit_group_weight", 0.0)) * L_commit

    L_eye_w = float(w["eye_recon_group_weight"]) * L_eye
    L_feat_w = float(w.get("manual_feature_group_weight", 0.0)) * L_feat

    L_total = L_eye_w + L_feat_w + L_commit_w

    stats = {**recon_stats, **feat_stats}
    stats["L_total"] = L_total.detach()
    stats["L_eye_weighted"] = L_eye_w.detach()
    stats["L_feat_weighted"] = L_feat_w.detach()
    stats["L_commit_weighted"] = L_commit_w.detach()

    return L_total, stats
