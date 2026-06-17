from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_mean(loss_tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_f = mask.to(loss_tensor.dtype)
    numerator = (loss_tensor * mask_f).sum()
    denominator = mask_f.sum()
    value = numerator / denominator.clamp_min(eps)
    value = torch.where(denominator > 0, value, numerator * 0.0)
    return value, numerator.detach(), denominator.detach()


def compute_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    mae_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    eye_token_valid: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg["loss"]
    missing = quality[..., 0].bool()
    target_blink = target[..., 3] > 0.5
    nonpad = (~pad_mask)[:, :, None, None]
    eye_valid = eye_token_valid[:, :, :, None]
    if bool(loss_cfg["loss_only_on_mae_mask"]):
        loss_token = mae_mask
    else:
        loss_token = torch.ones_like(mae_mask, dtype=torch.bool)
    loss_frame = loss_token[:, :, :, None].expand_as(missing)
    base_valid = loss_frame & nonpad & eye_valid
    if bool(loss_cfg["ignore_missing_for_all_losses"]):
        base_valid = base_valid & (~missing)
    coord_valid = base_valid
    if bool(loss_cfg["ignore_blink_for_xy_area_loss"]):
        coord_valid = coord_valid & (~target_blink)
    blink_valid = base_valid

    xy_raw = F.smooth_l1_loss(pred[..., 0:2], target[..., 0:2], reduction="none")
    xy_loss, xy_num, xy_den = masked_mean(xy_raw, coord_valid[..., None].expand_as(xy_raw))

    area_raw = F.smooth_l1_loss(pred[..., 2], target[..., 2], reduction="none")
    area_loss, area_num, area_den = masked_mean(area_raw, coord_valid)

    blink_raw = F.binary_cross_entropy_with_logits(pred[..., 3], target[..., 3], reduction="none")
    blink_loss, blink_num, blink_den = masked_mean(blink_raw, blink_valid)

    if bool(loss_cfg["velocity_within_patch_only"]):
        v_pred = pred[..., 1:, 0:2] - pred[..., :-1, 0:2]
        v_true = target[..., 1:, 0:2] - target[..., :-1, 0:2]
        v_valid = coord_valid[..., 1:] & coord_valid[..., :-1]
    else:
        bsz, n_patches, n_eyes, patch_samples, _ = pred.shape
        v_pred_xy = pred[..., 0:2].permute(0, 1, 3, 2, 4).reshape(bsz, n_patches * patch_samples, n_eyes, 2)
        v_true_xy = target[..., 0:2].permute(0, 1, 3, 2, 4).reshape(bsz, n_patches * patch_samples, n_eyes, 2)
        v_valid_frames = coord_valid.permute(0, 1, 3, 2).reshape(bsz, n_patches * patch_samples, n_eyes)
        v_pred = v_pred_xy[:, 1:] - v_pred_xy[:, :-1]
        v_true = v_true_xy[:, 1:] - v_true_xy[:, :-1]
        v_valid = v_valid_frames[:, 1:] & v_valid_frames[:, :-1]
    v_raw = F.smooth_l1_loss(v_pred, v_true, reduction="none")
    velocity_loss, velocity_num, velocity_den = masked_mean(v_raw, v_valid[..., None].expand_as(v_raw))

    total = (
        float(loss_cfg["xy_weight"]) * xy_loss
        + float(loss_cfg["area_weight"]) * area_loss
        + float(loss_cfg["blink_weight"]) * blink_loss
        + float(loss_cfg["velocity_weight"]) * velocity_loss
    )
    total_den = xy_den + area_den + blink_den + velocity_den
    stats = {
        "total_loss": total.detach(),
        "xy_loss": xy_loss.detach(),
        "area_loss": area_loss.detach(),
        "blink_loss": blink_loss.detach(),
        "velocity_loss": velocity_loss.detach(),
        "xy_denominator": xy_den,
        "area_denominator": area_den,
        "blink_denominator": blink_den,
        "velocity_denominator": velocity_den,
        "total_denominator": total_den,
        "xy_numerator": xy_num,
        "area_numerator": area_num,
        "blink_numerator": blink_num,
        "velocity_numerator": velocity_num,
    }
    return total, stats
