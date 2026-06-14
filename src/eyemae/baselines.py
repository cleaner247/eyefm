from __future__ import annotations

import math
from typing import Any

import torch


def new_baseline_sums(device: torch.device | None = None) -> dict[str, torch.Tensor]:
    d = device or torch.device("cpu")
    return {
        "previous_xy_sq_sum": torch.zeros((), dtype=torch.float64, device=d),
        "previous_xy_count": torch.zeros((), dtype=torch.float64, device=d),
        "linear_xy_sq_sum": torch.zeros((), dtype=torch.float64, device=d),
        "linear_xy_count": torch.zeros((), dtype=torch.float64, device=d),
    }


def update_baseline_sums(
    sums: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mae_mask: torch.Tensor,
    cfg: dict[str, Any],
    token_filter: torch.Tensor | None = None,
) -> None:
    target = batch["content"]
    valid = _valid_coord_mask(batch, mae_mask)
    if token_filter is not None:
        valid = valid & token_filter[:, :, :, None]
    source_visible = _source_coord_mask(batch, mae_mask)

    bsz, n_patches, eyes, patch = valid.shape
    total_frames = n_patches * patch
    true_xy = target[..., 0:2].permute(0, 2, 1, 3, 4).reshape(bsz, eyes, total_frames, 2)
    valid_flat = valid.permute(0, 2, 1, 3).reshape(bsz, eyes, total_frames)
    source_flat = source_visible.permute(0, 2, 1, 3).reshape(bsz, eyes, total_frames)

    idx_1d = torch.arange(total_frames, device=target.device)
    idx = idx_1d.view(1, 1, total_frames).expand(bsz, eyes, total_frames)
    missing_idx = torch.full_like(idx, -1)

    prev_idx = torch.cummax(torch.where(source_flat, idx, missing_idx), dim=-1).values
    has_prev = prev_idx >= 0
    prev_xy = _gather_time(true_xy, prev_idx.clamp_min(0))
    previous_valid = valid_flat & has_prev
    _accumulate_xy_error(sums, "previous", prev_xy, true_xy, previous_valid, cfg)

    rev_original_idx = idx.flip(-1)
    next_idx = torch.cummax(torch.where(source_flat.flip(-1), rev_original_idx, missing_idx), dim=-1).values.flip(-1)
    has_next = next_idx >= 0
    next_xy = _gather_time(true_xy, next_idx.clamp_min(0))

    both = has_prev & has_next
    denom = (next_idx - prev_idx).clamp_min(1).to(true_xy.dtype)
    alpha = ((idx - prev_idx).to(true_xy.dtype) / denom).clamp(0.0, 1.0).unsqueeze(-1)
    interp_xy = prev_xy * (1.0 - alpha) + next_xy * alpha
    linear_xy = torch.where(both.unsqueeze(-1), interp_xy, torch.where(has_prev.unsqueeze(-1), prev_xy, next_xy))
    linear_valid = valid_flat & (has_prev | has_next)
    _accumulate_xy_error(sums, "linear", linear_xy, true_xy, linear_valid, cfg)


def finalize_baseline_sums(sums: dict[str, torch.Tensor], prefix: str = "") -> dict[str, float]:
    def rmse(name: str) -> float:
        count = float(sums[f"{name}_xy_count"].detach().cpu())
        if count <= 0:
            return math.nan
        mse2 = float(sums[f"{name}_xy_sq_sum"].detach().cpu()) / count
        return math.sqrt(mse2)

    metrics = {
        "previous_value/masked_xy_rmse_deg": rmse("previous"),
        "previous_value/count": float(sums["previous_xy_count"].detach().cpu()),
        "linear_interpolation/masked_xy_rmse_deg": rmse("linear"),
        "linear_interpolation/count": float(sums["linear_xy_count"].detach().cpu()),
    }
    if prefix:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}
    return metrics


def previous_value_baseline(batch: dict[str, torch.Tensor], mae_mask: torch.Tensor, cfg: dict[str, Any]) -> dict[str, float]:
    sums = new_baseline_sums(batch["content"].device)
    update_baseline_sums(sums, batch, mae_mask, cfg)
    metrics = finalize_baseline_sums(sums)
    return {"masked_xy_rmse_deg": metrics["previous_value/masked_xy_rmse_deg"]}


def linear_interpolation_baseline(batch: dict[str, torch.Tensor], mae_mask: torch.Tensor, cfg: dict[str, Any]) -> dict[str, float]:
    sums = new_baseline_sums(batch["content"].device)
    update_baseline_sums(sums, batch, mae_mask, cfg)
    metrics = finalize_baseline_sums(sums)
    return {"masked_xy_rmse_deg": metrics["linear_interpolation/masked_xy_rmse_deg"]}


def _valid_coord_mask(batch: dict[str, torch.Tensor], mae_mask: torch.Tensor) -> torch.Tensor:
    target = batch["content"]
    missing = batch["quality"][..., 0].bool()
    blink = target[..., 3] > 0.5
    return (
        mae_mask[:, :, :, None].expand_as(missing)
        & (~batch["pad_mask"])[:, :, None, None]
        & batch["eye_token_valid"][:, :, :, None]
        & (~missing)
        & (~blink)
    )


def _source_coord_mask(batch: dict[str, torch.Tensor], mae_mask: torch.Tensor) -> torch.Tensor:
    target = batch["content"]
    missing = batch["quality"][..., 0].bool()
    blink = target[..., 3] > 0.5
    return (
        (~mae_mask)[:, :, :, None].expand_as(missing)
        & (~batch["pad_mask"])[:, :, None, None]
        & batch["eye_token_valid"][:, :, :, None]
        & (~missing)
        & (~blink)
    )


def _gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 2, indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]))


def _accumulate_xy_error(
    sums: dict[str, torch.Tensor],
    name: str,
    pred_xy: torch.Tensor,
    true_xy: torch.Tensor,
    valid: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    if not bool(valid.any()):
        return
    scale = torch.tensor(
        [float(cfg["normalization"]["x_clip_deg"]), float(cfg["normalization"]["y_clip_deg"])],
        dtype=pred_xy.dtype,
        device=pred_xy.device,
    )
    err = (pred_xy - true_xy) * scale
    sq = err.pow(2).sum(dim=-1)
    sums[f"{name}_xy_sq_sum"] += sq[valid].double().sum()
    sums[f"{name}_xy_count"] += valid.double().sum()
