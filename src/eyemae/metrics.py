from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def new_metric_sums(device: torch.device | None = None) -> dict[str, torch.Tensor]:
    d = device or torch.device("cpu")
    scalar_keys = [
        "loss_sum",
        "loss_count",
        "xy_loss_num",
        "xy_loss_den",
        "area_loss_num",
        "area_loss_den",
        "blink_loss_num",
        "blink_loss_den",
        "velocity_loss_num",
        "velocity_loss_den",
        "xy_sq_sum",
        "x_sq_sum",
        "y_sq_sum",
        "xy_count",
        "area_abs_sum",
        "area_count",
        "blink_bce_sum",
        "blink_count",
        "velocity_sq_sum",
        "velocity_count",
        "blink_pos_count",
        "blink_neg_count",
    ]
    sums = {key: torch.zeros((), dtype=torch.float64, device=d) for key in scalar_keys}
    sums["blink_pos_hist"] = torch.zeros(200, dtype=torch.float64, device=d)
    sums["blink_neg_hist"] = torch.zeros(200, dtype=torch.float64, device=d)
    return sums


def update_metric_sums(
    sums: dict[str, torch.Tensor],
    pred: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    mae_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    eye_token_valid: torch.Tensor,
    loss_stats: dict[str, torch.Tensor] | None,
    cfg: dict[str, Any],
    token_filter: torch.Tensor | None = None,
) -> None:
    missing = quality[..., 0].bool()
    target_blink = target[..., 3] > 0.5
    nonpad = (~pad_mask)[:, :, None, None]
    eye_valid = eye_token_valid[:, :, :, None]
    metric_valid = mae_mask[:, :, :, None].expand_as(missing) & nonpad & eye_valid & (~missing)
    if token_filter is not None:
        metric_valid = metric_valid & token_filter[:, :, :, None]
    metric_coord_valid = metric_valid & (~target_blink)

    loss_cfg = cfg["loss"]
    if bool(loss_cfg["loss_only_on_mae_mask"]):
        loss_token = mae_mask
    else:
        loss_token = torch.ones_like(mae_mask, dtype=torch.bool)
    loss_valid = loss_token[:, :, :, None].expand_as(missing) & nonpad & eye_valid
    if bool(loss_cfg["ignore_missing_for_all_losses"]):
        loss_valid = loss_valid & (~missing)
    if token_filter is not None:
        loss_valid = loss_valid & token_filter[:, :, :, None]
    loss_coord_valid = loss_valid
    if bool(loss_cfg["ignore_blink_for_xy_area_loss"]):
        loss_coord_valid = loss_coord_valid & (~target_blink)

    x_scale = float(cfg["normalization"]["x_clip_deg"])
    y_scale = float(cfg["normalization"]["y_clip_deg"])
    x_err = (pred[..., 0] - target[..., 0]) * x_scale
    y_err = (pred[..., 1] - target[..., 1]) * y_scale
    xy_sq = x_err.pow(2) + y_err.pow(2)
    xy_raw = F.smooth_l1_loss(pred[..., 0:2], target[..., 0:2], reduction="none")
    area_raw = F.smooth_l1_loss(pred[..., 2], target[..., 2], reduction="none")
    blink_raw = F.binary_cross_entropy_with_logits(pred[..., 3], target[..., 3], reduction="none")
    metric_v_pred = pred[..., 1:, 0:2] - pred[..., :-1, 0:2]
    metric_v_true = target[..., 1:, 0:2] - target[..., :-1, 0:2]
    metric_v_valid = metric_coord_valid[..., 1:] & metric_coord_valid[..., :-1]

    if loss_stats is not None and token_filter is None:
        sums["loss_sum"] += loss_stats["total_loss"].double()
        sums["loss_count"] += 1.0
        sums["xy_loss_num"] += loss_stats["xy_numerator"].double()
        sums["xy_loss_den"] += loss_stats["xy_denominator"].double()
        sums["area_loss_num"] += loss_stats["area_numerator"].double()
        sums["area_loss_den"] += loss_stats["area_denominator"].double()
        sums["blink_loss_num"] += loss_stats["blink_numerator"].double()
        sums["blink_loss_den"] += loss_stats["blink_denominator"].double()
        sums["velocity_loss_num"] += loss_stats["velocity_numerator"].double()
        sums["velocity_loss_den"] += loss_stats["velocity_denominator"].double()
    else:
        xy_loss_mask = loss_coord_valid[..., None].expand_as(xy_raw)
        if bool(loss_cfg["velocity_within_patch_only"]):
            loss_v_pred = pred[..., 1:, 0:2] - pred[..., :-1, 0:2]
            loss_v_true = target[..., 1:, 0:2] - target[..., :-1, 0:2]
            loss_v_valid = loss_coord_valid[..., 1:] & loss_coord_valid[..., :-1]
        else:
            bsz, n_patches, n_eyes, patch_samples, _ = pred.shape
            loss_pred_xy = pred[..., 0:2].permute(0, 1, 3, 2, 4).reshape(bsz, n_patches * patch_samples, n_eyes, 2)
            loss_true_xy = target[..., 0:2].permute(0, 1, 3, 2, 4).reshape(bsz, n_patches * patch_samples, n_eyes, 2)
            loss_valid_frames = loss_coord_valid.permute(0, 1, 3, 2).reshape(bsz, n_patches * patch_samples, n_eyes)
            loss_v_pred = loss_pred_xy[:, 1:] - loss_pred_xy[:, :-1]
            loss_v_true = loss_true_xy[:, 1:] - loss_true_xy[:, :-1]
            loss_v_valid = loss_valid_frames[:, 1:] & loss_valid_frames[:, :-1]
        loss_v_raw = F.smooth_l1_loss(loss_v_pred, loss_v_true, reduction="none")
        velocity_loss_mask = loss_v_valid[..., None].expand_as(loss_v_raw)
        sums["xy_loss_num"] += xy_raw[xy_loss_mask].double().sum()
        sums["xy_loss_den"] += xy_loss_mask.double().sum()
        sums["area_loss_num"] += area_raw[loss_coord_valid].double().sum()
        sums["area_loss_den"] += loss_coord_valid.double().sum()
        sums["blink_loss_num"] += blink_raw[loss_valid].double().sum()
        sums["blink_loss_den"] += loss_valid.double().sum()
        sums["velocity_loss_num"] += loss_v_raw[velocity_loss_mask].double().sum()
        sums["velocity_loss_den"] += velocity_loss_mask.double().sum()
    sums["xy_sq_sum"] += xy_sq[metric_coord_valid].double().sum()
    sums["x_sq_sum"] += x_err.pow(2)[metric_coord_valid].double().sum()
    sums["y_sq_sum"] += y_err.pow(2)[metric_coord_valid].double().sum()
    sums["xy_count"] += metric_coord_valid.double().sum()
    sums["area_abs_sum"] += (pred[..., 2] - target[..., 2]).abs()[metric_coord_valid].double().sum()
    sums["area_count"] += metric_coord_valid.double().sum()
    sums["blink_bce_sum"] += blink_raw[metric_valid].double().sum()
    sums["blink_count"] += metric_valid.double().sum()
    if bool(metric_valid.any()):
        blink_prob = torch.sigmoid(pred[..., 3].detach())
        pos = metric_valid & target_blink
        neg = metric_valid & (~target_blink)
        sums["blink_pos_count"] += pos.double().sum()
        sums["blink_neg_count"] += neg.double().sum()
        if bool(pos.any()):
            sums["blink_pos_hist"] += torch.histc(blink_prob[pos].double(), bins=200, min=0.0, max=1.0)
        if bool(neg.any()):
            sums["blink_neg_hist"] += torch.histc(blink_prob[neg].double(), bins=200, min=0.0, max=1.0)
    v_scale = torch.tensor([x_scale, y_scale], dtype=pred.dtype, device=pred.device) / 20.0
    v_err = (metric_v_pred - metric_v_true) * v_scale
    sums["velocity_sq_sum"] += v_err.pow(2)[metric_v_valid[..., None].expand_as(v_err)].double().sum()
    sums["velocity_count"] += (metric_v_valid.double().sum() * 2.0)


def reduce_metric_sums(sums: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for value in sums.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return sums


def _histogram_auc(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
    pos_total = float(pos_hist.sum().detach().cpu())
    neg_total = float(neg_hist.sum().detach().cpu())
    if pos_total <= 0 or neg_total <= 0:
        return math.nan
    neg_before = torch.cumsum(neg_hist, dim=0) - neg_hist
    auc_num = (pos_hist * (neg_before + 0.5 * neg_hist)).sum()
    return float((auc_num / (pos_total * neg_total)).detach().cpu())


def finalize_metric_sums(sums: dict[str, torch.Tensor], prefix: str = "", cfg: dict[str, Any] | None = None) -> dict[str, float]:
    def div(num: str, den: str) -> float:
        d = float(sums[den].detach().cpu())
        if d <= 0:
            return math.nan
        return float(sums[num].detach().cpu()) / d

    xy_loss = div("xy_loss_num", "xy_loss_den")
    area_loss = div("area_loss_num", "area_loss_den")
    blink_loss = div("blink_loss_num", "blink_loss_den")
    velocity_loss = div("velocity_loss_num", "velocity_loss_den")
    if cfg is not None and all(math.isfinite(v) for v in (xy_loss, area_loss, blink_loss, velocity_loss)):
        loss_cfg = cfg["loss"]
        total_loss = (
            float(loss_cfg["xy_weight"]) * xy_loss
            + float(loss_cfg["area_weight"]) * area_loss
            + float(loss_cfg["blink_weight"]) * blink_loss
            + float(loss_cfg["velocity_weight"]) * velocity_loss
        )
    else:
        total_loss = div("loss_sum", "loss_count")

    xy_mse2 = div("xy_sq_sum", "xy_count")
    metrics = {
        "total_loss": total_loss,
        "xy_loss": xy_loss,
        "area_loss": area_loss,
        "blink_loss": blink_loss,
        "velocity_loss": velocity_loss,
        "masked_xy_rmse_deg": math.sqrt(xy_mse2) if math.isfinite(xy_mse2) else math.nan,
        "masked_x_rmse_deg": math.sqrt(div("x_sq_sum", "xy_count")) if float(sums["xy_count"].cpu()) > 0 else math.nan,
        "masked_y_rmse_deg": math.sqrt(div("y_sq_sum", "xy_count")) if float(sums["xy_count"].cpu()) > 0 else math.nan,
        "masked_area_mae": div("area_abs_sum", "area_count"),
        "masked_blink_bce": div("blink_bce_sum", "blink_count"),
        "masked_blink_auc": _histogram_auc(sums["blink_pos_hist"], sums["blink_neg_hist"]),
        "masked_velocity_rmse_deg_per_ms": math.sqrt(div("velocity_sq_sum", "velocity_count"))
        if float(sums["velocity_count"].cpu()) > 0
        else math.nan,
    }
    if prefix:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}
    return metrics
