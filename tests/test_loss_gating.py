from __future__ import annotations

import torch

from eyemae.config import load_config
from eyemae.losses import compute_reconstruction_loss


def test_loss_gating_rules() -> None:
    cfg = load_config("configs/debug.yaml")
    pred = torch.ones(1, 2, 2, 20, 4)
    target = torch.zeros_like(pred)
    quality = torch.zeros(1, 2, 2, 20, 1)
    mae_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    pad_mask = torch.zeros(1, 2, dtype=torch.bool)
    eye_valid = torch.ones(1, 2, 2, dtype=torch.bool)
    quality[:, 0, 0, :, 0] = 1.0
    target[:, 0, 1, :, 3] = 1.0
    pad_mask[:, 1] = True
    loss, stats = compute_reconstruction_loss(pred, target, quality, mae_mask, pad_mask, eye_valid, cfg)
    assert torch.isfinite(loss)
    assert stats["xy_denominator"] == 0
    assert stats["area_denominator"] == 0
    assert stats["blink_denominator"] == 20


def test_denominator_zero_is_not_nan() -> None:
    cfg = load_config("configs/debug.yaml")
    pred = torch.ones(1, 1, 2, 20, 4)
    target = torch.zeros_like(pred)
    quality = torch.ones(1, 1, 2, 20, 1)
    mae_mask = torch.zeros(1, 1, 2, dtype=torch.bool)
    pad_mask = torch.ones(1, 1, dtype=torch.bool)
    eye_valid = torch.zeros(1, 1, 2, dtype=torch.bool)
    loss, stats = compute_reconstruction_loss(pred, target, quality, mae_mask, pad_mask, eye_valid, cfg)
    assert loss.item() == 0.0
    assert stats["total_denominator"].item() == 0.0
