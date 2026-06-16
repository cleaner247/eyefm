from __future__ import annotations

import torch


def eye_mean_pool(
    hidden_eye: torch.Tensor,
    eye_token_valid: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (~pad_mask)[:, :, None] & eye_token_valid
    weights = valid.to(hidden_eye.dtype)[..., None]
    denominator = weights.sum(dim=(1, 2)).squeeze(-1)
    pooled = (hidden_eye * weights).sum(dim=(1, 2)) / denominator.clamp_min(1.0)[:, None]
    return pooled, denominator > 0, denominator
