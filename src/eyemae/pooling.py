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


def eye_separate_pool(
    hidden_eye: torch.Tensor,
    eye_token_valid: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool left and right eye tokens separately, then concat.

    Args:
        hidden_eye: (B, n_patches, 2, d_model) — dim=2 is [left, right].
        eye_token_valid: (B, n_patches, 2) — valid eye token mask.
        pad_mask: (B, n_patches) — sequence padding mask.

    Returns:
        pooled: (B, 2 * d_model) — [left_pooled, right_pooled].
        has_valid: (B,) — True if any eye token is valid.
        valid_count: (B,) — total valid eye token count.
    """
    not_pad = ~pad_mask  # (B, n_patches)
    pooled_parts: list[torch.Tensor] = []
    total_valid = torch.zeros(hidden_eye.shape[0], device=hidden_eye.device)

    for eye_idx in range(2):
        h = hidden_eye[:, :, eye_idx, :]  # (B, n_patches, d_model)
        v = not_pad & eye_token_valid[:, :, eye_idx]  # (B, n_patches)
        w = v.to(h.dtype).unsqueeze(-1)  # (B, n_patches, 1)
        denom = w.sum(dim=1).clamp_min(1.0)  # (B, 1)
        p = (h * w).sum(dim=1) / denom  # (B, d_model)
        pooled_parts.append(p)
        total_valid = total_valid + w.sum(dim=(1, 2))

    pooled = torch.cat(pooled_parts, dim=-1)  # (B, 2 * d_model)
    has_valid = total_valid > 0
    return pooled, has_valid, total_valid


def eye_stim_separate_pool(
    hidden_seq: torch.Tensor,
    hidden_eye: torch.Tensor,
    eye_token_valid: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool stimulus, left-eye, and right-eye tokens separately, then concat.

    hidden_seq: (B, 3 * n, d_model) — interleaved [S₀, L₀, R₀, S₁, L₁, R₁, …].
    hidden_eye: (B, n, 2, d_model) — [left, right] eye tokens.
    """
    not_pad = ~pad_mask  # (B, n)
    pooled_parts: list[torch.Tensor] = []
    total_valid = torch.zeros(hidden_seq.shape[0], device=hidden_seq.device)

    # Stimulus tokens (positions 0::3) — all are valid (always present).
    s_tokens = hidden_seq[:, 0::3, :]  # (B, n, d_model)
    w_s = not_pad.to(s_tokens.dtype).unsqueeze(-1)
    denom_s = w_s.sum(dim=1).clamp_min(1.0)
    pooled_parts.append((s_tokens * w_s).sum(dim=1) / denom_s)
    total_valid = total_valid + w_s.sum(dim=(1, 2))

    # Left / right eye tokens.
    for eye_idx in range(2):
        h = hidden_eye[:, :, eye_idx, :]
        v = not_pad & eye_token_valid[:, :, eye_idx]
        w = v.to(h.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        pooled_parts.append((h * w).sum(dim=1) / denom)
        total_valid = total_valid + w.sum(dim=(1, 2))

    pooled = torch.cat(pooled_parts, dim=-1)  # (B, 3 * d_model)
    has_valid = total_valid > 0
    return pooled, has_valid, total_valid
