from __future__ import annotations

from typing import Any

import numpy as np
import torch


def patchify_preprocessed_trial(trial: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    samples = int(cfg["patch"]["samples"])
    stride = int(cfg["patch"]["stride"])
    if stride != samples:
        raise ValueError("First version requires patch.stride == patch.samples")
    content = np.asarray(trial["content"], dtype=np.float32)
    quality = np.asarray(trial["quality"], dtype=np.float32)
    stim = np.asarray(trial["stim"], dtype=np.float32)
    t = int(content.shape[0])
    n = t // samples
    if n == 0:
        return None
    keep = n * samples
    content_patch = content[:keep].reshape(n, samples, 2, 4).transpose(0, 2, 1, 3).copy()
    quality_patch = quality[:keep].reshape(n, samples, 2, 1).transpose(0, 2, 1, 3).copy()
    stim_patch = stim[:keep].reshape(n, samples, 4).copy()
    eye_nonmissing_frac = 1.0 - quality_patch[..., 0].mean(axis=-1)
    threshold = float(cfg["attention"]["min_nonmissing_frac_for_eye_token"])
    eye_token_valid = eye_nonmissing_frac >= threshold
    return {
        "content": content_patch.astype(np.float32),
        "quality": quality_patch.astype(np.float32),
        "stim": stim_patch.astype(np.float32),
        "task_id": int(trial["task_id"]),
        "eye_nonmissing_frac": eye_nonmissing_frac.astype(np.float32),
        "eye_token_valid": eye_token_valid.astype(bool),
        "subject_id": trial["subject_id"],
        "trial_id": trial["trial_id"],
        "path": trial.get("path", ""),
    }


def build_sequence_attention_mask(pad_mask: torch.Tensor, eye_token_valid: torch.Tensor) -> torch.Tensor:
    batch, n = pad_mask.shape
    seq_mask = torch.ones(batch, 3 * n, dtype=torch.bool, device=pad_mask.device)
    seq_mask[:, 0::3] = pad_mask
    seq_mask[:, 1::3] = pad_mask | (~eye_token_valid[:, :, 0])
    seq_mask[:, 2::3] = pad_mask | (~eye_token_valid[:, :, 1])
    return seq_mask
