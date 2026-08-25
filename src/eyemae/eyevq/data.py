"""Shared data collation for EyeVQ tokenizer and BERT stages."""

from __future__ import annotations

import torch


def collate_trials_fixed_nmax(items: list[dict], nmax: int = 256) -> dict:
    """Pad trials to a fixed patch count and truncate longer inputs."""
    if not items:
        raise ValueError("empty batch")
    batch_size = len(items)
    patch = int(items[0]["content"].shape[2])
    content = torch.zeros(batch_size, nmax, 2, patch, 4, dtype=torch.float32)
    quality = torch.ones(batch_size, nmax, 2, patch, 1, dtype=torch.float32)
    stim = torch.zeros(batch_size, nmax, patch, 4, dtype=torch.float32)
    pad_mask = torch.ones(batch_size, nmax, dtype=torch.bool)
    eye_nonmissing_frac = torch.zeros(batch_size, nmax, 2, dtype=torch.float32)
    global_trial_id: list[str] = []
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    for batch_index, item in enumerate(items):
        n_patches = min(int(item["content"].shape[0]), nmax)
        content[batch_index, :n_patches] = torch.as_tensor(
            item["content"][:n_patches], dtype=torch.float32
        )
        quality[batch_index, :n_patches] = torch.as_tensor(
            item["quality"][:n_patches], dtype=torch.float32
        )
        stim[batch_index, :n_patches] = torch.as_tensor(
            item["stim"][:n_patches], dtype=torch.float32
        )
        pad_mask[batch_index, :n_patches] = False
        eye_nonmissing_frac[batch_index, :n_patches] = torch.as_tensor(
            item["eye_nonmissing_frac"][:n_patches], dtype=torch.float32
        )
        global_trial_id.append(str(item.get("global_trial_id", item.get("trial_id", ""))))
        task_ids[batch_index] = int(item.get("task_id", 0))
    return {
        "content": content,
        "quality": quality,
        "stim": stim,
        "pad_mask": pad_mask,
        "eye_nonmissing_frac": eye_nonmissing_frac,
        "global_trial_id": global_trial_id,
        "task_id": task_ids,
    }
