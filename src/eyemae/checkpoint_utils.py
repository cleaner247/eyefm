from __future__ import annotations

from typing import Any

import torch


def extract_encoder_state_dict(checkpoint: dict[str, Any] | dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    raw_state = checkpoint.get("model", checkpoint)
    state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in raw_state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_key = clean_key[len("encoder.") :] if clean_key.startswith("encoder.") else clean_key
        if clean_key.startswith("pred_head.") or clean_key.startswith("head."):
            skipped.append(clean_key)
            continue
        state[clean_key] = value
    return state, skipped
