from __future__ import annotations

import torch

from eyemae.config import load_config
from eyemae.model import build_model
from eyemae.patching import build_sequence_attention_mask


def test_attention_mask_rules_and_invalid_hidden_zero() -> None:
    pad = torch.tensor([[False, True]])
    eye_valid = torch.tensor([[[False, True], [True, True]]])
    seq_mask = build_sequence_attention_mask(pad, eye_valid)
    assert not seq_mask[0, 0]
    assert seq_mask[0, 1]
    assert not seq_mask[0, 2]
    assert seq_mask[0, 3] and seq_mask[0, 4] and seq_mask[0, 5]

    cfg = load_config("configs/debug.yaml")
    cfg["model"]["d_model"] = 32
    cfg["model"]["n_layers"] = 1
    cfg["model"]["n_heads"] = 4
    cfg["model"]["ffn_hidden"] = 64
    model = build_model(cfg)
    batch = {
        "content": torch.randn(1, 2, 2, 20, 4),
        "quality": torch.zeros(1, 2, 2, 20, 1),
        "stim": torch.randn(1, 2, 20, 4),
        "task_id": torch.tensor([0]),
        "pad_mask": pad,
        "eye_token_valid": eye_valid,
    }
    mae_mask = torch.tensor([[[False, True], [False, False]]])
    out = model(**batch, mae_mask=mae_mask, return_hidden=True)
    assert out["seq_attn_pad_mask"][0, 2] == 0
    assert torch.all(out["hidden_seq"][0, seq_mask[0]] == 0)
