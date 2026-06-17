from __future__ import annotations

import torch

from eyemae.config import load_config
from eyemae.model import build_model


def test_model_forward_shapes() -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["model"]["d_model"] = 64
    cfg["model"]["n_layers"] = 1
    cfg["model"]["n_heads"] = 4
    cfg["model"]["ffn_hidden"] = 128
    model = build_model(cfg)
    b, n = 2, 5
    batch = {
        "content": torch.randn(b, n, 2, 20, 4),
        "quality": torch.zeros(b, n, 2, 20, 1),
        "stim": torch.randn(b, n, 20, 4),
        "task_id": torch.tensor([0, 1]),
        "pad_mask": torch.zeros(b, n, dtype=torch.bool),
        "eye_token_valid": torch.ones(b, n, 2, dtype=torch.bool),
    }
    mae_mask = torch.zeros(b, n, 2, dtype=torch.bool)
    out = model(**batch, mae_mask=mae_mask, return_hidden=True)
    assert out["pred"].shape == (b, n, 2, 20, 4)
    assert out["seq_attn_pad_mask"].shape == (b, 3 * n)
    assert out["hidden_eye"].shape == (b, n, 2, 64)


def test_model_forward_without_token_type_embedding() -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["model"]["d_model"] = 64
    cfg["model"]["n_layers"] = 1
    cfg["model"]["n_heads"] = 4
    cfg["model"]["ffn_hidden"] = 128
    cfg["model"]["use_token_type_embedding"] = False
    model = build_model(cfg)
    b, n = 2, 5
    batch = {
        "content": torch.randn(b, n, 2, 20, 4),
        "quality": torch.zeros(b, n, 2, 20, 1),
        "stim": torch.randn(b, n, 20, 4),
        "task_id": torch.tensor([0, 1]),
        "pad_mask": torch.zeros(b, n, dtype=torch.bool),
        "eye_token_valid": torch.ones(b, n, 2, dtype=torch.bool),
    }
    mae_mask = torch.zeros(b, n, 2, dtype=torch.bool)
    out = model(**batch, mae_mask=mae_mask, return_hidden=True)
    assert out["pred"].shape == (b, n, 2, 20, 4)
    assert out["seq_attn_pad_mask"].shape == (b, 3 * n)
