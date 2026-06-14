from __future__ import annotations

import torch

from eyemae.config import load_config
from eyemae.losses import compute_reconstruction_loss
from eyemae.model import build_model


def test_tiny_batch_training_step_is_finite() -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["model"]["d_model"] = 32
    cfg["model"]["n_layers"] = 1
    cfg["model"]["n_heads"] = 4
    cfg["model"]["ffn_hidden"] = 64
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "content": torch.randn(2, 4, 2, 20, 4),
        "quality": torch.zeros(2, 4, 2, 20, 1),
        "stim": torch.randn(2, 4, 20, 4),
        "task_id": torch.tensor([0, 1]),
        "pad_mask": torch.zeros(2, 4, dtype=torch.bool),
        "eye_token_valid": torch.ones(2, 4, 2, dtype=torch.bool),
    }
    mae_mask = torch.zeros(2, 4, 2, dtype=torch.bool)
    mae_mask[:, 1:3] = True
    losses = []
    for _ in range(3):
        opt.zero_grad()
        out = model(**batch, mae_mask=mae_mask)
        loss, _ = compute_reconstruction_loss(out["pred"], batch["content"], batch["quality"], mae_mask, batch["pad_mask"], batch["eye_token_valid"], cfg)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] <= losses[0] * 1.2
