from __future__ import annotations

import torch

from eyemae.config import load_config
from eyemae.masking import generate_mae_mask, mae_eligible_from_batch


def _batch():
    b, n = 4, 20
    quality = torch.zeros(b, n, 2, 20, 1)
    pad_mask = torch.zeros(b, n, dtype=torch.bool)
    pad_mask[:, -2:] = True
    eye_token_valid = torch.ones(b, n, 2, dtype=torch.bool)
    quality[0, 0, 0, :, 0] = 1
    eye_token_valid[0, 0, 0] = False
    quality[1, 1, 1, :12, 0] = 1
    return {
        "quality": quality,
        "pad_mask": pad_mask,
        "eye_token_valid": eye_token_valid,
    }


def test_masking_respects_eligibility_and_padding() -> None:
    cfg = load_config("configs/debug.yaml")
    gen = torch.Generator().manual_seed(0)
    batch = _batch()
    mae_mask, mask_type = generate_mae_mask(batch, cfg, generator=gen)
    eligible = mae_eligible_from_batch(batch, cfg)
    assert mae_mask.shape == (4, 20, 2)
    assert mask_type.shape == (4, 20, 2)
    assert not mae_mask[batch["pad_mask"]].any()
    assert not mae_mask[0, 0, 0]
    assert not mae_mask[1, 1, 1]
    assert (mae_mask & ~eligible).sum() == 0
    assert (eligible & ~mae_mask).any(dim=(1, 2)).all()


def test_masking_handles_too_few_eligible_tokens() -> None:
    cfg = load_config("configs/debug.yaml")
    batch = _batch()
    batch["eye_token_valid"][:] = False
    mae_mask, _ = generate_mae_mask(batch, cfg, generator=torch.Generator().manual_seed(0))
    assert not mae_mask.any()
