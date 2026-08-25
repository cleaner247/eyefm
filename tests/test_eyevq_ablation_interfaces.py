from copy import deepcopy
from pathlib import Path

import torch
import yaml

from eyemae.eyevq.config import build_bert, build_tokenizer, validate_bert_config
from eyemae.eyevq.pretrain.model import EyeVQBERT, TARGET_RAW_PATCH
from eyemae.eyevq.tokenizer.vqvae import VQVAEQuantizer


ROOT = Path(__file__).resolve().parents[1]


def test_raw_patch_target_runs_without_code_ids_and_backpropagates() -> None:
    model = EyeVQBERT(
        K_e=15,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        max_time=3,
        patch_samples=8,
        target_type=TARGET_RAW_PATCH,
        factorized_fsq=False,
    )
    batch, patches = 2, 3
    stim = torch.randn(batch, patches, 4, 8)
    eyes = torch.randn(batch, patches, 2, 4, 8)
    eyes[..., 3, :] = torch.randint(0, 2, eyes[..., 3, :].shape).float()
    quality = torch.zeros(batch, patches, 2, 8, 1)
    pad = torch.zeros(batch, patches, dtype=torch.bool)
    nonmissing = torch.ones(batch, patches, 2)
    task = torch.zeros(batch, dtype=torch.long)
    mask = torch.zeros(batch, 1 + patches * 3, dtype=torch.bool)
    mask[:, 2::3] = True
    mask[:, 3::3] = True
    loss, stats = model(
        stim, eyes, quality, pad, nonmissing, task, None, mask
    )
    assert torch.isfinite(loss)
    assert int(stats["n_masked"]) == batch * patches * 2
    loss.backward()
    assert model.pred_head.net.weight.grad is not None


def test_vqvae_quantizer_has_straight_through_and_codebook_gradients() -> None:
    quantizer = VQVAEQuantizer(d=4, codebook_size=17, commitment_beta=0.25)
    latent = torch.randn(3, 5, 4, requires_grad=True)
    straight_through, ids, quantized, qloss, stats = quantizer(latent)
    loss = straight_through.square().mean() + qloss
    loss.backward()
    assert ids.shape == (3, 5)
    assert quantized.shape == latent.shape
    assert latent.grad is not None
    assert quantizer.embedding.weight.grad is not None
    assert 0 < stats["active_code_fraction"] <= 1


def test_vqvae_tokenizer_and_raw_bert_configs_are_buildable() -> None:
    tokenizer_cfg = yaml.safe_load(
        (ROOT / "configs/eyevq/final/tokenizer.yaml").read_text(encoding="utf-8")
    )
    tokenizer_cfg["vq"] = {
        "type": "vqvae",
        "code_dim": 4,
        "codebook_size": 1575,
        "commitment_beta": 0.25,
        "min_nonmissing_frac": 0.5,
    }
    tokenizer_cfg["loss"]["eye_commit_group_weight"] = 1.0
    tokenizer = build_tokenizer(tokenizer_cfg)
    assert isinstance(tokenizer.eye_codebook, VQVAEQuantizer)

    bert_cfg = yaml.safe_load(
        (ROOT / "configs/eyevq/final/bert.yaml").read_text(encoding="utf-8")
    )
    bert_cfg["bert"]["target_type"] = "raw_patch"
    bert_cfg["bert"]["factorized_fsq"] = False
    bert_cfg["train"].pop("tokenizer_checkpoint")
    bert_cfg["train"].pop("code_ids_cache")
    validate_bert_config(bert_cfg)
    bert = build_bert(bert_cfg)
    assert bert.target_type == TARGET_RAW_PATCH
