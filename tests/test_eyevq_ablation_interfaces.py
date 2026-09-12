from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from eyemae.eyevq.config import build_bert, build_tokenizer, validate_bert_config
from eyemae.eyevq.pretrain.model import (
    EyeVQBERT,
    TARGET_DIRECT_RECONSTRUCTION,
    TARGET_NORMALIZED_LATENT,
    TARGET_RAW_PATCH,
    raw_patch_reconstruction_losses,
)
from eyemae.eyevq.tokenizer.vqvae import VQVAEQuantizer
from eyemae.eyevq.tokenizer.train import collate_vq_trials


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


def test_raw_patch_losses_match_tokenizer_eye_validity_rules() -> None:
    target = torch.zeros(1, 4, 4)
    target[:, 3, 1] = 1.0  # blink: no XY/area supervision
    prediction = torch.zeros_like(target)
    prediction[:, :3, 1] = 100.0
    missing = torch.zeros(1, 4, dtype=torch.bool)
    missing[:, 2] = True
    prediction[:, :3, 2] = 100.0  # missing: no supervision at all
    xy, area, blink, velocity, valid = raw_patch_reconstruction_losses(
        prediction, target, missing, blink_pos_weight=1.0
    )
    assert torch.equal(valid, ~missing)
    assert xy.item() == 0.0
    assert area.item() == 0.0
    assert torch.isfinite(blink).all()
    # Velocity crossing a blink or missing frame is excluded too.
    assert velocity.item() == 0.0


def test_direct_reconstruction_uses_decode_and_feature_losses_without_accuracy() -> None:
    loss_cfg = {
        "eye_xy_weight": 1.0,
        "eye_area_weight": 0.1,
        "eye_blink_weight": 0.1,
        "eye_blink_pos_weight": 1.0,
        "eye_velocity_weight": 0.0,
        "eye_recon_group_weight": 1.0,
        "manual_feature_group_weight": 0.0015,
        "manual_feature_binary_weight": 0.25,
        "manual_feature_continuous_weight": 1.5,
    }
    model = EyeVQBERT(
        K_e=15,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        max_time=3,
        patch_samples=8,
        target_type=TARGET_DIRECT_RECONSTRUCTION,
        factorized_fsq=False,
        manual_feature_dim=6,
        direct_loss_cfg=loss_cfg,
    )
    batch, patches, features = 2, 3, 6
    eyes = torch.randn(batch, patches, 2, 4, 8)
    eyes[..., 3, :] = torch.randint(0, 2, eyes[..., 3, :].shape).float()
    mask = torch.zeros(batch, 1 + patches * 3, dtype=torch.bool)
    mask[:, 2::3] = True
    mask[:, 3::3] = True
    loss, stats = model(
        torch.randn(batch, patches, 4, 8),
        eyes,
        torch.zeros(batch, patches, 2, 8, 1),
        torch.zeros(batch, patches, dtype=torch.bool),
        torch.ones(batch, patches, 2),
        torch.zeros(batch, dtype=torch.long),
        None,
        mask,
        manual_feature_targets=torch.randn(batch, features),
        manual_feature_loss_mask=torch.ones(batch, features, dtype=torch.bool),
        manual_feature_binary_mask=torch.zeros(features, dtype=torch.bool),
        manual_feature_count_mask=torch.zeros(features, dtype=torch.bool),
        manual_feature_pos_weight=torch.ones(features),
        manual_feature_bce_scale=torch.ones(features),
        manual_feature_weights=torch.ones(batch, features),
    )
    assert torch.isfinite(loss)
    assert "bert_acc" not in stats
    assert torch.allclose(
        loss,
        stats["L_eye_weighted"] + stats["L_feat_weighted"],
    )
    loss.backward()
    assert model.pred_head.net.weight.grad is not None
    assert model.manual_feat_head[-1].weight.grad is not None


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


def test_kmeans_collate_uses_runtime_patch_length_and_channels_first() -> None:
    items = []
    for patches in (3, 2):
        items.append({
            "content": np.zeros((patches, 2, 4, 40), dtype=np.float32),
            "quality": np.zeros((patches, 2, 40, 1), dtype=np.float32),
            "stim": np.zeros((patches, 4, 40), dtype=np.float32),
            "eye_nonmissing_frac": np.ones((patches, 2), dtype=np.float32),
        })
    batch = collate_vq_trials(items)
    assert batch["content"].shape == (2, 3, 2, 4, 40)
    assert batch["quality"].shape == (2, 3, 2, 40, 1)
    assert batch["stim"].shape == (2, 3, 4, 40)
    assert batch["pad_mask"].tolist() == [[False, False, False], [False, False, True]]


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


def test_continuous_ae_uses_decoder_layernorm_and_bert_latent_mse() -> None:
    tokenizer_cfg = yaml.safe_load(
        (ROOT / "configs/eyevq/final/tokenizer.yaml").read_text(encoding="utf-8")
    )
    tokenizer_cfg["vq"] = {
        "type": "ae", "code_dim": 64, "min_nonmissing_frac": 0.5,
    }
    tokenizer = build_tokenizer(tokenizer_cfg)
    assert isinstance(tokenizer.decoder_latent_norm, torch.nn.LayerNorm)
    assert tokenizer.decoder_latent_norm.normalized_shape == (64,)
    assert tokenizer.decoder_latent_norm.elementwise_affine is False

    model = EyeVQBERT(
        K_e=64, d_model=32, n_layers=1, n_heads=4, dim_ff=64,
        max_time=3, patch_samples=8, target_type=TARGET_NORMALIZED_LATENT,
        factorized_fsq=False, latent_dim=64,
    )
    batch, patches = 2, 3
    mask = torch.zeros(batch, 1 + patches * 3, dtype=torch.bool)
    mask[:, 2::3] = True
    mask[:, 3::3] = True
    targets = torch.nn.functional.layer_norm(
        torch.randn(batch, patches, 2, 64), (64,)
    )
    loss, stats = model(
        torch.randn(batch, patches, 4, 8),
        torch.randn(batch, patches, 2, 4, 8),
        torch.zeros(batch, patches, 2, 8, 1),
        torch.zeros(batch, patches, dtype=torch.bool),
        torch.ones(batch, patches, 2),
        torch.zeros(batch, dtype=torch.long),
        None,
        mask,
        eye_latent_targets=targets,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(stats["latent_mse"], loss)
    loss.backward()
    assert model.pred_head.net.weight.grad is not None
