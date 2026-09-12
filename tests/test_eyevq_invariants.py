from __future__ import annotations

import ast
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
import torch
import torch.nn as nn
import yaml

from eyemae.batching import TokenBatchSampler
from eyemae.eyevq.config import (
    build_bert,
    checkpoint_config,
    build_tokenizer,
    override_fsq_levels,
    tokenizer_architecture,
    validate_bert_config,
)
from eyemae.eyevq.pretrain.masking import (
    generate_eyemae_mask,
    generate_eyemae_mask_paired_multiblock,
    generate_eyemae_mask_paired_fixed_blocks,
    generate_eyemae_mask_paired_random,
    generate_eyemae_mask_paired_span,
    span_length_probabilities,
)
from eyemae.eyevq.pretrain.model import EyeVQBERT
from eyemae.eyevq.pretrain.model import (
    factorized_fsq_cross_entropy,
    mean_masked_loss_per_trial,
)
from eyemae.eyevq.pretrain.train import lookup_code_ids
from eyemae.eyevq.optim import build_adamw_param_groups
from eyemae.eyevq.artifacts import latest_step_checkpoint
from eyemae.eyevq.precompute_codes import mask_invalid_eye_codes
from eyemae.eyevq.pipeline import Pipeline as EyeVQPipeline
from eyemae.eyevq.artifacts import (
    CACHE_FORMAT_VERSION,
    assert_run_identity,
    sha256_file as artifact_sha256_file,
    validate_cache_identity,
    write_cache_manifest,
)
from eyemae.eyevq.downstream.runtime import get_encoder_head_lrs
from eyemae.eyevq.downstream.model import EyeVQForClassification
from eyemae.eyevq.tokenizer.model import (
    EyeVQTokenizer,
    build_stim_isolated_attn_mask,
)
from eyemae.eyevq.tokenizer.losses import (
    compute_eye_recon_loss,
    compute_manual_feature_loss,
)
from eyemae.eyevq.tokenizer.fsq import FSQ
from eyemae.eyevq.tokenizer.train import (
    build_manual_feature_loss_metadata,
    get_velocity_weight,
)


class _FakeEncoder(nn.Module):
    def forward(self, stim, eye_l, eye_r, eye_nonmissing_frac, pad_mask):
        del eye_l, eye_r, eye_nonmissing_frac, pad_mask
        batch, time = stim.shape[:2]
        left = torch.zeros(batch, time, 16)
        right = torch.zeros(batch, time, 16)
        left[..., 0] = torch.arange(time) + 10
        right[..., 0] = torch.arange(time) + 100
        return {
            "enc_cls": torch.zeros(batch, 1, 16),
            "s_hidden": torch.zeros(batch, time, 16),
            "l_hidden": left,
            "r_hidden": right,
        }


class _FirstCoordinate(nn.Module):
    def forward(self, value):
        return torch.cat([value[..., :1], torch.zeros_like(value[..., :1])], dim=-1)


class _IdentityCodebook(nn.Module):
    def forward(self, value):
        ids = value[..., 0].long()
        return value, ids, value, value.new_zeros(()), {}


def _tiny_tokenizer(architecture: str = "joint") -> EyeVQTokenizer:
    return EyeVQTokenizer(
        d_model=16,
        enc_n_layers=1,
        enc_n_heads=4,
        enc_dim_ff=32,
        eye_code_dim=2,
        fsq_L=[3, 3],
        dec_n_layers=1,
        dec_n_heads=4,
        dec_dim_ff=32,
        num_manual_features=4,
        max_patches=4,
        patch_samples=20,
        architecture=architecture,
        s_enc_n_layers=1,
        feat_dec_n_layers=1,
    )


def test_lr_codes_are_time_major_interleaved() -> None:
    model = _tiny_tokenizer()
    model.encoder = _FakeEncoder()
    model.vq_proj = _FirstCoordinate()
    model.eye_codebook = _IdentityCodebook()
    batch, time = 1, 3
    code_ids = model.encode_codes(
        torch.zeros(batch, time, 4, 20),
        torch.zeros(batch, time, 2, 4, 20),
        torch.zeros(batch, time, 2, 20, 1),
        torch.zeros(batch, time, dtype=torch.bool),
        torch.ones(batch, time, 2),
    )
    assert code_ids.tolist() == [[[10, 100], [11, 101], [12, 102]]]


def test_continuous_warmup_uses_fsq_bounded_latents() -> None:
    model = _tiny_tokenizer()
    model.encoder = _FakeEncoder()
    model.vq_proj = _FirstCoordinate()
    batch, time = 1, 3
    _enc, _stim, z_lr, code_ids, stats, commit = model._encode_and_quantize(
        torch.zeros(batch, time, 4, 20),
        torch.zeros(batch, time, 2, 4, 20),
        torch.zeros(batch, time, 2, 20, 1),
        torch.zeros(batch, time, dtype=torch.bool),
        torch.ones(batch, time, 2),
        quantize=False,
    )
    assert torch.all(z_lr.abs() <= 1.0)
    torch.testing.assert_close(z_lr[0, 0, 0, 0], torch.tanh(torch.tensor(10.0)))
    torch.testing.assert_close(z_lr[0, 0, 1, 0], torch.tanh(torch.tensor(100.0)))
    assert code_ids.count_nonzero().item() == 0
    assert stats == {}
    assert commit.item() == 0.0


def test_ifsq_matches_distribution_matched_odd_level_grid() -> None:
    quantizer = FSQ(d=2, L=[9, 5], activation="ifsq", ifsq_alpha=1.6)
    z = torch.tensor(
        [[-3.0, -1.0], [0.0, 0.25], [1.5, 4.0]], requires_grad=True
    )
    z_st, code_ids, z_q, commit, stats = quantizer(z)

    bounded = 2.0 * torch.sigmoid(1.6 * z) - 1.0
    half_width = torch.tensor([4.0, 2.0])
    q_integer = torch.round(bounded * half_width)
    expected_q = q_integer / half_width
    expected_digits = (q_integer + half_width).long()
    expected_ids = expected_digits[:, 0] + 9 * expected_digits[:, 1]

    torch.testing.assert_close(z_q, expected_q)
    torch.testing.assert_close(z_st, expected_q)
    torch.testing.assert_close(code_ids, expected_ids)
    assert quantizer.codebook_size == 45
    assert 0 <= code_ids.min() and code_ids.max() < quantizer.codebook_size
    assert commit.item() == 0.0
    assert stats["code_dim_perplexity_min"] >= 1.0

    z_st.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert (z.grad.abs() > 0).all()


def test_velocity_weight_is_delayed_then_cosine_ramped() -> None:
    loss_cfg = {
        "eye_velocity_weight": 5.0,
        "eye_velocity_start_step": 1000,
        "eye_velocity_ramp_steps": 1000,
    }
    assert get_velocity_weight(0, loss_cfg) == pytest.approx(0.0)
    assert get_velocity_weight(999, loss_cfg) == pytest.approx(0.0)
    assert get_velocity_weight(1000, loss_cfg) == pytest.approx(0.0)
    assert get_velocity_weight(1500, loss_cfg) == pytest.approx(2.5)
    assert get_velocity_weight(2000, loss_cfg) == pytest.approx(5.0)
    assert get_velocity_weight(50000, loss_cfg) == pytest.approx(5.0)


@pytest.mark.parametrize("architecture", ["joint", "cross_attention"])
def test_formal_tokenizers_keep_output_shapes(architecture: str) -> None:
    model = _tiny_tokenizer(architecture).eval()
    batch, time = 2, 4
    output = model(
        torch.randn(batch, time, 4, 20),
        torch.randn(batch, time, 2, 4, 20),
        torch.zeros(batch, time, 2, 20, 1),
        torch.tensor([[False, False, True, True], [False] * time]),
        torch.ones(batch, time, 2),
    )
    assert output["code_ids"].shape == (batch, time, 2)
    assert output["eye_recon"].shape == (batch, time, 2, 4, 20)
    assert output["manual_feat_pred"].shape == (batch, 4)


def test_true_cross_stimulus_stream_does_not_read_eye_tokens() -> None:
    model = _tiny_tokenizer("cross_attention").eval()
    batch, time = 1, 4
    stim = torch.randn(batch, time, 4, 20)
    quality = torch.zeros(batch, time, 2, 20, 1)
    pad = torch.zeros(batch, time, dtype=torch.bool)
    valid = torch.ones(batch, time, 2)
    out_a = model(stim, torch.randn(batch, time, 2, 4, 20), quality, pad, valid)
    out_b = model(stim, torch.randn(batch, time, 2, 4, 20), quality, pad, valid)
    torch.testing.assert_close(out_a["s_hidden"], out_b["s_hidden"])


def test_invalid_eye_content_cannot_change_valid_tokenizer_outputs() -> None:
    model = _tiny_tokenizer("joint").eval()
    batch, time = 1, 4
    stim = torch.randn(batch, time, 4, 20)
    eye_a = torch.randn(batch, time, 2, 4, 20)
    eye_b = eye_a.clone()
    eye_b[:, :, 1] = torch.randn_like(eye_b[:, :, 1]) * 10_000
    quality = torch.zeros(batch, time, 2, 20, 1)
    pad = torch.zeros(batch, time, dtype=torch.bool)
    nonmissing = torch.ones(batch, time, 2)
    nonmissing[:, :, 1] = 0.0
    with torch.no_grad():
        out_a = model(stim, eye_a, quality, pad, nonmissing)
        out_b = model(stim, eye_b, quality, pad, nonmissing)
    torch.testing.assert_close(out_a["code_ids"][:, :, 0], out_b["code_ids"][:, :, 0])
    torch.testing.assert_close(out_a["eye_recon"][:, :, 0], out_b["eye_recon"][:, :, 0])
    torch.testing.assert_close(out_a["manual_feat_pred"], out_b["manual_feat_pred"])


def test_invalid_eye_content_cannot_change_valid_bert_context() -> None:
    model = EyeVQBERT(
        K_e=15,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dim_ff=32,
        max_time=4,
        patch_samples=20,
        fsq_L=[3, 5],
        min_nonmissing_frac=0.5,
    ).eval()
    batch, time = 1, 4
    stim = torch.randn(batch, time, 4, 20)
    eye_a = torch.randn(batch, time, 2, 4, 20)
    eye_b = eye_a.clone()
    eye_b[:, :, 1] = torch.randn_like(eye_b[:, :, 1]) * 10_000
    pad = torch.zeros(batch, time, dtype=torch.bool)
    nonmissing = torch.ones(batch, time, 2)
    nonmissing[:, :, 1] = 0.0
    task = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        out_a = model.encode(stim, eye_a, pad, nonmissing, task)
        out_b = model.encode(stim, eye_b, pad, nonmissing, task)
    torch.testing.assert_close(out_a[:, 0], out_b[:, 0])
    torch.testing.assert_close(out_a[:, 2::3], out_b[:, 2::3])


def test_invalid_and_padding_code_ids_are_zeroed() -> None:
    codes = torch.tensor([[[11, 12], [13, 14], [15, 16]]])
    nonmissing = torch.tensor([[[1.0, 0.0], [0.9, 0.8], [1.0, 1.0]]])
    pad = torch.tensor([[False, False, True]])
    masked = mask_invalid_eye_codes(codes, nonmissing, pad, 0.85)
    assert masked.tolist() == [[[11, 0], [13, 0], [0, 0]]]


def test_joint_structural_mask_blocks_lr_keys_for_stim_queries() -> None:
    mask = build_stim_isolated_attn_mask(1 + 3 * 3, 3, "cpu", with_cls=True)
    stim_positions = torch.tensor([1, 4, 7])
    lr_positions = torch.tensor([2, 3, 5, 6, 8, 9])
    assert not mask[stim_positions][:, lr_positions].any()
    assert mask[stim_positions][:, stim_positions].all()
    assert mask[0].all()  # CLS and eye queries may read the complete sequence.


def test_tokenizer_stim_queries_can_explicitly_block_cls_key() -> None:
    mask = build_stim_isolated_attn_mask(
        1 + 3 * 3,
        3,
        "cpu",
        with_cls=True,
        stim_attend_cls=False,
    )
    stim_positions = torch.tensor([1, 4, 7])
    lr_positions = torch.tensor([2, 3, 5, 6, 8, 9])
    assert not mask[stim_positions, 0].any()
    assert not mask[stim_positions][:, lr_positions].any()
    assert mask[stim_positions][:, stim_positions].all()
    assert mask[0].all()  # The directed CLS->stim information path is retained.


def test_paired_random_mask_has_joint_valid_only_lr_quota() -> None:
    eye_valid = torch.zeros(2, 20, 2, dtype=torch.bool)
    eye_valid[0, [1, 4, 7, 9, 13, 19], 0] = True
    eye_valid[0, [0, 1, 4, 7, 9, 12, 15, 19], 1] = True
    eye_valid[1, [0, 5, 8], 0] = True
    eye_valid[1, [0, 8], 1] = True
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[0, 19] = True
    mask, loss_mask = generate_eyemae_mask_paired_random(
        eye_valid, pad, mask_ratio=0.5, generator=torch.Generator().manual_seed(7)
    )

    left_mask = mask[:, 2::3]
    right_mask = mask[:, 3::3]
    joint_eligible = eye_valid.all(dim=-1) & ~pad

    torch.testing.assert_close(mask, loss_mask)
    torch.testing.assert_close(left_mask, right_mask)
    assert not mask[:, 0].any()       # CLS
    assert not mask[:, 1::3].any()    # Stim
    assert not (left_mask & ~joint_eligible).any()
    assert not (right_mask & ~joint_eligible).any()
    assert mask[0, 2::3].sum().item() == 2
    assert mask[0, 3::3].sum().item() == 2
    assert mask[1, 2::3].sum().item() == 1
    assert mask[1, 3::3].sum().item() == 1


def test_paired_span_mask_obeys_joint_validity_and_two_to_six_bounds() -> None:
    eye_valid = torch.ones(2, 40, 2, dtype=torch.bool)
    eye_valid[1, 10:13, 0] = False
    eye_valid[1, 27, 1] = False
    pad = torch.zeros(2, 40, dtype=torch.bool)
    pad[1, 35:] = True
    mask, loss_mask = generate_eyemae_mask_paired_span(
        eye_valid,
        pad,
        mask_ratio=0.4,
        span_min=2,
        span_max=6,
        generator=torch.Generator().manual_seed(23),
    )
    left_mask = mask[:, 2::3]
    right_mask = mask[:, 3::3]
    joint_eligible = eye_valid.all(dim=-1) & ~pad

    torch.testing.assert_close(mask, loss_mask)
    torch.testing.assert_close(left_mask, right_mask)
    assert not mask[:, 0].any()
    assert not mask[:, 1::3].any()
    assert not (left_mask & ~joint_eligible).any()
    assert left_mask.sum().item() > 0

    for row in left_mask.tolist():
        run_lengths: list[int] = []
        run = 0
        for selected in row + [False]:
            if selected:
                run += 1
            elif run:
                run_lengths.append(run)
                run = 0
        assert run_lengths
        assert all(2 <= length <= 6 for length in run_lengths)


def test_paired_span_mask_returns_exact_span_length_metadata() -> None:
    eye_valid = torch.ones(8, 48, 2, dtype=torch.bool)
    pad = torch.zeros(8, 48, dtype=torch.bool)
    mask, loss_mask, span_lengths = generate_eyemae_mask_paired_span(
        eye_valid,
        pad,
        mask_ratio=0.6,
        span_min=1,
        span_max=5,
        generator=torch.Generator().manual_seed(123),
        return_span_lengths=True,
    )
    selected = mask[:, 2::3]
    torch.testing.assert_close(mask, loss_mask)
    assert span_lengths.shape == selected.shape
    assert not (span_lengths[~selected] != 0).any()
    assert not (span_lengths[selected] < 1).any()
    assert not (span_lengths[selected] > 5).any()
    for row_mask, row_lengths in zip(selected.tolist(), span_lengths.tolist()):
        start = 0
        while start < len(row_mask):
            if not row_mask[start]:
                start += 1
                continue
            end = start
            while end < len(row_mask) and row_mask[end]:
                end += 1
            run_length = end - start
            assert row_lengths[start:end] == [run_length] * run_length
            start = end


def test_predictor_span_length_embedding_conditions_shared_factorized_input() -> None:
    model = EyeVQBERT(
        K_e=9 * 7 * 5 * 5,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_ff=64,
        max_time=4,
        patch_samples=40,
        factorized_fsq=True,
        fsq_L=[9, 7, 5, 5],
        predictor_span_length_embedding=True,
        max_mask_span_length=5,
    )
    batch, time = 2, 4
    stim = torch.randn(batch, time, 4, 40)
    eyes = torch.randn(batch, time, 2, 4, 40)
    quality = torch.zeros(batch, time, 2, 40, 1)
    pad = torch.zeros(batch, time, dtype=torch.bool)
    nonmissing = torch.ones(batch, time, 2)
    tasks = torch.zeros(batch, dtype=torch.long)
    codes = torch.zeros(batch, time, 2, dtype=torch.long)
    mask = torch.zeros(batch, 1 + 3 * time, dtype=torch.bool)
    mask[:, 2::3] = True
    mask[:, 3::3] = True
    span_lengths = torch.tensor([[1, 2, 2, 1], [5, 5, 5, 5]])
    loss, stats = model(
        stim, eyes, quality, pad, nonmissing, tasks, codes, mask, span_lengths
    )
    loss.backward()
    assert stats["n_masked"] == batch * time * 2
    assert model.span_length_embed is not None
    assert model.span_length_embed.weight.grad is not None
    assert model.span_length_embed.weight.grad[1:].abs().sum() > 0


def test_symmetric_power_span_prior_suppresses_both_endpoints() -> None:
    probabilities = span_length_probabilities(
        1, 6, distribution="symmetric_power", power=1.0
    )
    torch.testing.assert_close(probabilities, probabilities.flip(0))
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[-1] < probabilities[-2] < probabilities[-3]
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))

    eye_valid = torch.ones(16, 48, 2, dtype=torch.bool)
    pad = torch.zeros(16, 48, dtype=torch.bool)
    mask, _ = generate_eyemae_mask_paired_span(
        eye_valid,
        pad,
        mask_ratio=0.5,
        span_min=1,
        span_max=6,
        span_length_distribution="symmetric_power",
        span_length_power=1.0,
        generator=torch.Generator().manual_seed(91),
    )
    torch.testing.assert_close(mask[:, 2::3], mask[:, 3::3])


def test_explicit_span_probabilities_are_normalized() -> None:
    probabilities = span_length_probabilities(
        1,
        6,
        distribution="explicit",
        explicit_probabilities=[0.314, 0.235, 0.174, 0.131, 0.089, 0.057],
    )
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        probabilities,
        torch.tensor([0.314, 0.235, 0.174, 0.131, 0.089, 0.057]),
    )


def test_token_balanced_span_prior_equalizes_expected_patch_contributions() -> None:
    probabilities = span_length_probabilities(
        1, 5, distribution="token_balanced"
    )
    lengths = torch.arange(1, 6, dtype=torch.float32)
    contributions = probabilities * lengths
    torch.testing.assert_close(
        contributions / contributions.sum(),
        torch.full((5,), 0.2),
    )
    torch.testing.assert_close(
        probabilities,
        torch.tensor([0.4379562, 0.2189781, 0.1459854, 0.1094891, 0.0875912]),
    )


def test_factorized_fsq_loss_is_sum_of_dimension_cross_entropies() -> None:
    coords = torch.tensor([[0, 0], [1, 3]])
    logits = [torch.zeros(2, 3, requires_grad=True), torch.zeros(2, 5, requires_grad=True)]
    per_dim_ce, per_token_loss = factorized_fsq_cross_entropy(logits, coords)
    expected = torch.tensor([math.log(3.0), math.log(5.0)])
    torch.testing.assert_close(per_dim_ce[0], expected)
    torch.testing.assert_close(
        per_token_loss,
        torch.full((2,), math.log(15.0)),
    )
    per_token_loss.mean().backward()
    assert all(value.grad is not None for value in logits)


def test_masked_loss_is_trial_mean_not_token_mean() -> None:
    losses = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
    trial_indices = torch.tensor([0, 0, 1])
    loss, n_trials = mean_masked_loss_per_trial(losses, trial_indices, batch_size=3)
    # Trial 0 mean=2, trial 1 mean=10; trial 2 has no target and is excluded.
    torch.testing.assert_close(loss, torch.tensor(6.0))
    assert n_trials == 2
    loss.backward()
    torch.testing.assert_close(
        losses.grad, torch.tensor([0.25, 0.25, 0.5])
    )


def test_paired_multiblock_is_nonoverlapping_gapped_and_valid_only() -> None:
    eye_valid = torch.ones(8, 64, 2, dtype=torch.bool)
    eye_valid[:, 28:31, 0] = False
    pad = torch.zeros(8, 64, dtype=torch.bool)
    pad[:, 60:] = True
    mask, loss_mask = generate_eyemae_mask_paired_multiblock(
        eye_valid,
        pad,
        mask_ratio=0.6,
        num_blocks=4,
        target_scale_min=0.15,
        target_scale_max=0.15,
        min_gap=1,
        generator=torch.Generator().manual_seed(17),
    )
    left = mask[:, 2::3]
    right = mask[:, 3::3]
    eligible = eye_valid.all(dim=-1) & ~pad
    torch.testing.assert_close(mask, loss_mask)
    torch.testing.assert_close(left, right)
    assert not mask[:, 0].any()
    assert not mask[:, 1::3].any()
    assert not (left & ~eligible).any()
    for row in left.tolist():
        components = 0
        previous = False
        for selected in row:
            if selected and not previous:
                components += 1
            previous = selected
        assert components == 4

    repeated, _ = generate_eyemae_mask_paired_multiblock(
        eye_valid,
        pad,
        mask_ratio=0.6,
        num_blocks=4,
        target_scale_min=0.15,
        target_scale_max=0.15,
        min_gap=1,
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(mask, repeated)


def test_paired_fixed_blocks_respect_count_span_gap_and_validity() -> None:
    eye_valid = torch.ones(3, 64, 2, dtype=torch.bool)
    pad = torch.zeros(3, 64, dtype=torch.bool)
    mask, loss_mask = generate_eyemae_mask_paired_fixed_blocks(
        eye_valid, pad, num_blocks=10, span_min=1, span_max=3, min_gap=1,
        generator=torch.Generator().manual_seed(11),
    )
    selected = mask[:, 2::3]
    torch.testing.assert_close(mask, loss_mask)
    torch.testing.assert_close(selected, mask[:, 3::3])
    for row in selected.tolist():
        components, lengths, current = 0, [], 0
        for value in row + [False]:
            if value:
                current += 1
            elif current:
                components += 1
                lengths.append(current)
                current = 0
        assert components == 10
        assert all(1 <= length <= 3 for length in lengths)

    long_mask, _ = generate_eyemae_mask_paired_fixed_blocks(
        eye_valid, pad, num_blocks=4, span_min=4, span_max=6, min_gap=1,
        generator=torch.Generator().manual_seed(12),
    )
    for row in long_mask[:, 2::3].tolist():
        lengths, current = [], 0
        for value in row + [False]:
            if value:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        assert len(lengths) == 4
        assert all(4 <= length <= 6 for length in lengths)


def test_generic_mask_dispatches_both_paired_modes() -> None:
    eye_valid = torch.ones(1, 20, 2, dtype=torch.bool)
    pad = torch.zeros(1, 20, dtype=torch.bool)
    for mode in ("paired_random", "paired_span", "paired_multiblock"):
        mask, _ = generate_eyemae_mask(
            eye_valid,
            pad,
            mask_ratio=0.25,
            mode=mode,
            generator=torch.Generator().manual_seed(3),
            span_min=2,
            span_max=6,
        )
        torch.testing.assert_close(mask[:, 2::3], mask[:, 3::3])


def test_bert_training_has_no_tokenizer_or_weighted_mask_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/eyemae/eyevq/pretrain/train.py").read_text(encoding="utf-8")
    assert "load_tokenizer_checkpoint" not in source
    assert "eyevq.tokenizer.train" not in source
    assert ".encode_codes(" not in source
    assert "CodeLossEMA" not in source
    assert "generate_eyemae_mask_freq" not in source
    assert "generate_eyemae_mask_loss" not in source


def test_downstream_freezes_pretraining_only_parameters() -> None:
    bert = EyeVQBERT(
        K_e=15,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dim_ff=32,
        max_time=4,
        patch_samples=20,
        fsq_L=[3, 5],
    )
    model = EyeVQForClassification(bert, num_classes=2, dropout=0.0)

    assert not model.bert.embed.mask_token.requires_grad
    assert all(not parameter.requires_grad for parameter in model.bert.pred_head.parameters())

    batch, time = 2, 4
    logits = model(
        stim_patches=torch.randn(batch, time, 4, 20),
        eye_patches=torch.randn(batch, time, 2, 4, 20),
        quality=torch.zeros(batch, time, 2, 20, 1),
        pad_mask=torch.zeros(batch, time, dtype=torch.bool),
        eye_nonmissing_frac=torch.ones(batch, time, 2),
        task_ids=torch.zeros(batch, dtype=torch.long),
    )
    logits.sum().backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert missing == []


def test_downstream_encoder_and_head_lr_schedules_are_independent() -> None:
    enc_lr, head_lr = get_encoder_head_lrs(
        step=500,
        warmup=100,
        max_steps=1000,
        encoder_lr=1e-6,
        head_lr=2e-5,
        encoder_min_lr=1e-7,
        head_min_lr=2e-6,
    )
    assert head_lr / enc_lr == pytest.approx(20.0)

    enc_end, head_end = get_encoder_head_lrs(
        step=1000,
        warmup=100,
        max_steps=1000,
        encoder_lr=1e-6,
        head_lr=2e-5,
        encoder_min_lr=1e-7,
        head_min_lr=2e-6,
    )
    assert enc_end == pytest.approx(1e-7)
    assert head_end == pytest.approx(2e-6)
















def test_manual_continuous_uses_smooth_l1_and_count_has_zero_gradient() -> None:
    prediction = torch.tensor([[2.0, 100.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss, stats = compute_manual_feature_loss(
        prediction,
        target,
        loss_mask=torch.ones_like(prediction, dtype=torch.bool),
        binary_mask=torch.tensor([False, False]),
        count_mask=torch.tensor([False, True]),
        cfg={"loss": {}},
    )

    # SmoothL1(2, 0) = |2| - 0.5 = 1.5.  The raw count target is excluded.
    assert loss.item() == pytest.approx(1.5)
    assert stats["L_feat_continuous"].item() == pytest.approx(1.5)
    assert stats["L_feat_binary"].item() == pytest.approx(0.0)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].item() == pytest.approx(1.0)
    assert prediction.grad[0, 1].item() == pytest.approx(0.0)


def test_manual_feature_loss_accumulates_mixed_bf16_losses_in_fp32() -> None:
    prediction = torch.tensor(
        [[0.25, 2.0, 100.0]], dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.zeros(1, 3, dtype=torch.float32)
    loss, _ = compute_manual_feature_loss(
        prediction,
        target,
        loss_mask=torch.ones_like(prediction, dtype=torch.bool),
        binary_mask=torch.tensor([True, False, False]),
        count_mask=torch.tensor([False, False, True]),
        cfg={"loss": {}},
    )
    assert loss.dtype == torch.float32
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].item() != 0.0
    assert prediction.grad[0, 1].item() != 0.0
    assert prediction.grad[0, 2].item() == 0.0


def test_manual_binary_balance_uses_train_prior_and_preserves_expected_scale() -> None:
    stats = {
        "continuous": {"feature_idx": 1, "is_binary": 0, "is_count": 0, "mean": 5.0},
        "rare_flag": {"feature_idx": 0, "is_binary": 1, "is_count": 0, "mean": 0.25},
        "count": {"feature_idx": 2, "is_binary": 0, "is_count": 1, "mean": 2.0},
    }
    binary, count, pos_weight, bce_scale = build_manual_feature_loss_metadata(
        stats,
        num_features=3,
        device=torch.device("cpu"),
        balanced_binary=True,
    )
    assert binary.tolist() == [True, False, False]
    assert count.tolist() == [False, False, True]
    assert pos_weight.tolist() == pytest.approx([3.0, 1.0, 1.0])
    assert bce_scale.tolist() == pytest.approx([2.0 / 3.0, 1.0, 1.0])
    expected_mass = (1.0 - 0.25) + 0.25 * pos_weight[0].item()
    assert expected_mass * bce_scale[0].item() == pytest.approx(1.0)


def test_normalized_blink_pos_weight_does_not_inflate_bce_scale() -> None:
    prediction = torch.zeros(1, 1, 1, 4, 4)
    target = torch.zeros_like(prediction)
    target[..., 3, 0] = 1.0
    quality = torch.zeros(1, 1, 1, 4, 1)
    pad_mask = torch.zeros(1, 1, dtype=torch.bool)
    cfg = {
        "loss": {
            "eye_xy_weight": 0.0,
            "eye_area_weight": 0.0,
            "eye_blink_weight": 1.0,
            "eye_velocity_weight": 0.0,
            "eye_blink_pos_weight": 3.0,
        }
    }
    loss, stats = compute_eye_recon_loss(
        prediction,
        target,
        quality,
        pad_mask,
        cfg,
    )
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert stats["blink_positive_fraction"].item() == pytest.approx(0.25)




def test_cache_lookup_never_substitutes_missing_labels() -> None:
    batch = {"global_trial_id": ["present", "missing"]}
    codes = torch.zeros(1, 4, 2, dtype=torch.int16).numpy()
    with pytest.raises(KeyError, match="misses 1 trial"):
        lookup_code_ids(batch, {"present": 0}, codes, torch.device("cpu"), 4)




class _SizedDataset:
    def __len__(self):
        return 97

    def get_num_patches(self, index):
        return index % 9 + 1


def test_rank_samplers_are_disjoint_with_shared_seed() -> None:
    dataset = _SizedDataset()
    per_rank = []
    for rank in range(4):
        sampler = TokenBatchSampler(
            dataset,
            max_seq_tokens=10_000,
            max_trials=8,
            shuffle=True,
            seed=123,
            rank=rank,
            world_size=4,
        )
        per_rank.append({index for batch in sampler for index in batch})
    for left in range(4):
        for right in range(left + 1, 4):
            assert per_rank[left].isdisjoint(per_rank[right])
    assert set.union(*per_rank) == set(range(len(dataset)))


def test_removed_architectures_are_rejected() -> None:
    with pytest.raises(ValueError, match="lightweight_recon"):
        tokenizer_architecture({"model": {"lightweight_recon": True}})
    with pytest.raises(ValueError, match="sep_decoders"):
        tokenizer_architecture({"arch": {"sep_decoders": True}})


def test_legacy_checkpoint_without_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy checkpoint"):
        checkpoint_config({"cfg": {}})


def test_installed_package_imports_outside_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", "import eyemae.eyevq; print(eyemae.eyevq.EyeVQTokenizer.__name__)"],
        cwd="/tmp",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "EyeVQTokenizer"


@pytest.mark.parametrize(
    "relative_path,forward_target",
    [
        ("src/eyemae/eyevq/tokenizer/train.py", "model"),
        ("src/eyemae/eyevq/pretrain/train.py", "model"),
        ("src/eyemae/eyevq/downstream/train_mil.py", "model"),
    ],
)
def test_training_forward_does_not_bypass_ddp(relative_path: str, forward_target: str) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert forward_target in called_names
    # raw_model remains valid for save/eval/state access, never for the training call.
    stripped_lines = [line.strip() for line in source.splitlines()]
    assert not any(line.startswith("out = raw_model(") for line in stripped_lines)
    assert not any(line.startswith("loss, stats = raw_model(") for line in stripped_lines)
    assert not any(line.startswith("logits = raw_model(") for line in stripped_lines)


def test_adamw_groups_exclude_bias_norm_and_vectors_from_decay() -> None:
    module = nn.Sequential(
        nn.Linear(4, 3),
        nn.LayerNorm(3),
        nn.Linear(3, 2, bias=False),
    )
    module.register_parameter("cls_token", nn.Parameter(torch.zeros(1, 1, 4)))
    groups, audit = build_adamw_param_groups(module, weight_decay=0.05)
    by_decay = {float(group["weight_decay"]): group for group in groups}
    decay_ids = {id(parameter) for parameter in by_decay[0.05]["params"]}
    no_decay_ids = {id(parameter) for parameter in by_decay[0.0]["params"]}
    named = dict(module.named_parameters())
    assert id(named["0.weight"]) in decay_ids
    assert id(named["2.weight"]) in decay_ids
    assert id(named["0.bias"]) in no_decay_ids
    assert id(named["1.weight"]) in no_decay_ids
    assert id(named["1.bias"]) in no_decay_ids
    assert id(named["cls_token"]) in no_decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == {id(parameter) for parameter in module.parameters()}
    assert audit["num_trainable_parameters"] == sum(
        parameter.numel() for parameter in module.parameters()
    )


def test_latest_step_checkpoint_uses_numeric_step(tmp_path: Path) -> None:
    for name in ("ckpt_step7500.pt", "ckpt_step040000.pt", "ckpt_step10000.pt"):
        (tmp_path / name).touch()
    assert latest_step_checkpoint(tmp_path).name == "ckpt_step040000.pt"








def test_content_addressed_cache_rejects_same_path_tokenizer_replacement(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.pt"
    tokenizer.write_bytes(b"tokenizer-a")
    cache = tmp_path / "codes.npz"
    contract_sha = "c" * 64
    np.savez_compressed(
        cache,
        format_version=np.int64(CACHE_FORMAT_VERSION),
        lr_layout=np.array("time_major_lr_v1"),
        invalid_eye_code_id=np.int64(0),
        patch_samples=np.int64(40),
        tokenizer_checkpoint=np.array(str(tokenizer.resolve())),
        tokenizer_sha256=np.array(artifact_sha256_file(tokenizer)),
        cache_contract_sha256=np.array(contract_sha),
        gids=np.array(["trial"], dtype=object),
        code_ids=np.zeros((1, 1, 2), dtype=np.int16),
        num_patches=np.ones(1, dtype=np.int16),
    )
    write_cache_manifest(
        cache,
        tokenizer_checkpoint=tokenizer,
        contract_sha256=contract_sha,
        num_trials=1,
    )
    validate_cache_identity(
        cache,
        tokenizer_checkpoint=tokenizer,
        contract_sha256=contract_sha,
    )
    tokenizer.write_bytes(b"tokenizer-b")
    with pytest.raises(ValueError, match="tokenizer SHA256"):
        validate_cache_identity(
            cache,
            tokenizer_checkpoint=tokenizer,
            contract_sha256=contract_sha,
        )


def test_strict_resume_identity_rejects_configuration_change() -> None:
    expected = {
        "identity_version": 1,
        "stage": "bert",
        "config_sha256": "a",
        "dependencies": {},
    }
    checkpoint = {"run_identity": {**expected, "config_sha256": "b"}}
    with pytest.raises(ValueError, match="Unsafe checkpoint resume rejected"):
        assert_run_identity(checkpoint, expected)
    assert_run_identity(checkpoint, expected, allow_mismatch=True)
