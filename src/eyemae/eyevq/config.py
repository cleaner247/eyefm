"""Validated model factories shared by every EyeVQ stage."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch

from .pretrain.model import (
    EyeVQBERT,
    SUPPORTED_TARGET_TYPES,
    TARGET_DIRECT_RECONSTRUCTION,
    TARGET_FACTORIZED_CODE,
    TARGET_JOINT_CODE,
    TARGET_NORMALIZED_LATENT,
    TARGET_RAW_PATCH,
)
from .tokenizer.model import EyeVQTokenizer


CONFIG_VERSION = 1


def override_fsq_levels(cfg: dict[str, Any], levels_text: str | None) -> None:
    """Apply one auditable FSQ CLI override to tokenizer/BERT configs."""
    if levels_text is None:
        return
    try:
        levels = [int(value.strip()) for value in levels_text.split(",")]
    except ValueError as error:
        raise ValueError("FSQ levels must be comma-separated integers") from error
    if not levels or any(level < 2 for level in levels):
        raise ValueError("Every FSQ level must be at least 2")
    if str(cfg.get("vq", {}).get("type", "fsq")) != "fsq":
        raise ValueError("--fsq-levels can only override vq.type=fsq")
    cfg["vq"]["fsq_d"] = len(levels)
    cfg["vq"]["fsq_L"] = levels


def _quantizer_spec(cfg: dict[str, Any]) -> tuple[str, int, int, list[int] | None]:
    vq = cfg["vq"]
    vq_type = str(vq.get("type", "fsq"))
    if vq_type == "fsq":
        code_dim = int(vq["fsq_d"])
        levels = vq["fsq_L"]
        levels_list = [int(levels)] * code_dim if isinstance(levels, int) else [int(v) for v in levels]
        if len(levels_list) != code_dim:
            raise ValueError(f"vq.fsq_L has {len(levels_list)} dimensions, expected {code_dim}")
        return vq_type, code_dim, math.prod(levels_list), levels_list
    if vq_type == "vqvae":
        return vq_type, int(vq["code_dim"]), int(vq["codebook_size"]), None
    if vq_type == "ae":
        code_dim = int(vq["code_dim"])
        return vq_type, code_dim, code_dim, None
    raise ValueError(f"Unsupported vq.type={vq_type!r}; expected 'fsq', 'vqvae', or 'ae'")


# Kept as a private compatibility alias for older callers.
_fsq_spec = _quantizer_spec


def tokenizer_architecture(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("lightweight_recon", False)) or bool(
        cfg.get("encoder", {}).get("lightweight_recon", False)
    ):
        raise ValueError("lightweight_recon was removed; trial features must pass through FSQ")
    arch_cfg = cfg.get("arch", {})
    if "sep_decoders" in arch_cfg:
        raise ValueError("arch.sep_decoders was removed; use model.architecture=cross_attention")
    architecture = str(model_cfg.get("architecture", "joint"))
    if architecture not in {"joint", "cross_attention"}:
        raise ValueError(
            "model.architecture must be 'joint' or 'cross_attention', "
            f"got {architecture!r}"
        )
    return architecture


def validate_tokenizer_config(cfg: dict[str, Any]) -> None:
    for section in ("model", "attention", "encoder", "decoder", "vq", "patch", "manual_features"):
        if section not in cfg:
            raise ValueError(f"Tokenizer config is missing required section: {section}")
    if "architecture" not in cfg["model"]:
        raise ValueError("Tokenizer config must explicitly declare model.architecture")
    if "stim_isolated" not in cfg["attention"]:
        raise ValueError("Tokenizer config must explicitly declare attention.stim_isolated")
    if "stim_attend_cls" in cfg["attention"] and not isinstance(
        cfg["attention"]["stim_attend_cls"], bool
    ):
        raise ValueError("attention.stim_attend_cls must be a boolean")
    attention_layout = str(cfg["attention"].get("layout", "joint"))
    if attention_layout not in {"joint", "axial_cross"}:
        raise ValueError("attention.layout must be 'joint' or 'axial_cross'")
    if "num_features" not in cfg["manual_features"]:
        raise ValueError("Tokenizer config must explicitly declare manual_features.num_features")
    encoder = cfg["encoder"]
    decoder = cfg["decoder"]
    if int(encoder["d_model"]) != int(decoder.get("d_model", encoder["d_model"])):
        raise ValueError("encoder.d_model and decoder.d_model must match")
    if int(cfg["patch"]["samples"]) != int(cfg["patch"].get("stride", cfg["patch"]["samples"])):
        raise ValueError("EyeVQ currently requires non-overlapping patches (samples == stride)")
    vq_type, code_dim, codebook_size, levels = _quantizer_spec(cfg)
    if vq_type == "fsq":
        if any(int(level) % 2 == 0 for level in levels):
            raise ValueError("EyeVQ FSQ/iFSQ currently requires odd levels")
        activation = str(cfg["vq"].get("fsq_activation", "tanh"))
        if activation not in {"tanh", "ifsq"}:
            raise ValueError(
                "vq.fsq_activation must be 'tanh' or 'ifsq', "
                f"got {activation!r}"
            )
        alpha = float(cfg["vq"].get("ifsq_alpha", 1.6))
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"vq.ifsq_alpha must be finite and positive, got {alpha}")
    elif vq_type == "vqvae":
        if code_dim < 1 or codebook_size < 2:
            raise ValueError("VQ-VAE requires vq.code_dim >= 1 and vq.codebook_size >= 2")
        beta = float(cfg["vq"].get("commitment_beta", 0.25))
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("vq.commitment_beta must be finite and non-negative")
    elif code_dim < 1:
        raise ValueError("Continuous AE requires vq.code_dim >= 1")
    tokenizer_architecture(cfg)


def build_tokenizer(cfg: dict[str, Any]) -> EyeVQTokenizer:
    validate_tokenizer_config(cfg)
    encoder = cfg["encoder"]
    decoder = cfg["decoder"]
    vq = cfg["vq"]
    features = cfg["manual_features"]
    architecture = tokenizer_architecture(cfg)
    vq_type, code_dim, codebook_size, levels = _quantizer_spec(cfg)
    return EyeVQTokenizer(
        d_model=int(encoder["d_model"]),
        enc_n_layers=int(encoder["n_layers"]),
        enc_n_heads=int(encoder["n_heads"]),
        enc_dim_ff=int(encoder["dim_ff"]),
        vq_type=vq_type,
        eye_code_dim=code_dim,
        eye_codebook_size=codebook_size,
        fsq_L=levels if levels is not None else 5,
        fsq_activation=str(vq.get("fsq_activation", "tanh")),
        ifsq_alpha=float(vq.get("ifsq_alpha", 1.6)),
        commitment_beta=float(vq.get("commitment_beta", 0.25)),
        dec_n_layers=int(decoder["n_layers"]),
        dec_n_heads=int(decoder["n_heads"]),
        dec_dim_ff=int(decoder["dim_ff"]),
        num_manual_features=int(features["num_features"]),
        max_patches=int(encoder["max_patches"]),
        patch_samples=int(cfg["patch"]["samples"]),
        dropout=float(encoder.get("dropout", 0.0)),
        stim_isolated_attn=bool(cfg.get("attention", {}).get("stim_isolated", True)),
        stim_attend_cls=bool(cfg.get("attention", {}).get("stim_attend_cls", True)),
        attention_layout=str(cfg.get("attention", {}).get("layout", "joint")),
        architecture=architecture,
        s_enc_n_layers=int(model_section(cfg).get("stim_layers", 3)),
        feat_dec_n_layers=int(model_section(cfg).get("cross_layers", decoder["n_layers"])),
        min_nonmissing_frac=float(vq.get("min_nonmissing_frac", 0.50)),
    )


def model_section(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("model", {})


def validate_bert_config(cfg: dict[str, Any]) -> None:
    for section in ("attention", "bert", "vq", "patch", "mask", "loss"):
        if section not in cfg:
            raise ValueError(f"Pretrain config is missing required section: {section}")
    bert = cfg["bert"]
    if "stim_isolated" not in cfg["attention"]:
        raise ValueError("Pretrain config must explicitly declare attention.stim_isolated")
    if "stim_attend_cls" in cfg["attention"] and not isinstance(
        cfg["attention"]["stim_attend_cls"], bool
    ):
        raise ValueError("attention.stim_attend_cls must be a boolean")
    attention_layout = str(cfg["attention"].get("layout", "joint"))
    if attention_layout not in {"joint", "axial_cross"}:
        raise ValueError("attention.layout must be 'joint' or 'axial_cross'")
    if int(bert["max_time"]) != int(bert["max_patches"]):
        raise ValueError("bert.max_time and bert.max_patches must match")
    if int(cfg["patch"]["samples"]) != int(cfg["patch"].get("stride", cfg["patch"]["samples"])):
        raise ValueError("EyeVQ pretraining requires non-overlapping patches")
    vq_type, _code_dim, _codebook_size, _levels = _quantizer_spec(cfg)
    factorized = bool(bert.get("factorized_fsq", False))
    target_type = str(
        bert.get(
            "target_type",
            TARGET_FACTORIZED_CODE if factorized else TARGET_JOINT_CODE,
        )
    )
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise ValueError(
            f"bert.target_type must be one of {sorted(SUPPORTED_TARGET_TYPES)}"
        )
    if factorized and vq_type != "fsq":
        raise ValueError("bert.factorized_fsq requires vq.type=fsq")
    if (target_type == TARGET_FACTORIZED_CODE) != factorized:
        raise ValueError("factorized_code target and bert.factorized_fsq must be enabled together")
    if target_type == TARGET_NORMALIZED_LATENT and vq_type != "ae":
        raise ValueError("normalized_latent target requires vq.type=ae")
    if vq_type == "ae" and target_type != TARGET_NORMALIZED_LATENT:
        raise ValueError("vq.type=ae requires bert.target_type=normalized_latent")
    if target_type in {TARGET_RAW_PATCH, TARGET_DIRECT_RECONSTRUCTION}:
        raw_defaults = {
            "raw_xy_weight": cfg["loss"].get("raw_continuous_weight", 1.0),
            "raw_area_weight": cfg["loss"].get("raw_continuous_weight", 1.0),
            "raw_blink_weight": 1.0,
            "raw_blink_pos_weight": 1.0,
            "raw_velocity_weight": 0.0,
        }
        for key, default in raw_defaults.items():
            value = float(cfg["loss"].get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss.{key} must be finite and non-negative")
        if float(cfg["loss"].get("raw_blink_pos_weight", 1.0)) <= 0.0:
            raise ValueError("loss.raw_blink_pos_weight must be positive")
    if target_type == TARGET_DIRECT_RECONSTRUCTION:
        feature_cfg = cfg.get("manual_features", {})
        if not bool(feature_cfg.get("enabled", False)):
            raise ValueError(
                "direct_reconstruction requires manual_features.enabled=true"
            )
        if int(feature_cfg.get("num_features", 0)) <= 0:
            raise ValueError(
                "direct_reconstruction requires manual_features.num_features > 0"
            )
        for key in (
            "eye_recon_group_weight",
            "manual_feature_group_weight",
            "manual_feature_binary_weight",
            "manual_feature_continuous_weight",
        ):
            value = float(cfg["loss"].get(key, math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"loss.{key} must be finite and non-negative")
        if (
            float(cfg["loss"]["eye_recon_group_weight"]) == 0.0
            and float(cfg["loss"]["manual_feature_group_weight"]) == 0.0
        ):
            raise ValueError("direct_reconstruction requires a nonzero loss group")
    mode = str(cfg["mask"].get("mode", "paired_random"))
    supported_modes = {
        "uniform", "paired_random", "paired_span", "paired_multiblock",
        "paired_dual_scale",
    }
    if mode not in supported_modes:
        raise ValueError(
            f"mask.mode must be one of {sorted(supported_modes)}, got {mode!r}; "
            "dynamic/frequency/loss-weighted sampling is unsupported"
        )
    ratio = float(cfg["mask"].get("eye_masking_ratio", 0.25))
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask.eye_masking_ratio must be in (0, 1), got {ratio}")
    span_min = int(cfg["mask"].get("span_min_patches", 2))
    span_max = int(cfg["mask"].get("span_max_patches", 6))
    if span_min < 1 or span_max < span_min:
        raise ValueError(
            "mask span bounds must satisfy 1 <= span_min_patches <= "
            f"span_max_patches, got [{span_min}, {span_max}]"
        )
    if mode == "paired_dual_scale":
        for prefix in ("dual_short", "dual_long"):
            blocks = int(cfg["mask"].get(f"{prefix}_num_blocks", 0))
            lower = int(cfg["mask"].get(f"{prefix}_span_min_patches", 0))
            upper = int(cfg["mask"].get(f"{prefix}_span_max_patches", 0))
            if blocks < 1 or lower < 1 or upper < lower:
                raise ValueError(
                    f"mask.{prefix} requires positive blocks and valid span bounds"
                )
        if int(cfg["mask"].get("dual_min_gap", 1)) < 0:
            raise ValueError("mask.dual_min_gap must be non-negative")
    predictor_span_embedding = bool(
        bert.get("predictor_span_length_embedding", False)
    )
    if predictor_span_embedding and mode != "paired_span":
        raise ValueError(
            "bert.predictor_span_length_embedding requires mask.mode=paired_span"
        )
    span_distribution = str(cfg["mask"].get("span_length_distribution", "uniform"))
    if span_distribution not in {
        "uniform", "symmetric_power", "explicit", "token_balanced"
    }:
        raise ValueError(
            "mask.span_length_distribution must be 'uniform', 'symmetric_power', "
            "'token_balanced', or 'explicit', "
            f"got {span_distribution!r}"
        )
    span_power = float(cfg["mask"].get("span_length_power", 1.0))
    if not math.isfinite(span_power) or span_power <= 0:
        raise ValueError(
            f"mask.span_length_power must be finite and positive, got {span_power}"
        )
    probabilities = cfg["mask"].get("span_length_probabilities")
    if span_distribution == "explicit":
        if not isinstance(probabilities, list) or len(probabilities) != span_max - span_min + 1:
            raise ValueError(
                "mask.span_length_probabilities must contain one value per span length"
            )
        values = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise ValueError(
                "mask.span_length_probabilities must be finite, non-negative, and sum positive"
            )
    multiblock_num_blocks = int(cfg["mask"].get("multiblock_num_blocks", 4))
    multiblock_scale_min = float(
        cfg["mask"].get("multiblock_target_scale_min", 0.12)
    )
    multiblock_scale_max = float(
        cfg["mask"].get("multiblock_target_scale_max", 0.18)
    )
    multiblock_min_gap = int(cfg["mask"].get("multiblock_min_gap", 1))
    if multiblock_num_blocks < 1:
        raise ValueError("mask.multiblock_num_blocks must be positive")
    if not 0.0 < multiblock_scale_min <= multiblock_scale_max < 1.0:
        raise ValueError(
            "mask multiblock target scales must satisfy 0 < min <= max < 1"
        )
    if multiblock_min_gap < 0:
        raise ValueError("mask.multiblock_min_gap must be non-negative")


def build_bert(cfg: dict[str, Any]) -> EyeVQBERT:
    validate_bert_config(cfg)
    bert = cfg["bert"]
    vq_type, code_dim, codebook_size, levels = _quantizer_spec(cfg)
    fsq_levels = levels if vq_type == "fsq" and isinstance(levels, list) else None
    return EyeVQBERT(
        K_e=codebook_size,
        d_model=int(bert["d_model"]),
        n_layers=int(bert["n_layers"]),
        n_heads=int(bert["n_heads"]),
        dim_ff=int(bert["dim_ff"]),
        dropout=float(bert.get("dropout", 0.0)),
        max_time=int(bert.get("max_time", bert.get("max_patches", 512))),
        patch_samples=int(cfg["patch"]["samples"]),
        label_smoothing=float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
        stim_isolated_attn=bool(cfg.get("attention", {}).get("stim_isolated", True)),
        stim_attend_cls=bool(cfg.get("attention", {}).get("stim_attend_cls", True)),
        attention_layout=str(cfg.get("attention", {}).get("layout", "joint")),
        factorized_fsq=bool(bert.get("factorized_fsq", False)),
        fsq_L=fsq_levels,
        min_nonmissing_frac=float(cfg["vq"].get("min_nonmissing_frac", 0.50)),
        predictor_span_length_embedding=bool(
            bert.get("predictor_span_length_embedding", False)
        ),
        max_mask_span_length=int(cfg["mask"].get("span_max_patches", 0)),
        target_type=str(
            bert.get(
                "target_type",
                TARGET_FACTORIZED_CODE
                if bool(bert.get("factorized_fsq", False))
                else TARGET_JOINT_CODE,
            )
        ),
        latent_dim=code_dim,
        raw_xy_weight=float(
            cfg["loss"].get(
                "raw_xy_weight", cfg["loss"].get("raw_continuous_weight", 1.0)
            )
        ),
        raw_area_weight=float(
            cfg["loss"].get(
                "raw_area_weight", cfg["loss"].get("raw_continuous_weight", 1.0)
            )
        ),
        raw_blink_weight=float(cfg["loss"].get("raw_blink_weight", 1.0)),
        raw_blink_pos_weight=float(
            cfg["loss"].get("raw_blink_pos_weight", 1.0)
        ),
        raw_velocity_weight=float(cfg["loss"].get("raw_velocity_weight", 0.0)),
        manual_feature_dim=int(
            cfg.get("manual_features", {}).get("num_features", 38)
        ),
        direct_loss_cfg=deepcopy(cfg["loss"]),
    )


def normalized_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        key = key.replace("_orig_mod.", "")
        normalized[key] = value
    return normalized


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    version = checkpoint.get("config_version")
    if version != CONFIG_VERSION:
        raise ValueError(
            f"Unsupported or legacy checkpoint config_version={version!r}; "
            f"expected {CONFIG_VERSION}. Retrain or explicitly convert the checkpoint."
        )
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint has no embedded cfg; legacy checkpoints require explicit conversion")
    return deepcopy(cfg)


def load_tokenizer_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[EyeVQTokenizer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint_config(checkpoint)
    model = build_tokenizer(cfg).to(device)
    model.load_state_dict(normalized_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, checkpoint


def load_bert_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[EyeVQBERT, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint_config(checkpoint)
    model = build_bert(cfg).to(device)
    model.load_state_dict(normalized_state_dict(checkpoint["model_state_dict"]), strict=True)
    return model, cfg, checkpoint
