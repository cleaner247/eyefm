from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _require(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing required config field: {dotted}")
        cur = cur[part]
    return cur


def validate_config(cfg: dict[str, Any], *, require_splits: bool = True) -> None:
    required = [
        "experiment.name",
        "experiment.output_dir",
        "data.format",
        "data.data_dir",
        "data.sampling_rate",
        "data.npz_keys",
        "label.nonblink_value",
        "label.blink_value",
        "label.missing_value",
        "normalization.x_clip_deg",
        "normalization.y_clip_deg",
        "area.stats_path",
        "stim.stim_dim",
        "input.content_dim",
        "input.quality_dim",
        "input.stim_dim",
        "patch.samples",
        "patch.stride",
        "attention.min_nonmissing_frac_for_eye_token",
        "mask.min_nonmissing_frac_for_mae",
        "model.sequence_format",
        "model.pretrain_style",
        "model.use_cls",
        "model.use_stim_tokens",
        "model.broadcast_stim_to_eye",
        "train.precision",
    ]
    for field in required:
        _require(cfg, field)

    if cfg["data"]["format"] != "npz_per_trial":
        raise ValueError("data.format must be npz_per_trial")
    nan_policy = cfg["data"].get("nan_policy", "error")
    if nan_policy not in {"error", "mark_missing"}:
        raise ValueError("data.nan_policy must be one of error, mark_missing")

    data_dir = Path(cfg["data"]["data_dir"])
    synthetic_ok = "tests/fixtures/synthetic_npz" in str(data_dir)
    if not data_dir.exists() and not synthetic_ok:
        raise ValueError(f"data.data_dir does not exist: {data_dir}")

    if require_splits:
        for field in ("data.pretrain_train_split", "data.pretrain_val_split"):
            split_path = Path(_require(cfg, field))
            if not split_path.exists():
                raise ValueError(f"{field} does not exist: {split_path}")

    labels = [
        cfg["label"]["nonblink_value"],
        cfg["label"]["blink_value"],
        cfg["label"]["missing_value"],
    ]
    if len(set(labels)) != 3:
        raise ValueError("label nonblink/blink/missing values must be distinct")

    if int(cfg["input"]["content_dim"]) != 4:
        raise ValueError("input.content_dim must be 4")
    if int(cfg["input"]["quality_dim"]) != 1:
        raise ValueError("input.quality_dim must be 1")
    if int(cfg["input"]["stim_dim"]) != int(cfg["stim"]["stim_dim"]):
        raise ValueError("input.stim_dim must equal stim.stim_dim")
    if int(cfg["stim"]["stim_dim"]) != 4:
        raise ValueError("stim.stim_dim must be 4 for first version")

    if int(cfg["patch"]["samples"]) <= 0:
        raise ValueError("patch.samples must be > 0")
    if int(cfg["patch"]["stride"]) != int(cfg["patch"]["samples"]):
        raise ValueError("patch.stride must equal patch.samples for first version")

    if cfg["model"]["sequence_format"] != "stim_eye_triplet_no_cls":
        raise ValueError("model.sequence_format must be stim_eye_triplet_no_cls")
    if cfg["model"]["pretrain_style"] != "bert_masked_reconstruction":
        raise ValueError("model.pretrain_style must be bert_masked_reconstruction")
    if bool(cfg["model"]["use_cls"]):
        raise ValueError("model.use_cls must be false")
    if not bool(cfg["model"]["use_stim_tokens"]):
        raise ValueError("model.use_stim_tokens must be true")
    if bool(cfg["model"]["broadcast_stim_to_eye"]):
        raise ValueError("model.broadcast_stim_to_eye must be false")

    for field in ("attention.min_nonmissing_frac_for_eye_token", "mask.min_nonmissing_frac_for_mae"):
        value = float(_require(cfg, field))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if float(cfg["mask"]["min_nonmissing_frac_for_mae"]) < float(cfg["attention"]["min_nonmissing_frac_for_eye_token"]):
        raise ValueError("mask.min_nonmissing_frac_for_mae must be >= attention.min_nonmissing_frac_for_eye_token")

    if cfg["train"]["precision"] not in {"bf16", "fp32", "fp16"}:
        raise ValueError("train.precision must be one of bf16, fp32, fp16")


def split_path_for_name(cfg: dict[str, Any], split: str) -> Path:
    key = f"{split}_split" if split.startswith("pretrain_") else f"pretrain_{split}_split"
    if key not in cfg["data"]:
        raise ValueError(f"Unknown split: {split}")
    return Path(cfg["data"][key])
