"""Shared loading and normalization for the 38D trial-level targets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def build_manual_feature_loss_metadata(
    stats: dict[str, dict[str, Any]],
    num_features: int,
    device: torch.device,
    *,
    balanced_binary: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indexed = sorted(
        (
            int(values.get("feature_idx", fallback)),
            name,
            values,
        )
        for fallback, (name, values) in enumerate(stats.items())
    )
    if [item[0] for item in indexed] != list(range(num_features)):
        raise ValueError("manual feature stats must cover every feature exactly once")
    binary = torch.tensor(
        [bool(values.get("is_binary", 0)) for _, _, values in indexed],
        dtype=torch.bool,
        device=device,
    )
    count = torch.tensor(
        [bool(values.get("is_count", 0)) for _, _, values in indexed],
        dtype=torch.bool,
        device=device,
    )
    if torch.any(binary & count):
        raise ValueError("a manual feature cannot be both binary and count")
    pos_weight = torch.ones(num_features, dtype=torch.float32, device=device)
    bce_scale = torch.ones_like(pos_weight)
    if balanced_binary:
        for index, name, values in indexed:
            if not bool(values.get("is_binary", 0)):
                continue
            fraction = float(values.get("mean", math.nan))
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    f"binary feature {name} has invalid positive fraction {fraction}"
                )
            odds = (1.0 - fraction) / fraction
            pos_weight[index] = odds
            bce_scale[index] = 1.0 / ((1.0 - fraction) + fraction * odds)
    return binary, count, pos_weight, bce_scale


class ManualFeatureTargetStore:
    """Train-only-normalized trial targets with tokenizer-identical metadata."""

    def __init__(self, cfg: dict, loss_cfg: dict, device: torch.device) -> None:
        if not bool(cfg.get("enabled", False)):
            raise ValueError("manual feature target store requires enabled=true")
        self.num_features = int(cfg["num_features"])
        cache_path = Path(str(cfg.get("cache_path", "")))
        stats_path = Path(str(cfg.get("stats_path", "")))
        weight_path = Path(str(cfg.get("task_metric_weight_path", "")))
        for label, path in (
            ("cache_path", cache_path),
            ("stats_path", stats_path),
            ("task_metric_weight_path", weight_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"manual_features.{label} does not exist: {path}")
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if len(raw_stats) != self.num_features:
            raise ValueError(
                f"manual feature stats contain {len(raw_stats)} entries, "
                f"expected {self.num_features}"
            )
        self.stats = dict(
            sorted(
                raw_stats.items(),
                key=lambda item: int(item[1].get("feature_idx", 0)),
            )
        )
        loaded_weights = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(loaded_weights, dict):
            loaded_weights = loaded_weights.get("task_metric_weight")
        if not isinstance(loaded_weights, torch.Tensor):
            raise TypeError("task metric weight file has no tensor")
        self.task_weights = loaded_weights.float().to(device)
        if self.task_weights.ndim != 2 or self.task_weights.shape[1] != self.num_features:
            raise ValueError(
                f"task metric weights must have shape [tasks,{self.num_features}]"
            )
        (
            self.binary_mask,
            self.count_mask,
            self.pos_weight,
            self.bce_scale,
        ) = build_manual_feature_loss_metadata(
            self.stats,
            self.num_features,
            device,
            balanced_binary=bool(loss_cfg.get("manual_feature_balanced_bce", False)),
        )
        self.device = device

    def lookup(self, batch: dict) -> dict[str, torch.Tensor]:
        values, masks = [], []
        for global_id in batch.get("global_trial_id", []):
            cached = self.cache.get(global_id)
            if cached is None:
                values.append(torch.zeros(self.num_features))
                masks.append(torch.zeros(self.num_features, dtype=torch.bool))
                continue
            value = torch.as_tensor(cached[0], dtype=torch.float32)
            mask = torch.as_tensor(cached[1], dtype=torch.bool)
            if value.numel() != self.num_features or mask.numel() != self.num_features:
                raise ValueError(f"manual feature shape mismatch for trial {global_id}")
            values.append(value)
            masks.append(mask)
        if not values:
            raise ValueError("manual feature lookup received an empty batch")
        targets = torch.stack(values).to(self.device)
        loss_mask = torch.stack(masks).to(self.device)
        for index, (_, stat) in enumerate(self.stats.items()):
            if stat.get("is_binary", 0) or stat.get("is_count", 0):
                continue
            mad = float(stat.get("mad", 0.0))
            if mad <= 0.0:
                continue
            valid = loss_mask[:, index]
            targets[valid, index] = (
                (targets[valid, index] - float(stat["median"]))
                / (mad * 1.4826 + 1e-8)
            ).clamp(-5.0, 5.0)
        task_ids = torch.as_tensor(
            batch["task_id"], dtype=torch.long, device=self.device
        )
        return {
            "targets": targets,
            "loss_mask": loss_mask,
            "binary_mask": self.binary_mask,
            "count_mask": self.count_mask,
            "pos_weight": self.pos_weight,
            "bce_scale": self.bce_scale,
            "weights": self.task_weights[task_ids],
        }
