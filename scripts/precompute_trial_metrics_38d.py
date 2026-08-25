#!/usr/bin/env python3
"""Build the per-trial 38D metrics cache (.pt) and per-feature stats (.json)
that the EyeVQ tokenizer consumes, from a task-aware 38D pt sidecar.

Replicates the format consumed by ``eyemae.eyevq.tokenizer.train``:

  cache : dict { global_trial_id: (continuous_f32[38], loss_mask_bool[38]) }
  stats : dict { feature_name: {median, mad, n_valid, mean, std,
                                is_binary, feature_idx, is_count} }

Notes to keep the format aligned with the original:
  - cache value[0] = ``continuous`` row (float32)
  - cache value[1] = ``loss_mask`` row (bool)  (not raw metric_valid)
  - stats are computed over positions where ``loss_mask`` is True
  - ``mad`` is the RAW median absolute deviation (NOT scaled by 1.4826);
    training scales it by 1.4826 itself
  - ``is_binary`` from metric kind == "binary", ``is_count`` from kind == "count"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_cache_and_stats(
    pt_path: Path,
    extra_pt_paths: tuple[Path, ...] = (),
) -> tuple[dict[str, tuple], dict]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    continuous = payload["continuous"].numpy()
    loss_mask = payload["loss_mask"].numpy()
    metric_types = payload["metric_types"]
    feature_names = list(payload["feature_names"])

    cache: dict[str, tuple] = {}
    for source_path, source in [(pt_path, payload)] + [
        (path, torch.load(path, map_location="cpu", weights_only=False))
        for path in extra_pt_paths
    ]:
        if list(source["feature_names"]) != feature_names:
            raise ValueError(f"feature_names mismatch in {source_path}")
        source_continuous = source["continuous"].numpy()
        source_mask = source["loss_mask"].numpy()
        for i, gid in enumerate(source["global_trial_id"]):
            key = str(gid)
            if key in cache:
                raise ValueError(f"duplicate global_trial_id across sidecars: {key}")
            cache[key] = (
                np.ascontiguousarray(source_continuous[i], dtype=np.float32),
                np.ascontiguousarray(source_mask[i], dtype=bool),
            )

    stats: dict[str, dict] = {}
    for j, name in enumerate(feature_names):
        vals = continuous[loss_mask[:, j], j]
        n_valid = int(vals.size)
        if n_valid == 0:
            median = 0.0
            mad = 0.0
            mean = 0.0
            std = 0.0
        else:
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median)))  # RAW MAD, no 1.4826
            mean = float(np.mean(vals))
            std = float(np.std(vals))
        stats[name] = {
            "median": median,
            "mad": mad,
            "n_valid": n_valid,
            "mean": mean,
            "std": std,
            "is_binary": int(metric_types[j] == "binary"),
            "feature_idx": j,
            "is_count": int(metric_types[j] == "count"),
        }
    return cache, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path, required=True,
                        help="task-aware 38D pt sidecar (e.g. train.pt)")
    parser.add_argument(
        "--extra-pt", type=Path, action="append", default=[],
        help="additional sidecar included in the cache; stats remain train-only from --pt",
    )
    parser.add_argument("--out-cache", type=Path, required=True,
                        help="output .pt cache path")
    parser.add_argument("--out-stats", type=Path, required=True,
                        help="output .json stats path")
    args = parser.parse_args()

    cache, stats = build_cache_and_stats(args.pt, tuple(args.extra_pt))
    args.out_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out_cache)
    args.out_stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"cache: {len(cache)} trials -> {args.out_cache}")
    print(f"stats: {len(stats)} features -> {args.out_stats}")


if __name__ == "__main__":
    main()
