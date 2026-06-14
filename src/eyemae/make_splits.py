from __future__ import annotations

import argparse
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import load_config
from .data import (
    get_split_subject_key,
    infer_subject_from_path,
    infer_suffix_from_name,
    infer_task_from_path,
    load_npz_trial,
)
from .utils import setup_logging, write_json


LOGGER = logging.getLogger(__name__)


def _trial_meta(path: Path, data_dir: Path, cfg: dict[str, Any]) -> tuple[str, int, str]:
    if cfg["data"].get("metadata_from_path", False):
        return infer_subject_from_path(path, data_dir), infer_task_from_path(path), infer_suffix_from_name(path)
    trial = load_npz_trial(path, data_dir, cfg)
    return str(trial["subject_id"]), int(trial["task_id"]), str(trial["subject_id"][-1])


def _write_split(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _summary(
    splits: dict[str, list[Path]],
    metas: dict[Path, tuple[str, int, str]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "strategy": cfg["split"]["strategy"],
        "seed": int(cfg["split"]["seed"]),
        "train_ratio": float(cfg["split"]["train_ratio"]),
        "val_ratio": float(cfg["split"]["val_ratio"]),
        "test_ratio": float(cfg["split"]["test_ratio"]),
        "group_by_base_subject_id": bool(cfg["split"]["group_by_base_subject_id"]),
        "num_train_trials": len(splits["train"]),
        "num_val_trials": len(splits["val"]),
        "num_test_trials": len(splits["test"]),
        "num_train_subjects": 0,
        "num_val_subjects": 0,
        "num_test_subjects": 0,
        "task_counts": {},
        "eye_availability_counts": {},
        "warnings": [],
    }
    for split_name, paths in splits.items():
        subjects = {
            get_split_subject_key(metas[p][0], bool(cfg["split"]["group_by_base_subject_id"]))
            for p in paths
        }
        out[f"num_{split_name}_subjects"] = len(subjects)
        task_counts = Counter(str(metas[p][1]) for p in paths)
        out["task_counts"][split_name] = {str(i): int(task_counts.get(str(i), 0)) for i in range(4)}
        eye_counts = Counter(metas[p][2] if metas[p][2] in {"D", "L", "R"} else "unknown" for p in paths)
        out["eye_availability_counts"][split_name] = {
            "D": int(eye_counts.get("D", 0)),
            "L": int(eye_counts.get("L", 0)),
            "R": int(eye_counts.get("R", 0)),
            "unknown": int(eye_counts.get("unknown", 0)),
        }
        missing_tasks = [str(i) for i in range(4) if task_counts.get(str(i), 0) == 0 and paths]
        if split_name in {"val", "test"} and missing_tasks:
            warning = f"{split_name} split missing task_ids: {','.join(missing_tasks)}"
            LOGGER.warning(warning)
            out["warnings"].append(warning)
    return out


def make_splits(cfg: dict[str, Any], *, out_dir: str | Path | None = None) -> dict[str, Any]:
    data_dir = Path(cfg["data"]["data_dir"])
    split_dir = Path(out_dir or cfg["split"]["out_dir"])
    paths = sorted(p for p in data_dir.rglob("*.npz") if p.is_file())
    if not paths:
        raise RuntimeError(f"No .npz files found under {data_dir}")
    metas = {p: _trial_meta(p, data_dir, cfg) for p in paths}
    rel = {p: str(p.relative_to(data_dir)) for p in paths}
    rng = random.Random(int(cfg["split"]["seed"]))
    strategy = cfg["split"]["strategy"]
    train_ratio = float(cfg["split"]["train_ratio"])
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = float(cfg["split"]["test_ratio"])
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if strategy == "trial_random":
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n, n_train)
        n_val = min(n - n_train, n_val)
        splits["train"] = shuffled[:n_train]
        splits["val"] = shuffled[n_train : n_train + n_val]
        splits["test"] = shuffled[n_train + n_val :]
    elif strategy == "subject_heldout":
        grouped: dict[str, list[Path]] = defaultdict(list)
        group_base = bool(cfg["split"]["group_by_base_subject_id"])
        for p in paths:
            grouped[get_split_subject_key(metas[p][0], group_base)].append(p)
        keys = sorted(grouped)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n, n_train)
        n_val = min(n - n_train, n_val)
        key_splits = {
            "train": keys[:n_train],
            "val": keys[n_train : n_train + n_val],
            "test": keys[n_train + n_val :],
        }
        for name, split_keys in key_splits.items():
            for key in split_keys:
                splits[name].extend(grouped[key])
    else:
        raise ValueError(f"Unknown split.strategy: {strategy}")

    for name in splits:
        splits[name].sort(key=lambda p: rel[p])
    _write_split(split_dir / "pretrain_train.txt", [rel[p] for p in splits["train"]])
    _write_split(split_dir / "pretrain_val.txt", [rel[p] for p in splits["val"]])
    _write_split(split_dir / "pretrain_test.txt", [rel[p] for p in splits["test"]])
    summary = _summary(splits, metas, cfg)
    write_json(split_dir / "split_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--group_by_base_subject_id", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    for key in ("strategy", "seed", "train_ratio", "val_ratio", "test_ratio"):
        value = getattr(args, key)
        if value is not None:
            cfg["split"][key] = value
    if args.group_by_base_subject_id:
        cfg["split"]["group_by_base_subject_id"] = True
    summary = make_splits(cfg, out_dir=args.out_dir)
    print(summary)


if __name__ == "__main__":
    main()
