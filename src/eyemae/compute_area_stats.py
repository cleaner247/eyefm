from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data import infer_subject_from_path, load_npz_trial, read_split_file
from .utils import setup_logging, write_json


LOGGER = logging.getLogger(__name__)


def _reservoir_extend(bucket: list[float], values: np.ndarray, max_items: int | None, rng: random.Random) -> None:
    if max_items is None or max_items <= 0:
        bucket.extend(values.astype(float).tolist())
        return
    for value in values.astype(float).tolist():
        if len(bucket) < max_items:
            bucket.append(value)
        else:
            j = rng.randint(0, len(bucket))
            if j < max_items:
                bucket[j] = value


def _valid_log_area(trial: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    from .data import parse_subject_eye_availability

    eye = trial["eye"]
    miss = int(cfg["label"]["missing_value"])
    blink_value = int(cfg["label"]["blink_value"])
    availability = parse_subject_eye_availability(str(trial["subject_id"]))
    out: dict[str, np.ndarray] = {}
    for name, offset, available in (
        ("left", 0, availability["left_available"]),
        ("right", 4, availability["right_available"]),
    ):
        area = eye[:, offset + 2]
        label = eye[:, offset + 3].astype(np.int64)
        missing = label == miss
        if not available:
            missing[:] = True
        blink = (label == blink_value) & (~missing)
        valid = (~missing) & (~blink) & (area > 0)
        out[name] = np.log1p(area[valid]).astype(np.float64)
    return out


def _median_mad(values: list[float], eps: float) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad < eps:
        mad = 1.0
    return median, mad


def _group_rels_by_subject(rels: list[str], data_dir: Path, cfg: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    if cfg["data"].get("metadata_from_path", False):
        for rel in rels:
            grouped[infer_subject_from_path(data_dir / rel, data_dir)].append(rel)
        return dict(grouped)
    for rel in rels:
        trial = load_npz_trial(data_dir / rel, data_dir, cfg)
        grouped[str(trial["subject_id"])].append(rel)
    return dict(grouped)


def compute_area_stats(cfg: dict[str, Any], split: str = "pretrain_train", out: str | Path | None = None) -> dict[str, Any]:
    split_key = f"{split}_split"
    split_file = Path(cfg["data"][split_key])
    data_dir = Path(cfg["data"]["data_dir"])
    rels = read_split_file(split_file)
    eps = float(cfg["area"]["eps"])
    rng = random.Random(int(cfg["split"].get("seed", 42)))
    max_subject = cfg["area"].get("max_frames_per_subject")
    max_global = cfg["area"].get("max_global_frames")
    grouped = _group_rels_by_subject(rels, data_dir, cfg)
    global_values: list[float] = []
    global_count = 0
    subjects_payload: dict[str, dict[str, float | int]] = {}
    max_subject_items = int(max_subject) if max_subject else None
    max_global_items = int(max_global) if max_global else None
    for subject_index, (subject_id, subject_rels) in enumerate(sorted(grouped.items()), start=1):
        subject_values: list[float] = []
        subject_count = 0
        for rel in subject_rels:
            trial = load_npz_trial(data_dir / rel, data_dir, cfg)
            vals_by_eye = _valid_log_area(trial, cfg)
            vals = (
                np.concatenate([v for v in vals_by_eye.values() if v.size > 0])
                if any(v.size for v in vals_by_eye.values())
                else np.asarray([], dtype=np.float64)
            )
            if vals.size == 0:
                continue
            subject_count += int(vals.size)
            global_count += int(vals.size)
            _reservoir_extend(subject_values, vals, max_subject_items, rng)
            _reservoir_extend(global_values, vals, max_global_items, rng)
        median, mad = _median_mad(subject_values, eps)
        subjects_payload[subject_id] = {
            "median": median,
            "mad": mad,
            "num_valid_frames": int(subject_count),
        }
        if subject_index % 250 == 0:
            LOGGER.info(
                "area stats progress: %s/%s subjects, %s valid frames",
                subject_index,
                len(grouped),
                global_count,
            )
    global_median, global_mad = _median_mad(global_values, eps)
    payload = {
        "global": {"median": global_median, "mad": global_mad, "num_valid_frames": int(global_count)},
        "subjects": {},
    }
    for subject, stats in subjects_payload.items():
        mad = float(stats["mad"])
        payload["subjects"][subject] = {
            "median": float(stats["median"]),
            "mad": mad if mad >= eps else global_mad,
            "num_valid_frames": int(stats["num_valid_frames"]),
        }
    write_json(out or cfg["area"]["stats_path"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="pretrain_train")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    payload = compute_area_stats(cfg, args.split, args.out)
    print(payload["global"])


if __name__ == "__main__":
    main()
