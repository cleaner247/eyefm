from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, split_path_for_name
from .data import (
    PackedTrialStore,
    filter_packed_rows_with_usable_eye,
    infer_subject_from_path,
    load_npz_trial,
    read_packed_index,
    read_split_file,
)
from .utils import read_json, setup_logging, write_json


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
    enforce_eye_availability = bool(cfg.get("data", {}).get("enforce_suffix_eye_availability", True))
    try:
        suffix_availability = parse_subject_eye_availability(str(trial["subject_id"]))
    except ValueError:
        suffix_availability = {"left_available": True, "right_available": True}
    # Per-trial final_keep is authoritative.  A D-suffix subject can still
    # have one rejected eye (or both rejected) in an individual trial.
    availability = {
        "left_available": bool(
            trial.get("left_eye_available", suffix_availability["left_available"])
        ),
        "right_available": bool(
            trial.get("right_eye_available", suffix_availability["right_available"])
        ),
    }
    out: dict[str, np.ndarray] = {}
    for name, offset, available in (
        ("left", 0, availability["left_available"]),
        ("right", 4, availability["right_available"]),
    ):
        area = eye[:, offset + 2]
        label = eye[:, offset + 3].astype(np.int64)
        missing = label == miss
        if enforce_eye_availability and not available:
            missing[:] = True
        blink = (label == blink_value) & (~missing)
        valid = (~missing) & (~blink) & np.isfinite(area) & (area > 0)
        values = area[valid].astype(np.float64)
        if bool(cfg["area"].get("use_log1p", True)):
            values = np.log1p(values)
        out[name] = values
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
    if cfg["data"].get("format") == "packed_mmap":
        return compute_packed_area_stats(cfg, split=split, out=out)
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
    global_eye_values: dict[str, list[float]] = {"left": [], "right": []}
    global_count = 0
    global_eye_count = {"left": 0, "right": 0}
    subjects_payload: dict[str, dict[str, Any]] = {}
    max_subject_items = int(max_subject) if max_subject else None
    max_global_items = int(max_global) if max_global else None
    for subject_index, (subject_id, subject_rels) in enumerate(sorted(grouped.items()), start=1):
        subject_values: list[float] = []
        subject_eye_values: dict[str, list[float]] = {"left": [], "right": []}
        subject_count = 0
        subject_eye_count = {"left": 0, "right": 0}
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
            for eye_name, eye_values in vals_by_eye.items():
                if eye_values.size == 0:
                    continue
                count = int(eye_values.size)
                subject_eye_count[eye_name] += count
                global_eye_count[eye_name] += count
                _reservoir_extend(
                    subject_eye_values[eye_name], eye_values, max_subject_items, rng
                )
                _reservoir_extend(
                    global_eye_values[eye_name], eye_values, max_global_items, rng
                )
        median, mad = _median_mad(subject_values, eps)
        subjects_payload[subject_id] = {
            "median": median,
            "mad": mad,
            "num_valid_frames": int(subject_count),
            "eyes": {
                eye_name: {
                    "median": _median_mad(subject_eye_values[eye_name], eps)[0],
                    "mad": _median_mad(subject_eye_values[eye_name], eps)[1],
                    "num_valid_frames": int(subject_eye_count[eye_name]),
                }
                for eye_name in ("left", "right")
            },
        }
        if subject_index % 250 == 0:
            LOGGER.info(
                "area stats progress: %s/%s subjects, %s valid frames",
                subject_index,
                len(grouped),
                global_count,
            )
    global_median, global_mad = _median_mad(global_values, eps)
    global_by_eye = {}
    for eye_name in ("left", "right"):
        eye_median, eye_mad = _median_mad(global_eye_values[eye_name], eps)
        global_by_eye[eye_name] = {
            "median": eye_median,
            "mad": eye_mad,
            "num_valid_frames": int(global_eye_count[eye_name]),
        }
    payload = {
        "global": {"median": global_median, "mad": global_mad, "num_valid_frames": int(global_count)},
        "global_by_eye": global_by_eye,
        "subjects": {},
    }
    for subject, stats in subjects_payload.items():
        mad = float(stats["mad"])
        payload["subjects"][subject] = {
            "median": float(stats["median"]),
            "mad": mad if mad >= eps else global_mad,
            "num_valid_frames": int(stats["num_valid_frames"]),
            "eyes": stats["eyes"],
        }
    write_json(out or cfg["area"]["stats_path"], payload)
    return payload


def compute_packed_area_stats(cfg: dict[str, Any], split: str = "train", out: str | Path | None = None) -> dict[str, Any]:
    data_dir = Path(cfg["data"]["data_dir"])
    index_file = split_path_for_name(cfg, split)
    raw_rows = read_packed_index(index_file)
    rows, excluded_both_invalid = filter_packed_rows_with_usable_eye(raw_rows)
    eps = float(cfg["area"]["eps"])
    rng = random.Random(int(cfg.get("split", {}).get("seed", 42)))
    max_subject = cfg["area"].get("max_frames_per_subject")
    max_global = cfg["area"].get("max_global_frames")
    max_subject_items = int(max_subject) if max_subject else None
    max_global_items = int(max_global) if max_global else None
    global_reference_path = cfg["area"].get("global_reference_path")
    global_reference_payload = read_json(global_reference_path) if global_reference_path else None
    global_reference = (
        global_reference_payload.get("global", {})
        if global_reference_payload is not None
        else None
    )
    global_eye_reference = (
        global_reference_payload.get("global_by_eye", {})
        if global_reference_payload is not None
        else None
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["ml_subject_id"]].append(row)
    store = PackedTrialStore(
        data_dir,
        max_open_shards_per_worker=int(cfg["data"].get("max_open_shards_per_worker", 16)),
        validate_offsets=bool(cfg["data"].get("validate_offsets", True)),
    )
    global_chunks: list[np.ndarray] = []
    global_eye_chunks: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    global_sample_per_subject = (
        max(1, int(np.ceil(max_global_items / max(len(grouped), 1))))
        if global_reference is None and max_global_items is not None
        else None
    )
    global_count = 0
    global_eye_count = {"left": 0, "right": 0}
    subjects_payload: dict[str, dict[str, Any]] = {}
    for subject_index, (subject_id, subject_rows) in enumerate(sorted(grouped.items()), start=1):
        subject_eye_values: dict[str, list[float]] = {"left": [], "right": []}
        # Full-subject mode keeps NumPy chunks instead of converting every one
        # of the ~2.2B valid eye-frames into a Python float and back again.
        subject_eye_chunks: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        subject_count = 0
        subject_eye_count = {"left": 0, "right": 0}
        for row in subject_rows:
            trial = store.read_trial(row)
            vals_by_eye = _valid_log_area(trial, cfg)
            for eye_name, eye_values in vals_by_eye.items():
                if eye_values.size == 0:
                    continue
                count = int(eye_values.size)
                subject_count += count
                subject_eye_count[eye_name] += count
                global_count += count
                global_eye_count[eye_name] += count
                if max_subject_items is None:
                    subject_eye_chunks[eye_name].append(eye_values)
                else:
                    _reservoir_extend(
                        subject_eye_values[eye_name], eye_values, max_subject_items, rng
                    )

        subject_eye_arrays: dict[str, np.ndarray] = {}
        subject_eye_payload: dict[str, dict[str, float | int]] = {}
        for eye_name in ("left", "right"):
            eye_array = (
                np.concatenate(subject_eye_chunks[eye_name])
                if subject_eye_chunks[eye_name]
                else np.asarray(subject_eye_values[eye_name], dtype=np.float64)
            )
            subject_eye_arrays[eye_name] = eye_array
            if eye_array.size:
                eye_median = float(np.median(eye_array))
                eye_mad = float(np.median(np.abs(eye_array - eye_median)))
                if eye_mad < eps:
                    eye_mad = 1.0
            else:
                eye_median, eye_mad = 0.0, 1.0
            subject_eye_payload[eye_name] = {
                "median": eye_median,
                "mad": eye_mad,
                "num_valid_frames": int(subject_eye_count[eye_name]),
            }

        nonempty_eye_arrays = [
            subject_eye_arrays[eye_name]
            for eye_name in ("left", "right")
            if subject_eye_arrays[eye_name].size
        ]
        subject_array = (
            np.concatenate(nonempty_eye_arrays)
            if nonempty_eye_arrays
            else np.asarray([], dtype=np.float64)
        )
        if subject_array.size:
            median = float(np.median(subject_array))
            mad = float(np.median(np.abs(subject_array - median)))
            if mad < eps:
                mad = 1.0
        else:
            median, mad = 0.0, 1.0

        if global_reference is None and subject_array.size:
            for sample_name, sample_array, output_chunks, seed_offset in (
                ("pooled", subject_array, global_chunks, 0),
                ("left", subject_eye_arrays["left"], global_eye_chunks["left"], 1_000_000),
                ("right", subject_eye_arrays["right"], global_eye_chunks["right"], 2_000_000),
            ):
                if sample_array.size == 0:
                    continue
                if global_sample_per_subject is not None and sample_array.size > global_sample_per_subject:
                    subject_rng = np.random.default_rng(
                        int(cfg.get("split", {}).get("seed", 42))
                        + seed_offset
                        + subject_index
                    )
                    sample_indices = subject_rng.choice(
                        sample_array.size, global_sample_per_subject, replace=False
                    )
                    output_chunks.append(sample_array[sample_indices])
                else:
                    output_chunks.append(sample_array)
        subjects_payload[subject_id] = {
            "median": median,
            "mad": mad,
            "num_valid_frames": int(subject_count),
            "eyes": subject_eye_payload,
        }
        if subject_index % 250 == 0:
            LOGGER.info(
                "packed area stats progress: %s/%s subjects, sampled valid frames=%s",
                subject_index,
                len(grouped),
                global_count,
            )
    global_fallback_sample_size = 0
    global_eye_fallback_sample_size = {"left": 0, "right": 0}
    if global_reference is None:
        global_array = (
            np.concatenate(global_chunks) if global_chunks else np.asarray([], dtype=np.float64)
        )
        if max_global_items is not None and global_array.size > max_global_items:
            global_array = global_array[:max_global_items]
        global_fallback_sample_size = int(global_array.size)
        if global_array.size:
            global_median = float(np.median(global_array))
            global_mad = float(np.median(np.abs(global_array - global_median)))
            if global_mad < eps:
                global_mad = 1.0
        else:
            global_median, global_mad = 0.0, 1.0
        global_by_eye: dict[str, dict[str, float | int]] = {}
        for eye_name in ("left", "right"):
            eye_global_array = (
                np.concatenate(global_eye_chunks[eye_name])
                if global_eye_chunks[eye_name]
                else np.asarray([], dtype=np.float64)
            )
            if max_global_items is not None and eye_global_array.size > max_global_items:
                eye_global_array = eye_global_array[:max_global_items]
            global_eye_fallback_sample_size[eye_name] = int(eye_global_array.size)
            if eye_global_array.size:
                eye_median = float(np.median(eye_global_array))
                eye_mad = float(np.median(np.abs(eye_global_array - eye_median)))
                if eye_mad < eps:
                    eye_mad = 1.0
            else:
                eye_median, eye_mad = global_median, global_mad
            global_by_eye[eye_name] = {
                "median": eye_median,
                "mad": eye_mad,
                "num_valid_frames": int(global_eye_count[eye_name]),
            }
    else:
        global_median = float(global_reference["median"])
        global_mad = float(global_reference["mad"])
        global_by_eye = {}
        for eye_name in ("left", "right"):
            eye_reference = (global_eye_reference or {}).get(eye_name, global_reference)
            global_by_eye[eye_name] = {
                "median": float(eye_reference["median"]),
                "mad": float(eye_reference["mad"]),
                "num_valid_frames": int(eye_reference.get("num_valid_frames", 0)),
            }
    payload = {
        "global": {"median": global_median, "mad": global_mad, "num_valid_frames": int(global_count)},
        "global_by_eye": global_by_eye,
        "subjects": {},
        "source": {
            "scope": str(cfg["area"].get("source_scope", "unspecified")),
            "normalization_scope": "per_subject_eye_median_mad",
            "format": "packed_mmap",
            "index": str(index_file),
            "num_subjects": len(grouped),
            "num_trials": len(rows),
            "num_trials_raw": len(raw_rows),
            "dropped_both_eyes_invalid": len(excluded_both_invalid),
            "global_reference_path": str(global_reference_path or ""),
            "global_fallback_sampling": (
                "equal_per_subject" if global_reference is None else "external_reference"
            ),
            "global_fallback_sample_size": global_fallback_sample_size,
            "global_eye_fallback_sample_size": global_eye_fallback_sample_size,
            "invalid_eye_policy": "trial_final_keep_then_frame_qc",
            "normalization": {
                "transform": "log1p" if bool(cfg["area"].get("use_log1p", True)) else "identity",
                "center": "per_subject_eye_median",
                "scale": "per_subject_eye_raw_mad",
                "fallback": "subject_pooled_then_global_eye_then_global_pooled",
                "mad_to_sigma": float(cfg["area"].get("mad_scale", 1.4826)),
                "mad_floor": float(cfg["area"].get("mad_floor", 0.0)),
                "clip": float(cfg["area"].get("clip", 5.0)),
            },
        },
    }
    for subject, stats in subjects_payload.items():
        mad = float(stats["mad"])
        payload["subjects"][subject] = {
            "median": float(stats["median"]),
            "mad": mad if mad >= eps else global_mad,
            "num_valid_frames": int(stats["num_valid_frames"]),
            "eyes": stats["eyes"],
        }
    write_json(out or cfg["area"]["stats_path"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    payload = compute_area_stats(cfg, args.split, args.out)
    print(payload["global"])


if __name__ == "__main__":
    main()
