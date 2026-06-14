from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_json


def load_area_stats(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"global": {"median": 0.0, "mad": 1.0, "num_valid_frames": 0}, "subjects": {}}
    return read_json(p)


def _subject_stats(subject_id: str, stats: dict[str, Any], eps: float) -> tuple[float, float]:
    global_stats = stats.get("global", {"median": 0.0, "mad": 1.0})
    subject_stats = stats.get("subjects", {}).get(subject_id)
    median = float(global_stats.get("median", 0.0))
    mad = float(global_stats.get("mad", 1.0))
    if subject_stats is not None:
        median = float(subject_stats.get("median", median))
        subject_mad = float(subject_stats.get("mad", mad))
        mad = subject_mad if subject_mad >= eps else mad
    if mad < eps:
        mad = 1.0
    return median, mad


def preprocess_trial(trial: dict[str, Any], cfg: dict[str, Any], area_stats: dict[str, Any]) -> dict[str, Any]:
    from .data import parse_subject_eye_availability

    eye = np.asarray(trial["eye"], dtype=np.float32)
    fix_on = np.asarray(trial["fix_on"], dtype=np.float32)
    stim_raw = np.asarray(trial["stim"], dtype=np.float32)
    subject_id = str(trial["subject_id"])
    availability = parse_subject_eye_availability(subject_id)

    x_clip = float(cfg["normalization"]["x_clip_deg"])
    y_clip = float(cfg["normalization"]["y_clip_deg"])
    area_clip = float(cfg["area"]["clip"])
    eps = float(cfg["area"]["eps"])
    label_missing = int(cfg["label"]["missing_value"])
    label_blink = int(cfg["label"]["blink_value"])

    left = {
        "x": eye[:, 0],
        "y": eye[:, 1],
        "area": eye[:, 2],
        "label": eye[:, 3].astype(np.int64),
    }
    right = {
        "x": eye[:, 4],
        "y": eye[:, 5],
        "area": eye[:, 6],
        "label": eye[:, 7].astype(np.int64),
    }

    content = np.zeros((eye.shape[0], 2, 4), dtype=np.float32)
    quality = np.zeros((eye.shape[0], 2, 1), dtype=np.float32)
    median, mad = _subject_stats(subject_id, area_stats, eps)

    for e, source in enumerate((left, right)):
        missing = source["label"] == label_missing
        if e == 0 and not availability["left_available"]:
            missing[:] = True
        if e == 1 and not availability["right_available"]:
            missing[:] = True
        blink = (source["label"] == label_blink) & (~missing)
        valid_area = (~missing) & (~blink) & (source["area"] > 0)

        x_norm = np.clip(source["x"], -x_clip, x_clip) / x_clip
        y_norm = np.clip(source["y"], -y_clip, y_clip) / y_clip
        if bool(cfg["area"].get("use_log1p", True)):
            u = np.log1p(np.maximum(source["area"], 0.0))
        else:
            u = source["area"].astype(np.float32)
        area_norm = (u - median) / (1.4826 * mad + eps)
        area_norm = np.clip(area_norm, -area_clip, area_clip)
        zero_mask = missing | blink
        x_norm = x_norm.astype(np.float32)
        y_norm = y_norm.astype(np.float32)
        area_norm = area_norm.astype(np.float32)
        x_norm[zero_mask] = 0.0
        y_norm[zero_mask] = 0.0
        area_norm[zero_mask | (~valid_area)] = 0.0
        content[:, e, 0] = x_norm
        content[:, e, 1] = y_norm
        content[:, e, 2] = area_norm
        content[:, e, 3] = blink.astype(np.float32)
        quality[:, e, 0] = missing.astype(np.float32)

    stim_on = stim_raw[:, 0].astype(np.float32)
    stim_x = stim_raw[:, 1].astype(np.float32)
    stim_y = stim_raw[:, 2].astype(np.float32)
    stim_absent = stim_on <= 0.0
    stim_x = np.clip(stim_x, -x_clip, x_clip) / x_clip
    stim_y = np.clip(stim_y, -y_clip, y_clip) / y_clip
    stim_x[stim_absent] = 0.0
    stim_y[stim_absent] = 0.0
    stim = np.stack([fix_on, stim_on, stim_x, stim_y], axis=1).astype(np.float32)

    return {
        "content": content,
        "quality": quality,
        "stim": stim,
        "task_id": int(np.asarray(trial["task_id"]).item()),
        "subject_id": subject_id,
        "trial_id": str(trial["trial_id"]),
        "path": trial.get("path", ""),
    }
