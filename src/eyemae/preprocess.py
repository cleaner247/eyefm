from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_json


def load_area_stats(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"global": {"median": 0.0, "mad": 1.0}, "subjects": {}}
    return read_json(p)


def validate_area_normalization_contract(
    stats: dict[str, Any],
    area_cfg: dict[str, Any],
) -> None:
    """Reject a stats/config transform mismatch before any training starts."""
    if not bool(area_cfg.get("require_contract", False)):
        return
    normalization = stats.get("source", {}).get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Area statistics have no normalization contract")
    expected_transform = "log1p" if bool(area_cfg.get("use_log1p", True)) else "identity"
    per_eye = bool(area_cfg.get("per_eye", False))
    checks = {
        "transform": expected_transform,
        "center": "per_subject_eye_median" if per_eye else "per_subject_median",
        "scale": "per_subject_eye_raw_mad" if per_eye else "per_subject_raw_mad",
        "mad_to_sigma": float(area_cfg.get("mad_scale", 1.4826)),
        "mad_floor": float(area_cfg.get("mad_floor", 0.0)),
        "clip": float(area_cfg.get("clip", 5.0)),
    }
    if per_eye:
        checks["fallback"] = "subject_pooled_then_global_eye_then_global_pooled"
    for key, expected in checks.items():
        observed = normalization.get(key)
        if isinstance(expected, float):
            matches = observed is not None and abs(float(observed) - expected) <= 1e-9
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"Area normalization mismatch for {key}: stats={observed!r}, config={expected!r}"
            )
    if stats.get("source", {}).get("invalid_eye_policy") != "trial_final_keep_then_frame_qc":
        raise ValueError("Area statistics do not enforce the valid-eye policy")
    if per_eye:
        global_by_eye = stats.get("global_by_eye")
        if not isinstance(global_by_eye, dict) or not {"left", "right"} <= set(global_by_eye):
            raise ValueError("Per-eye area statistics require global_by_eye.left/right")
        missing_subject_eyes = [
            subject_id
            for subject_id, subject_stats in stats.get("subjects", {}).items()
            if not {"left", "right"} <= set(subject_stats.get("eyes", {}))
        ]
        if missing_subject_eyes:
            raise ValueError(
                "Per-eye area statistics are missing left/right entries for subject(s): "
                + ", ".join(missing_subject_eyes[:5])
            )


def _subject_stats(
    subject_id: str,
    stats: dict[str, Any],
    eps: float,
    min_valid_frames: int = 0,
    eye_name: str | None = None,
) -> tuple[float, float]:
    global_stats = stats.get("global", {"median": 0.0, "mad": 1.0})
    subject_stats = stats.get("subjects", {}).get(subject_id)
    median = float(global_stats.get("median", 0.0))
    mad = float(global_stats.get("mad", 1.0))
    if eye_name is not None:
        if eye_name not in {"left", "right"}:
            raise ValueError(f"Unsupported eye_name: {eye_name}")
        eye_stats = (subject_stats or {}).get("eyes", {}).get(eye_name)
        if eye_stats is not None and int(eye_stats.get("num_valid_frames", 0)) >= min_valid_frames:
            median = float(eye_stats.get("median", median))
            eye_mad = float(eye_stats.get("mad", mad))
            mad = eye_mad if eye_mad >= eps else mad
            return median, mad if mad >= eps else 1.0
        # Sparse-eye fallback retains the subject calibration when enough
        # pooled valid frames exist, then falls back to the global eye scale.
        if subject_stats is not None and int(subject_stats.get("num_valid_frames", 0)) >= min_valid_frames:
            median = float(subject_stats.get("median", median))
            subject_mad = float(subject_stats.get("mad", mad))
            mad = subject_mad if subject_mad >= eps else mad
            return median, mad if mad >= eps else 1.0
        global_eye_stats = stats.get("global_by_eye", {}).get(eye_name)
        if global_eye_stats is not None:
            median = float(global_eye_stats.get("median", median))
            eye_mad = float(global_eye_stats.get("mad", mad))
            mad = eye_mad if eye_mad >= eps else mad
            return median, mad if mad >= eps else 1.0
    if subject_stats is not None and int(subject_stats.get("num_valid_frames", 0)) >= min_valid_frames:
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
    try:
        availability = parse_subject_eye_availability(subject_id)
    except ValueError:
        if "left_eye_available" not in trial and "right_eye_available" not in trial:
            raise
        availability = {"left_available": True, "right_available": True, "suffix": "explicit"}
    left_available = bool(trial.get("left_eye_available", availability["left_available"]))
    right_available = bool(trial.get("right_eye_available", availability["right_available"]))
    enforce_eye_availability = bool(cfg.get("data", {}).get("enforce_suffix_eye_availability", True))

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
    min_valid_frames = int(cfg["area"].get("min_subject_valid_frames", 0))
    per_eye = bool(cfg["area"].get("per_eye", False))
    mad_floor = float(cfg["area"].get("mad_floor", 0.0))
    mad_scale = float(cfg["area"].get("mad_scale", 1.4826))

    for e, source in enumerate((left, right)):
        eye_name = "left" if e == 0 else "right"
        median, mad = _subject_stats(
            subject_id,
            area_stats,
            eps,
            min_valid_frames,
            eye_name=eye_name if per_eye else None,
        )
        missing = source["label"] == label_missing
        if enforce_eye_availability and e == 0 and not left_available:
            missing[:] = True
        if enforce_eye_availability and e == 1 and not right_available:
            missing[:] = True
        blink = (source["label"] == label_blink) & (~missing)
        # A non-blink frame with a non-positive/non-finite pupil area is not a
        # usable eye measurement. Treat it as missing so it cannot enter the
        # encoder, attention context, reconstruction loss, or code targets.
        invalid_measurement = (~blink) & (
            (~np.isfinite(source["area"])) | (source["area"] <= 0)
        )
        missing |= invalid_measurement
        blink &= ~missing
        valid_area = (~missing) & (~blink) & (source["area"] > 0)

        x_norm = np.clip(source["x"], -x_clip, x_clip) / x_clip
        y_norm = np.clip(source["y"], -y_clip, y_clip) / y_clip
        if bool(cfg["area"].get("use_log1p", True)):
            u = np.log1p(np.maximum(source["area"], 0.0))
        else:
            u = source["area"].astype(np.float32)
        # Median/MAD must be computed after the configured transform (identity
        # or log1p).  Scaling raw MAD by 1.4826 makes it a robust standard-
        # deviation estimate under an approximately Gaussian transformed-area
        # distribution.  The floor only guards tracker-constant subjects
        # against exploding normalized noise.
        area_norm = (u - median) / (mad_scale * max(mad, mad_floor) + eps)
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


# ──────────────────────────────────────────────
# Outlier spike masking (tracker noise) — applied BEFORE training
# ──────────────────────────────────────────────

def _detect_spikes(v: np.ndarray, valid: np.ndarray, thr: float, window: int = 2) -> np.ndarray:
    """Flag isolated spikes: sample i jumps away from AND back to neighbors.

    A tracker spike is v[i] deviating strongly from v[i-1] and v[i+1], while the
    flanking samples v[i-window] and v[i+window] agree (jump-out-and-back).
    Real saccades are continuous multi-sample motion, so they do NOT satisfy the
    'jump out and back' condition (span stays large). Returns bool[T]."""
    T = len(v)
    flag = np.zeros(T, dtype=bool)
    if T < 2 * window + 1:
        return flag
    d = np.abs(np.diff(v))                      # d[j] = |v[j+1]-v[j]|
    # center i in [window, T-1-window]: dev_l=d[i-1], dev_r=d[i]
    jump = (d[window - 1: T - 1 - window] > thr) & (d[window: T - window] > thr)  # [T-2w]
    span = np.abs(v[2 * window:] - v[:T - 2 * window])                            # [T-2w]
    spk = jump & (span < thr)
    # require all samples in [i-window, i+window] valid
    cum = np.concatenate([[0], np.cumsum(valid.astype(np.int32))])
    win_valid = (cum[2 * window + 1:] - cum[:-(2 * window + 1)]) == (2 * window + 1)
    center_sel = np.zeros(T, dtype=bool)
    center_sel[window:T - window] = spk & win_valid
    return center_sel


def mask_outlier_spikes(content: np.ndarray, quality: np.ndarray,
                        thr_xy: float = 0.0236, thr_area: float = 0.2501,
                        window: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Mark isolated-spike frames (tracker noise) as MISSING so training loss
    ignores them. content: [T,2,4] (x,y,area_norm,label), quality: [T,2,1].
    Thresholds (normalized space) calibrated at p99.9 of |diff| on 400 trials.
    Flagged frames get quality=1 (missing) and label=2 (missing)."""
    content = np.asarray(content).copy()
    quality = np.asarray(quality).copy()
    T = content.shape[0]
    if T < 5:
        return content, quality
    for e in range(2):
        missing = quality[:, e, 0] > 0.5
        blink = content[:, e, 3] > 0.5
        valid = (~missing) & (~blink)
        flag = np.zeros(T, dtype=bool)
        for ch, thr in [(0, thr_xy), (1, thr_xy), (2, thr_area)]:
            flag |= _detect_spikes(content[:, e, ch], valid, thr, window)
        if flag.any():
            quality[flag, e, 0] = 1.0
            content[flag, e, 3] = 2.0  # label = missing
    return content, quality
