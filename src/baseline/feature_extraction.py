"""EyeFM paper-feature baseline: saccade feature extraction.

Extract 6 paper-replicable saccade features (× 4 task) = 24 dim per subject
from the v2 packed-mmap eye-movement data, following the 5/5 paper consensus
analysis in `docs/saccade_papers_metrics_review.md`.

Pipeline:

    trial row (csv) → shard npy (mmap) → binocular extraction (L+R mean) →
    aggregate per-subject per-task mean.

Reference:  docs/eyemae_baseline.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# 4 saccade task types (per-subject per-task aggregation, v2 layout)
SACCADE_TASKS = ("ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade")

# 6 paper-replicable features (4 paper-consensus + 2 paper-extra)
PAPER_FEATS_A = (
    "first_saccade_latency_ms",  # ≡ reaction time (5/5 paper)
    "primary_amp_deg",           # ≡ 1st-step saccade amplitude (5/5)
    "primary_peak_v",            # ≡ peak velocity (5/5)
    "primary_dur_ms",            # ≡ main sequence duration (5/5)
)
PAPER_FEATS_B = (
    "blink_ratio",               # ≡ blink rate (2/5)
    "missing_ratio",             # ≡ non-reactive rate (2/5)
)
PAPER_FEATS_A_B = PAPER_FEATS_A + PAPER_FEATS_B
N_PAPER_FEATS = len(PAPER_FEATS_A_B)  # 6

# 10 column layout (per-frame, packed_mmap X_data.npy)
COL_LEFT_X, COL_LEFT_Y, COL_LEFT_AREA = 0, 1, 2
COL_RIGHT_X, COL_RIGHT_Y, COL_RIGHT_AREA = 3, 4, 5
COL_STIM_X, COL_STIM_Y, COL_STIM_ON, COL_FIX_ON = 6, 7, 8, 9

# 2 column layout (per-frame, packed_mmap y_frame.npy)
COL_LEFT_QC, COL_RIGHT_QC = 0, 1

# Saccade detection hyper-params
SAMPLING_RATE_HZ = 1000
SPEED_THRESH_DEG_PER_S = 30.0
MIN_SACCADE_SAMPLES = 10  # ≥10 ms


@dataclass(frozen=True)
class TrialFeat:
    """Per-trial 10-dim saccade feature dict (NaN if undefined)."""

    first_saccade_latency_ms: float
    primary_amp_deg: float
    primary_peak_v: float
    primary_dur_ms: float
    total_amp_deg: float
    endpoint_error_deg: float
    displacement_max: float
    n_saccades: int
    missing_ratio: float
    blink_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "first_saccade_latency_ms": self.first_saccade_latency_ms,
            "primary_amp_deg": self.primary_amp_deg,
            "primary_peak_v": self.primary_peak_v,
            "primary_dur_ms": self.primary_dur_ms,
            "total_amp_deg": self.total_amp_deg,
            "endpoint_error_deg": self.endpoint_error_deg,
            "displacement_max": self.displacement_max,
            "n_saccades": self.n_saccades,
            "missing_ratio": self.missing_ratio,
            "blink_ratio": self.blink_ratio,
        }


def _empty_trial_feat() -> TrialFeat:
    nan = float("nan")
    return TrialFeat(nan, nan, nan, nan, nan, nan, nan, 0, nan, nan)


def _detect_saccades(xv: np.ndarray, yv: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Detect saccade intervals (start, end) using 30°/s speed threshold.

    Speed is computed via numpy gradient × sampling rate.
    """
    if len(xv) < 5:
        return []
    vx = np.gradient(xv) * SAMPLING_RATE_HZ
    vy = np.gradient(yv) * SAMPLING_RATE_HZ
    speed = np.sqrt(vx * vx + vy * vy)
    is_sac = speed > SPEED_THRESH_DEG_PER_S
    saccades: list[tuple[int, int]] = []
    in_sac = False
    s = 0
    for i, flag in enumerate(is_sac):
        if flag and not in_sac:
            s = i
            in_sac = True
        elif not flag and in_sac:
            if i - s >= min_samples:
                saccades.append((s, i))
            in_sac = False
    if in_sac and len(is_sac) - s >= min_samples:
        saccades.append((s, len(is_sac)))
    return saccades


def _extract_one_eye(x: np.ndarray, y: np.ndarray, qc: np.ndarray) -> TrialFeat:
    """Extract per-eye 10-dim saccade features; NaN if undefined."""
    n = len(x)
    if n == 0:
        return _empty_trial_feat()
    valid = (qc == 0) & np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())
    blink = int((qc == 1).sum())
    missing_ratio = 1.0 - n_valid / n
    blink_ratio = blink / n
    if n_valid < 5:
        return TrialFeat(float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"),
                         0, missing_ratio, blink_ratio)
    valid_idx = np.where(valid)[0]
    x_v = x[valid_idx]
    y_v = y[valid_idx]
    saccades = _detect_saccades(x_v, y_v, MIN_SACCADE_SAMPLES)
    if not saccades:
        return TrialFeat(float("nan"), float("nan"), float("nan"), float("nan"),
                         0.0, float("nan"), float("nan"),
                         0, missing_ratio, blink_ratio)
    s0, e0 = saccades[0]
    sxs, sys_ = x_v[s0:e0], y_v[s0:e0]
    amp = float(np.hypot(sxs[-1] - sxs[0], sys_[-1] - sys_[0]))
    if len(sxs) >= 2:
        vx = np.gradient(sxs) * SAMPLING_RATE_HZ
        vy = np.gradient(sys_) * SAMPLING_RATE_HZ
        speed = np.sqrt(vx * vx + vy * vy)
        peak_v = float(np.max(speed))
        dur_ms = float(e0 - s0)
    else:
        peak_v = float("nan")
        dur_ms = float("nan")
    latency_ms = float(valid_idx[s0])
    total_amp = 0.0
    for ss, ee in saccades:
        if ee - ss >= 2:
            dx = x_v[ee - 1] - x_v[ss]
            dy = y_v[ee - 1] - y_v[ss]
            total_amp += float(np.hypot(dx, dy))
    if len(x_v) >= 2:
        disp = np.hypot(x_v - x_v[0], y_v - y_v[0])
        displacement_max = float(np.max(disp))
    else:
        displacement_max = float("nan")
    return TrialFeat(
        first_saccade_latency_ms=latency_ms,
        primary_amp_deg=amp,
        primary_peak_v=peak_v,
        primary_dur_ms=dur_ms,
        total_amp_deg=total_amp,
        endpoint_error_deg=float("nan"),  # no stim in this layer
        displacement_max=displacement_max,
        n_saccades=len(saccades),
        missing_ratio=missing_ratio,
        blink_ratio=blink_ratio,
    )


def _binocular_mean(left: TrialFeat, right: TrialFeat) -> TrialFeat:
    """Average left & right eye per field (NaN-aware)."""
    d_l, d_r = left.as_dict(), right.as_dict()
    out: dict[str, float] = {}
    for k in d_l:
        l, r = d_l[k], d_r[k]
        if np.isfinite(l) and np.isfinite(r):
            out[k] = (l + r) / 2
        elif np.isfinite(l):
            out[k] = l
        elif np.isfinite(r):
            out[k] = r
        else:
            out[k] = float("nan")
    return TrialFeat(**out)


def extract_trial_feats(
    trial_x: np.ndarray,
    trial_y: np.ndarray,
) -> TrialFeat:
    """Extract 10-dim binocular mean saccade features from one trial.

    Args:
        trial_x: (T, 10) per-frame packed features (left/right xy+area, stim).
        trial_y: (T, 2) per-frame quality codes (left_qc, right_qc).

    Returns:
        TrialFeat dataclass with 10 fields; missing fields are NaN.
    """
    l_feat = _extract_one_eye(trial_x[:, COL_LEFT_X], trial_x[:, COL_LEFT_Y], trial_y[:, COL_LEFT_QC])
    r_feat = _extract_one_eye(trial_x[:, COL_RIGHT_X], trial_x[:, COL_RIGHT_Y], trial_y[:, COL_RIGHT_QC])
    return _binocular_mean(l_feat, r_feat)


def load_trial_npy(
    shard_dir: Path,
    frame_offset: int,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one trial window from packed_mmap shards.

    Returns:
        (trial_x, trial_y) of shapes (T, 10) and (T, 2), or None if out-of-range.
    """
    x_path = shard_dir / "X_data.npy"
    y_path = shard_dir / "y_frame.npy"
    if not x_path.exists() or not y_path.exists():
        return None
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    if frame_offset + frame_length > len(X):
        return None
    return np.asarray(X[frame_offset:frame_offset + frame_length]), np.asarray(Y[frame_offset:frame_offset + frame_length])


def extract_subject_task(
    rows: Iterable[dict],
    finetune_task: str,
    saccade_task: str,
    data_root: Path,
) -> dict[str, list[float]]:
    """Extract per-subject 24-dim feature vectors from a list of trial rows.

    Args:
        rows: trial-level rows from a v2 split csv (already filtered to one saccade task).
        finetune_task: top-level finetune task dir (e.g. ``detox_binary``). Used to locate shards.
        saccade_task: saccade paradigm (e.g. ``AntiSaccade``). Only used for logging; trials
            are expected to be pre-filtered to this task.
        data_root: v2 data root.

    For each trial: load shard window → extract 10-dim binocular → store.
    Then aggregate per-subject per-task as mean of each paper feat across trials.
    Returns dict[subject_id -> list of 6 paper feats (mean per saccade task)].
    Skips trials whose shard window cannot be loaded.
    """
    by_subj: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        shard_id = row["shard_id"]
        shard_dir_name = shard_id if shard_id.startswith("shard_") else f"shard_{shard_id}"
        shard_dir = data_root / "finetune" / finetune_task / "shards" / shard_dir_name
        try:
            offset = int(row["frame_offset"])
            length = int(row["frame_length"])
        except (KeyError, ValueError):
            continue
        loaded = load_trial_npy(shard_dir, offset, length)
        if loaded is None:
            continue
        trial_x, trial_y = loaded
        feat = extract_trial_feats(trial_x, trial_y)
        d = feat.as_dict()
        subj = row.get("ml_subject_id") or row.get("subject")
        if subj is None:
            continue
        for f in PAPER_FEATS_A_B:
            v = d.get(f, float("nan"))
            if np.isfinite(v):
                by_subj[subj][f].append(float(v))
    return {
        subj: [float(np.mean(by_subj[subj][f])) if by_subj[subj][f] else float("nan") for f in PAPER_FEATS_A_B]
        for subj in by_subj
    }


def load_split_csv(task: str, split: str, data_root: Path) -> list[dict]:
    """Load v2 split csv (rows = trials) for a given task & split."""
    csv_path = data_root / "finetune" / task / f"{split}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_label_map(task: str) -> dict[str, int]:
    """Return string label → int map per task."""
    if task == "pd_related_5class":
        return {"-1": 0, "0": 1, "1": 2, "2": 3, "3": 4}
    if task in ("detox_binary", "pd_binary"):
        return {"0": 0, "1": 1}
    raise ValueError(f"Unknown task: {task}")


def build_subject_feature_table(
    task: str,
    data_root: Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """Build per-(subject, saccade_task) feature matrix (X), label vector (y), subj ids, split tags.

    Each row of the output corresponds to one (subject, saccade_task) pair, so a
    subject who participated in all 4 saccade tasks contributes 4 rows. The
    6 paper features are aggregated per (subject, saccade_task) by mean across
    trials. Total dimensionality is therefore **6 per row**; the 4 saccade
    tasks × 6 features = 24-dim *coverage* is achieved across the 4 rows per
    subject. The downstream ML pipeline sees the 6-dim per row.

    Returns:
        X: (n_rows, 6) subject-level aggregated features.
        y: (n_rows,) integer labels (per row, taken from the first matching trial csv row).
        subj_ids: (n_rows,) ml_subject_id strings.
        split_tags: (n_rows,) split name per row.
    """
    label_map = get_label_map(task)
    X: list[list[float]] = []
    y: list[int] = []
    subj_ids: list[str] = []
    split_tags: list[str] = []
    for split in splits:
        rows = load_split_csv(task, split, data_root)
        if not rows:
            continue
        for saccade_task in SACCADE_TASKS:
            sub_rows = [r for r in rows if r.get("task") == saccade_task]
            if not sub_rows:
                continue
            per_subj = extract_subject_task(sub_rows, task, saccade_task, data_root)
            for subj, feat_vec in per_subj.items():
                X.append(feat_vec)
                label_str = "0"
                for r in sub_rows:
                    if r.get("ml_subject_id") == subj:
                        label_str = r.get("health_label") or r.get("label") or "0"
                        break
                y.append(int(label_map.get(label_str, 0)))
                subj_ids.append(subj)
                split_tags.append(split)
    return X, y, subj_ids, split_tags
