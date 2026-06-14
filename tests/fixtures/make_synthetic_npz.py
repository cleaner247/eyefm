#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TASK_NAMES = ["ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade"]


def make_synthetic_dataset(out_dir: str | Path, num_trials: int = 128, seed: int = 42) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    suffixes = ["D", "L", "R"]
    for i in range(num_trials):
        task_id = i % 4
        suffix = suffixes[i % 3]
        subject_id = f"subj{i % 16:03d}{suffix}"
        task_name = TASK_NAMES[task_id]
        t = int(rng.integers(12, 1800)) if i % 17 == 0 else int(rng.integers(160, 1200))
        time = np.arange(t, dtype=np.float32) / 1000.0
        base_x = np.sin(time * (2.0 + task_id)) * (3 + task_id)
        base_y = np.cos(time * (1.5 + task_id)) * (2 + task_id)
        left_x = base_x + rng.normal(0, 0.05, size=t)
        left_y = base_y + rng.normal(0, 0.05, size=t)
        right_x = base_x + 0.2 + rng.normal(0, 0.05, size=t)
        right_y = base_y - 0.1 + rng.normal(0, 0.05, size=t)
        left_area = 2400 + 120 * np.sin(time * 3.0) + rng.normal(0, 20, size=t)
        right_area = 2300 + 100 * np.cos(time * 2.0) + rng.normal(0, 20, size=t)
        left_label = np.zeros(t, dtype=np.float32)
        right_label = np.zeros(t, dtype=np.float32)
        if t > 80:
            left_label[20:30] = 1
            right_label[45:55] = 1
            left_label[70:80] = 2
        if suffix == "L":
            right_label[:] = 2
        if suffix == "R":
            left_label[:] = 2
        eye = np.stack(
            [left_x, left_y, left_area, left_label, right_x, right_y, right_area, right_label],
            axis=1,
        ).astype(np.float32)
        fix_on = np.zeros(t, dtype=np.float32)
        fix_on[: min(t, 100)] = 1.0
        stim = np.zeros((t, 3), dtype=np.float32)
        if t > 100:
            stim_start = min(100, t - 1)
            stim_end = min(t, stim_start + 120)
            stim[stim_start:stim_end, 0] = 1.0
            stim[stim_start:stim_end, 1] = [10, -10, 0, 8][task_id]
            stim[stim_start:stim_end, 2] = [0, 0, 10, -8][task_id]
        path = out / subject_id / task_name / f"trial_{i:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            eye=eye,
            task_id=np.array(task_id, dtype=np.int64),
            fix_on=fix_on,
            stim=stim,
            subject_id=np.array(subject_id),
            trial_id=np.array(f"trial_{i:04d}"),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_synthetic_dataset(args.out_dir, args.num_trials, args.seed)


if __name__ == "__main__":
    main()
