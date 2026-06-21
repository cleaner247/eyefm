from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from eyemae.build_fast_packed_dataset import build_fast_packed_dataset


def _write_ml_ready_view(root: Path, view: str, split: str, *, source_uid: str, subject: str, label: int) -> None:
    view_dir = root / view
    view_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([40, 60], dtype=np.int32)
    offsets = np.asarray([0, 40], dtype=np.int64)
    total = int(lengths.sum())
    x = np.zeros((total, 10), dtype=np.float32)
    x[:, 0] = np.arange(total, dtype=np.float32)
    x[:, 2] = 100.0
    x[:, 5] = 120.0
    x[:, 8] = 1.0
    y = np.zeros((total, 2), dtype=np.int8)
    y[5, 0] = 1
    y[10, 1] = 2
    np.savez(view_dir / f"{split}.npz", X_data=x, X_offsets=offsets, X_lengths=lengths, y_frame=y, y_trial=np.zeros((2, 1), dtype=np.int16))
    rows = []
    for i, direction in enumerate(["2", "4"]):
        rows.append(
            {
                "view": "matched_groups_full",
                "disease": view,
                "group": "患病" if label else "对照组",
                "subtype": view,
                "source_top": view,
                "source_dataset": "synthetic",
                "source_group": "患病" if label else "对照组",
                "source_subtype": view,
                "subject": subject,
                "info_id": subject,
                "info_id_with_suffix": f"{subject}_D",
                "task": "ProSaccade" if i == 0 else "AntiSaccade",
                "source_suffix": "D",
                "source_stem": f"{source_uid}_{i}",
                "source_csv": "",
                "relative_csv": f"{view}/synthetic/{subject}/trial_{i}.csv",
                "source_file_uid": source_uid,
                "original_trial_index": str(i),
                "direction": direction,
                "success": "1",
                "n_samples": str(int(lengths[i])),
                "pipeline": "",
                "trial_npz": "",
                "gaze_shape": "",
                "stimulus_shape": "",
                "left_final_keep": "True",
                "right_final_keep": "True",
                "left_final_reject": "False",
                "right_final_reject": "False",
                "left_highfreq_x": "",
                "left_highfreq_y": "",
                "right_highfreq_x": "",
                "right_highfreq_y": "",
                "left_blink_points": "1",
                "left_missing_points": "0",
                "right_blink_points": "0",
                "right_missing_points": "1",
                "ml_view": view,
                "ml_subject_id": f"{view}|synthetic|患病|{view}|{subject}",
                "health_label": str(label),
                "pd_disease_label": "-1",
                "split": split,
                "packed_trial_index": str(i),
                "X_offset": str(int(offsets[i])),
                "X_length": str(int(lengths[i])),
            }
        )
    with (view_dir / f"manifest_{split}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_remaining_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 50
    gaze = np.zeros((n, 8), dtype=np.float32)
    gaze[:, 2] = 200.0
    gaze[:, 5] = 210.0
    stimulus = np.zeros((n, 4), dtype=np.float32)
    stimulus[:, 2] = 1.0
    np.savez(path, gaze=gaze, stimulus=stimulus)


def test_build_fast_packed_dataset_dedup_and_indexes(tmp_path: Path) -> None:
    ml_ready = tmp_path / "ml_ready"
    remaining = tmp_path / "remaining" / "剩余对照组"
    _write_ml_ready_view(ml_ready, "AD", "train", source_uid="u1", subject="s001", label=1)
    _write_ml_ready_view(ml_ready, "AD匹配后", "train", source_uid="u1", subject="s001", label=1)
    _write_remaining_npz(remaining / "ctrl001" / "ProSaccade" / "src000001_D_trial00_dir2.npz")

    out = tmp_path / "fast"
    audit = build_fast_packed_dataset(
        ml_ready_root=ml_ready,
        remaining_control_root=remaining,
        out_dir=out,
        shard_target_gib=0.001,
        splits=["train"],
        seed=1,
    )

    assert audit["counters"]["downstream_manifest_rows"] == 4
    assert audit["counters"]["unique_downstream_trials"] == 2
    assert audit["counters"]["duplicate_downstream_rows"] == 2
    assert audit["counters"]["remaining_control_trials"] == 1
    assert (out / "README.md").exists()

    with (out / "trials.csv").open(newline="", encoding="utf-8") as f:
        trial_rows = list(csv.DictReader(f))
    assert len(trial_rows) == 3

    with (out / "downstream" / "AD" / "train.csv").open(newline="", encoding="utf-8") as f:
        ad_rows = list(csv.DictReader(f))
    with (out / "downstream" / "AD匹配后" / "train.csv").open(newline="", encoding="utf-8") as f:
        matched_rows = list(csv.DictReader(f))
    assert ad_rows[0]["global_trial_id"] == matched_rows[0]["global_trial_id"]
    assert ad_rows[1]["global_trial_id"] == matched_rows[1]["global_trial_id"]

    shard_dir = out / "shards" / trial_rows[0]["shard_id"]
    x = np.load(shard_dir / "X_data.npy", mmap_mode="r")
    y = np.load(shard_dir / "y_frame.npy", mmap_mode="r")
    start = int(trial_rows[0]["frame_offset"])
    end = start + int(trial_rows[0]["frame_length"])
    assert x[start:end].shape == (40, 10)
    assert y[start:end].shape == (40, 2)

    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "packed_mmap"
    assert manifest["num_trials"] == 3
