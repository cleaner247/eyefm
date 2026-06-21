from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import task_name_to_id
from .manifest import parse_suffix_from_name


DEFAULT_ML_READY_ROOT = Path(
    "/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_ml_ready_subjectkey_20260619"
)
DEFAULT_REMAINING_CONTROL_ROOT = Path(
    "/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_fixed_pd_20260618/"
    "analysis_subject_unique/剩余对照组"
)
DEFAULT_OUT_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1")
SPLITS = ("train", "validation", "test")
TASK_NAMES = {"ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade"}
LABEL_VALID = 0
LABEL_BLINK = 1
LABEL_MISSING = 2


TRIAL_FIELDNAMES = [
    "global_trial_id",
    "source_kind",
    "dedupe_key",
    "shard_id",
    "local_trial_index",
    "frame_offset",
    "frame_length",
    "num_patches_20ms",
    "ml_subject_id",
    "subject",
    "trial_id",
    "task",
    "task_id",
    "source_top",
    "source_dataset",
    "source_group",
    "source_subtype",
    "source_suffix",
    "source_file_uid",
    "original_trial_index",
    "direction",
    "relative_source_path",
    "health_label",
    "pd_disease_label",
    "left_final_keep",
    "right_final_keep",
    "left_blink_points",
    "left_missing_points",
    "right_blink_points",
    "right_missing_points",
]

DOWNSTREAM_FIELDNAMES = [
    "global_trial_id",
    "view",
    "split",
    "shard_id",
    "local_trial_index",
    "frame_offset",
    "frame_length",
    "num_patches_20ms",
    "ml_subject_id",
    "subject",
    "trial_id",
    "task",
    "task_id",
    "health_label",
    "pd_disease_label",
    "source_top",
    "source_dataset",
    "source_group",
    "source_subtype",
    "source_file_uid",
    "original_trial_index",
    "direction",
    "relative_source_path",
]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            count += 1
    return count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_int(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _view_dirs(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [root / view for view in requested]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name != "00_run_info")


def _stable_dedupe_key(row: dict[str, str]) -> str:
    source_file_uid = row.get("source_file_uid", "").strip()
    task = row.get("task", "").strip()
    original_trial_index = row.get("original_trial_index", "").strip()
    direction = row.get("direction", "").strip()
    if source_file_uid:
        return f"uid:{source_file_uid}|trial:{original_trial_index}|dir:{direction}|task:{task}"
    parts = [
        row.get("source_top", ""),
        row.get("source_dataset", ""),
        row.get("source_group", ""),
        row.get("source_subtype", ""),
        row.get("subject", ""),
        row.get("relative_csv", ""),
        original_trial_index,
        direction,
        task,
    ]
    return "path:" + "|".join(str(part).strip() for part in parts)


def _remaining_dedupe_key(rel_path: str) -> str:
    return "remaining:" + rel_path


def _remaining_task_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part in TASK_NAMES:
            return part
    raise ValueError(f"Cannot infer task from remaining-control path: {path}")


def _source_suffix_from_path(path: Path) -> str:
    suffix = parse_suffix_from_name(path.name)
    if suffix != "unknown":
        return suffix
    match = re.search(r"_(D|L|R)(?:_|$)", path.stem)
    return match.group(1) if match else "D"


def _packed_arrays_to_trial(x_data: np.ndarray, y_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_data, dtype=np.float32)
    y = np.asarray(y_frame, dtype=np.int8)
    if x.ndim != 2 or x.shape[1] != 10:
        raise ValueError(f"X_data trial slice must have shape [T, 10], got {x.shape}")
    if y.ndim != 2 or y.shape[1] != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y_frame trial slice must have shape [T, 2], got {y.shape} for T={x.shape[0]}")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, y


def _gaze_stimulus_to_packed(gaze: np.ndarray, stimulus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gaze, dtype=np.float32)
    s = np.asarray(stimulus, dtype=np.float32)
    if g.ndim != 2 or g.shape[1] != 8:
        raise ValueError(f"gaze must have shape [T, 8], got {g.shape}")
    if s.ndim != 2 or s.shape[1] != 4 or s.shape[0] != g.shape[0]:
        raise ValueError(f"stimulus must have shape [T, 4], got {s.shape} for T={g.shape[0]}")

    x = np.empty((g.shape[0], 10), dtype=np.float32)
    x[:, 0] = g[:, 0]
    x[:, 1] = g[:, 1]
    x[:, 2] = g[:, 2]
    x[:, 3] = g[:, 3]
    x[:, 4] = g[:, 4]
    x[:, 5] = g[:, 5]
    x[:, 6] = s[:, 0]
    x[:, 7] = s[:, 1]
    x[:, 8] = s[:, 2]
    x[:, 9] = s[:, 3]

    y = np.empty((g.shape[0], 2), dtype=np.int8)
    left_label = g[:, 6].copy()
    right_label = g[:, 7].copy()
    left_bad = ~np.isfinite(g[:, 0:3]).all(axis=1) | ~np.isfinite(left_label)
    right_bad = ~np.isfinite(g[:, 3:6]).all(axis=1) | ~np.isfinite(right_label)
    left_label[left_bad] = LABEL_MISSING
    right_label[right_bad] = LABEL_MISSING
    x[left_bad, 0:3] = 0.0
    x[right_bad, 3:6] = 0.0
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y[:, 0] = np.clip(left_label, LABEL_VALID, LABEL_MISSING).astype(np.int8)
    y[:, 1] = np.clip(right_label, LABEL_VALID, LABEL_MISSING).astype(np.int8)
    return x, y


@dataclass
class ShardRef:
    shard_id: str
    local_trial_index: int
    frame_offset: int
    frame_length: int


class ShardWriter:
    def __init__(self, out_dir: Path, target_bytes: int) -> None:
        self.out_dir = out_dir
        self.target_bytes = int(target_bytes)
        self.shard_index = 0
        self.frame_offset = 0
        self.x_parts: list[np.ndarray] = []
        self.y_parts: list[np.ndarray] = []
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.rows: list[dict[str, Any]] = []
        self.shard_summaries: list[dict[str, Any]] = []

    @property
    def current_bytes(self) -> int:
        return sum(part.nbytes for part in self.x_parts) + sum(part.nbytes for part in self.y_parts)

    @property
    def current_trial_count(self) -> int:
        return len(self.lengths)

    def add(self, x: np.ndarray, y: np.ndarray, row: dict[str, Any]) -> ShardRef:
        if self.current_trial_count > 0 and self.current_bytes + x.nbytes + y.nbytes > self.target_bytes:
            self.flush()
        shard_id = f"shard_{self.shard_index:06d}"
        local_index = self.current_trial_count
        offset = self.frame_offset
        length = int(x.shape[0])
        self.x_parts.append(x)
        self.y_parts.append(y)
        self.offsets.append(offset)
        self.lengths.append(length)
        self.rows.append(
            {
                **row,
                "shard_id": shard_id,
                "local_trial_index": local_index,
                "frame_offset": offset,
                "frame_length": length,
                "num_patches_20ms": length // 20,
            }
        )
        self.frame_offset += length
        return ShardRef(shard_id=shard_id, local_trial_index=local_index, frame_offset=offset, frame_length=length)

    def flush(self) -> None:
        if not self.x_parts:
            return
        shard_id = f"shard_{self.shard_index:06d}"
        shard_dir = self.out_dir / "shards" / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        x = np.concatenate(self.x_parts, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(self.y_parts, axis=0).astype(np.int8, copy=False)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        lengths = np.asarray(self.lengths, dtype=np.int32)
        np.save(shard_dir / "X_data.npy", x)
        np.save(shard_dir / "y_frame.npy", y)
        np.save(shard_dir / "X_offsets.npy", offsets)
        np.save(shard_dir / "X_lengths.npy", lengths)
        _write_csv(shard_dir / "trial_index.csv", self.rows, TRIAL_FIELDNAMES)
        self.shard_summaries.append(
            {
                "shard_id": shard_id,
                "num_trials": int(lengths.shape[0]),
                "num_frames": int(x.shape[0]),
                "x_data_shape": list(x.shape),
                "y_frame_shape": list(y.shape),
                "bytes": int(x.nbytes + y.nbytes + offsets.nbytes + lengths.nbytes),
            }
        )
        self.shard_index += 1
        self.frame_offset = 0
        self.x_parts.clear()
        self.y_parts.clear()
        self.offsets.clear()
        self.lengths.clear()
        self.rows.clear()


def _downstream_storage_row(row: dict[str, str], global_trial_id: str, dedupe_key: str) -> dict[str, Any]:
    task = row["task"].strip()
    length = _safe_int(row.get("X_length"))
    return {
        "global_trial_id": global_trial_id,
        "source_kind": "downstream_view",
        "dedupe_key": dedupe_key,
        "ml_subject_id": row.get("ml_subject_id", "").strip(),
        "subject": row.get("subject", "").strip(),
        "trial_id": dedupe_key,
        "task": task,
        "task_id": task_name_to_id(task),
        "source_top": row.get("source_top", "").strip(),
        "source_dataset": row.get("source_dataset", "").strip(),
        "source_group": row.get("source_group", "").strip(),
        "source_subtype": row.get("source_subtype", "").strip(),
        "source_suffix": row.get("source_suffix", "").strip(),
        "source_file_uid": row.get("source_file_uid", "").strip(),
        "original_trial_index": row.get("original_trial_index", "").strip(),
        "direction": row.get("direction", "").strip(),
        "relative_source_path": row.get("relative_csv", "").strip(),
        "health_label": row.get("health_label", "").strip(),
        "pd_disease_label": row.get("pd_disease_label", "").strip(),
        "left_final_keep": row.get("left_final_keep", "").strip(),
        "right_final_keep": row.get("right_final_keep", "").strip(),
        "left_blink_points": row.get("left_blink_points", "").strip(),
        "left_missing_points": row.get("left_missing_points", "").strip(),
        "right_blink_points": row.get("right_blink_points", "").strip(),
        "right_missing_points": row.get("right_missing_points", "").strip(),
        "frame_length": length,
        "num_patches_20ms": length // 20,
    }


def _remaining_storage_row(path: Path, root: Path, global_trial_id: str, dedupe_key: str) -> dict[str, Any]:
    rel_path = path.relative_to(root).as_posix()
    task = _remaining_task_from_path(path)
    subject = path.parent.parent.name
    suffix = _source_suffix_from_path(path)
    ml_subject_id = f"剩余对照组|剩余对照组|对照组|剩余对照组|{subject}"
    return {
        "global_trial_id": global_trial_id,
        "source_kind": "remaining_control",
        "dedupe_key": dedupe_key,
        "ml_subject_id": ml_subject_id,
        "subject": subject,
        "trial_id": f"remaining:{rel_path}",
        "task": task,
        "task_id": task_name_to_id(task),
        "source_top": "剩余对照组",
        "source_dataset": "剩余对照组",
        "source_group": "对照组",
        "source_subtype": "剩余对照组",
        "source_suffix": suffix,
        "source_file_uid": "",
        "original_trial_index": "",
        "direction": "",
        "relative_source_path": rel_path,
        "health_label": "",
        "pd_disease_label": "",
        "left_final_keep": str(suffix != "R"),
        "right_final_keep": str(suffix != "L"),
        "left_blink_points": "",
        "left_missing_points": "",
        "right_blink_points": "",
        "right_missing_points": "",
    }


def _with_ref(row: dict[str, Any], ref: ShardRef) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "shard_id": ref.shard_id,
            "local_trial_index": ref.local_trial_index,
            "frame_offset": ref.frame_offset,
            "frame_length": ref.frame_length,
            "num_patches_20ms": ref.frame_length // 20,
        }
    )
    return out


def _downstream_index_row(row: dict[str, str], view: str, split: str, storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_trial_id": storage["global_trial_id"],
        "view": view,
        "split": split,
        "shard_id": storage["shard_id"],
        "local_trial_index": storage["local_trial_index"],
        "frame_offset": storage["frame_offset"],
        "frame_length": storage["frame_length"],
        "num_patches_20ms": storage["num_patches_20ms"],
        "ml_subject_id": row.get("ml_subject_id", "").strip(),
        "subject": row.get("subject", "").strip(),
        "trial_id": storage["trial_id"],
        "task": row.get("task", "").strip(),
        "task_id": task_name_to_id(row.get("task", "").strip()),
        "health_label": row.get("health_label", "").strip(),
        "pd_disease_label": row.get("pd_disease_label", "").strip(),
        "source_top": row.get("source_top", "").strip(),
        "source_dataset": row.get("source_dataset", "").strip(),
        "source_group": row.get("source_group", "").strip(),
        "source_subtype": row.get("source_subtype", "").strip(),
        "source_file_uid": row.get("source_file_uid", "").strip(),
        "original_trial_index": row.get("original_trial_index", "").strip(),
        "direction": row.get("direction", "").strip(),
        "relative_source_path": row.get("relative_csv", "").strip(),
    }


def _write_pretrain_indices(out_dir: Path, rows: list[dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> dict[str, Any]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ml_subject_id"])].append(row)
    subjects = sorted(grouped)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n, n_train)
    n_val = min(n - n_train, n_val)
    split_subjects = {
        "train": set(subjects[:n_train]),
        "validation": set(subjects[n_train : n_train + n_val]),
        "test": set(subjects[n_train + n_val :]),
    }
    split_rows = {
        split: [row for subject in sorted(subjects_) for row in sorted(grouped[subject], key=lambda r: str(r["global_trial_id"]))]
        for split, subjects_ in split_subjects.items()
    }
    pretrain_dir = out_dir / "pretrain"
    _write_csv(pretrain_dir / "pretrain_all_unique.csv", rows, TRIAL_FIELDNAMES)
    for split, split_row_list in split_rows.items():
        _write_csv(pretrain_dir / f"pretrain_{split}.csv", split_row_list, TRIAL_FIELDNAMES)
    overlaps = {
        "train_validation": len(split_subjects["train"] & split_subjects["validation"]),
        "train_test": len(split_subjects["train"] & split_subjects["test"]),
        "validation_test": len(split_subjects["validation"] & split_subjects["test"]),
    }
    summary = {
        "strategy": "subject_heldout",
        "seed": seed,
        "subject_key": "ml_subject_id",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "num_subjects": len(subjects),
        "num_trials": len(rows),
        "splits": {},
        "subject_overlap_counts": overlaps,
        "no_subject_overlap": all(value == 0 for value in overlaps.values()),
    }
    for split, split_row_list in split_rows.items():
        summary["splits"][split] = {
            "subjects": len(split_subjects[split]),
            "trials": len(split_row_list),
            "frames": int(sum(int(row["frame_length"]) for row in split_row_list)),
            "task_counts": dict(Counter(str(row["task_id"]) for row in split_row_list)),
            "source_kind_counts": dict(Counter(str(row["source_kind"]) for row in split_row_list)),
            "source_top_counts": dict(Counter(str(row["source_top"]) for row in split_row_list)),
        }
    _write_json(pretrain_dir / "pretrain_split_summary.json", summary)
    return summary


def _write_subjects(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ml_subject_id"])].append(row)
    fieldnames = [
        "ml_subject_id",
        "source_top_values",
        "source_dataset_values",
        "source_group_values",
        "source_subtype_values",
        "subject_values",
        "num_trials",
        "num_frames",
        "available_tasks",
    ]
    subject_rows = []
    for subject, subject_rows_raw in sorted(grouped.items()):
        subject_rows.append(
            {
                "ml_subject_id": subject,
                "source_top_values": ";".join(sorted({str(row["source_top"]) for row in subject_rows_raw})),
                "source_dataset_values": ";".join(sorted({str(row["source_dataset"]) for row in subject_rows_raw})),
                "source_group_values": ";".join(sorted({str(row["source_group"]) for row in subject_rows_raw})),
                "source_subtype_values": ";".join(sorted({str(row["source_subtype"]) for row in subject_rows_raw})),
                "subject_values": ";".join(sorted({str(row["subject"]) for row in subject_rows_raw})),
                "num_trials": len(subject_rows_raw),
                "num_frames": int(sum(int(row["frame_length"]) for row in subject_rows_raw)),
                "available_tasks": ";".join(sorted({str(row["task"]) for row in subject_rows_raw})),
            }
        )
    _write_csv(out_dir / "subjects.csv", subject_rows, fieldnames)
    return {"num_subjects": len(subject_rows)}


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_readme(out_dir: Path, manifest: dict[str, Any], audit: dict[str, Any] | None = None) -> None:
    pretrain_table = ""
    downstream_table = ""
    if audit:
        pretrain_rows = []
        for split, info in audit.get("pretrain", {}).get("splits", {}).items():
            pretrain_rows.append(
                [
                    split,
                    f"{int(info.get('trials', 0)):,}",
                    f"{int(info.get('subjects', 0)):,}",
                    f"{int(info.get('frames', 0)):,}",
                ]
            )
        if pretrain_rows:
            pretrain_table = "\n\n" + _markdown_table(["split", "trials", "subjects", "frames"], pretrain_rows)
        downstream_rows = []
        for view, view_info in sorted(audit.get("downstream", {}).items()):
            split_info = view_info.get("splits", {})
            for split, info in split_info.items():
                labels = info.get("health_label_counts", {})
                downstream_rows.append(
                    [
                        view,
                        split,
                        f"{int(info.get('rows', 0)):,}",
                        f"{int(info.get('subjects', 0)):,}",
                        f"{int(info.get('frames', 0)):,}",
                        f"0={labels.get('0', 0)}, 1={labels.get('1', 0)}",
                    ]
                )
        if downstream_rows:
            downstream_table = "\n\n" + _markdown_table(
                ["view", "split", "rows", "subjects", "frames", "health labels"],
                downstream_rows,
            )
    text = f"""# EyeMAE Fast Packed Dataset v1

This dataset is organized for high-throughput EyeMAE pretraining and downstream fine-tuning.
It replaces per-trial NPZ files with mmap-friendly shard arrays plus lightweight CSV indexes.

## Data Sources

- Downstream packed views: `{manifest["ml_ready_root"]}`
- Extra pretrain control-only trials: `{manifest["remaining_control_root"]}`

## Layout

```text
dataset_manifest.json
columns.json
label_maps.json
audit_summary.json
trials.csv
subjects.csv
README.md

shards/
  shard_000000/
    X_data.npy
    y_frame.npy
    X_offsets.npy
    X_lengths.npy
    trial_index.csv
  ...

pretrain/
  pretrain_all_unique.csv
  pretrain_train.csv
  pretrain_validation.csv
  pretrain_test.csv
  pretrain_split_summary.json

downstream/
  <view>/
    train.csv
    validation.csv
    test.csv
    split_summary.json
```

The shard arrays store the actual frame data. The CSV files are indexes that point into those
arrays. A trial is identified by `shard_id`, `local_trial_index`, `frame_offset`, and
`frame_length`.

## Shard Arrays

`X_data.npy` has shape `(total_frames_in_shard, 10)` and dtype `float32`.

| column | name |
| --- | --- |
| 0 | left_x |
| 1 | left_y |
| 2 | left_s |
| 3 | right_x |
| 4 | right_y |
| 5 | right_s |
| 6 | stimulus_x |
| 7 | stimulus_y |
| 8 | stimulus_on |
| 9 | cross_on |

`y_frame.npy` has shape `(total_frames_in_shard, 2)` and dtype `int8`.

| column | name | values |
| --- | --- | --- |
| 0 | left_qc_label | 0=VALID, 1=BLINK, 2=MISSING |
| 1 | right_qc_label | 0=VALID, 1=BLINK, 2=MISSING |

`X_offsets.npy` and `X_lengths.npy` locate each local trial in the shard.

## Reading One Trial

```python
import csv
from pathlib import Path
import numpy as np

root = Path("/path/to/eyemae_fast_dataset_v1")
with (root / "pretrain" / "pretrain_train.csv").open(newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))

shard_dir = root / "shards" / row["shard_id"]
X = np.load(shard_dir / "X_data.npy", mmap_mode="r")
Y = np.load(shard_dir / "y_frame.npy", mmap_mode="r")

start = int(row["frame_offset"])
end = start + int(row["frame_length"])
x_trial = X[start:end]
y_frame_trial = Y[start:end]
```

Mapping to the current EyeMAE internal trial dict:

```text
eye[:, 0] = X[:, 0]      left_x
eye[:, 1] = X[:, 1]      left_y
eye[:, 2] = X[:, 2]      left_s
eye[:, 3] = Y[:, 0]      left_qc_label
eye[:, 4] = X[:, 3]      right_x
eye[:, 5] = X[:, 4]      right_y
eye[:, 6] = X[:, 5]      right_s
eye[:, 7] = Y[:, 1]      right_qc_label
fix_on    = X[:, 9]      cross_on
stim      = X[:, [8,6,7]]  stimulus_on, stimulus_x, stimulus_y
```

## Pretraining

Use:

```text
pretrain/pretrain_train.csv
pretrain/pretrain_validation.csv
pretrain/pretrain_test.csv
```

The pretraining indexes are subject-heldout by `ml_subject_id`. `pretrain_all_unique.csv`
contains all unique pretraining trials.

## Downstream Fine-Tuning

Use:

```text
downstream/<view>/train.csv
downstream/<view>/validation.csv
downstream/<view>/test.csv
```

Each downstream view keeps its own health labels and subject-heldout split. The actual frame
data are shared through `shards/`; matched and overall views do not duplicate frame arrays.

`PD相关` includes `pd_disease_label`:

```text
-1 control/NA
 0 帕金森病
 1 震颤
 2 特发性震颤
 3 运动障碍
```

## Manifest Summary

```json
{json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)}
```

## Pretrain Split Summary
{pretrain_table}

## Downstream View Summary
{downstream_table}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def build_fast_packed_dataset(
    *,
    ml_ready_root: Path,
    remaining_control_root: Path,
    out_dir: Path,
    shard_target_gib: float = 2.0,
    views: list[str] | None = None,
    splits: list[str] | None = None,
    max_downstream_trials_per_split: int | None = None,
    max_remaining_control_trials: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_names = tuple(splits or SPLITS)
    invalid_splits = set(split_names) - set(SPLITS)
    if invalid_splits:
        raise ValueError(f"Unknown splits: {sorted(invalid_splits)}")

    writer = ShardWriter(out_dir, int(shard_target_gib * (1024**3)))
    storage_by_key: dict[str, dict[str, Any]] = {}
    trial_rows: list[dict[str, Any]] = []
    downstream_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    view_summaries: dict[str, Any] = {}

    for view_dir in _view_dirs(ml_ready_root, views):
        if not view_dir.exists():
            raise FileNotFoundError(view_dir)
        view = view_dir.name
        view_summaries[view] = {"splits": {}}
        for split in split_names:
            manifest_path = view_dir / f"manifest_{split}.csv"
            npz_path = view_dir / f"{split}.npz"
            if not manifest_path.exists() or not npz_path.exists():
                continue
            print(f"[packed] reading view={view} split={split}", flush=True)
            with manifest_path.open(newline="", encoding="utf-8") as f:
                manifest_rows = list(csv.DictReader(f))
            if max_downstream_trials_per_split is not None:
                manifest_rows = manifest_rows[: int(max_downstream_trials_per_split)]
            counters["downstream_manifest_rows"] += len(manifest_rows)
            split_new_unique = 0
            split_duplicates = 0
            with np.load(npz_path, allow_pickle=False) as z:
                x_all = z["X_data"]
                y_all = z["y_frame"]
                offsets = z["X_offsets"]
                lengths = z["X_lengths"]
                for row in manifest_rows:
                    dedupe_key = _stable_dedupe_key(row)
                    storage = storage_by_key.get(dedupe_key)
                    if storage is None:
                        global_trial_id = f"gt_{len(trial_rows):09d}"
                        packed_i = _safe_int(row.get("packed_trial_index"))
                        start = int(offsets[packed_i])
                        length = int(lengths[packed_i])
                        x, y = _packed_arrays_to_trial(x_all[start : start + length], y_all[start : start + length])
                        storage_base = _downstream_storage_row(row, global_trial_id, dedupe_key)
                        ref = writer.add(x, y, storage_base)
                        storage = _with_ref(storage_base, ref)
                        storage_by_key[dedupe_key] = storage
                        trial_rows.append(storage)
                        counters["unique_downstream_trials"] += 1
                        split_new_unique += 1
                    else:
                        counters["duplicate_downstream_rows"] += 1
                        split_duplicates += 1
                    downstream_rows[(view, split)].append(_downstream_index_row(row, view, split, storage))
            writer.flush()
            view_summaries[view]["splits"][split] = {
                "manifest_rows": len(manifest_rows),
                "new_unique_trials": split_new_unique,
                "duplicate_rows": split_duplicates,
                "index_rows": len(downstream_rows[(view, split)]),
            }
            print(
                f"[packed] done view={view} split={split} rows={len(manifest_rows)} "
                f"new={split_new_unique} dup={split_duplicates}",
                flush=True,
            )

    remaining_paths = sorted(remaining_control_root.rglob("*.npz"))
    if max_remaining_control_trials is not None:
        remaining_paths = remaining_paths[: int(max_remaining_control_trials)]
    print(f"[packed] reading remaining control trials={len(remaining_paths)}", flush=True)
    for i, path in enumerate(remaining_paths, start=1):
        rel_path = path.relative_to(remaining_control_root).as_posix()
        dedupe_key = _remaining_dedupe_key(rel_path)
        if dedupe_key in storage_by_key:
            counters["duplicate_remaining_control_rows"] += 1
            continue
        global_trial_id = f"gt_{len(trial_rows):09d}"
        with np.load(path, allow_pickle=False) as z:
            x, y = _gaze_stimulus_to_packed(z["gaze"], z["stimulus"])
        storage_base = _remaining_storage_row(path, remaining_control_root, global_trial_id, dedupe_key)
        ref = writer.add(x, y, storage_base)
        storage = _with_ref(storage_base, ref)
        storage_by_key[dedupe_key] = storage
        trial_rows.append(storage)
        counters["remaining_control_trials"] += 1
        if i % 50000 == 0:
            print(f"[packed] remaining control processed={i}", flush=True)
    writer.flush()
    print("[packed] writing indexes and summaries", flush=True)

    _write_csv(out_dir / "trials.csv", trial_rows, TRIAL_FIELDNAMES)
    subjects_summary = _write_subjects(out_dir, trial_rows)

    downstream_summary: dict[str, Any] = {}
    for (view, split), rows in sorted(downstream_rows.items()):
        _write_csv(out_dir / "downstream" / view / f"{split}.csv", rows, DOWNSTREAM_FIELDNAMES)
        view_summary = downstream_summary.setdefault(view, {"splits": {}})
        subjects = {str(row["ml_subject_id"]) for row in rows}
        view_summary["splits"][split] = {
            "rows": len(rows),
            "subjects": len(subjects),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "health_label_counts": dict(Counter(str(row["health_label"]) for row in rows)),
            "task_counts": dict(Counter(str(row["task_id"]) for row in rows)),
        }
    for view, view_summary in downstream_summary.items():
        split_subjects = {
            split: {str(row["ml_subject_id"]) for row in downstream_rows[(view, split)]}
            for split in split_names
            if (view, split) in downstream_rows
        }
        overlaps = {}
        for left in split_subjects:
            for right in split_subjects:
                if left >= right:
                    continue
                overlaps[f"{left}_{right}"] = len(split_subjects[left] & split_subjects[right])
        view_summary["subject_overlap_counts"] = overlaps
        view_summary["no_subject_overlap"] = all(value == 0 for value in overlaps.values())
        _write_json(out_dir / "downstream" / view / "split_summary.json", view_summary)

    pretrain_summary = _write_pretrain_indices(out_dir, trial_rows, seed, train_ratio, val_ratio)

    columns = {
        "X_data": [
            "left_x",
            "left_y",
            "left_s",
            "right_x",
            "right_y",
            "right_s",
            "stimulus_x",
            "stimulus_y",
            "stimulus_on",
            "cross_on",
        ],
        "y_frame": ["left_qc_label", "right_qc_label"],
        "qc_label_map": {"0": "VALID", "1": "BLINK", "2": "MISSING"},
        "internal_mapping": {
            "eye": [
                "left_x",
                "left_y",
                "left_s",
                "left_qc_label",
                "right_x",
                "right_y",
                "right_s",
                "right_qc_label",
            ],
            "fix_on": "cross_on",
            "stim": ["stimulus_on", "stimulus_x", "stimulus_y"],
        },
    }
    label_maps = {
        "health_label": {"0": "healthy/control", "1": "disease/patient"},
        "pd_disease_label": {
            "-1": "not_applicable_or_control",
            "0": "帕金森病",
            "1": "震颤",
            "2": "特发性震颤",
            "3": "运动障碍",
        },
    }
    manifest = {
        "format": "packed_mmap",
        "version": 1,
        "ml_ready_root": str(ml_ready_root),
        "remaining_control_root": str(remaining_control_root),
        "shard_target_gib": shard_target_gib,
        "views": sorted(downstream_summary),
        "splits": list(split_names),
        "num_shards": len(writer.shard_summaries),
        "num_trials": len(trial_rows),
        "num_subjects": subjects_summary["num_subjects"],
        "num_frames": int(sum(int(row["frame_length"]) for row in trial_rows)),
    }
    audit = {
        "status": "ok",
        "counters": dict(counters),
        "manifest": manifest,
        "shards": writer.shard_summaries,
        "view_conversion": view_summaries,
        "downstream": downstream_summary,
        "pretrain": pretrain_summary,
    }
    _write_json(out_dir / "columns.json", columns)
    _write_json(out_dir / "label_maps.json", label_maps)
    _write_json(out_dir / "dataset_manifest.json", manifest)
    _write_json(out_dir / "audit_summary.json", audit)
    _write_readme(out_dir, manifest, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EyeMAE packed mmap shard dataset.")
    parser.add_argument("--ml-ready-root", type=Path, default=DEFAULT_ML_READY_ROOT)
    parser.add_argument("--remaining-control-root", type=Path, default=DEFAULT_REMAINING_CONTROL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-target-gib", type=float, default=2.0)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", choices=SPLITS, default=None)
    parser.add_argument("--max-downstream-trials-per-split", type=int, default=None)
    parser.add_argument("--max-remaining-control-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()
    total = args.train_ratio + args.val_ratio
    if total >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1.0")
    audit = build_fast_packed_dataset(
        ml_ready_root=args.ml_ready_root,
        remaining_control_root=args.remaining_control_root,
        out_dir=args.out_dir,
        shard_target_gib=args.shard_target_gib,
        views=args.views,
        splits=args.splits,
        max_downstream_trials_per_split=args.max_downstream_trials_per_split,
        max_remaining_control_trials=args.max_remaining_control_trials,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(audit["manifest"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
