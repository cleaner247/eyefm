from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eyemae.build_fast_packed_dataset import (  # noqa: E402
    ShardWriter,
    _gaze_stimulus_to_packed,
    _with_ref,
    _write_csv,
    _write_json,
)
from eyemae.data import task_name_to_id  # noqa: E402


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/disk_sde/data-260606/extracted/cd_speed4_hard_blink_fixed_pd_20260618/"
    "matched_groups_full_BACKUP_intermediate_20260618"
)
DEFAULT_OUT_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2")
DEFAULT_V1_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1")
TASKS = ("ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade")
SPLITS = ("train", "validation", "test")
RANDOM_SPLIT_RATIOS = {"train": 0.64, "validation": 0.16, "test": 0.20}
PRETRAIN_SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}
EDUCATION_CODES = {"XX", "CZ", "GZ", "ZZ", "DZ", "BK", "SS", "BS", "WM"}
IDENTITY_EDUCATION_OVERRIDES = {
    # Source metadata is inconsistent across task files, but these are the same
    # raw subject and should not be split into multiple downstream identities.
    ("FangDeXiu", "77"): "MIXED",
    ("LuXingQiong", "61"): "MIXED",
    ("雷妮莎", "23"): "MIXED",
}
PD_DATASET_TO_LABEL = {
    "帕金森病": 0,
    "震颤": 1,
    "特发性震颤": 2,
    "运动障碍": 3,
}
PD_CONTROL_PRIORITY = {
    "帕金森匹配后": 0,
    "震颤匹配后": 1,
    "特发性震颤匹配后": 2,
    "运动障碍匹配后": 3,
}
PD_MATCHED_DATASETS = set(PD_CONTROL_PRIORITY)
PRETRAIN_INDEX_FIELDNAMES = [
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
    "raw_identity",
    "identity_name",
    "identity_age",
    "identity_education",
    "source_stem",
]
DOWNSTREAM_INDEX_FIELDNAMES = [
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
    "source_suffix",
    "source_file_uid",
    "original_trial_index",
    "direction",
    "relative_source_path",
    "left_final_keep",
    "right_final_keep",
    "left_blink_points",
    "left_missing_points",
    "right_blink_points",
    "right_missing_points",
    "raw_identity",
    "identity_name",
    "identity_age",
    "identity_education",
    "source_stem",
]


@dataclass(frozen=True)
class Identity:
    key: str
    name: str
    age: str
    education: str


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def iter_manifest(source_root: Path) -> Iterable[dict[str, str]]:
    manifest = source_root / "manifest_all.csv"
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def norm_token(value: str) -> str:
    text = str(value or "").strip().lstrip("_").strip()
    return re.sub(r"\s+", "", text)


def source_stem(row: dict[str, str]) -> str:
    value = row.get("source_stem", "").strip()
    if value:
        return value
    rel = row.get("relative_csv", "").strip()
    return Path(rel).stem


def parse_identity(row: dict[str, str]) -> Identity:
    stem = source_stem(row)
    parts = stem.split("_")
    task_index = next((i for i, part in enumerate(parts) if part in TASKS), -1)
    if task_index < 0:
        name = norm_token(row.get("subject", ""))
        return Identity(key=f"{name}|age:unknown|edu:unknown", name=name, age="unknown", education="unknown")

    before = parts[:task_index]
    education_index = -1
    for i in range(2, len(before) - 2):
        education = norm_token(before[i]).upper()
        age_token = before[i + 2].strip()
        if education in EDUCATION_CODES and age_token.isdigit() and 0 < int(age_token) < 130:
            education_index = i
            break

    if education_index >= 0:
        education = norm_token(before[education_index]).upper()
        age = before[education_index + 2].strip()
        sex_index = education_index - 1
        name_tokens = before[2:max(2, sex_index)]
        name = norm_token("_".join(name_tokens)) or norm_token(row.get("subject", ""))
    else:
        age_index = -1
        for i in range(len(before) - 1, -1, -1):
            token = before[i].strip()
            if token.isdigit() and 0 < int(token) < 130:
                age_index = i
                break
        if age_index >= 0:
            age = before[age_index].strip()
            education = norm_token(before[age_index - 2]).upper() if age_index >= 2 else "unknown"
            name_tokens = before[2 : max(2, age_index - 3)]
            name = norm_token("_".join(name_tokens)) or norm_token(row.get("subject", ""))
        else:
            age = "unknown"
            education = "unknown"
            name = norm_token(row.get("subject", ""))

    if not name:
        name = norm_token(row.get("subject", "unknown")) or "unknown"
    education = IDENTITY_EDUCATION_OVERRIDES.get((name, age), education)
    return Identity(key=f"{name}|age:{age}|edu:{education}", name=name, age=age, education=education)


def source_subject_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            row.get("source_top", "").strip(),
            row.get("source_dataset", "").strip(),
            row.get("source_group", "").strip(),
            row.get("source_subtype", "").strip(),
            row.get("subject", "").strip(),
        ]
    )


def stable_trial_key(row: dict[str, str]) -> str:
    source_file_uid = row.get("source_file_uid", "").strip()
    original_trial_index = row.get("original_trial_index", "").strip()
    direction = row.get("direction", "").strip()
    task = row.get("task", "").strip()
    if source_file_uid:
        return f"uid:{source_file_uid}|trial:{original_trial_index}|dir:{direction}|task:{task}"
    return "|".join(
        [
            row.get("source_top", ""),
            row.get("source_dataset", ""),
            row.get("source_group", ""),
            row.get("source_subtype", ""),
            row.get("subject", ""),
            source_stem(row),
            original_trial_index,
            direction,
            task,
        ]
    )


def is_control_row(row: dict[str, str]) -> bool:
    if row.get("source_top", "").strip() == "对照组数据汇总":
        return True
    return row.get("source_group", "").strip() in {"对照组", "健康"}


def is_pd_matched_experimental(row: dict[str, str]) -> bool:
    return (
        row.get("source_top", "").strip() == "PD相关"
        and row.get("source_dataset", "").strip() in PD_MATCHED_DATASETS
        and row.get("source_group", "").strip() == "实验组"
    )


def is_ad_matched_experimental(row: dict[str, str]) -> bool:
    return (
        row.get("source_top", "").strip() == "AD"
        and row.get("source_dataset", "").strip() == "匹配后"
        and row.get("source_group", "").strip() == "实验组"
    )


def is_mci_matched(row: dict[str, str]) -> bool:
    return row.get("source_top", "").strip() == "MCI" and row.get("source_dataset", "").strip() == "匹配后"


def is_ad_gly_matched_control(row: dict[str, str], identity: Identity) -> bool:
    return (
        row.get("source_top", "").strip() == "AD"
        and row.get("source_dataset", "").strip() == "匹配后"
        and row.get("source_group", "").strip() == "对照组"
        and identity.name == "GaoLianYing"
    )


def control_priority(row: dict[str, str]) -> tuple[int, str]:
    if row.get("source_top", "").strip() == "对照组数据汇总":
        return (0, source_subject_key(row))
    dataset = row.get("source_dataset", "").strip()
    if "匹配后" not in dataset:
        return (1, source_subject_key(row))
    return (2, source_subject_key(row))


def pd_control_priority(row: dict[str, str]) -> tuple[int, str]:
    dataset = row.get("source_dataset", "").strip()
    return (PD_CONTROL_PRIORITY.get(dataset, 99), source_subject_key(row))


def resolve_npz_path(source_root: Path, row: dict[str, str]) -> Path:
    raw_path = Path(row["trial_npz"])
    parts = raw_path.parts
    if "matched_groups_full" in parts:
        index = parts.index("matched_groups_full")
        rel = Path(*parts[index + 1 :])
        candidate = source_root / rel
    else:
        candidate = raw_path
    if not candidate.exists():
        raise FileNotFoundError(f"Missing trial npz for {row.get('relative_csv')}: {candidate}")
    return candidate


def split_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: n * ratios[split] for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    remaining = n - sum(counts.values())
    for split, _value in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    if n >= len(SPLITS):
        for split in SPLITS:
            if counts[split] == 0:
                donor = max(SPLITS, key=lambda item: counts[item])
                counts[donor] -= 1
                counts[split] += 1
    return counts


def stratified_subject_split(
    subject_labels: dict[str, str],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for subject, label in subject_labels.items():
        by_label[str(label)].append(subject)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label, subjects in sorted(by_label.items()):
        shuffled = sorted(subjects)
        rng.shuffle(shuffled)
        counts = split_counts(len(shuffled), ratios)
        split_names = (
            ["train"] * counts["train"]
            + ["validation"] * counts["validation"]
            + ["test"] * (len(shuffled) - counts["train"] - counts["validation"])
        )
        for subject, split in zip(shuffled, split_names):
            assignment[subject] = split
    return assignment


def random_subject_split(subjects: Iterable[str], *, seed: int, ratios: dict[str, float]) -> dict[str, str]:
    shuffled = sorted(set(subjects))
    random.Random(seed).shuffle(shuffled)
    counts = split_counts(len(shuffled), ratios)
    split_names = (
        ["train"] * counts["train"]
        + ["validation"] * counts["validation"]
        + ["test"] * (len(shuffled) - counts["train"] - counts["validation"])
    )
    return {subject: split for subject, split in zip(shuffled, split_names)}


def v1_identity_from_row(row: dict[str, str]) -> str:
    stem = Path(row.get("relative_source_path", "")).stem
    synthetic = {"source_stem": stem, "subject": row.get("subject", "")}
    return parse_identity(synthetic).key


def load_preserved_v1_split(v1_dir: Path, view: str) -> dict[str, str]:
    identity_to_split: dict[str, str] = {}
    for split in SPLITS:
        rows, _fields = read_csv(v1_dir / "downstream" / view / f"{split}.csv")
        for row in rows:
            identity = v1_identity_from_row(row)
            previous = identity_to_split.get(identity)
            if previous is not None and previous != split:
                raise ValueError(f"v1 {view} identity appears in multiple splits: {identity}: {previous} vs {split}")
            identity_to_split[identity] = split
    return identity_to_split


def base_row(
    row: dict[str, str],
    *,
    identity: Identity,
    global_trial_id: str,
    source_kind: str,
    view: str = "",
    split: str = "",
    health_label: str = "",
    pd_disease_label: str = "",
) -> dict[str, Any]:
    task = row.get("task", "").strip()
    source_suffix = row.get("source_suffix", "").strip() or "D"
    frame_length = int(float(row.get("n_samples", "0") or 0))
    return {
        "global_trial_id": global_trial_id,
        "view": view,
        "split": split,
        "source_kind": source_kind,
        "dedupe_key": stable_trial_key(row),
        "shard_id": "",
        "local_trial_index": "",
        "frame_offset": "",
        "frame_length": frame_length,
        "num_patches_20ms": frame_length // 20,
        "ml_subject_id": identity.key,
        "subject": identity.name,
        "trial_id": stable_trial_key(row),
        "task": task,
        "task_id": task_name_to_id(task),
        "source_top": row.get("source_top", "").strip(),
        "source_dataset": row.get("source_dataset", "").strip(),
        "source_group": row.get("source_group", "").strip(),
        "source_subtype": row.get("source_subtype", "").strip(),
        "source_suffix": source_suffix,
        "source_file_uid": row.get("source_file_uid", "").strip(),
        "original_trial_index": row.get("original_trial_index", "").strip(),
        "direction": row.get("direction", "").strip(),
        "relative_source_path": row.get("relative_csv", "").strip(),
        "health_label": health_label,
        "pd_disease_label": pd_disease_label,
        "left_final_keep": row.get("left_final_keep", "").strip(),
        "right_final_keep": row.get("right_final_keep", "").strip(),
        "left_blink_points": row.get("left_blink_points", "").strip(),
        "left_missing_points": row.get("left_missing_points", "").strip(),
        "right_blink_points": row.get("right_blink_points", "").strip(),
        "right_missing_points": row.get("right_missing_points", "").strip(),
        "raw_identity": identity.key,
        "identity_name": identity.name,
        "identity_age": identity.age,
        "identity_education": identity.education,
        "source_stem": source_stem(row),
        "_npz_path": "",
    }


def load_packed_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return _gaze_stimulus_to_packed(data["gaze"], data["stimulus"])


def pack_index_rows(
    out_root: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    target_bytes: int,
    split_filenames: dict[str, str],
    all_filename: str | None,
    fieldnames: list[str],
) -> dict[str, Any]:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_root, target_bytes=target_bytes)
    packed_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    all_packed: list[dict[str, Any]] = []
    total_rows = sum(len(rows) for rows in rows_by_split.values())
    done_rows = 0
    for split in SPLITS:
        for row in rows_by_split[split]:
            x, y = load_packed_arrays(Path(row["_npz_path"]))
            storage_row = {key: value for key, value in row.items() if not key.startswith("_")}
            ref = writer.add(x, y, storage_row)
            packed = _with_ref(storage_row, ref)
            packed_by_split[split].append(packed)
            all_packed.append(packed)
            done_rows += 1
            if done_rows == 1 or done_rows % 10000 == 0 or done_rows == total_rows:
                print(f"[pack] {out_root.name}: {done_rows}/{total_rows} trials", file=sys.stderr, flush=True)
    writer.flush()
    for split, filename in split_filenames.items():
        _write_csv(out_root / filename, packed_by_split[split], fieldnames)
    if all_filename is not None:
        _write_csv(out_root / all_filename, all_packed, fieldnames)
    return {
        "num_trials": len(all_packed),
        "num_frames": int(sum(int(row["frame_length"]) for row in all_packed)),
        "num_shards": len(writer.shard_summaries),
        "shards": writer.shard_summaries,
        "rows_by_split": packed_by_split,
        "all_rows": all_packed,
    }


def summarize_rows(rows_by_split: dict[str, list[dict[str, Any]]], *, label_type: str) -> dict[str, Any]:
    subject_sets = {split: {str(row["ml_subject_id"]) for row in rows} for split, rows in rows_by_split.items()}
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    summary: dict[str, Any] = {
        "no_subject_overlap": all(value == 0 for value in overlaps.values()),
        "subject_overlap_counts": overlaps,
        "label_type": label_type,
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        subject_labels: dict[str, str] = {}
        subject_classes: dict[str, str] = {}
        for row in rows:
            subject_labels.setdefault(str(row["ml_subject_id"]), str(row.get("health_label", "")))
            if label_type == "multiclass":
                if str(row.get("health_label")) == "0":
                    cls = "0"
                else:
                    cls = str(int(row.get("pd_disease_label", "0")) + 1)
                subject_classes.setdefault(str(row["ml_subject_id"]), cls)
        payload = {
            "rows": len(rows),
            "subjects": len(subject_sets[split]),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "task_counts": dict(sorted(Counter(str(row["task_id"]) for row in rows).items())),
            "source_dataset_counts": dict(sorted(Counter(str(row.get("source_dataset", "")) for row in rows).items())),
            "source_group_counts": dict(sorted(Counter(str(row.get("source_group", "")) for row in rows).items())),
        }
        if label_type in {"binary", "multiclass"}:
            payload["health_label_trial_counts"] = dict(sorted(Counter(str(row.get("health_label", "")) for row in rows).items()))
            payload["health_label_subject_counts"] = dict(sorted(Counter(subject_labels.values()).items()))
        if label_type == "multiclass":
            payload["class_trial_counts"] = dict(
                sorted(
                    Counter(
                        "0" if str(row.get("health_label")) == "0" else str(int(row.get("pd_disease_label", "0")) + 1)
                        for row in rows
                    ).items()
                )
            )
            payload["class_subject_counts"] = dict(sorted(Counter(subject_classes.values()).items()))
        summary["splits"][split] = payload
    return summary


def write_subjects(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ml_subject_id"])].append(row)
    fields = [
        "ml_subject_id",
        "identity_name",
        "identity_age",
        "identity_education",
        "num_trials",
        "num_frames",
        "available_tasks",
        "health_labels",
        "pd_disease_labels",
        "source_top_values",
        "source_dataset_values",
        "source_group_values",
    ]
    subject_rows = []
    for subject, subject_rows_raw in sorted(grouped.items()):
        first = subject_rows_raw[0]
        subject_rows.append(
            {
                "ml_subject_id": subject,
                "identity_name": first.get("identity_name", ""),
                "identity_age": first.get("identity_age", ""),
                "identity_education": first.get("identity_education", ""),
                "num_trials": len(subject_rows_raw),
                "num_frames": int(sum(int(row.get("frame_length", 0)) for row in subject_rows_raw)),
                "available_tasks": ";".join(sorted({str(row.get("task", "")) for row in subject_rows_raw})),
                "health_labels": ";".join(sorted({str(row.get("health_label", "")) for row in subject_rows_raw})),
                "pd_disease_labels": ";".join(sorted({str(row.get("pd_disease_label", "")) for row in subject_rows_raw})),
                "source_top_values": ";".join(sorted({str(row.get("source_top", "")) for row in subject_rows_raw})),
                "source_dataset_values": ";".join(sorted({str(row.get("source_dataset", "")) for row in subject_rows_raw})),
                "source_group_values": ";".join(sorted({str(row.get("source_group", "")) for row in subject_rows_raw})),
            }
        )
    _write_csv(path, subject_rows, fields)


def write_common_metadata(root: Path, *, source_root: Path, task_name: str, audit: dict[str, Any]) -> None:
    columns = {
        "X_data": [
            "left_x",
            "left_y",
            "left_area",
            "right_x",
            "right_y",
            "right_area",
            "stim_x",
            "stim_y",
            "stim_on",
            "fix_on",
        ],
        "y_frame": ["left_qc_label", "right_qc_label"],
        "qc_label_map": {"0": "VALID", "1": "BLINK", "2": "MISSING"},
    }
    label_maps = {
        "health_label": {"0": "healthy/control", "1": "disease/patient"},
        "pd_disease_label": {
            "-1": "not_pd_patient_or_control",
            "0": "帕金森病",
            "1": "震颤",
            "2": "特发性震颤",
            "3": "运动障碍",
        },
    }
    manifest = {
        "dataset_version": "eyemae_fast_dataset_v2",
        "task_name": task_name,
        "source_root": str(source_root),
        "format": "packed_mmap",
        "shards_dir": "shards",
        "identity_key": "name|age|education_code parsed from source_stem; education_code is the field two positions before age, e.g. XX/CZ/GZ/ZZ/DZ/BK/SS/BS/WM",
    }
    _write_json(root / "columns.json", columns)
    _write_json(root / "label_maps.json", label_maps)
    _write_json(root / "dataset_manifest.json", manifest)
    _write_json(root / "audit_summary.json", audit)


def make_pretrain(
    source_root: Path,
    out_dir: Path,
    *,
    seed: int,
    target_bytes: int,
    audit_only: bool,
) -> dict[str, Any]:
    best_control_source: dict[str, tuple[tuple[int, str], str]] = {}
    raw_counts = Counter()
    total_manifest_rows = 0
    identity_subjects: set[str] = set()

    for row in iter_manifest(source_root):
        total_manifest_rows += 1
        identity = parse_identity(row)
        identity_subjects.add(identity.key)
        raw_counts[(row["source_top"], row["source_dataset"], row["source_group"])] += 1
        if is_control_row(row):
            key = identity.key
            candidate = (control_priority(row), source_subject_key(row))
            previous = best_control_source.get(key)
            if previous is None or candidate < previous:
                best_control_source[key] = candidate

    selected: list[dict[str, Any]] = []
    drops = Counter()
    for row in iter_manifest(source_root):
        identity = parse_identity(row)
        if is_ad_matched_experimental(row):
            drops["drop_ad_matched_experimental"] += 1
            continue
        if is_pd_matched_experimental(row):
            drops["drop_pd_matched_experimental"] += 1
            continue
        if is_mci_matched(row):
            drops["drop_mci_matched_for_pretrain"] += 1
            continue
        if is_ad_gly_matched_control(row, identity):
            drops["drop_ad_gly_matched_control"] += 1
            continue
        if is_control_row(row):
            best = best_control_source.get(identity.key)
            if best is None or source_subject_key(row) != best[1]:
                drops["drop_duplicate_noncanonical_control_identity"] += 1
                continue
        global_trial_id = f"pt2_{len(selected):09d}"
        item = base_row(row, identity=identity, global_trial_id=global_trial_id, source_kind="pretrain_v2")
        item["_npz_path"] = str(resolve_npz_path(source_root, row))
        selected.append(item)

    assignment = random_subject_split((row["ml_subject_id"] for row in selected), seed=seed, ratios=PRETRAIN_SPLIT_RATIOS)
    rows_by_split = {split: [] for split in SPLITS}
    for row in selected:
        rows_by_split[assignment[row["ml_subject_id"]]].append(row)
    for split in SPLITS:
        rows_by_split[split].sort(key=lambda item: item["global_trial_id"])

    split_summary = summarize_rows(rows_by_split, label_type="unlabeled")
    split_summary.update(
        {
            "strategy": "identity_heldout_random",
            "seed": seed,
            "ratios": PRETRAIN_SPLIT_RATIOS,
            "manifest_rows": total_manifest_rows,
            "manifest_identities": len(identity_subjects),
            "selected_rows_before_packing": len(selected),
            "selected_identities": len({row["ml_subject_id"] for row in selected}),
            "drop_counts": dict(sorted(drops.items())),
            "source_counts_after_cleaning": dict(
                sorted(
                    Counter(
                        f"{row['source_top']}/{row['source_dataset']}/{row['source_group']}" for row in selected
                    ).items()
                )
            ),
            "raw_source_counts": {"/".join(key): value for key, value in sorted(raw_counts.items())},
        }
    )
    if audit_only:
        return split_summary

    pretrain_root = out_dir / "pretrain"
    packed = pack_index_rows(
        pretrain_root,
        rows_by_split,
        target_bytes=target_bytes,
        split_filenames={split: f"pretrain/pretrain_{split}.csv" for split in SPLITS},
        all_filename="pretrain/pretrain_all_unique.csv",
        fieldnames=PRETRAIN_INDEX_FIELDNAMES,
    )
    packed_rows_by_split = packed["rows_by_split"]
    packed_all = packed["all_rows"]
    split_summary = summarize_rows(packed_rows_by_split, label_type="unlabeled")
    split_summary.update(
        {
            "strategy": "identity_heldout_random",
            "seed": seed,
            "ratios": PRETRAIN_SPLIT_RATIOS,
            "manifest_rows": total_manifest_rows,
            "manifest_identities": len(identity_subjects),
            "selected_rows": len(packed_all),
            "selected_identities": len({row["ml_subject_id"] for row in packed_all}),
            "drop_counts": dict(sorted(drops.items())),
            "source_counts_after_cleaning": dict(
                sorted(
                    Counter(
                        f"{row['source_top']}/{row['source_dataset']}/{row['source_group']}" for row in packed_all
                    ).items()
                )
            ),
            "pack": {key: value for key, value in packed.items() if key not in {"rows_by_split", "all_rows"}},
        }
    )
    _write_csv(pretrain_root / "trials.csv", packed_all, PRETRAIN_INDEX_FIELDNAMES)
    write_subjects(pretrain_root / "subjects.csv", packed_all)
    _write_json(pretrain_root / "pretrain" / "pretrain_split_summary.json", split_summary)
    write_common_metadata(pretrain_root, source_root=source_root, task_name="pretrain", audit=split_summary)
    return split_summary


def add_task_row(
    rows: list[dict[str, Any]],
    source_root: Path,
    row: dict[str, str],
    *,
    identity: Identity,
    task_name: str,
    health_label: int,
    pd_disease_label: int = -1,
) -> None:
    item = base_row(
        row,
        identity=identity,
        global_trial_id=f"{task_name}_{len(rows):09d}",
        source_kind="downstream_v2",
        view=task_name,
        health_label=str(health_label),
        pd_disease_label=str(pd_disease_label),
    )
    item["_npz_path"] = str(resolve_npz_path(source_root, row))
    rows.append(item)


def collect_downstream_rows(source_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pd_best_control: dict[str, tuple[tuple[int, str], str]] = {}
    for row in iter_manifest(source_root):
        if row.get("source_top", "").strip() != "PD相关":
            continue
        if row.get("source_group", "").strip() != "对照组":
            continue
        if row.get("source_dataset", "").strip() not in PD_MATCHED_DATASETS:
            continue
        identity = parse_identity(row)
        candidate = (pd_control_priority(row), source_subject_key(row))
        previous = pd_best_control.get(identity.key)
        if previous is None or candidate < previous:
            pd_best_control[identity.key] = candidate

    tasks: dict[str, list[dict[str, Any]]] = {
        "pd_related_5class": [],
        "pd_binary": [],
        "epilepsy_binary": [],
        "detox_binary": [],
        "migraine_binary": [],
        "ad_binary": [],
        "mci_binary": [],
        "mci_matched_binary": [],
    }
    drops: dict[str, Counter[str]] = defaultdict(Counter)

    for row in iter_manifest(source_root):
        identity = parse_identity(row)
        top = row.get("source_top", "").strip()
        dataset = row.get("source_dataset", "").strip()
        group = row.get("source_group", "").strip()

        if top == "PD相关":
            if group == "实验组" and dataset in PD_MATCHED_DATASETS:
                drops["pd"]["drop_pd_matched_experimental"] += 1
            elif group == "对照组" and dataset in PD_MATCHED_DATASETS:
                best = pd_best_control.get(identity.key)
                if best is None or source_subject_key(row) != best[1]:
                    drops["pd"]["drop_pd_duplicate_control_identity_noncanonical"] += 1
                else:
                    add_task_row(
                        tasks["pd_related_5class"],
                        source_root,
                        row,
                        identity=identity,
                        task_name="pd_related_5class",
                        health_label=0,
                        pd_disease_label=-1,
                    )
                    add_task_row(
                        tasks["pd_binary"],
                        source_root,
                        row,
                        identity=identity,
                        task_name="pd_binary",
                        health_label=0,
                        pd_disease_label=-1,
                    )
            elif group == "患病" and dataset in PD_DATASET_TO_LABEL:
                pd_label = PD_DATASET_TO_LABEL[dataset]
                add_task_row(
                    tasks["pd_related_5class"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="pd_related_5class",
                    health_label=1,
                    pd_disease_label=pd_label,
                )
                add_task_row(
                    tasks["pd_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="pd_binary",
                    health_label=1,
                    pd_disease_label=pd_label,
                )

        elif top == "AD":
            if dataset == "AD组" and group == "患病":
                add_task_row(tasks["ad_binary"], source_root, row, identity=identity, task_name="ad_binary", health_label=1)
            elif dataset == "匹配后" and group == "实验组":
                drops["ad"]["drop_ad_matched_experimental"] += 1
            elif dataset == "匹配后" and group == "对照组":
                if is_ad_gly_matched_control(row, identity):
                    drops["ad"]["drop_ad_gly_matched_control"] += 1
                else:
                    add_task_row(
                        tasks["ad_binary"],
                        source_root,
                        row,
                        identity=identity,
                        task_name="ad_binary",
                        health_label=0,
                    )

        elif top == "MCI":
            if dataset == "对照组" and group == "对照组":
                add_task_row(tasks["mci_binary"], source_root, row, identity=identity, task_name="mci_binary", health_label=0)
            elif dataset == "实验组" and group == "实验组":
                add_task_row(tasks["mci_binary"], source_root, row, identity=identity, task_name="mci_binary", health_label=1)
            elif dataset == "匹配后":
                # The matched-MCI source label direction is known to be reversed.
                if group == "对照组":
                    final_label = 1
                elif group == "实验组":
                    final_label = 0
                else:
                    drops["mci_matched"]["drop_unknown_group"] += 1
                    continue
                add_task_row(
                    tasks["mci_matched_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="mci_matched_binary",
                    health_label=final_label,
                )

        elif top == "癫痫":
            if dataset == "对照组" and group == "对照组":
                add_task_row(
                    tasks["epilepsy_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="epilepsy_binary",
                    health_label=0,
                )
            elif dataset == "癫痫组" and group == "患病":
                add_task_row(
                    tasks["epilepsy_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="epilepsy_binary",
                    health_label=1,
                )

        elif top == "偏头痛":
            if dataset == "对照组" and group == "对照组":
                add_task_row(
                    tasks["migraine_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="migraine_binary",
                    health_label=0,
                )
            elif dataset == "偏头痛" and group == "患病":
                add_task_row(
                    tasks["migraine_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="migraine_binary",
                    health_label=1,
                )

        elif top == "戒毒所":
            if dataset == "对照组原始数据" and group == "对照组":
                add_task_row(
                    tasks["detox_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="detox_binary",
                    health_label=0,
                )
            elif dataset == "戒毒所数据" and group == "患病":
                add_task_row(
                    tasks["detox_binary"],
                    source_root,
                    row,
                    identity=identity,
                    task_name="detox_binary",
                    health_label=1,
                )

    audit = {task: {"drop_counts": dict(counter)} for task, counter in drops.items()}
    return tasks, audit


def validate_no_subject_label_conflict(rows: list[dict[str, Any]], *, task_name: str, multiclass: bool) -> dict[str, str]:
    subject_labels: dict[str, str] = {}
    for row in rows:
        if multiclass:
            label = "0" if row["health_label"] == "0" else str(int(row["pd_disease_label"]) + 1)
        else:
            label = str(row["health_label"])
        previous = subject_labels.get(row["ml_subject_id"])
        if previous is not None and previous != label:
            raise ValueError(f"{task_name} subject has conflicting labels: {row['ml_subject_id']}: {previous} vs {label}")
        subject_labels[row["ml_subject_id"]] = label
    return subject_labels


def make_task_splits(
    tasks: dict[str, list[dict[str, Any]]],
    *,
    v1_dir: Path,
    seed: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows_by_task_split: dict[str, dict[str, list[dict[str, Any]]]] = {}

    preserved_views = {
        "epilepsy_binary": "癫痫",
        "migraine_binary": "偏头痛",
        "detox_binary": "戒毒所",
    }
    for task_name, view in preserved_views.items():
        split_by_identity = load_preserved_v1_split(v1_dir, view)
        rows_by_split = {split: [] for split in SPLITS}
        missing = sorted({row["ml_subject_id"] for row in tasks[task_name]} - set(split_by_identity))
        if missing:
            raise ValueError(f"{task_name} has identities missing from v1 preserved split: {missing[:20]}")
        for row in tasks[task_name]:
            split = split_by_identity[row["ml_subject_id"]]
            row["split"] = split
            rows_by_split[split].append(row)
        rows_by_task_split[task_name] = rows_by_split

    pd_subject_classes = validate_no_subject_label_conflict(tasks["pd_related_5class"], task_name="pd_related_5class", multiclass=True)
    pd_assignment = stratified_subject_split(pd_subject_classes, seed=seed, ratios=RANDOM_SPLIT_RATIOS)
    for task_name in ("pd_related_5class", "pd_binary"):
        rows_by_split = {split: [] for split in SPLITS}
        for row in tasks[task_name]:
            split = pd_assignment[row["ml_subject_id"]]
            row["split"] = split
            rows_by_split[split].append(row)
        rows_by_task_split[task_name] = rows_by_split

    for task_name in ("ad_binary", "mci_binary", "mci_matched_binary"):
        subject_labels = validate_no_subject_label_conflict(tasks[task_name], task_name=task_name, multiclass=False)
        assignment = stratified_subject_split(subject_labels, seed=seed, ratios=RANDOM_SPLIT_RATIOS)
        rows_by_split = {split: [] for split in SPLITS}
        for row in tasks[task_name]:
            split = assignment[row["ml_subject_id"]]
            row["split"] = split
            rows_by_split[split].append(row)
        rows_by_task_split[task_name] = rows_by_split

    for rows_by_split in rows_by_task_split.values():
        for split in SPLITS:
            rows_by_split[split].sort(key=lambda item: item["global_trial_id"])
    return rows_by_task_split


def make_downstream(
    source_root: Path,
    out_dir: Path,
    *,
    v1_dir: Path,
    seed: int,
    target_bytes: int,
    audit_only: bool,
) -> dict[str, Any]:
    tasks, cleaning_audit = collect_downstream_rows(source_root)
    rows_by_task_split = make_task_splits(tasks, v1_dir=v1_dir, seed=seed)
    task_audits: dict[str, Any] = {}
    for task_name, rows_by_split in rows_by_task_split.items():
        label_type = "multiclass" if task_name == "pd_related_5class" else "binary"
        summary = summarize_rows(rows_by_split, label_type=label_type)
        summary.update(
            {
                "task_name": task_name,
                "seed": seed,
                "ratios": RANDOM_SPLIT_RATIOS,
                "cleaning": cleaning_audit.get(
                    "pd" if task_name.startswith("pd_") else task_name.replace("_binary", "").replace("_matched", "_matched"),
                    {},
                ),
                "rows_before_packing": sum(len(rows) for rows in rows_by_split.values()),
                "subjects_before_packing": len({row["ml_subject_id"] for rows in rows_by_split.values() for row in rows}),
                "split_policy": (
                    "preserve_v1_identity_split"
                    if task_name in {"epilepsy_binary", "detox_binary", "migraine_binary"}
                    else "identity_stratified_random_64_16_20"
                ),
            }
        )
        if audit_only:
            task_audits[task_name] = summary
            continue

        task_root = out_dir / "finetune" / task_name
        packed = pack_index_rows(
            task_root,
            rows_by_split,
            target_bytes=target_bytes,
            split_filenames={split: f"{split}.csv" for split in SPLITS},
            all_filename="trials.csv",
            fieldnames=DOWNSTREAM_INDEX_FIELDNAMES,
        )
        packed_rows_by_split = packed["rows_by_split"]
        packed_all = packed["all_rows"]
        packed_summary = summarize_rows(packed_rows_by_split, label_type=label_type)
        packed_summary.update(
            {
                "task_name": task_name,
                "seed": seed,
                "ratios": RANDOM_SPLIT_RATIOS,
                "cleaning": summary.get("cleaning", {}),
                "split_policy": summary["split_policy"],
                "selected_rows": len(packed_all),
                "selected_subjects": len({row["ml_subject_id"] for row in packed_all}),
                "pack": {key: value for key, value in packed.items() if key not in {"rows_by_split", "all_rows"}},
            }
        )
        write_subjects(task_root / "subjects.csv", packed_all)
        _write_json(task_root / "split_summary.json", packed_summary)
        write_common_metadata(task_root, source_root=source_root, task_name=task_name, audit=packed_summary)
        task_audits[task_name] = packed_summary
    return task_audits


def write_root_readme(out_dir: Path, pretrain: dict[str, Any], downstream: dict[str, Any]) -> None:
    lines = [
        "# EyeMAE Fast Dataset v2",
        "",
        "This dataset is rebuilt from the raw `matched_groups_full_BACKUP_intermediate_20260618` source after applying the AD/PD/MCI cleaning rules identified during the v3 audit.",
        "",
        "## Layout",
        "",
        "- `pretrain/`: unlabeled pretraining packed mmap shards and `pretrain/pretrain_{train,validation,test}.csv`.",
        "- `finetune/<task>/`: one independent packed mmap dataset per downstream task, each with `train.csv`, `validation.csv`, `test.csv`, `split_summary.json`, `trials.csv`, and `subjects.csv`.",
        "",
        "## Key Cleaning Rules",
        "",
        "- AD matched experimental rows are dropped because they duplicate `AD/AD组/患病`; `AD/匹配后/对照组/GaoLianYing` is dropped and GaoLianYing remains an AD patient only.",
        "- PD matched experimental rows are dropped. PD matched controls are deduplicated by `name + age + education`; ZhangMingSha/LiuWenFang/ZhangMingLin are not treated as PD patients through matched experimental folders.",
        "- Original MCI excludes all `MCI/匹配后` rows. Matched MCI is a separate task and reverses the source matched label direction.",
        "- Epilepsy, migraine, and detox preserve the v1 identity-level train/validation/test split.",
        "",
        "## Summary",
        "",
        f"- Pretrain rows: {pretrain.get('selected_rows', pretrain.get('selected_rows_before_packing', 'NA'))}",
        f"- Pretrain identities: {pretrain.get('selected_identities', 'NA')}",
        "",
        "| downstream task | rows | subjects | split policy |",
        "| --- | ---: | ---: | --- |",
    ]
    for task_name, info in sorted(downstream.items()):
        rows = info.get("selected_rows", info.get("rows_before_packing", "NA"))
        subjects = info.get("selected_subjects", info.get("subjects_before_packing", "NA"))
        lines.append(f"| `{task_name}` | {rows} | {subjects} | {info.get('split_policy', '')} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--shard-target-gib", type=float, default=2.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    target_bytes = int(args.shard_target_gib * (1024**3))
    result: dict[str, Any] = {
        "source_root": str(args.source_root),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "shard_target_gib": args.shard_target_gib,
        "audit_only": args.audit_only,
    }
    pretrain_summary: dict[str, Any] = {}
    downstream_summary: dict[str, Any] = {}
    if not args.skip_pretrain:
        pretrain_summary = make_pretrain(
            args.source_root,
            args.out_dir,
            seed=args.seed,
            target_bytes=target_bytes,
            audit_only=args.audit_only,
        )
        result["pretrain"] = pretrain_summary
    if not args.skip_downstream:
        downstream_summary = make_downstream(
            args.source_root,
            args.out_dir,
            v1_dir=args.v1_dir,
            seed=args.seed,
            target_bytes=target_bytes,
            audit_only=args.audit_only,
        )
        result["downstream"] = downstream_summary
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / ("audit_only_summary.json" if args.audit_only else "v2_build_summary.json")
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        existing.update({key: value for key, value in result.items() if key not in {"pretrain", "downstream"}})
        if "pretrain" in result:
            existing["pretrain"] = result["pretrain"]
        if "downstream" in result:
            existing["downstream"] = result["downstream"]
        result = existing
    _write_json(summary_path, result)
    if not args.audit_only:
        write_root_readme(args.out_dir, result.get("pretrain", {}), result.get("downstream", {}))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
