from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import PackedTrialStore, collate_trials, load_npz_trial, read_packed_index, read_split_file
from .manifest import TrialRecord, build_record_index, scan_trial_records
from .patching import patchify_preprocessed_trial
from .preprocess import load_area_stats, preprocess_trial


DEFAULT_DISEASES = ["AD", "MCI", "PD相关", "偏头痛", "戒毒所", "癫痫"]
GROUP_TO_LABEL = {"健康": 0, "患病": 1}


def downstream_label(record: TrialRecord) -> int | None:
    return GROUP_TO_LABEL.get(record.group.strip())


def is_downstream_record(record: TrialRecord, disease: str) -> bool:
    return (
        record.disease.strip() == disease
        and downstream_label(record) is not None
        and (record.left_final_keep or record.right_final_keep)
    )


def filter_downstream_records(records: list[TrialRecord], disease: str) -> list[TrialRecord]:
    return [record for record in records if is_downstream_record(record, disease)]


def load_downstream_records(data_dir: str | Path, disease: str) -> list[TrialRecord]:
    return filter_downstream_records(scan_trial_records(data_dir, exclude_no_eye_keep=True), disease)


def summarize_downstream_records(records: list[TrialRecord]) -> dict[str, Any]:
    label_counts = Counter(str(downstream_label(record)) for record in records)
    subject_labels: dict[str, int] = {}
    subject_trial_counts: Counter[str] = Counter()
    for record in records:
        label = downstream_label(record)
        if label is None:
            continue
        existing = subject_labels.get(record.base_subject_id)
        if existing is not None and existing != label:
            raise ValueError(f"Subject has conflicting labels: {record.base_subject_id}")
        subject_labels[record.base_subject_id] = label
        subject_trial_counts[record.base_subject_id] += 1
    subject_label_counts = Counter(str(label) for label in subject_labels.values())
    return {
        "num_trials": len(records),
        "num_subjects": len(subject_labels),
        "trial_label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
        "subject_label_counts": {
            "0": int(subject_label_counts.get("0", 0)),
            "1": int(subject_label_counts.get("1", 0)),
        },
        "task_counts": dict(Counter(str(record.task_id) for record in records)),
        "group_counts": dict(Counter(record.group for record in records)),
        "disease_counts": dict(Counter(record.disease for record in records)),
        "eye_pattern_counts": dict(Counter(record.usable_eye_pattern for record in records)),
        "source_suffix_counts": dict(Counter(record.source_suffix for record in records)),
        "max_trials_per_subject": int(max(subject_trial_counts.values(), default=0)),
        "min_trials_per_subject": int(min(subject_trial_counts.values(), default=0)),
    }


def apply_final_keep_to_trial(trial: dict[str, Any], record: TrialRecord, cfg: dict[str, Any]) -> dict[str, Any]:
    eye = np.asarray(trial["eye"], dtype=np.float32).copy()
    missing_value = float(cfg["label"]["missing_value"])
    if not record.left_final_keep:
        eye[:, 0:3] = 0.0
        eye[:, 3] = missing_value
    if not record.right_final_keep:
        eye[:, 4:7] = 0.0
        eye[:, 7] = missing_value
    out = dict(trial)
    out["eye"] = eye
    out["subject_id"] = record.subject_id
    out["task_id"] = np.asarray(record.task_id, dtype=np.int64)
    out["trial_id"] = record.trial_id
    out["path"] = record.path
    out["left_eye_available"] = bool(record.left_final_keep)
    out["right_eye_available"] = bool(record.right_final_keep)
    return out


class DownstreamTrialDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        cfg: dict[str, Any],
        *,
        disease: str,
        area_stats: dict[str, Any] | None = None,
        max_trials: int | None = None,
        record_index: dict[str, TrialRecord] | None = None,
        train_subject_trial_counts: dict[str, int] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.disease = disease
        self.rel_paths = read_split_file(split_file)
        if max_trials is not None:
            self.rel_paths = self.rel_paths[: int(max_trials)]
        self.record_index = record_index if record_index is not None else build_record_index(data_dir, exclude_no_eye_keep=True)
        self.records = [self._record_for_rel_path(rel_path) for rel_path in self.rel_paths]
        self.area_stats = area_stats if area_stats is not None else load_area_stats(cfg["area"]["stats_path"])
        self._length_cache: dict[int, int] = {}
        if train_subject_trial_counts is None:
            train_subject_trial_counts = dict(Counter(record.base_subject_id for record in self.records))
        self.train_subject_trial_counts = train_subject_trial_counts

    def _record_for_rel_path(self, rel_path: str) -> TrialRecord:
        normalized = Path(rel_path).as_posix()
        record = self.record_index.get(normalized)
        if record is None:
            raise KeyError(f"Split row not found in manifest index: {rel_path}")
        if not is_downstream_record(record, self.disease):
            raise ValueError(f"Split row is not eligible for disease={self.disease}: {rel_path}")
        return record

    def __len__(self) -> int:
        return len(self.records)

    def get_num_patches(self, index: int) -> int:
        if index in self._length_cache:
            return self._length_cache[index]
        path = Path(self.records[index].path)
        schema = self.cfg["data"].get("npz_schema", "canonical")
        key = self.cfg["data"]["npz_keys"].get("eye", "eye")
        if schema == "cd_no_cond2_gaze_stimulus":
            key = self.cfg["data"]["npz_keys"].get("eye", "gaze")
        with np.load(path, allow_pickle=True) as z:
            t = int(z[key].shape[0])
        n = t // int(self.cfg["patch"]["samples"])
        self._length_cache[index] = n
        return n

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = downstream_label(record)
        if label is None:
            raise ValueError(f"Record has no downstream label: {record.rel_path}")
        trial = load_npz_trial(record.path, self.data_dir, self.cfg)
        trial = apply_final_keep_to_trial(trial, record, self.cfg)
        processed = preprocess_trial(trial, self.cfg, self.area_stats)
        patched = patchify_preprocessed_trial(processed, self.cfg)
        if patched is None:
            raise ValueError(f"No valid patches in trial: {record.path}")
        count = max(1, int(self.train_subject_trial_counts.get(record.base_subject_id, 1)))
        patched.update(
            {
                "label": float(label),
                "sample_weight": 1.0 / float(count),
                "base_subject_id": record.base_subject_id,
                "subject_key": record.base_subject_id,
                "record_subject_id": record.subject_id,
                "disease": record.disease,
                "group": record.group,
                "usable_eye_pattern": record.usable_eye_pattern,
            }
        )
        return patched


def packed_downstream_label(row: dict[str, str], cfg: dict[str, Any]) -> int:
    label_cfg = cfg.get("label", {})
    label_type = label_cfg.get("type", "binary")
    health = int(row.get("health_label", ""))
    if label_type == "binary":
        if health not in {0, 1}:
            raise ValueError(f"binary health_label must be 0/1, got {health} for {row.get('global_trial_id')}")
        return health
    if label_type == "multiclass":
        if health == 0:
            return 0
        pd_label = int(row.get("pd_disease_label", ""))
        if pd_label not in {0, 1, 2, 3}:
            raise ValueError(f"pd_disease_label must be 0..3 for patient row {row.get('global_trial_id')}, got {pd_label}")
        class_id = pd_label + 1
        num_classes = int(label_cfg.get("num_classes", 5))
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"class_id out of range for {row.get('global_trial_id')}: {class_id}")
        return class_id
    raise ValueError(f"Unsupported label.type: {label_type}")


class PackedDownstreamDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        index_file: str | Path,
        cfg: dict[str, Any],
        *,
        area_stats: dict[str, Any] | None = None,
        max_trials: int | None = None,
        train_subject_trial_counts: dict[str, int] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_file = Path(index_file)
        self.cfg = cfg
        self.rows = read_packed_index(self.index_file)
        if max_trials is not None:
            self.rows = self.rows[: int(max_trials)]
        self.area_stats = area_stats if area_stats is not None else load_area_stats(cfg["area"]["stats_path"])
        self.store = PackedTrialStore(
            self.data_dir,
            max_open_shards_per_worker=int(cfg["data"].get("max_open_shards_per_worker", 16)),
            validate_offsets=bool(cfg["data"].get("validate_offsets", True)),
        )
        if train_subject_trial_counts is None:
            train_subject_trial_counts = dict(Counter(row["ml_subject_id"] for row in self.rows))
        self.train_subject_trial_counts = train_subject_trial_counts
        self.labels = [packed_downstream_label(row, cfg) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def get_num_patches(self, index: int) -> int:
        row = self.rows[index]
        if row.get("num_patches_20ms"):
            return int(row["num_patches_20ms"])
        return int(row["frame_length"]) // int(self.cfg["patch"]["samples"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        trial = self.store.read_trial(row)
        processed = preprocess_trial(trial, self.cfg, self.area_stats)
        processed["ml_subject_id"] = trial["ml_subject_id"]
        processed["global_trial_id"] = trial["global_trial_id"]
        processed["source_suffix"] = trial.get("source_suffix", "")
        patched = patchify_preprocessed_trial(processed, self.cfg)
        if patched is None:
            raise ValueError(f"No valid patches in packed downstream trial: {trial['path']}")
        subject = row["ml_subject_id"]
        count = max(1, int(self.train_subject_trial_counts.get(subject, 1)))
        label = int(self.labels[index])
        patched.update(
            {
                "label": label,
                "label_type": self.cfg.get("label", {}).get("type", "binary"),
                "sample_weight": 1.0 / float(count),
                "base_subject_id": subject,
                "ml_subject_id": subject,
                "subject_key": subject,
                "record_subject_id": row.get("subject", subject),
                "global_trial_id": row["global_trial_id"],
                "disease": self.cfg.get("label", {}).get("task_name", self.cfg.get("downstream", {}).get("task_name", "")),
                "group": str(row.get("health_label", "")),
                "usable_eye_pattern": row.get("source_suffix", ""),
            }
        )
        return patched


def collate_downstream_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_trials(items)
    label_values = [item["label"] for item in items]
    label_dtype = torch.long if any(isinstance(value, int) and value not in {0, 1} for value in label_values) else torch.float32
    if any(str(item.get("label_type", "")) == "multiclass" for item in items):
        label_dtype = torch.long
    batch["label"] = torch.as_tensor(label_values, dtype=label_dtype)
    batch["sample_weight"] = torch.as_tensor([item["sample_weight"] for item in items], dtype=torch.float32)
    batch["base_subject_id"] = [str(item["base_subject_id"]) for item in items]
    batch["ml_subject_id"] = [str(item.get("ml_subject_id", item["base_subject_id"])) for item in items]
    batch["subject_key"] = [str(item["subject_key"]) for item in items]
    batch["record_subject_id"] = [str(item["record_subject_id"]) for item in items]
    batch["global_trial_id"] = [str(item.get("global_trial_id", item["trial_id"])) for item in items]
    batch["disease"] = [str(item["disease"]) for item in items]
    batch["group"] = [str(item["group"]) for item in items]
    batch["usable_eye_pattern"] = [str(item["usable_eye_pattern"]) for item in items]
    return batch
