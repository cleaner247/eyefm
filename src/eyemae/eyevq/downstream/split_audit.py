"""Leakage checks shared by Subject-MIL training and final evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _trial_identity(row: Mapping[str, Any]) -> str:
    source_uid = str(row.get("source_file_uid", "")).strip()
    trial_index = str(row.get("original_trial_index", "")).strip()
    if source_uid and trial_index:
        return f"source:{source_uid}|trial:{trial_index}"
    relative_path = str(row.get("relative_source_path", "")).strip()
    if relative_path and trial_index:
        return f"path:{relative_path}|trial:{trial_index}"
    global_trial_id = str(row.get("global_trial_id", "")).strip()
    if not global_trial_id:
        raise ValueError("Split row has no usable trial identity")
    return f"global:{global_trial_id}"


def audit_split_rows(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    required = ("train", "val", "test")
    missing = [name for name in required if name not in split_rows]
    if missing:
        raise ValueError(f"Missing split rows for audit: {missing}")
    subjects = {
        name: {str(row["ml_subject_id"]) for row in split_rows[name]}
        for name in required
    }
    trials = {
        name: {_trial_identity(row) for row in split_rows[name]}
        for name in required
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    subject_overlap = {
        f"{left}_{right}": sorted(subjects[left] & subjects[right])
        for left, right in pairs
    }
    trial_overlap = {
        f"{left}_{right}": sorted(trials[left] & trials[right])
        for left, right in pairs
    }
    return {
        "subject_counts": {name: len(values) for name, values in subjects.items()},
        "trial_identity_counts": {name: len(values) for name, values in trials.items()},
        "subject_overlap": subject_overlap,
        "trial_overlap": trial_overlap,
        "passed": not any(subject_overlap.values()) and not any(trial_overlap.values()),
    }


def assert_clean_splits(audit: Mapping[str, Any]) -> None:
    if bool(audit.get("passed", False)):
        return
    summary = {
        "subject_overlap": {
            key: len(value) for key, value in audit["subject_overlap"].items()
        },
        "trial_overlap": {
            key: len(value) for key, value in audit["trial_overlap"].items()
        },
    }
    raise ValueError(f"Downstream split leakage detected: {summary}")
