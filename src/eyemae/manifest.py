from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .utils import str_to_bool, task_name_to_id


TASK_NAMES = {"ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade"}
AUX_DIRS = {"_manifest_parts", "direction_minus1_diagnostics", "supplement_direction_minus1_20260610"}


@dataclass(frozen=True)
class TrialRecord:
    path: str
    rel_path: str
    subject: str
    subject_id: str
    base_subject_id: str
    task_name: str
    task_id: int
    trial_id: str
    source_suffix: str
    left_final_keep: bool
    right_final_keep: bool
    n_samples: int | None = None
    disease: str = ""
    group: str = ""
    subtype: str = ""
    original_trial_index: str = ""
    direction: str = ""

    @property
    def usable_eye_pattern(self) -> str:
        if self.left_final_keep and self.right_final_keep:
            return "LR"
        if self.left_final_keep:
            return "L"
        if self.right_final_keep:
            return "R"
        return "none"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["usable_eye_pattern"] = self.usable_eye_pattern
        return out


def normalize_rel_path(path: str | Path, data_dir: str | Path) -> str:
    p = Path(path)
    base = Path(data_dir)
    try:
        rel = p.relative_to(base)
    except ValueError:
        rel = p
    return rel.as_posix()


def parse_suffix_from_name(name: str) -> str:
    match = re.search(r"_(D|L|R)_trial", name)
    if match:
        return match.group(1)
    stem = Path(name).stem
    parts = stem.split("_")
    for part in parts:
        if part in {"D", "L", "R"}:
            return part
    return "unknown"


def parse_fallback_record(npz_path: str | Path, data_dir: str | Path) -> TrialRecord:
    p = Path(npz_path)
    task_name = p.parent.name if p.parent.name in TASK_NAMES else "ProSaccade"
    subject = p.parent.parent.name if p.parent.parent else p.parent.name
    suffix = parse_suffix_from_name(p.name)
    left_keep = suffix != "R"
    right_keep = suffix != "L"
    if suffix == "unknown":
        left_keep = True
        right_keep = True
    subject_id = f"{subject}_{suffix}" if suffix != "unknown" else subject
    return TrialRecord(
        path=str(p),
        rel_path=normalize_rel_path(p, data_dir),
        subject=subject,
        subject_id=subject_id,
        base_subject_id=subject,
        task_name=task_name,
        task_id=task_name_to_id(task_name),
        trial_id=p.stem,
        source_suffix=suffix,
        left_final_keep=left_keep,
        right_final_keep=right_keep,
    )


def _record_from_manifest_row(row: dict[str, str], manifest_dir: Path, data_dir: Path) -> TrialRecord:
    trial_npz = row.get("trial_npz", "").strip()
    if trial_npz:
        path = Path(trial_npz)
    else:
        path = manifest_dir / row["relative_npz"]
    if not path.is_absolute():
        path = data_dir / path
    subject = (row.get("subject") or manifest_dir.name).strip()
    source_suffix = (row.get("source_suffix") or parse_suffix_from_name(path.name)).strip()
    info_id = (row.get("info_id") or subject).strip()
    info_id_with_suffix = (row.get("info_id_with_suffix") or "").strip()
    subject_id = info_id_with_suffix or (f"{info_id}_{source_suffix}" if source_suffix else info_id)
    task_name = (row.get("task") or path.parent.name).strip()
    return TrialRecord(
        path=str(path),
        rel_path=normalize_rel_path(path, data_dir),
        subject=subject,
        subject_id=subject_id,
        base_subject_id=info_id,
        task_name=task_name,
        task_id=task_name_to_id(task_name),
        trial_id=Path(path).stem,
        source_suffix=source_suffix,
        left_final_keep=str_to_bool(row.get("left_final_keep")),
        right_final_keep=str_to_bool(row.get("right_final_keep")),
        n_samples=int(row["n_samples"]) if row.get("n_samples") else None,
        disease=(row.get("disease") or "").strip(),
        group=(row.get("group") or "").strip(),
        subtype=(row.get("subtype") or "").strip(),
        original_trial_index=(row.get("original_trial_index") or "").strip(),
        direction=(row.get("direction") or "").strip(),
    )


def iter_manifest_records(data_dir: str | Path) -> Iterable[TrialRecord]:
    root = Path(data_dir).resolve()
    for dirpath, _, filenames in os.walk(root):
        if "manifest.csv" not in filenames:
            continue
        manifest_path = Path(dirpath) / "manifest.csv"
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield _record_from_manifest_row(row, manifest_path.parent, root)


def iter_npz_fallback_records(data_dir: str | Path) -> Iterable[TrialRecord]:
    root = Path(data_dir).resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in AUX_DIRS]
        for filename in filenames:
            if not filename.endswith(".npz"):
                continue
            p = Path(dirpath) / filename
            if p.parent.name not in TASK_NAMES and not any(key in p.name for key in ["trial", "synthetic"]):
                continue
            yield parse_fallback_record(p, root)


def scan_trial_records(data_dir: str | Path, exclude_no_eye_keep: bool = True) -> list[TrialRecord]:
    records = list(iter_manifest_records(data_dir))
    if not records:
        records = list(iter_npz_fallback_records(data_dir))
    if exclude_no_eye_keep:
        records = [r for r in records if r.left_final_keep or r.right_final_keep]
    return records


def build_record_index(data_dir: str | Path, exclude_no_eye_keep: bool = False) -> dict[str, TrialRecord]:
    records = scan_trial_records(data_dir, exclude_no_eye_keep=exclude_no_eye_keep)
    return {r.rel_path: r for r in records}


def summarize_records(records: list[TrialRecord], excluded_no_eye_keep: int = 0) -> dict:
    suffix_counts = Counter(r.source_suffix for r in records)
    eye_patterns = Counter(r.usable_eye_pattern for r in records)
    task_counts = Counter(str(r.task_id) for r in records)
    subjects = {r.base_subject_id for r in records}
    return {
        "usable_trials": len(records),
        "excluded_no_eye_keep": excluded_no_eye_keep,
        "num_subjects": len(subjects),
        "source_suffix_counts": dict(suffix_counts),
        "final_usable_eye_pattern_counts": dict(eye_patterns),
        "task_counts": dict(task_counts),
    }


def get_split_subject_key(record: TrialRecord | str, group_by_base_subject_id: bool = True) -> str:
    if isinstance(record, TrialRecord):
        return record.base_subject_id if group_by_base_subject_id else record.subject_id
    subject_id = record
    if group_by_base_subject_id and subject_id and subject_id[-1] in {"D", "L", "R"}:
        return subject_id[:-1]
    if group_by_base_subject_id and subject_id.endswith(("_D", "_L", "_R")):
        return subject_id[:-2]
    return subject_id
