from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1")
DEFAULT_SOURCE_VIEW = "AD"
DEFAULT_TARGET_VIEW = "AD_dedup_rawsubject"
CONFLICT_CONTROL_SUBJECTS = {"GaoLianYing"}
MODES = ("scratch", "linear_probe", "partial", "full")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_view_rows(data_dir: Path, source_view: str) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for split in ("train", "validation", "test"):
        split_rows, fields = read_csv(data_dir / "downstream" / source_view / f"{split}.csv")
        if fieldnames is None:
            fieldnames = fields
        elif fieldnames != fields:
            raise ValueError(f"CSV columns differ for {source_view} split={split}")
        rows.extend(split_rows)
    if fieldnames is None:
        raise ValueError(f"No source rows found for view={source_view}")
    return rows, fieldnames


def source_basename(row: dict[str, str]) -> str:
    return Path(row["relative_source_path"]).name


def raw_file_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["subject"], source_basename(row))


def raw_trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["subject"],
        source_basename(row),
        row["original_trial_index"],
        row["direction"],
    )


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    rows_by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        rows_by_split[row["split"]].append(row)
    for split_rows_ in rows_by_split.values():
        split_rows_.sort(key=lambda item: item["global_trial_id"])
    return rows_by_split


def audit_rows(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    raw_subject_labels: dict[str, set[str]] = defaultdict(set)
    raw_file_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_trial_labels: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    ml_subject_splits: dict[str, set[str]] = defaultdict(set)
    raw_subject_splits: dict[str, set[str]] = defaultdict(set)
    raw_file_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_trial_splits: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    for split, rows in rows_by_split.items():
        for row in rows:
            label = row["health_label"]
            raw_subject_labels[row["subject"]].add(label)
            raw_file_labels[raw_file_key(row)].add(label)
            raw_trial_labels[raw_trial_key(row)].add(label)
            ml_subject_splits[row["ml_subject_id"]].add(split)
            raw_subject_splits[row["subject"]].add(split)
            raw_file_splits[raw_file_key(row)].add(split)
            raw_trial_splits[raw_trial_key(row)].add(split)

    def conflict_keys(mapping: dict[Any, set[str]]) -> set[Any]:
        return {key for key, values in mapping.items() if len(values) > 1}

    def row_count(keys: set[Any], key_fn: Any) -> int:
        return sum(1 for row in all_rows if key_fn(row) in keys)

    subject_conflicts = conflict_keys(raw_subject_labels)
    file_conflicts = conflict_keys(raw_file_labels)
    trial_conflicts = conflict_keys(raw_trial_labels)
    ml_split_overlap = conflict_keys(ml_subject_splits)
    subject_split_overlap = conflict_keys(raw_subject_splits)
    file_split_overlap = conflict_keys(raw_file_splits)
    trial_split_overlap = conflict_keys(raw_trial_splits)
    return {
        "rows": len(all_rows),
        "raw_subjects": len({row["subject"] for row in all_rows}),
        "ml_subjects": len({row["ml_subject_id"] for row in all_rows}),
        "raw_subject_label_conflict_keys": len(subject_conflicts),
        "raw_subject_label_conflict_rows": row_count(subject_conflicts, lambda row: row["subject"]),
        "raw_file_label_conflict_keys": len(file_conflicts),
        "raw_file_label_conflict_rows": row_count(file_conflicts, raw_file_key),
        "raw_trial_label_conflict_keys": len(trial_conflicts),
        "raw_trial_label_conflict_rows": row_count(trial_conflicts, raw_trial_key),
        "ml_subject_split_overlap_keys": len(ml_split_overlap),
        "ml_subject_split_overlap_rows": row_count(ml_split_overlap, lambda row: row["ml_subject_id"]),
        "raw_subject_split_overlap_keys": len(subject_split_overlap),
        "raw_subject_split_overlap_rows": row_count(subject_split_overlap, lambda row: row["subject"]),
        "raw_file_split_overlap_keys": len(file_split_overlap),
        "raw_file_split_overlap_rows": row_count(file_split_overlap, raw_file_key),
        "raw_trial_split_overlap_keys": len(trial_split_overlap),
        "raw_trial_split_overlap_rows": row_count(trial_split_overlap, raw_trial_key),
    }


def summarize(rows_by_split: dict[str, list[dict[str, str]]], extra: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **extra,
        "splits": {},
        "audit": audit_rows(rows_by_split),
    }
    for split, rows in rows_by_split.items():
        subject_labels: dict[str, str] = {}
        raw_subject_labels: dict[str, str] = {}
        for row in rows:
            subject_labels.setdefault(row["ml_subject_id"], row["health_label"])
            raw_subject_labels.setdefault(row["subject"], row["health_label"])
        summary["splits"][split] = {
            "rows": len(rows),
            "ml_subjects": len(subject_labels),
            "raw_subjects": len(raw_subject_labels),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "binary_trial_counts": dict(sorted(Counter(row["health_label"] for row in rows).items())),
            "binary_raw_subject_counts": dict(sorted(Counter(raw_subject_labels.values()).items())),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
            "source_dataset_counts": dict(sorted(Counter(row.get("source_dataset", "") for row in rows).items())),
            "source_group_counts": dict(sorted(Counter(row.get("source_group", "") for row in rows).items())),
        }
    return summary


def build_ad_dedup_view(
    *,
    data_dir: Path,
    source_view: str,
    target_view: str,
    force: bool,
) -> dict[str, Any]:
    rows, fieldnames = load_view_rows(data_dir, source_view)
    original_patient_rows = [
        row
        for row in rows
        if row["source_dataset"] == "AD组" and row["source_group"] == "患病" and row["health_label"] == "1"
    ]
    original_patient_keys = {raw_trial_key(row) for row in original_patient_rows}
    matched_exp_rows = [
        row
        for row in rows
        if row["source_dataset"] == "匹配后" and row["source_group"] == "实验组"
    ]
    unmatched_exp_rows = [row for row in matched_exp_rows if raw_trial_key(row) not in original_patient_keys]
    if unmatched_exp_rows:
        examples = [row["relative_source_path"] for row in unmatched_exp_rows[:10]]
        raise ValueError(
            "AD matched experimental rows are not fully contained in AD patient rows. "
            f"unmatched={len(unmatched_exp_rows)} examples={examples}"
        )

    kept_rows: list[dict[str, str]] = []
    removed_counts: Counter[str] = Counter()
    for row in rows:
        if row["source_dataset"] == "匹配后" and row["source_group"] == "实验组":
            removed_counts["matched_experimental_duplicate_rows"] += 1
            continue
        if (
            row["source_dataset"] == "匹配后"
            and row["source_group"] == "对照组"
            and row["subject"] in CONFLICT_CONTROL_SUBJECTS
        ):
            removed_counts["conflicting_matched_control_rows"] += 1
            continue
        out = dict(row)
        out["view"] = target_view
        kept_rows.append(out)

    rows_by_split = split_rows(kept_rows)
    target_dir = data_dir / "downstream" / target_view
    if target_dir.exists() and not force:
        raise FileExistsError(f"Target view already exists; pass --force to overwrite: {target_dir}")
    for split, split_rows_ in rows_by_split.items():
        write_csv(target_dir / f"{split}.csv", split_rows_, fieldnames)

    summary = summarize(
        rows_by_split,
        {
            "source_view": source_view,
            "target_view": target_view,
            "split_policy": "preserve_source_split_after_raw_trial_dedup",
            "dedup_rule": {
                "matched_experimental": "drop because every raw trial is contained in AD组/患病",
                "matched_control_conflicts": sorted(CONFLICT_CONTROL_SUBJECTS),
            },
            "source_rows": len(rows),
            "kept_rows": len(kept_rows),
            "removed_counts": dict(sorted(removed_counts.items())),
            "containment": {
                "original_patient_rows": len(original_patient_rows),
                "original_patient_raw_trial_keys": len(original_patient_keys),
                "matched_experimental_rows": len(matched_exp_rows),
                "matched_experimental_unmatched_rows": len(unmatched_exp_rows),
            },
        },
    )
    (target_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def mode_output_name(mode: str) -> str:
    return {
        "scratch": "scratch_full",
        "linear_probe": "pretrained_linear_probe",
        "partial": "pretrained_partial",
        "full": "pretrained_full",
    }[mode]


def update_ad_configs(*, target_view: str, output_root: Path) -> list[Path]:
    written: list[Path] = []
    for mode in MODES:
        path = Path("configs/downstream") / f"ad_binary_{mode}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["experiment"]["name"] = f"downstream_v3_fast_ad_binary_dedup_rawsubject_{mode}"
        cfg["experiment"]["output_dir"] = str(output_root / mode_output_name(mode))
        cfg["data"]["train_index"] = f"downstream/{target_view}/train.csv"
        cfg["data"]["val_index"] = f"downstream/{target_view}/validation.csv"
        cfg["data"]["test_index"] = f"downstream/{target_view}/test.csv"
        cfg["data"]["subject_key"] = "subject"
        cfg["split"]["subject_key"] = "subject"
        cfg["split"]["split_summary"] = f"downstream/{target_view}/split_summary.json"
        cfg["label"]["view"] = target_view
        cfg["downstream"]["task_name"] = "ad_binary_dedup_rawsubject"
        cfg["downstream"]["disease"] = "ad_binary_dedup_rawsubject"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-view", default=DEFAULT_SOURCE_VIEW)
    parser.add_argument("--target-view", default=DEFAULT_TARGET_VIEW)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update-configs", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/downstream_v3_fast/ad_binary_dedup_rawsubject"),
    )
    args = parser.parse_args()

    summary = build_ad_dedup_view(
        data_dir=args.data_dir,
        source_view=args.source_view,
        target_view=args.target_view,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.update_configs:
        written = update_ad_configs(target_view=args.target_view, output_root=args.output_root)
        print("updated_configs")
        for path in written:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
