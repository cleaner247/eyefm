from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1")
DEFAULT_SOURCE_VIEW = "AD"
DEFAULT_TARGET_VIEW = "AD_dedup_rawsubject"
DEFAULT_SEED = 20260622
DEFAULT_TRAIN_RATIO = 0.64
DEFAULT_VAL_RATIO = 0.16
DEFAULT_TEST_RATIO = 0.20
CONFLICT_CONTROL_SUBJECTS = {"GaoLianYing"}
MODES = ("scratch", "linear_probe", "partial", "full")
PRESERVE_SPLIT_POLICY = "preserve_source_split_after_raw_trial_dedup"
RANDOM_SPLIT_POLICY = "random_raw_subject_stratified"


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


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    raw = {
        "train": n * train_ratio,
        "validation": n * val_ratio,
        "test": n * test_ratio,
    }
    counts = {key: int(value) for key, value in raw.items()}
    remaining = n - sum(counts.values())
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1
    if n >= 3:
        for key in ("train", "validation", "test"):
            if counts[key] == 0:
                donor = max(counts, key=lambda item: counts[item])
                counts[donor] -= 1
                counts[key] += 1
    return counts["train"], counts["validation"], counts["test"]


def health_label(row: dict[str, str]) -> int:
    label = int(row["health_label"])
    if label not in {0, 1}:
        raise ValueError(f"Invalid AD health_label={label} for row {row.get('global_trial_id')}")
    return label


def make_raw_subject_split(
    rows: list[dict[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    subject_labels: dict[str, int] = {}
    by_label: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        subject = row["subject"]
        label = health_label(row)
        previous = subject_labels.get(subject)
        if previous is not None and previous != label:
            raise ValueError(f"Raw subject has conflicting AD labels: {subject}: {previous} vs {label}")
        if previous is None:
            subject_labels[subject] = label
    for subject, label in subject_labels.items():
        by_label[label].append(subject)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label, subjects in sorted(by_label.items()):
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        n_train, n_val, _n_test = split_counts(len(shuffled), train_ratio, val_ratio, test_ratio)
        split_names = ["train"] * n_train + ["validation"] * n_val + ["test"] * (
            len(shuffled) - n_train - n_val
        )
        for subject, split in zip(shuffled, split_names):
            assignment[subject] = split
    return assignment


def split_rows_by_assignment(
    rows: list[dict[str, str]],
    *,
    assignment: dict[str, str],
    target_view: str,
) -> dict[str, list[dict[str, str]]]:
    rows_by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        out = dict(row)
        split = assignment[out["subject"]]
        out["split"] = split
        out["view"] = target_view
        rows_by_split[split].append(out)
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
    ml_subject_sets = {split: {row["ml_subject_id"] for row in rows} for split, rows in rows_by_split.items()}
    raw_subject_sets = {split: {row["subject"] for row in rows} for split, rows in rows_by_split.items()}
    subject_overlap_counts = {
        "test_train": len(ml_subject_sets["test"] & ml_subject_sets["train"]),
        "test_validation": len(ml_subject_sets["test"] & ml_subject_sets["validation"]),
        "train_validation": len(ml_subject_sets["train"] & ml_subject_sets["validation"]),
    }
    raw_subject_overlap_counts = {
        "test_train": len(raw_subject_sets["test"] & raw_subject_sets["train"]),
        "test_validation": len(raw_subject_sets["test"] & raw_subject_sets["validation"]),
        "train_validation": len(raw_subject_sets["train"] & raw_subject_sets["validation"]),
    }
    summary: dict[str, Any] = {
        **extra,
        "no_subject_overlap": all(value == 0 for value in subject_overlap_counts.values()),
        "subject_overlap_counts": subject_overlap_counts,
        "raw_subject_overlap_counts": raw_subject_overlap_counts,
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
    split_policy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
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

    if split_policy == PRESERVE_SPLIT_POLICY:
        rows_by_split = split_rows(kept_rows)
        split_metadata: dict[str, Any] = {}
    elif split_policy == RANDOM_SPLIT_POLICY:
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1, got {ratio_sum}")
        assignment = make_raw_subject_split(
            kept_rows,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        rows_by_split = split_rows_by_assignment(
            kept_rows,
            assignment=assignment,
            target_view=target_view,
        )
        split_metadata = {
            "seed": seed,
            "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        }
    else:
        raise ValueError(f"Unknown split_policy={split_policy}")

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
            "split_policy": split_policy,
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
            **split_metadata,
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


def downstream_task_name(target_view: str) -> str:
    normalized = target_view.lower()
    if normalized.startswith("ad_binary_"):
        return normalized
    if normalized.startswith("ad_"):
        normalized = normalized[len("ad_") :]
    return f"ad_binary_{normalized}"


def update_ad_configs(
    *,
    target_view: str,
    output_root: Path,
    seed: int | None,
    ratios: dict[str, float] | None,
) -> list[Path]:
    written: list[Path] = []
    task_name = downstream_task_name(target_view)
    for mode in MODES:
        path = Path("configs/downstream") / f"ad_binary_{mode}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["experiment"]["name"] = f"downstream_v3_fast_{task_name}_{mode}"
        cfg["experiment"]["output_dir"] = str(output_root / mode_output_name(mode))
        cfg["data"]["train_index"] = f"downstream/{target_view}/train.csv"
        cfg["data"]["val_index"] = f"downstream/{target_view}/validation.csv"
        cfg["data"]["test_index"] = f"downstream/{target_view}/test.csv"
        cfg["data"]["subject_key"] = "subject"
        if seed is not None:
            cfg["split"]["seed"] = seed
        if ratios is not None:
            cfg["split"]["train_ratio"] = ratios["train"]
            cfg["split"]["val_ratio"] = ratios["validation"]
            cfg["split"]["test_ratio"] = ratios["test"]
        cfg["split"]["subject_key"] = "subject"
        cfg["split"]["split_summary"] = f"downstream/{target_view}/split_summary.json"
        cfg["label"]["view"] = target_view
        cfg["downstream"]["task_name"] = task_name
        cfg["downstream"]["disease"] = task_name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-view", default=DEFAULT_SOURCE_VIEW)
    parser.add_argument("--target-view", default=DEFAULT_TARGET_VIEW)
    parser.add_argument(
        "--split-policy",
        choices=(PRESERVE_SPLIT_POLICY, RANDOM_SPLIT_POLICY),
        default=PRESERVE_SPLIT_POLICY,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
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
        split_policy=args.split_policy,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.update_configs:
        written = update_ad_configs(
            target_view=args.target_view,
            output_root=args.output_root,
            seed=summary.get("seed"),
            ratios=summary.get("ratios"),
        )
        print("updated_configs")
        for path in written:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
