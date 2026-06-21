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
DEFAULT_SOURCE_VIEW = "PD相关"
DEFAULT_SEED = 20260620
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


def class_id(row: dict[str, str]) -> int:
    health = int(row["health_label"])
    if health == 0:
        return 0
    pd_label = int(row["pd_disease_label"])
    if pd_label not in {0, 1, 2, 3}:
        raise ValueError(f"Invalid pd_disease_label={pd_label} for patient row {row.get('global_trial_id')}")
    return pd_label + 1


def load_source_rows(data_dir: Path, source_view: str) -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    source_subject_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_rows, fields = read_csv(data_dir / "downstream" / source_view / f"{split}.csv")
        if fieldnames is None:
            fieldnames = fields
        elif fieldnames != fields:
            raise ValueError(f"CSV columns differ for split={split}")
        subjects = {row["ml_subject_id"] for row in split_rows}
        source_subject_counts[split] = len(subjects)
        rows.extend(split_rows)
    if fieldnames is None:
        raise ValueError("No source rows found")
    return rows, fieldnames, source_subject_counts


def default_ratios(source_subject_counts: dict[str, int]) -> tuple[float, float, float]:
    total = sum(source_subject_counts.values())
    if total <= 0:
        raise ValueError("No source subjects found")
    return (
        source_subject_counts.get("train", 0) / total,
        source_subject_counts.get("validation", 0) / total,
        source_subject_counts.get("test", 0) / total,
    )


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


def make_subject_split(
    rows: list[dict[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    subject_labels: dict[str, int] = {}
    by_class: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        subject = row["ml_subject_id"]
        label = class_id(row)
        previous = subject_labels.get(subject)
        if previous is not None and previous != label:
            raise ValueError(f"Subject has conflicting PD class labels: {subject}: {previous} vs {label}")
        if previous is None:
            subject_labels[subject] = label
    for subject, label in subject_labels.items():
        by_class[label].append(subject)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label, subjects in sorted(by_class.items()):
        subjects = list(subjects)
        rng.shuffle(subjects)
        n_train, n_val, _n_test = split_counts(len(subjects), train_ratio, val_ratio, test_ratio)
        split_names = (
            ["train"] * n_train
            + ["validation"] * n_val
            + ["test"] * (len(subjects) - n_train - n_val)
        )
        for subject, split in zip(subjects, split_names):
            assignment[subject] = split
    return assignment


def summarize_split(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    subject_sets = {split: {row["ml_subject_id"] for row in rows} for split, rows in rows_by_split.items()}
    summary: dict[str, Any] = {
        "no_subject_overlap": True,
        "subject_overlap_counts": {
            "test_train": len(subject_sets["test"] & subject_sets["train"]),
            "test_validation": len(subject_sets["test"] & subject_sets["validation"]),
            "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        },
        "splits": {},
    }
    summary["no_subject_overlap"] = all(value == 0 for value in summary["subject_overlap_counts"].values())
    for split, rows in rows_by_split.items():
        subjects = {row["ml_subject_id"] for row in rows}
        subject_classes: dict[str, int] = {}
        for row in rows:
            subject_classes[row["ml_subject_id"]] = class_id(row)
        summary["splits"][split] = {
            "rows": len(rows),
            "subjects": len(subjects),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "health_label_counts": dict(sorted(Counter(row["health_label"] for row in rows).items())),
            "pd_class_trial_counts": dict(sorted(Counter(str(class_id(row)) for row in rows).items())),
            "pd_class_subject_counts": dict(sorted(Counter(str(label) for label in subject_classes.values()).items())),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
            "source_dataset_counts": dict(sorted(Counter(row.get("source_dataset", "") for row in rows).items())),
        }
    return summary


def generate_split(
    *,
    data_dir: Path,
    source_view: str,
    target_view: str,
    seed: int,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    force: bool,
) -> dict[str, Any]:
    rows, fieldnames, source_subject_counts = load_source_rows(data_dir, source_view)
    if train_ratio is None or val_ratio is None or test_ratio is None:
        train_ratio, val_ratio, test_ratio = default_ratios(source_subject_counts)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratio_sum}")

    target_dir = data_dir / "downstream" / target_view
    if target_dir.exists() and not force:
        raise FileExistsError(f"Target split exists; pass --force to overwrite: {target_dir}")
    assignment = make_subject_split(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    rows_by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        out = dict(row)
        split = assignment[out["ml_subject_id"]]
        out["split"] = split
        out["view"] = target_view
        rows_by_split[split].append(out)
    for split_rows in rows_by_split.values():
        split_rows.sort(key=lambda row: row["global_trial_id"])
    for split, split_rows in rows_by_split.items():
        write_csv(target_dir / f"{split}.csv", split_rows, fieldnames)

    summary = summarize_split(rows_by_split)
    summary.update(
        {
            "source_view": source_view,
            "target_view": target_view,
            "seed": seed,
            "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
            "source_subject_counts": source_subject_counts,
        }
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


def generate_configs(
    *,
    target_view: str,
    seed: int,
    output_root: Path,
    config_dir: Path,
    force: bool,
) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mode in MODES:
        source = Path("configs/downstream") / f"pd_related_5class_{mode}.yaml"
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        cfg["experiment"]["name"] = f"downstream_v3_pd_related_5class_random_seed{seed}_fast_{mode}"
        cfg["experiment"]["output_dir"] = str(output_root / mode_output_name(mode))
        cfg["data"]["train_index"] = f"downstream/{target_view}/train.csv"
        cfg["data"]["val_index"] = f"downstream/{target_view}/validation.csv"
        cfg["data"]["test_index"] = f"downstream/{target_view}/test.csv"
        cfg["split"]["seed"] = seed
        cfg["split"]["split_summary"] = f"downstream/{target_view}/split_summary.json"
        cfg["label"]["view"] = target_view
        cfg["downstream_train"]["max_epochs"] = 100
        cfg["downstream_train"]["early_stopping_patience"] = 10
        cfg["downstream_train"]["min_epochs_before_early_stopping"] = 0
        out = config_dir / f"pd_related_5class_random_seed{seed}_fast_{mode}.yaml"
        if out.exists() and not force:
            raise FileExistsError(f"Config exists; pass --force to overwrite: {out}")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-view", default=DEFAULT_SOURCE_VIEW)
    parser.add_argument("--target-view", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/downstream"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_view = args.target_view or f"{args.source_view}_random_seed{args.seed}"
    output_root = args.output_root or Path("outputs/downstream_v3") / f"pd_related_5class_random_seed{args.seed}_fast"

    summary = generate_split(
        data_dir=args.data_dir,
        source_view=args.source_view,
        target_view=target_view,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=args.force,
    )
    configs = generate_configs(
        target_view=target_view,
        seed=args.seed,
        output_root=output_root,
        config_dir=args.config_dir,
        force=args.force,
    )
    print(json.dumps({"split_summary": summary, "configs": [str(path) for path in configs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
