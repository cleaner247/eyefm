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
DEFAULT_SEED = 20260621
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


def load_view_rows(data_dir: Path, view: str) -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    source_subject_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_rows, fields = read_csv(data_dir / "downstream" / view / f"{split}.csv")
        if fieldnames is None:
            fieldnames = fields
        elif fieldnames != fields:
            raise ValueError(f"CSV columns differ for {view} split={split}")
        source_subject_counts[split] = len({row["ml_subject_id"] for row in split_rows})
        rows.extend(split_rows)
    if fieldnames is None:
        raise ValueError(f"No source rows found for view={view}")
    return rows, fieldnames, source_subject_counts


def default_ratios(source_subject_counts: dict[str, int]) -> dict[str, float]:
    total = sum(source_subject_counts.values())
    if total <= 0:
        raise ValueError("No source subjects found")
    return {
        "train": source_subject_counts.get("train", 0) / total,
        "validation": source_subject_counts.get("validation", 0) / total,
        "test": source_subject_counts.get("test", 0) / total,
    }


def ratios_from_rows(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, float]:
    subject_counts = {split: len({row["ml_subject_id"] for row in rows}) for split, rows in rows_by_split.items()}
    total = sum(subject_counts.values())
    if total <= 0:
        raise ValueError("No subjects found")
    return {split: subject_counts[split] / total for split in ("train", "validation", "test")}


def split_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: n * ratios[key] for key in ("train", "validation", "test")}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = n - sum(counts.values())
    for key, _value in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
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
    return counts


def binary_label(row: dict[str, str]) -> int:
    label = int(row["health_label"])
    if label not in {0, 1}:
        raise ValueError(f"Invalid health_label={label} for row {row.get('global_trial_id')}")
    return label


def mci_source_group_for_label(label: str) -> str:
    if label == "0":
        return "对照组"
    if label == "1":
        return "实验组"
    raise ValueError(f"Invalid MCI anchor label={label}")


def flip_binary_label(label: str) -> str:
    if label == "0":
        return "1"
    if label == "1":
        return "0"
    raise ValueError(f"Invalid binary label={label}")


def raw_file_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("subject", ""), row.get("source_stem", ""))


def raw_trial_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("subject", ""),
        row.get("source_stem", ""),
        row.get("task", ""),
        row.get("original_trial_index", ""),
        row.get("direction", ""),
    )


def audit_rows(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    subject_labels: dict[str, set[str]] = defaultdict(set)
    raw_subject_labels: dict[str, set[str]] = defaultdict(set)
    raw_file_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_trial_labels: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    raw_subject_splits: dict[str, set[str]] = defaultdict(set)
    raw_file_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            label = row["health_label"]
            subject_labels[row["ml_subject_id"]].add(label)
            raw_subject_labels[row["subject"]].add(label)
            raw_file_labels[raw_file_key(row)].add(label)
            raw_trial_labels[raw_trial_key(row)].add(label)
            subject_splits[row["ml_subject_id"]].add(split)
            raw_subject_splits[row["subject"]].add(split)
            raw_file_splits[raw_file_key(row)].add(split)

    def conflict_count(mapping: dict[Any, set[str]]) -> int:
        return sum(1 for values in mapping.values() if len(values) > 1)

    def overlap_count(mapping: dict[Any, set[str]]) -> int:
        return sum(1 for values in mapping.values() if len(values) > 1)

    return {
        "rows": len(all_rows),
        "ml_subject_label_conflicts": conflict_count(subject_labels),
        "raw_subject_label_conflicts": conflict_count(raw_subject_labels),
        "raw_file_label_conflicts": conflict_count(raw_file_labels),
        "raw_trial_label_conflicts": conflict_count(raw_trial_labels),
        "ml_subject_split_overlap": overlap_count(subject_splits),
        "raw_subject_split_overlap": overlap_count(raw_subject_splits),
        "raw_file_split_overlap": overlap_count(raw_file_splits),
    }


def summarize_split(
    rows_by_split: dict[str, list[dict[str, str]]],
    *,
    task_name: str,
    source_view: str,
    target_view: str,
    split_policy: str,
    seed: int | None,
    ratios: dict[str, float] | None,
    source_subject_counts: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subject_sets = {split: {row["ml_subject_id"] for row in rows} for split, rows in rows_by_split.items()}
    summary: dict[str, Any] = {
        "task_name": task_name,
        "source_view": source_view,
        "target_view": target_view,
        "split_policy": split_policy,
        "seed": seed,
        "ratios": ratios,
        "source_subject_counts": source_subject_counts,
        "label_semantics": {"0": "control", "1": "MCI"},
        "subject_overlap_counts": {
            "test_train": len(subject_sets["test"] & subject_sets["train"]),
            "test_validation": len(subject_sets["test"] & subject_sets["validation"]),
            "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        },
        "splits": {},
        "audit": audit_rows(rows_by_split),
    }
    summary["no_subject_overlap"] = all(value == 0 for value in summary["subject_overlap_counts"].values())
    for split, rows in rows_by_split.items():
        subject_labels: dict[str, str] = {}
        for row in rows:
            subject = row["ml_subject_id"]
            label = row["health_label"]
            previous = subject_labels.get(subject)
            if previous is not None and previous != label:
                raise ValueError(f"Subject has conflicting labels in split summary: {subject}")
            subject_labels[subject] = label
        summary["splits"][split] = {
            "rows": len(rows),
            "subjects": len(subject_labels),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "binary_trial_counts": dict(sorted(Counter(row["health_label"] for row in rows).items())),
            "binary_subject_counts": dict(sorted(Counter(subject_labels.values()).items())),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
            "source_dataset_counts": dict(sorted(Counter(row.get("source_dataset", "") for row in rows).items())),
            "source_group_counts": dict(sorted(Counter(row.get("source_group", "") for row in rows).items())),
        }
    if extra:
        summary.update(extra)
    return summary


def write_rows_by_split(
    *,
    data_dir: Path,
    target_view: str,
    rows_by_split: dict[str, list[dict[str, str]]],
    fieldnames: list[str],
    force: bool,
) -> None:
    target_dir = data_dir / "downstream" / target_view
    if target_dir.exists() and not force:
        raise FileExistsError(f"Target split exists; pass --force to overwrite: {target_dir}")
    for split_rows in rows_by_split.values():
        split_rows.sort(key=lambda row: row["global_trial_id"])
    for split, split_rows in rows_by_split.items():
        write_csv(target_dir / f"{split}.csv", split_rows, fieldnames)


def collect_original_subject_labels(data_dir: Path, source_view: str) -> dict[str, str]:
    rows, _fieldnames, _source_subject_counts = load_view_rows(data_dir, source_view)
    original_subject_labels: dict[str, str] = {}
    for row in rows:
        if row.get("source_dataset") == "匹配后":
            continue
        subject = row["subject"]
        label = row["health_label"]
        previous = original_subject_labels.get(subject)
        if previous is not None and previous != label:
            raise ValueError(f"Original MCI subject has conflicting labels: {subject}: {previous} vs {label}")
        original_subject_labels[subject] = label
    return original_subject_labels


def generate_original_only(
    *,
    data_dir: Path,
    source_view: str,
    target_view: str,
    force: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    rows, fieldnames, source_subject_counts = load_view_rows(data_dir, source_view)
    rows_by_split = {"train": [], "validation": [], "test": []}
    removed_rows = 0
    original_subject_labels: dict[str, str] = {}
    for row in rows:
        if row.get("source_dataset") == "匹配后":
            removed_rows += 1
            continue
        subject = row["subject"]
        label = row["health_label"]
        previous = original_subject_labels.get(subject)
        if previous is not None and previous != label:
            raise ValueError(f"Original MCI subject has conflicting labels: {subject}: {previous} vs {label}")
        original_subject_labels[subject] = label
        out = dict(row)
        out["view"] = target_view
        rows_by_split[out["split"]].append(out)
    write_rows_by_split(
        data_dir=data_dir,
        target_view=target_view,
        rows_by_split=rows_by_split,
        fieldnames=fieldnames,
        force=force,
    )
    summary = summarize_split(
        rows_by_split,
        task_name="mci_original_only_binary",
        source_view=source_view,
        target_view=target_view,
        split_policy="filtered original MCI split; rows with source_dataset=匹配后 removed",
        seed=None,
        ratios=ratios_from_rows(rows_by_split),
        source_subject_counts=source_subject_counts,
        extra={
            "label_source": "original MCI rows only; source_dataset=匹配后 labels are ignored",
            "removed_matched_rows": removed_rows,
            "original_label_subjects": len(original_subject_labels),
        },
    )
    (data_dir / "downstream" / target_view / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary, original_subject_labels


def make_subject_split(
    rows: list[dict[str, str]],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    subject_labels: dict[str, int] = {}
    by_label: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        subject = row["ml_subject_id"]
        label = binary_label(row)
        previous = subject_labels.get(subject)
        if previous is not None and previous != label:
            raise ValueError(f"Subject has conflicting MCI labels: {subject}: {previous} vs {label}")
        if previous is None:
            subject_labels[subject] = label
    for subject, label in subject_labels.items():
        by_label[label].append(subject)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label, subjects in sorted(by_label.items()):
        shuffled = list(subjects)
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


def generate_matched_random(
    *,
    data_dir: Path,
    source_view: str,
    target_view: str,
    task_name: str,
    seed: int,
    original_subject_labels: dict[str, str],
    invert_anchor_label: bool,
    force: bool,
) -> dict[str, Any]:
    rows, fieldnames, source_subject_counts = load_view_rows(data_dir, source_view)
    ratios = default_ratios(source_subject_counts)
    relabeled_rows: list[dict[str, str]] = []
    dropped_unmapped_rows = 0
    changed_label_rows = 0
    flipped_anchor_rows = 0
    original_source_label_counts = Counter(row["health_label"] for row in rows)
    anchor_label_counts: Counter[str] = Counter()
    final_label_counts: Counter[str] = Counter()
    for row in rows:
        # Matched MCI rows are samples only. Their source labels are ignored.
        # The raw subject must exist in the original-MCI anchor; this optional
        # inversion handles the matched view whose healthy/disease coding was
        # later identified as reversed.
        anchor_label = original_subject_labels.get(row["subject"])
        if anchor_label is None:
            dropped_unmapped_rows += 1
            continue
        final_label = flip_binary_label(anchor_label) if invert_anchor_label else anchor_label
        out = dict(row)
        if invert_anchor_label:
            flipped_anchor_rows += 1
        if out["health_label"] != final_label:
            changed_label_rows += 1
        anchor_label_counts[anchor_label] += 1
        final_label_counts[final_label] += 1
        out["health_label"] = final_label
        out["source_group"] = mci_source_group_for_label(final_label)
        out["ml_subject_id"] = f"MCI_ORIGINAL_LABEL|{out['subject']}"
        relabeled_rows.append(out)
    assignment = make_subject_split(relabeled_rows, seed=seed, ratios=ratios)
    rows_by_split = {"train": [], "validation": [], "test": []}
    for row in relabeled_rows:
        out = dict(row)
        out["split"] = assignment[out["ml_subject_id"]]
        out["view"] = target_view
        rows_by_split[out["split"]].append(out)
    write_rows_by_split(
        data_dir=data_dir,
        target_view=target_view,
        rows_by_split=rows_by_split,
        fieldnames=fieldnames,
        force=force,
    )
    summary = summarize_split(
        rows_by_split,
        task_name=task_name,
        source_view=source_view,
        target_view=target_view,
        split_policy="subject-level stratified random split by health_label, preserving source split ratios",
        seed=seed,
        ratios=ratios,
        source_subject_counts=source_subject_counts,
        extra={
            "label_source": (
                "health_label set to the inverted original MCI subject label because "
                "the MCI matched healthy/disease coding was identified as reversed; "
                "source MCI匹配后 labels remain ignored"
                if invert_anchor_label
                else "health_label overwritten from original MCI subject label; source MCI匹配后 labels are ignored"
            ),
            "source_group_source": (
                "source_group overwritten to match the final inverted label"
                if invert_anchor_label
                else "source_group overwritten to match the original MCI subject label"
            ),
            "invert_anchor_label": invert_anchor_label,
            "source_rows": len(rows),
            "kept_rows_with_original_mci_subject": len(relabeled_rows),
            "dropped_unmapped_rows": dropped_unmapped_rows,
            "changed_label_rows": changed_label_rows,
            "flipped_anchor_rows": flipped_anchor_rows,
            "source_label_counts_before_relabel": dict(sorted(original_source_label_counts.items())),
            "anchor_label_counts_before_optional_inversion": dict(sorted(anchor_label_counts.items())),
            "final_label_counts": dict(sorted(final_label_counts.items())),
        },
    )
    (data_dir / "downstream" / target_view / "split_summary.json").write_text(
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


def generate_configs_for_task(
    *,
    base_task: str,
    task_name: str,
    target_view: str,
    output_root: Path,
    config_dir: Path,
    split_seed: int | None,
    split_ratios: dict[str, float] | None,
    force: bool,
) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mode in MODES:
        source = config_dir / f"{base_task}_{mode}.yaml"
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        cfg["experiment"]["name"] = f"downstream_v3_fast_{task_name}_{mode}"
        cfg["experiment"]["output_dir"] = str(output_root / mode_output_name(mode))
        cfg["data"]["train_index"] = f"downstream/{target_view}/train.csv"
        cfg["data"]["val_index"] = f"downstream/{target_view}/validation.csv"
        cfg["data"]["test_index"] = f"downstream/{target_view}/test.csv"
        cfg["split"]["split_summary"] = f"downstream/{target_view}/split_summary.json"
        cfg["split"]["seed"] = split_seed
        if split_ratios is not None:
            cfg["split"]["train_ratio"] = split_ratios["train"]
            cfg["split"]["val_ratio"] = split_ratios["validation"]
            cfg["split"]["test_ratio"] = split_ratios["test"]
        cfg["label"]["task_name"] = task_name
        cfg["label"]["view"] = target_view
        cfg["downstream"]["task_name"] = task_name
        cfg["downstream"]["disease"] = task_name
        cfg["downstream_train"]["max_epochs"] = 100
        cfg["downstream_train"]["early_stopping_patience"] = 10
        cfg["downstream_train"]["min_epochs_before_early_stopping"] = 0
        out = config_dir / f"{task_name}_{mode}.yaml"
        if out.exists() and not force:
            raise FileExistsError(f"Config exists; pass --force to overwrite: {out}")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/downstream"))
    parser.add_argument(
        "--matched-task-suffix",
        default="",
        help="Optional suffix appended to the generated matched view/task, e.g. label_fixed.",
    )
    parser.add_argument(
        "--invert-matched-anchor-labels",
        action="store_true",
        help="Flip final labels for the matched task after confirming the raw subject exists in original MCI.",
    )
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Only generate the matched task/config queue; do not rewrite original-only CSV/configs.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    original_view = "MCI_original_only_no_matched"
    matched_suffix = args.matched_task_suffix.strip()
    if matched_suffix and not matched_suffix.startswith("_"):
        matched_suffix = f"_{matched_suffix}"
    matched_random_view = f"MCI匹配后_random_seed{args.seed}{matched_suffix}"
    original_task = "mci_original_only_binary"
    matched_task = f"mci_matched_binary_random_seed{args.seed}{matched_suffix}"

    original_summary: dict[str, Any] | None = None
    original_configs: list[Path] = []
    if args.matched_only:
        original_subject_labels = collect_original_subject_labels(args.data_dir, "MCI")
    else:
        original_summary, original_subject_labels = generate_original_only(
            data_dir=args.data_dir,
            source_view="MCI",
            target_view=original_view,
            force=args.force,
        )
    matched_summary = generate_matched_random(
        data_dir=args.data_dir,
        source_view="MCI匹配后",
        target_view=matched_random_view,
        task_name=matched_task,
        seed=args.seed,
        original_subject_labels=original_subject_labels,
        invert_anchor_label=args.invert_matched_anchor_labels,
        force=args.force,
    )

    if original_summary is not None:
        original_configs = generate_configs_for_task(
            base_task="mci_binary",
            task_name=original_task,
            target_view=original_view,
            output_root=Path("outputs/downstream_v3_fast") / original_task,
            config_dir=args.config_dir,
            split_seed=None,
            split_ratios=original_summary["ratios"],
            force=args.force,
        )
    matched_configs = generate_configs_for_task(
        base_task="mci_matched_binary",
        task_name=matched_task,
        target_view=matched_random_view,
        output_root=Path("outputs/downstream_v3_fast") / matched_task,
        config_dir=args.config_dir,
        split_seed=args.seed,
        split_ratios=matched_summary["ratios"],
        force=args.force,
    )
    queue_name = f"queue_{matched_task}.txt" if args.matched_only or matched_suffix else f"queue_mci_followup_seed{args.seed}.txt"
    queue_path = args.config_dir / queue_name
    queue_lines = ["# MCI follow-up experiments requested after MCI label/identity audit."]
    if original_configs:
        queue_lines.extend(
            [
                "# Original MCI with source_dataset=匹配后 removed.",
                *[str(path) for path in original_configs],
                "",
            ]
        )
    queue_lines.extend(
        [
            "# MCI matched rows relabeled from original-MCI raw subject anchors.",
            "# If invert_anchor_label=true in split_summary.json, final labels are the anchor-label inverse.",
            *[str(path) for path in matched_configs],
        ]
    )
    if queue_path.exists() and not args.force:
        raise FileExistsError(f"Queue exists; pass --force to overwrite: {queue_path}")
    queue_path.write_text("\n".join(queue_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "original_summary": original_summary,
                "matched_random_summary": matched_summary,
                "original_configs": [str(path) for path in original_configs],
                "matched_random_configs": [str(path) for path in matched_configs],
                "queue": str(queue_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
