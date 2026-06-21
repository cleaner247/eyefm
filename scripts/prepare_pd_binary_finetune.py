from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v1")
DEFAULT_SOURCE_VIEW = "PD相关_random_seed20260620"
DEFAULT_TARGET_VIEW = "PD相关_binary_random_seed20260620"
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


def mode_output_name(mode: str) -> str:
    return {
        "scratch": "scratch_full",
        "linear_probe": "pretrained_linear_probe",
        "partial": "pretrained_partial",
        "full": "pretrained_full",
    }[mode]


def summarize_split(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    subject_sets = {split: {row["ml_subject_id"] for row in rows} for split, rows in rows_by_split.items()}
    summary: dict[str, Any] = {
        "task_name": "pd_binary",
        "label_semantics": {
            "0": "healthy control",
            "1": "PD-related patient; all four pd_disease_label subclasses merged",
        },
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
        subject_labels: dict[str, str] = {}
        subject_pd_classes: dict[str, str] = {}
        for row in rows:
            subject = row["ml_subject_id"]
            label = str(row["health_label"])
            previous = subject_labels.get(subject)
            if previous is not None and previous != label:
                raise ValueError(f"Subject has conflicting binary labels: {subject}: {previous} vs {label}")
            subject_labels[subject] = label
            subject_pd_classes[subject] = str(row.get("pd_disease_label", ""))
        summary["splits"][split] = {
            "rows": len(rows),
            "subjects": len(subject_labels),
            "frames": int(sum(int(row["frame_length"]) for row in rows)),
            "binary_trial_counts": dict(sorted(Counter(row["health_label"] for row in rows).items())),
            "binary_subject_counts": dict(sorted(Counter(subject_labels.values()).items())),
            "pd_disease_trial_counts": dict(sorted(Counter(row.get("pd_disease_label", "") for row in rows).items())),
            "pd_disease_subject_counts": dict(sorted(Counter(subject_pd_classes.values()).items())),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
            "source_dataset_counts": dict(sorted(Counter(row.get("source_dataset", "") for row in rows).items())),
        }
    return summary


def generate_binary_view(
    *,
    data_dir: Path,
    source_view: str,
    target_view: str,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    target_dir = data_dir / "downstream" / target_view
    if target_dir.exists() and not force:
        raise FileExistsError(f"Target view exists; pass --force to overwrite: {target_dir}")

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    fieldnames: list[str] | None = None
    for split in ("train", "validation", "test"):
        rows, fields = read_csv(data_dir / "downstream" / source_view / f"{split}.csv")
        if fieldnames is None:
            fieldnames = fields
        elif fieldnames != fields:
            raise ValueError(f"CSV columns differ for split={split}")
        out_rows: list[dict[str, str]] = []
        for row in rows:
            health = int(row["health_label"])
            pd_label = int(row["pd_disease_label"])
            if health == 0 and pd_label != -1:
                raise ValueError(f"Healthy row has unexpected pd_disease_label={pd_label}: {row['global_trial_id']}")
            if health == 1 and pd_label not in {0, 1, 2, 3}:
                raise ValueError(f"Patient row has invalid pd_disease_label={pd_label}: {row['global_trial_id']}")
            out = dict(row)
            out["view"] = target_view
            out["split"] = split
            out_rows.append(out)
        rows_by_split[split] = out_rows

    assert fieldnames is not None
    for split, rows in rows_by_split.items():
        write_csv(target_dir / f"{split}.csv", rows, fieldnames)

    summary = summarize_split(rows_by_split)
    summary.update(
        {
            "source_view": source_view,
            "target_view": target_view,
            "seed": seed,
            "split_policy": "reuses the source PD subject split; only label semantics are binary",
        }
    )
    (target_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


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
        source = config_dir / f"pd_related_5class_random_seed{seed}_fast_{mode}.yaml"
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        cfg["experiment"]["name"] = f"downstream_v3_fast_pd_binary_random_seed{seed}_{mode}"
        cfg["experiment"]["output_dir"] = str(output_root / mode_output_name(mode))
        cfg["data"]["train_index"] = f"downstream/{target_view}/train.csv"
        cfg["data"]["val_index"] = f"downstream/{target_view}/validation.csv"
        cfg["data"]["test_index"] = f"downstream/{target_view}/test.csv"
        cfg["split"]["seed"] = seed
        cfg["split"]["split_summary"] = f"downstream/{target_view}/split_summary.json"
        cfg["label"] = {
            "nonblink_value": cfg["label"].get("nonblink_value", 0),
            "blink_value": cfg["label"].get("blink_value", 1),
            "missing_value": cfg["label"].get("missing_value", 2),
            "type": "binary",
            "task_name": "pd_binary",
            "view": target_view,
            "label_column": "health_label",
            "negative_label": 0,
            "positive_label": 1,
        }
        cfg["downstream"]["task_name"] = "pd_binary"
        cfg["downstream"]["disease"] = "pd_binary"
        cfg["downstream_checkpoint"]["monitor"] = "validation/subject_auroc"
        cfg["class_weighting"]["mode"] = "subject_pos_weight"
        out = config_dir / f"pd_binary_random_seed{seed}_fast_{mode}.yaml"
        if out.exists() and not force:
            raise FileExistsError(f"Config exists; pass --force to overwrite: {out}")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-view", default=DEFAULT_SOURCE_VIEW)
    parser.add_argument("--target-view", default=DEFAULT_TARGET_VIEW)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/downstream_v3_fast") / f"pd_binary_random_seed{DEFAULT_SEED}",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs/downstream"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = generate_binary_view(
        data_dir=args.data_dir,
        source_view=args.source_view,
        target_view=args.target_view,
        seed=args.seed,
        force=args.force,
    )
    configs = generate_configs(
        target_view=args.target_view,
        seed=args.seed,
        output_root=args.output_root,
        config_dir=args.config_dir,
        force=args.force,
    )
    print(json.dumps({"split_summary": summary, "configs": [str(path) for path in configs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
