from __future__ import annotations

import argparse
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .downstream_config import load_downstream_config
from .downstream_data import DEFAULT_DISEASES, downstream_label, filter_downstream_records, summarize_downstream_records
from .manifest import TrialRecord, scan_trial_records
from .utils import setup_logging, write_json


LOGGER = logging.getLogger(__name__)


def _write_split(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
    return n_train, n_val, n_test


def _subject_labels(records: list[TrialRecord]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for record in records:
        label = downstream_label(record)
        if label is None:
            continue
        previous = labels.get(record.base_subject_id)
        if previous is not None and previous != label:
            raise ValueError(f"Subject has conflicting labels for downstream split: {record.base_subject_id}")
        labels[record.base_subject_id] = label
    return labels


def _split_subjects(
    subject_to_label: dict[str, int],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, set[str]]:
    rng = random.Random(seed)
    subjects_by_label: dict[int, list[str]] = defaultdict(list)
    for subject, label in subject_to_label.items():
        subjects_by_label[int(label)].append(subject)
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for label in sorted(subjects_by_label):
        subjects = sorted(subjects_by_label[label])
        rng.shuffle(subjects)
        n_train, n_val, _n_test = _split_counts(len(subjects), train_ratio, val_ratio)
        out["train"].update(subjects[:n_train])
        out["val"].update(subjects[n_train : n_train + n_val])
        out["test"].update(subjects[n_train + n_val :])
    return out


def _kfold_subject_buckets(
    subject_to_label: dict[str, int],
    *,
    seed: int,
    num_folds: int,
) -> list[set[str]]:
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for subject_stratified_kfold")
    rng = random.Random(seed)
    subjects_by_label: dict[int, list[str]] = defaultdict(list)
    for subject, label in subject_to_label.items():
        subjects_by_label[int(label)].append(subject)
    folds = [set() for _ in range(num_folds)]
    for label in sorted(subjects_by_label):
        subjects = sorted(subjects_by_label[label])
        rng.shuffle(subjects)
        if len(subjects) < num_folds:
            LOGGER.warning("label=%s has fewer subjects (%s) than num_folds=%s", label, len(subjects), num_folds)
        for index, subject in enumerate(subjects):
            folds[index % num_folds].add(subject)
    return folds


def _kfold_subject_splits(
    subject_to_label: dict[str, int],
    *,
    seed: int,
    num_folds: int,
) -> list[dict[str, set[str]]]:
    fold_buckets = _kfold_subject_buckets(subject_to_label, seed=seed, num_folds=num_folds)
    all_subjects = set(subject_to_label)
    out: list[dict[str, set[str]]] = []
    for fold_index in range(num_folds):
        test_subjects = set(fold_buckets[fold_index])
        val_subjects = set(fold_buckets[(fold_index + 1) % num_folds])
        train_subjects = all_subjects - test_subjects - val_subjects
        out.append({"train": train_subjects, "val": val_subjects, "test": test_subjects})
    return out


def _split_summary(
    disease: str,
    records: list[TrialRecord],
    subject_splits: dict[str, set[str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strategy: str = "subject_stratified_by_label",
    fold_index: int | None = None,
    num_folds: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "disease": disease,
        "seed": seed,
        "strategy": strategy,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "overall": summarize_downstream_records(records),
        "splits": {},
    }
    if fold_index is not None:
        summary["fold_index"] = int(fold_index)
    if num_folds is not None:
        summary["num_folds"] = int(num_folds)
    subject_to_label = _subject_labels(records)
    for split_name, subjects in subject_splits.items():
        split_records = [record for record in records if record.base_subject_id in subjects]
        label_counts = Counter(str(downstream_label(record)) for record in split_records)
        subject_label_counts = Counter(str(label) for subject, label in subject_to_label.items() if subject in subjects)
        summary["splits"][split_name] = {
            "num_trials": len(split_records),
            "num_subjects": len(subjects),
            "trial_label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
            "subject_label_counts": {
                "0": int(subject_label_counts.get("0", 0)),
                "1": int(subject_label_counts.get("1", 0)),
            },
            "task_counts": dict(Counter(str(record.task_id) for record in split_records)),
            "eye_pattern_counts": dict(Counter(record.usable_eye_pattern for record in split_records)),
            "source_suffix_counts": dict(Counter(record.source_suffix for record in split_records)),
        }
    return summary


def _make_downstream_holdout_splits(
    cfg: dict[str, Any],
    *,
    diseases: list[str] | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
) -> dict[str, Any]:
    split_cfg = cfg.get("downstream_split", {})
    data_dir = Path(cfg["data"]["data_dir"])
    split_out = Path(out_dir or split_cfg.get("out_dir", "splits/downstream_disease_binary_seed42"))
    selected_diseases = diseases or split_cfg.get("diseases") or cfg.get("downstream", {}).get("diseases") or DEFAULT_DISEASES
    split_seed = int(seed if seed is not None else split_cfg.get("seed", 42))
    tr = float(train_ratio if train_ratio is not None else split_cfg.get("train_ratio", 0.70))
    vr = float(val_ratio if val_ratio is not None else split_cfg.get("val_ratio", 0.15))
    ter = float(test_ratio if test_ratio is not None else split_cfg.get("test_ratio", 0.15))
    if abs((tr + vr + ter) - 1.0) > 1e-6:
        raise ValueError("downstream split ratios must sum to 1.0")

    all_records = scan_trial_records(data_dir, exclude_no_eye_keep=True)
    all_summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "out_dir": str(split_out),
        "strategy": "subject_stratified_by_label",
        "diseases": {},
    }
    for disease in selected_diseases:
        records = filter_downstream_records(all_records, disease)
        if not records:
            raise RuntimeError(f"No eligible downstream records for disease={disease}")
        subject_to_label = _subject_labels(records)
        subject_splits = _split_subjects(subject_to_label, seed=split_seed, train_ratio=tr, val_ratio=vr)
        disease_dir = split_out / disease
        for split_name, subjects in subject_splits.items():
            split_rows = sorted(record.rel_path for record in records if record.base_subject_id in subjects)
            _write_split(disease_dir / f"{split_name}.txt", split_rows)
        summary = _split_summary(
            disease,
            records,
            subject_splits,
            seed=split_seed,
            train_ratio=tr,
            val_ratio=vr,
            test_ratio=ter,
        )
        write_json(disease_dir / "split_summary.json", summary)
        all_summary["diseases"][disease] = summary
        LOGGER.info(
            "%s: train=%s val=%s test=%s subjects",
            disease,
            len(subject_splits["train"]),
            len(subject_splits["val"]),
            len(subject_splits["test"]),
        )
    write_json(split_out / "split_summary_all.json", all_summary)
    return all_summary


def make_downstream_kfold_splits(
    cfg: dict[str, Any],
    *,
    diseases: list[str] | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    num_folds: int | None = None,
) -> dict[str, Any]:
    split_cfg = cfg.get("downstream_split", {})
    data_dir = Path(cfg["data"]["data_dir"])
    split_out = Path(out_dir or split_cfg.get("out_dir", "splits/downstream_disease_binary_kfold_seed42"))
    selected_diseases = diseases or split_cfg.get("diseases") or cfg.get("downstream", {}).get("diseases") or DEFAULT_DISEASES
    split_seed = int(seed if seed is not None else split_cfg.get("seed", 42))
    folds = int(num_folds if num_folds is not None else split_cfg.get("num_folds", 5))
    all_records = scan_trial_records(data_dir, exclude_no_eye_keep=True)
    all_summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "out_dir": str(split_out),
        "strategy": "subject_stratified_kfold",
        "seed": split_seed,
        "num_folds": folds,
        "diseases": {},
    }
    for disease in selected_diseases:
        records = filter_downstream_records(all_records, disease)
        if not records:
            raise RuntimeError(f"No eligible downstream records for disease={disease}")
        subject_to_label = _subject_labels(records)
        fold_splits = _kfold_subject_splits(subject_to_label, seed=split_seed, num_folds=folds)
        disease_summaries: list[dict[str, Any]] = []
        for fold_index, subject_splits in enumerate(fold_splits):
            fold_dir = split_out / f"fold_{fold_index}" / disease
            for split_name, subjects in subject_splits.items():
                split_rows = sorted(record.rel_path for record in records if record.base_subject_id in subjects)
                _write_split(fold_dir / f"{split_name}.txt", split_rows)
            summary = _split_summary(
                disease,
                records,
                subject_splits,
                seed=split_seed,
                train_ratio=max(0, folds - 2) / float(folds),
                val_ratio=1.0 / float(folds),
                test_ratio=1.0 / float(folds),
                strategy="subject_stratified_kfold",
                fold_index=fold_index,
                num_folds=folds,
            )
            write_json(fold_dir / "split_summary.json", summary)
            disease_summaries.append(summary)
            LOGGER.info(
                "%s fold_%s: train=%s val=%s test=%s subjects",
                disease,
                fold_index,
                len(subject_splits["train"]),
                len(subject_splits["val"]),
                len(subject_splits["test"]),
            )
        all_summary["diseases"][disease] = {"overall": summarize_downstream_records(records), "folds": disease_summaries}
    write_json(split_out / "split_summary_all.json", all_summary)
    return all_summary


def make_downstream_splits(
    cfg: dict[str, Any],
    *,
    diseases: list[str] | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    strategy: str | None = None,
    num_folds: int | None = None,
) -> dict[str, Any]:
    split_cfg = cfg.get("downstream_split", {})
    split_strategy = strategy or split_cfg.get("strategy", "subject_stratified")
    if split_strategy == "subject_stratified_kfold":
        return make_downstream_kfold_splits(
            cfg,
            diseases=diseases,
            out_dir=out_dir,
            seed=seed,
            num_folds=num_folds,
        )
    if split_strategy not in {"subject_stratified", "subject_stratified_by_label"}:
        raise ValueError(f"Unknown downstream split strategy: {split_strategy}")
    return _make_downstream_holdout_splits(
        cfg,
        diseases=diseases,
        out_dir=out_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--disease", action="append", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_downstream_config(args.config)
    summary = make_downstream_splits(
        cfg,
        diseases=args.disease,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        strategy=args.strategy,
        num_folds=args.num_folds,
    )
    print(summary)


if __name__ == "__main__":
    main()
