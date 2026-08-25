#!/usr/bin/env python3
"""Train-only frozen-CLS proxy for ranking BERT checkpoints.

The proxy deliberately excludes demographics and never opens Val/Test.  Trial
CLS vectors are normalized exactly before subject aggregation, averaged within
task and then equally across the four tasks.  A fixed regularized linear model
is evaluated with subject-stratified folds shared by every checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eyemae.utils import write_json


def subject_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    if raw.get("selection_source") != "train_only_subject_folds":
        raise ValueError("Cache is not authorized for Train-only screening")
    split = raw.get("splits", {}).get("train")
    if split is None:
        raise ValueError("Cache has no Train split")
    features = F.layer_norm(split["features"].float(), (split["features"].shape[-1],))
    labels = split["labels"].long()
    tasks = split["task_ids"].long()
    subjects = list(map(str, split["subject_keys"]))
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    subject_label: dict[str, int] = {}
    for index, (subject, task, label) in enumerate(
        zip(subjects, tasks.tolist(), labels.tolist(), strict=True)
    ):
        if subject in subject_label and subject_label[subject] != int(label):
            raise ValueError(f"Inconsistent labels for subject {subject}")
        subject_label[subject] = int(label)
        grouped[subject][int(task)].append(index)
    ordered = sorted(subject_label)
    output = []
    output_labels = []
    for subject in ordered:
        if set(grouped[subject]) != {0, 1, 2, 3}:
            raise ValueError(f"Subject {subject} does not contain all four tasks")
        task_means = [features[grouped[subject][task]].mean(dim=0) for task in range(4)]
        output.append(torch.stack(task_means).mean(dim=0))
        output_labels.append(subject_label[subject])
    return (
        torch.stack(output).numpy(),
        np.asarray(output_labels, dtype=np.int64),
        ordered,
        str(raw["bert_sha256"]),
    )


def screen(
    cache_path: Path, *, folds: int = 5, seed: int = 42, c_value: float = 0.1
) -> dict[str, Any]:
    x, y, subjects, bert_sha = subject_features(cache_path)
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if len(nonzero) < 2 or int(nonzero.min()) < folds:
        raise ValueError(f"Every class needs at least {folds} subjects; counts={counts.tolist()}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros((len(y), len(counts)), dtype=np.float64)
    predictions = np.full(len(y), -1, dtype=np.int64)
    fold_rows = []
    for fold, (train_index, held_index) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        model.fit(x[train_index], y[train_index])
        fold_probability = model.predict_proba(x[held_index])
        classes = model[-1].classes_.astype(int)
        probabilities[np.ix_(held_index, classes)] = fold_probability
        predictions[held_index] = model.predict(x[held_index])
        fold_score = (
            roc_auc_score(y[held_index], fold_probability[:, 1])
            if len(counts) == 2
            else roc_auc_score(
                y[held_index], fold_probability, multi_class="ovr", average="macro",
                labels=classes,
            )
        )
        fold_rows.append({"fold": fold, "auroc": float(fold_score), "subjects": len(held_index)})
    if len(counts) == 2:
        auroc = roc_auc_score(y, probabilities[:, 1])
    else:
        auroc = roc_auc_score(
            y, probabilities, multi_class="ovr", average="macro",
            labels=np.arange(len(counts)),
        )
    return {
        "selection_source": "train_subject_stratified_folds_only",
        "validation_used_for_selection": False,
        "test_used_for_selection": False,
        "cache_path": str(cache_path.resolve()),
        "bert_sha256": bert_sha,
        "subjects": len(subjects),
        "class_counts": {str(i): int(value) for i, value in enumerate(counts)},
        "feature_definition": "LayerNorm(CLS), mean trials within task, equal mean over four tasks",
        "demographics_used": False,
        "classifier": {"type": "logistic_regression", "C": c_value, "class_weight": "balanced"},
        "folds": folds,
        "fold_seed": seed,
        "fold_auroc_mean": float(np.mean([row["auroc"] for row in fold_rows])),
        "fold_auroc_std": float(np.std([row["auroc"] for row in fold_rows], ddof=1)),
        "oof_auroc": float(auroc),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "fold_results": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=0.1)
    args = parser.parse_args()
    result = screen(
        Path(args.cache), folds=int(args.folds), seed=int(args.seed), c_value=float(args.c)
    )
    write_json(Path(args.output), result)
    print(result)


if __name__ == "__main__":
    main()
