from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_auroc(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def binary_average_precision(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(1 for label in labels if label == 1)
    if n_pos == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hits += 1
            precision_sum += hits / float(rank)
    return precision_sum / float(n_pos)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def compute_binary_metrics(
    labels: Iterable[int | float],
    logits: Iterable[float],
    *,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, float]:
    label_list = [int(label) for label in labels]
    logit_list = [float(logit) for logit in logits]
    prob_list = [sigmoid(logit) for logit in logit_list]
    if len(label_list) != len(logit_list):
        raise ValueError("labels and logits must have the same length")
    preds = [1 if prob >= threshold else 0 for prob in prob_list]
    tp = sum(1 for y, pred in zip(label_list, preds) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(label_list, preds) if y == 0 and pred == 0)
    fp = sum(1 for y, pred in zip(label_list, preds) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(label_list, preds) if y == 1 and pred == 0)
    n = len(label_list)
    eps = 1e-12
    bce = 0.0
    brier = 0.0
    for label, prob in zip(label_list, prob_list):
        p = min(1.0 - eps, max(eps, prob))
        bce += -(label * math.log(p) + (1 - label) * math.log(1.0 - p))
        brier += (prob - label) ** 2
    metrics = {
        "n": float(n),
        "loss_bce": bce / n if n > 0 else math.nan,
        "brier": brier / n if n > 0 else math.nan,
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp)),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "auroc": binary_auroc(label_list, prob_list),
        "auprc": binary_average_precision(label_list, prob_list),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }
    if prefix:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}
    return metrics


def binary_confusion_matrix(
    labels: Iterable[int | float],
    logits: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, int]:
    label_list = [int(label) for label in labels]
    pred_list = [1 if sigmoid(float(logit)) >= threshold else 0 for logit in logits]
    return {
        "tn": sum(1 for y, pred in zip(label_list, pred_list) if y == 0 and pred == 0),
        "fp": sum(1 for y, pred in zip(label_list, pred_list) if y == 0 and pred == 1),
        "fn": sum(1 for y, pred in zip(label_list, pred_list) if y == 1 and pred == 0),
        "tp": sum(1 for y, pred in zip(label_list, pred_list) if y == 1 and pred == 1),
        "threshold": threshold,
    }


def aggregate_subject_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_key"])].append(row)
    out: list[dict[str, Any]] = []
    for subject_key in sorted(grouped):
        group = grouped[subject_key]
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"Subject has inconsistent downstream labels: {subject_key}")
        mean_logit = sum(float(row["logit"]) for row in group) / float(len(group))
        out.append(
            {
                "base_subject_id": subject_key,
                "subject_key": subject_key,
                "label": int(next(iter(labels))),
                "logit": mean_logit,
                "prob": sigmoid(mean_logit),
                "num_trials": len(group),
                "disease": group[0].get("disease", ""),
                "group": group[0].get("group", ""),
            }
        )
    return out


def write_prediction_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
