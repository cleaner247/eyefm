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


def _cohen_kappa_from_confusion(matrix: list[list[int]]) -> float:
    n = sum(sum(row) for row in matrix)
    if n <= 0:
        return math.nan
    observed = sum(matrix[i][i] for i in range(len(matrix))) / float(n)
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[row][col] for row in range(len(matrix))) for col in range(len(matrix))]
    expected = sum(row_total * col_total for row_total, col_total in zip(row_totals, col_totals)) / float(n * n)
    return _safe_div(observed - expected, 1.0 - expected)


def _per_class_f1_from_confusion(matrix: list[list[int]]) -> list[float]:
    num_classes = len(matrix)
    f1s: list[float] = []
    for c in range(num_classes):
        tp = matrix[c][c]
        fn = sum(matrix[c][j] for j in range(num_classes) if j != c)
        fp = sum(matrix[i][c] for i in range(num_classes) if i != c)
        f1s.append(_safe_div(2 * tp, 2 * tp + fp + fn))
    return f1s


def _weighted_f1_from_confusion(matrix: list[list[int]], f1s: list[float]) -> float:
    supports = [sum(row) for row in matrix]
    total = sum(supports)
    if total <= 0:
        return math.nan
    numerator = 0.0
    for support, f1 in zip(supports, f1s):
        if support > 0 and math.isfinite(f1):
            numerator += support * f1
    return numerator / float(total)


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
    matrix = [[tn, fp], [fn, tp]]
    class_f1s = _per_class_f1_from_confusion(matrix)
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
        "weighted_f1": _weighted_f1_from_confusion(matrix, class_f1s),
        "cohen_kappa": _cohen_kappa_from_confusion(matrix),
        "auroc": binary_auroc(label_list, prob_list),
        "auprc": binary_average_precision(label_list, prob_list),
        "class_0_f1": class_f1s[0],
        "class_1_f1": class_f1s[1],
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


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    den = sum(exps)
    return [v / den for v in exps]


def multiclass_confusion_matrix(labels: Iterable[int], preds: Iterable[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        matrix[int(label)][int(pred)] += 1
    return matrix


def compute_multiclass_metrics(
    labels: Iterable[int | float],
    logits: Iterable[Iterable[float]],
    *,
    num_classes: int,
    prefix: str = "",
) -> dict[str, float]:
    label_list = [int(label) for label in labels]
    logit_rows = [[float(v) for v in row] for row in logits]
    if len(label_list) != len(logit_rows):
        raise ValueError("labels and logits must have the same length")
    probs = [softmax(row) for row in logit_rows]
    preds = [max(range(num_classes), key=lambda c: prob[c]) for prob in probs]
    matrix = multiclass_confusion_matrix(label_list, preds, num_classes)
    n = len(label_list)
    accuracy = sum(1 for y, pred in zip(label_list, preds) if y == pred) / n if n > 0 else math.nan
    recalls: list[float] = []
    f1s = _per_class_f1_from_confusion(matrix)
    per_class_auroc: list[float] = []
    per_class_auprc: list[float] = []
    skipped: list[int] = []
    for c in range(num_classes):
        tp = matrix[c][c]
        fn = sum(matrix[c][j] for j in range(num_classes) if j != c)
        fp = sum(matrix[i][c] for i in range(num_classes) if i != c)
        recall = _safe_div(tp, tp + fn)
        recalls.append(recall)
        one_vs_rest = [1 if label == c else 0 for label in label_list]
        scores = [prob[c] for prob in probs]
        auc = binary_auroc(one_vs_rest, scores)
        auprc = binary_average_precision(one_vs_rest, scores)
        if not math.isfinite(auc):
            skipped.append(c)
        per_class_auroc.append(auc)
        per_class_auprc.append(auprc)

    def finite_mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else math.nan

    metrics: dict[str, float] = {
        "n": float(n),
        "accuracy": float(accuracy),
        "balanced_accuracy": finite_mean(recalls),
        "macro_f1": finite_mean(f1s),
        "weighted_f1": _weighted_f1_from_confusion(matrix, f1s),
        "cohen_kappa": _cohen_kappa_from_confusion(matrix),
        "macro_auroc_ovr": finite_mean(per_class_auroc),
        "macro_auprc_ovr": finite_mean(per_class_auprc),
        "num_skipped_auc_classes": float(len(skipped)),
    }
    for c in range(num_classes):
        metrics[f"class_{c}_auroc_ovr"] = per_class_auroc[c]
        metrics[f"class_{c}_auprc_ovr"] = per_class_auprc[c]
        metrics[f"class_{c}_f1"] = f1s[c]
    if prefix:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}
    return metrics


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


def aggregate_subject_predictions_multiclass(rows: list[dict[str, Any]], num_classes: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_key"])].append(row)
    out: list[dict[str, Any]] = []
    for subject_key in sorted(grouped):
        group = grouped[subject_key]
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"Subject has inconsistent downstream labels: {subject_key}")
        mean_logits = [
            sum(float(row[f"logit_{c}"]) for row in group) / float(len(group))
            for c in range(num_classes)
        ]
        probs = softmax(mean_logits)
        payload: dict[str, Any] = {
            "ml_subject_id": subject_key,
            "base_subject_id": subject_key,
            "subject_key": subject_key,
            "label": int(next(iter(labels))),
            "pred": int(max(range(num_classes), key=lambda c: probs[c])),
            "num_trials": len(group),
            "disease": group[0].get("disease", ""),
            "group": group[0].get("group", ""),
        }
        for c in range(num_classes):
            payload[f"logit_{c}"] = mean_logits[c]
            payload[f"prob_{c}"] = probs[c]
        out.append(payload)
    return out


def write_prediction_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
