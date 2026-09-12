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


def _finite_support_weighted_mean(
    values: list[float], supports: list[int]
) -> float:
    """Average finite per-class metrics using true-label subject support.

    Classes absent from a split have zero support and are ignored.  Keeping
    this separate from the macro average makes the evaluation policy explicit
    and prevents historical macro metrics from silently changing meaning.
    """
    numerator = 0.0
    denominator = 0
    for value, support in zip(values, supports):
        if support > 0 and math.isfinite(value):
            numerator += float(support) * value
            denominator += int(support)
    return numerator / float(denominator) if denominator > 0 else math.nan


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
    class_bce = [0.0, 0.0]
    class_brier = [0.0, 0.0]
    class_count = [0, 0]
    for label, prob in zip(label_list, prob_list):
        p = min(1.0 - eps, max(eps, prob))
        sample_bce = -(label * math.log(p) + (1 - label) * math.log(1.0 - p))
        sample_brier = (prob - label) ** 2
        bce += sample_bce
        brier += sample_brier
        class_bce[label] += sample_bce
        class_brier[label] += sample_brier
        class_count[label] += 1
    balanced_bce = (
        0.5
        * sum(class_bce[class_id] / class_count[class_id] for class_id in (0, 1))
        if all(class_count)
        else math.nan
    )
    balanced_brier = (
        0.5
        * sum(class_brier[class_id] / class_count[class_id] for class_id in (0, 1))
        if all(class_count)
        else math.nan
    )
    metrics = {
        "n": float(n),
        "loss_bce": bce / n if n > 0 else math.nan,
        "brier": brier / n if n > 0 else math.nan,
        "balanced_loss_bce": balanced_bce,
        "balanced_brier": balanced_brier,
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
    predictions: Iterable[int] | None = None,
) -> dict[str, float]:
    label_list = [int(label) for label in labels]
    logit_rows = [[float(v) for v in row] for row in logits]
    if len(label_list) != len(logit_rows):
        raise ValueError("labels and logits must have the same length")
    probs = [softmax(row) for row in logit_rows]
    preds = (
        [int(value) for value in predictions]
        if predictions is not None
        else [max(range(num_classes), key=lambda c: prob[c]) for prob in probs]
    )
    if len(preds) != len(label_list):
        raise ValueError("predictions and labels must have the same length")
    matrix = multiclass_confusion_matrix(label_list, preds, num_classes)
    supports = [sum(row) for row in matrix]
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
        # This requested support-weighted form is mathematically identical to
        # ordinary multiclass accuracy.  Emit both names so reports can state
        # that equivalence without relabelling the historical accuracy field.
        "weighted_balanced_accuracy": _finite_support_weighted_mean(
            recalls, supports
        ),
        "macro_f1": finite_mean(f1s),
        "weighted_f1": _weighted_f1_from_confusion(matrix, f1s),
        "cohen_kappa": _cohen_kappa_from_confusion(matrix),
        "macro_auroc_ovr": finite_mean(per_class_auroc),
        "weighted_auroc_ovr": _finite_support_weighted_mean(
            per_class_auroc, supports
        ),
        "macro_auprc_ovr": finite_mean(per_class_auprc),
        "weighted_auprc_ovr": _finite_support_weighted_mean(
            per_class_auprc, supports
        ),
        "num_skipped_auc_classes": float(len(skipped)),
    }
    for c in range(num_classes):
        metrics[f"class_{c}_auroc_ovr"] = per_class_auroc[c]
        metrics[f"class_{c}_auprc_ovr"] = per_class_auprc[c]
        metrics[f"class_{c}_f1"] = f1s[c]
        metrics[f"class_{c}_recall"] = recalls[c]
        metrics[f"class_{c}_support"] = float(supports[c])
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


def aggregate_subject_by_task(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group trials by subject and task_id, mean-pool per task → T task-logits.

    Returns subject rows with logit_0..logit_{T-1}, where T = number of task types.
    """
    all_tasks = sorted({int(row["task_id"]) for row in rows})
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, int] = {}
    for row in rows:
        key = str(row["subject_key"])
        grouped[key][int(row["task_id"])].append(float(row["logit"]))
        labels[key] = int(row["label"])

    out = []
    for subject_key in sorted(grouped):
        task_logits = []
        for tid in all_tasks:
            vals = grouped[subject_key].get(tid, [0.0])
            task_logits.append(sum(vals) / len(vals))
        r = {
            "base_subject_id": subject_key,
            "subject_key": subject_key,
            "label": labels[subject_key],
            "num_trials": sum(len(v) for v in grouped[subject_key].values()),
        }
        for c, v in enumerate(task_logits):
            r[f"logit_{c}"] = v
        out.append(r)
    return out


def aggregate_subject_by_task_multiclass(
    rows: list[dict[str, Any]],
    num_classes: int,
) -> list[dict[str, Any]]:
    """Multiclass variant: per-task mean logits → T*C task-logits per subject."""
    all_tasks = sorted({int(row["task_id"]) for row in rows})
    grouped: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, int] = {}
    for row in rows:
        key = str(row["subject_key"])
        trial_logits = [float(row[f"logit_{c}"]) for c in range(num_classes)]
        grouped[key][int(row["task_id"])].append(trial_logits)
        labels[key] = int(row["label"])

    out = []
    for subject_key in sorted(grouped):
        T = len(all_tasks)
        task_means = []
        for tid in all_tasks:
            vals = grouped[subject_key].get(tid, [[0.0] * num_classes])
            mean = [sum(v[c] for v in vals) / len(vals) for c in range(num_classes)]
            task_means.append(mean)
        r = {
            "base_subject_id": subject_key,
            "subject_key": subject_key,
            "label": labels[subject_key],
            "num_trials": sum(len(v) for v in grouped[subject_key].values()),
        }
        idx = 0
        for t in range(T):
            for c in range(num_classes):
                r[f"logit_{idx}"] = task_means[t][c]
                idx += 1
        out.append(r)
    return out


def write_prediction_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_calibration_binary(
    subject_rows: list[dict[str, Any]],
    lr: float = 0.01,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Fit w,b for binary logit calibration. Returns (w, b)."""
    import torch
    from torch import nn

    device = torch.device("cpu")
    logits_t = torch.tensor([[float(r["logit"])] for r in subject_rows], device=device)
    labels_t = torch.tensor([int(r["label"]) for r in subject_rows], device=device, dtype=torch.float32)
    w = nn.Parameter(torch.ones(1, device=device))
    b = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        cal = logits_t.squeeze(-1) * w + b
        loss = nn.functional.binary_cross_entropy_with_logits(cal, labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return w.item(), b.item()


def _fit_calibration_linear(
    subject_rows: list[dict[str, Any]],
    num_classes: int,
    lr: float = 0.01,
    max_iter: int = 200,
    bias_only: bool = False,
    weight_decay: float = 0.0,
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Fit Linear(in_dim, out_dim) on subject_rows with per-task logits.

    If bias_only=True, fix W=1/in_dim (equal-weight mean) and only learn b.
    weight_decay adds L2 penalty to W (not b).
    """
    import torch
    from torch import nn

    if not subject_rows:
        return torch.eye(1), torch.zeros(1), 1

    device = torch.device("cpu")
    in_dim = sum(1 for k in subject_rows[0] if k.startswith("logit_"))
    out_dim = 1 if num_classes <= 2 else num_classes - 1

    X = torch.tensor([[float(r[f"logit_{c}"]) for c in range(in_dim)] for r in subject_rows], device=device)
    if out_dim == 1:
        y = torch.tensor([int(r["label"]) for r in subject_rows], device=device, dtype=torch.float32)
    else:
        y = torch.tensor([int(r["label"]) for r in subject_rows], device=device, dtype=torch.long)

    if bias_only:
        W_fixed = torch.ones(out_dim, in_dim, device=device) / in_dim
        b = nn.Parameter(torch.zeros(out_dim, device=device))
        opt = torch.optim.LBFGS([b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            out = X @ W_fixed.T + b
            if out_dim == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y)
            else:
                loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            return loss
        opt.step(closure)
        return W_fixed, b.detach(), in_dim

    W = nn.Parameter(torch.randn(out_dim, in_dim, device=device) * 0.01)
    b = nn.Parameter(torch.zeros(out_dim, device=device))
    opt = torch.optim.LBFGS([W, b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        out = X @ W.T + b
        if out_dim == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y)
        else:
            loss = nn.functional.cross_entropy(out, y)
        if weight_decay > 0:
            loss = loss + weight_decay * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach(), in_dim


def _apply_calibration_linear(
    subject_rows: list[dict[str, Any]],
    W: "torch.Tensor",
    b: "torch.Tensor",
    in_dim: int,
    num_classes: int,
) -> list[dict[str, Any]]:
    """Apply Linear(in_dim, out_dim) calibration to subject rows."""
    import torch

    out_dim = W.shape[0]
    out = []
    for r in subject_rows:
        x = torch.tensor([float(r[f"logit_{c}"]) for c in range(in_dim)])
        cal = (x @ W.T + b).tolist()
        rr = dict(r)
        if out_dim == 1:
            rr["logit"] = float(cal[0]) if isinstance(cal, list) else float(cal)
            rr["prob"] = sigmoid(rr["logit"])
        else:
            probs = softmax(cal)
            for c in range(out_dim):
                rr[f"logit_{c}"] = float(cal[c])
                rr[f"prob_{c}"] = float(probs[c])
            rr["pred"] = int(max(range(out_dim), key=lambda c: probs[c]))
        out.append(rr)
    return out


def _fit_calibration_multiclass(
    subject_rows: list[dict[str, Any]],
    num_classes: int,
    lr: float = 0.01,
    max_iter: int = 200,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Fit W,b for multiclass logit calibration. Returns (W, b) tensors."""
    import torch
    from torch import nn

    device = torch.device("cpu")
    logits_t = torch.tensor([[float(row[f"logit_{c}"]) for c in range(num_classes)] for row in subject_rows], device=device)
    labels_t = torch.tensor([int(row["label"]) for row in subject_rows], device=device, dtype=torch.long)
    W = nn.Parameter(torch.eye(num_classes, device=device))
    b = nn.Parameter(torch.zeros(num_classes, device=device))
    opt = torch.optim.LBFGS([W, b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        cal = logits_t @ W.T + b
        loss = nn.functional.cross_entropy(cal, labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def _apply_calibration_binary(
    subject_rows: list[dict[str, Any]], w: float, b: float
) -> list[dict[str, Any]]:
    """Apply learned w,b to binary subject rows."""
    out = []
    for row in subject_rows:
        cal_logit = float(row["logit"]) * w + b
        r = dict(row)
        r["logit"] = cal_logit
        r["prob"] = sigmoid(cal_logit)
        out.append(r)
    return out


def _apply_calibration_multiclass(
    subject_rows: list[dict[str, Any]], W: "torch.Tensor", b: "torch.Tensor", num_classes: int
) -> list[dict[str, Any]]:
    """Apply learned W,b to multiclass subject rows."""
    out = []
    for row in subject_rows:
        raw = [float(row[f"logit_{c}"]) for c in range(num_classes)]
        cal = (W @ torch.tensor(raw) + b).tolist()
        probs = softmax(cal)
        r = dict(row)
        for c in range(num_classes):
            r[f"logit_{c}"] = float(cal[c])
            r[f"prob_{c}"] = float(probs[c])
        r["pred"] = int(max(range(num_classes), key=lambda c: probs[c]))
        out.append(r)
    return out


def calibrate_subject_logits(
    subject_rows: list[dict[str, Any]],
    num_classes: int,
    *,
    lr: float = 0.01,
    max_iter: int = 200,
) -> list[dict[str, Any]]:
    """Fit AND apply linear calibration (convenience, use fit+apply separately for proper val/test split)."""
    if not subject_rows:
        return subject_rows
    if num_classes == 1:
        w, b = _fit_calibration_binary(subject_rows, lr, max_iter)
        return _apply_calibration_binary(subject_rows, w, b)
    W, b = _fit_calibration_multiclass(subject_rows, num_classes, lr, max_iter)
    return _apply_calibration_multiclass(subject_rows, W, b, num_classes)
