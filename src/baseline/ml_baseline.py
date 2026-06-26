"""EyeFM ML baseline: LR / RF / SVM with 30-combo val grid search.

Protocol:
  1. For each hp combo in 30-combo grid: fit on train, score on val (AUROC)
  2. Pick best hp per model on val
  3. Refit best per-model on train, evaluate on test (extended metrics)

Reference:  docs/eyemae_baseline.md
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LOGGER = logging.getLogger(__name__)


# ==== Grid search space (30 combos) ====
def build_grid() -> list[dict[str, Any]]:
    """Return 30-combo grid: LR(8) + RF(16) + SVM(6)."""
    grid: list[dict[str, Any]] = []
    for c in (0.01, 0.1, 1, 10):
        for balanced in (True, False):
            grid.append({"model": "LR", "C": c, "balanced": balanced})
    for n_est in (100, 300):
        for max_d in (None, 20):
            for min_leaf in (1, 5):
                for balanced in (True, False):
                    grid.append({"model": "RF", "n_est": n_est, "max_d": max_d, "min_leaf": min_leaf, "balanced": balanced})
    for c in (0.1, 1, 10):
        for balanced in (True, False):
            grid.append({"model": "SVM", "C": c, "balanced": balanced})
    return grid


def make_classifier(name: str, n_classes: int, **hp: Any):
    """Build a sklearn classifier from hp dict (without 'model' key)."""
    cw = "balanced" if hp.get("balanced", True) else None
    if name == "LR":
        solver = "lbfgs" if n_classes > 2 else "lbfgs"
        return LogisticRegression(solver=solver, max_iter=1000, C=float(hp["C"]), class_weight=cw)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=int(hp["n_est"]),
            max_depth=hp["max_d"],
            min_samples_leaf=int(hp["min_leaf"]),
            class_weight=cw,
            n_jobs=-1,
            random_state=42,
        )
    if name == "SVM":
        base = LinearSVC(C=float(hp["C"]), class_weight=cw, max_iter=2000, dual="auto")
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(f"Unknown model: {name}")


def _proba_from(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return (n, n_classes) probability matrix from a fitted classifier."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    scores = clf.decision_function(X)
    if n_classes == 2:
        return np.column_stack([1 - scores, scores])
    return scores


# ==== Metrics ====
def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap CI for AUROC (binary)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: list[float] = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    """Extended metrics: AUROC + 95% CI + Bal Acc + F1-macro + Cohen Kappa + Sensitivity + Specificity + AUC-MR."""
    y_pred = probs.argmax(axis=1)
    out: dict[str, float] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if n_classes == 2:
        try:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            out["auroc"] = float("nan")
        lo, hi = bootstrap_auc_ci(y_true, probs[:, 1])
        out["auroc_ci_low"] = lo
        out["auroc_ci_high"] = hi
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0
        out["sensitivity"] = float(tp / max(tp + fn, 1))
        out["specificity"] = float(tn / max(tn + fp, 1))
        try:
            out["auc_mr"] = float((out["sensitivity"] + out["specificity"]) / 2)
        except Exception:
            out["auc_mr"] = float("nan")
    else:
        try:
            out["auroc_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            out["auroc_macro"] = float("nan")
        out["auroc"] = out["auroc_macro"]
        # binary sens/spec only meaningful for binary; leave NaN for 5-class
        out["sensitivity"] = float("nan")
        out["specificity"] = float("nan")
        out["auc_mr"] = float("nan")
    return out


# ==== Pipeline ====
@dataclass
class SplitBundle:
    """X / y for one split, after impute + standardize."""
    X: np.ndarray
    y: np.ndarray


def impute_and_standardize(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> tuple[SplitBundle, SplitBundle, SplitBundle]:
    """Impute NaN with train column mean; StandardScaler fit on train only."""
    col_mean = np.nanmean(X_tr, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)

    def impute(X: np.ndarray) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_mean[j]
        return X

    Xtr = impute(X_tr)
    Xva = impute(X_va)
    Xte = impute(X_te)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    return (
        SplitBundle(Xtr_s, y_tr),
        SplitBundle(Xva_s, y_va),
        SplitBundle(Xte_s, y_te),
    )


@dataclass
class RunResult:
    model: str
    hp_str: str
    val_score: float
    train_time_sec: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def run_grid_search(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_classes: int,
    grid: list[dict[str, Any]] | None = None,
) -> list[RunResult]:
    """Run val grid search → best hp per model → refit → test final."""
    if grid is None:
        grid = build_grid()
    n_models = len({hp["model"] for hp in grid})
    LOGGER.info("Grid: %d combos across %d models", len(grid), n_models)

    # Fit per grid combo, score on val
    best_per_model: dict[str, dict[str, Any]] = {}
    for i, hp in enumerate(grid):
        name = hp["model"]
        hp_clean = {k: v for k, v in hp.items() if k != "model"}
        try:
            clf = make_classifier(name, n_classes, **hp_clean)
            clf.fit(X_tr, y_tr)
            pva = _proba_from(clf, X_va, n_classes)
            va_metrics = compute_metrics(y_va, pva, n_classes)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("grid[%d/%d] FAIL %s: %s", i + 1, len(grid), hp, e)
            continue
        score = va_metrics.get("auroc", va_metrics.get("auroc_macro", float("-inf")))
        cur = best_per_model.get(name)
        if cur is None or score > cur["score"]:
            best_per_model[name] = {
                "hp": hp,
                "score": score,
                "val_metrics": va_metrics,
            }

    # Refit best per model on train, evaluate on test
    results: list[RunResult] = []
    for name, info in best_per_model.items():
        hp = info["hp"]
        hp_clean = {k: v for k, v in hp.items() if k != "model"}
        clf = make_classifier(name, n_classes, **hp_clean)
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        train_time = time.time() - t0
        pte = _proba_from(clf, X_te, n_classes)
        te_metrics = compute_metrics(y_te, pte, n_classes)
        hp_str = name + "|" + "|".join(f"{k}={v}" for k, v in hp.items() if k != "model")
        results.append(RunResult(
            model=name,
            hp_str=hp_str,
            val_score=float(info["score"]),
            train_time_sec=round(train_time, 1),
            val_metrics=info["val_metrics"],
            test_metrics=te_metrics,
        ))
        LOGGER.info(
            "%s: val=%.4f test_auroc=%.4f test_bal_acc=%.4f (%s)",
            name, info["score"],
            te_metrics.get("auroc", te_metrics.get("auroc_macro", float("nan"))),
            te_metrics.get("balanced_accuracy", float("nan")),
            hp_str,
        )
    return results


CSV_FIELDS = (
    "model", "hp", "n_classes", "train_time_sec", "val_score", "n",
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa",
    "auroc", "auroc_macro", "auroc_ci_low", "auroc_ci_high",
    "sensitivity", "specificity", "auc_mr",
)


def write_test_csv(path: Path, results: list[RunResult]) -> None:
    """Write per-model best-on-val final test csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            row = {
                "model": r.model,
                "hp": r.hp_str,
                "n_classes": r.test_metrics.get("n_classes", 0) or len({0, 1}),  # placeholder
                "train_time_sec": r.train_time_sec,
                "val_score": r.val_score,
                **{k: r.test_metrics.get(k, float("nan")) for k in CSV_FIELDS if k not in {"model", "hp", "train_time_sec", "val_score"}},
            }
            w.writerow(row)
    LOGGER.info("Wrote %s", path)
