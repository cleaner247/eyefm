#!/usr/bin/env python3
"""Select Stage C from validation predictions only and bootstrap the delta."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from eyemae.utils import write_json


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _val_auc(run: Path) -> float:
    payload = _read_json(run / "metrics_val_best.json")
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"Stage C selection input is not validation-only: {run}")
    return float(payload["best_val_auroc"])


def _predictions(run: Path) -> dict[str, tuple[int, float]]:
    rows = {}
    with (run / "predictions_val_best.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["subject_key"])] = (int(row["label"]), float(row["logit"]))
    return rows


def _ensemble(runs: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    sources = [_predictions(run) for run in runs]
    subjects = sorted(sources[0])
    if any(sorted(source) != subjects for source in sources[1:]):
        raise ValueError("Validation subjects differ across Stage C seeds")
    labels = np.asarray([sources[0][subject][0] for subject in subjects])
    if any(
        source[subject][0] != labels[index]
        for source in sources[1:]
        for index, subject in enumerate(subjects)
    ):
        raise ValueError("Validation labels differ across Stage C seeds")
    logits = np.mean([
        [source[subject][1] for subject in subjects] for source in sources
    ], axis=0)
    return subjects, labels, logits


def _paired_bootstrap(
    labels: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20260824,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        positions = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[positions]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(
            roc_auc_score(sampled_labels, candidate_logits[positions])
            - roc_auc_score(sampled_labels, baseline_logits[positions])
        )
    values = np.asarray(deltas)
    return {
        "samples": int(len(values)),
        "mean_delta": float(values.mean()),
        "ci_2p5": float(np.quantile(values, 0.025)),
        "ci_97p5": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float((values > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs=3, required=True)
    parser.add_argument("--candidate", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = [Path(path) for path in args.baseline]
    candidate = [Path(path) for path in args.candidate]
    baseline_auc = np.asarray([_val_auc(path) for path in baseline])
    candidate_auc = np.asarray([_val_auc(path) for path in candidate])
    baseline_subjects, baseline_labels, baseline_logits = _ensemble(baseline)
    candidate_subjects, candidate_labels, candidate_logits = _ensemble(candidate)
    if baseline_subjects != candidate_subjects or not np.array_equal(
        baseline_labels, candidate_labels
    ):
        raise ValueError("Baseline and candidate validation cohorts differ")
    delta = float(candidate_auc.mean() - baseline_auc.mean())
    baseline_std = float(baseline_auc.std(ddof=1))
    candidate_std = float(candidate_auc.std(ddof=1))
    stability_gain = (
        candidate_std <= 0.8 * baseline_std if baseline_std > 0 else False
    )
    bootstrap = _paired_bootstrap(
        baseline_labels, baseline_logits, candidate_logits
    )
    bootstrap_supports_gain = bootstrap["ci_2p5"] > 0.0
    candidate_passes = bootstrap_supports_gain or (
        delta >= 0.005 and stability_gain
    )
    selected = "4xk4_mean" if candidate_passes else "k4_baseline"
    result = {
        "selection_source": "validation_only_three_seed_mean",
        "test_used_for_selection": False,
        "baseline": {
            "runs": list(map(str, baseline)),
            "seed_auroc": baseline_auc.tolist(),
            "mean_auroc": float(baseline_auc.mean()),
            "std_auroc": baseline_std,
            "ensemble_auroc": float(roc_auc_score(baseline_labels, baseline_logits)),
        },
        "candidate": {
            "runs": list(map(str, candidate)),
            "seed_auroc": candidate_auc.tolist(),
            "mean_auroc": float(candidate_auc.mean()),
            "std_auroc": candidate_std,
            "ensemble_auroc": float(roc_auc_score(candidate_labels, candidate_logits)),
        },
        "candidate_minus_baseline_mean_auroc": delta,
        "candidate_stability_gain_20pct": bool(stability_gain),
        "paired_subject_bootstrap": bootstrap,
        "bootstrap_supports_gain": bool(bootstrap_supports_gain),
        "selection_rule": (
            "paired bootstrap lower 95% bound > 0, or mean delta>=0.005 "
            "with >=20% seed-std reduction"
        ),
        "selected_variant": selected,
        "selected_runs": list(map(str, candidate if candidate_passes else baseline)),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
