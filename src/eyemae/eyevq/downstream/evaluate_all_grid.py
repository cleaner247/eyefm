"""Evaluate every completed validation-only Subject-MIL grid run on test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from eyemae.eyevq.downstream.evaluate_mil import evaluate_checkpoint
from eyemae.eyevq.downstream.finalize_grid import select_best_run
from eyemae.utils import write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_summary_row(
    result_path: Path,
    validation: dict[str, Any],
    test_result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one run's validation and test results for cross-run comparison."""
    cfg = validation["cfg"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    frozen_layers = int(model_cfg["freeze_bottom_layers"])
    embedding_frozen = bool(model_cfg.get("freeze_embedding", True))
    fixed = test_result.get("test", test_result.get("test_default_05"))
    if fixed is None:
        fixed = test_result["test_tuned"]
    default = test_result.get("test_default_05", fixed)
    decision_threshold = float(test_result.get("decision_threshold", 0.5))
    # Some historical fixtures/results omitted AUROC from the duplicate 0.5
    # metric block because AUROC is threshold independent.
    auroc = fixed.get("test/subject/auroc")
    if auroc is None:
        auroc = test_result["test_tuned"]["test/subject/auroc"]
    return {
        "run": result_path.parent.name,
        "unfrozen_transformer_layers": 12 - frozen_layers,
        "embedding_unfrozen": not embedding_frozen,
        "full_finetune": frozen_layers == 0 and not embedding_frozen,
        "encoder_lr": float(train_cfg["encoder_lr"]),
        "head_lr": float(train_cfg["head_lr"]),
        "best_epoch": validation.get("best_epoch"),
        "best_step": int(validation.get("best_step", -1)),
        "best_smoothed_val_auroc": validation.get("best_smoothed_val_auroc"),
        "best_val_auroc": float(validation["best_val_auroc"]),
        "decision_threshold": decision_threshold,
        "test_auroc": float(auroc),
        "test_balanced_accuracy": float(
            fixed["test/subject/balanced_accuracy"]
        ),
        "test_f1": float(fixed["test/subject/f1"]),
        "test_accuracy": float(fixed["test/subject/accuracy"]),
        "test_balanced_accuracy_at_05": float(
            default["test/subject/balanced_accuracy"]
        ),
        "test_f1_at_05": float(default["test/subject/f1"]),
        "test_accuracy_at_05": float(default["test/subject/accuracy"]),
    }


def evaluate_all_runs(root: Path, expected_runs: int) -> list[dict[str, Any]]:
    result_paths = sorted(root.glob("*/metrics_val_best.json"))
    if len(result_paths) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} completed validation runs, "
            f"found {len(result_paths)}"
        )

    best_path, best_validation = select_best_run(result_paths)
    rows: list[dict[str, Any]] = []
    for index, result_path in enumerate(result_paths, start=1):
        run_dir = result_path.parent
        validation = _load_json(result_path)
        checkpoint_path = run_dir / "ckpt_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
        metrics_path = run_dir / "metrics_test.json"
        if metrics_path.is_file():
            print(f"[{index}/{expected_runs}] Reusing test result: {run_dir.name}")
            test_result = _load_json(metrics_path)
        else:
            print(f"[{index}/{expected_runs}] Evaluating test: {run_dir.name}")
            test_result = evaluate_checkpoint(checkpoint_path, output_dir=run_dir)
        rows.append(make_summary_row(result_path, validation, test_result))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: float(row["best_val_auroc"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["validation_rank"] = rank
        row["selected_by_validation"] = row["run"] == best_path.parent.name

    csv_path = root / "all_test_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "all_test_summary.json", rows)
    write_json(root / "final_selection.json", {
        "selection_metric": "val/subject/auroc",
        "selected_run": best_path.parent.name,
        "selected_val_auroc": float(best_validation["best_val_auroc"]),
        "selected_best_epoch": best_validation.get("best_epoch"),
        "selected_best_step": int(best_validation.get("best_step", -1)),
        "test_evaluated_for_all_runs": True,
        "test_results_used_for_selection": False,
        "num_tested_runs": len(rows),
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--expected-runs", type=int, default=5)
    args = parser.parse_args()
    rows = evaluate_all_runs(Path(args.grid_root), args.expected_runs)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
