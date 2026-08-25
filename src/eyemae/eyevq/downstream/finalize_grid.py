"""Select the validation winner from a complete grid and test it exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eyemae.eyevq.downstream.evaluate_mil import evaluate_checkpoint
from eyemae.utils import write_json


def select_best_run(result_paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    if not result_paths:
        raise ValueError("No validation results were provided")
    candidates = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in result_paths
    ]
    return max(candidates, key=lambda item: float(item[1]["best_val_auroc"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--expected-runs", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.grid_root)
    paths = sorted(root.glob("*/metrics_val_best.json"))
    if len(paths) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} completed validation runs, found {len(paths)}"
        )

    lock_path = root / ".finalize.lock"
    try:
        os.mkdir(lock_path)
    except FileExistsError:
        print(f"Finalization already claimed: {lock_path}")
        return

    best_path, best_val = select_best_run(paths)
    run_dir = best_path.parent
    selection: dict[str, Any] = {
        "selection_metric": "val/subject/auroc",
        "selected_run": run_dir.name,
        "selected_val_auroc": float(best_val["best_val_auroc"]),
        "selected_best_epoch": best_val.get("best_epoch"),
        "selected_best_step": int(best_val.get("best_step", -1)),
        "test_evaluated_for_other_runs": False,
    }
    write_json(root / "final_selection.json", selection)

    metrics_path = run_dir / "metrics_test.json"
    if metrics_path.exists():
        test_result = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        test_result = evaluate_checkpoint(run_dir / "ckpt_best.pt", output_dir=run_dir)
    fixed_test = test_result.get("test", test_result.get("test_default_05"))
    if fixed_test is None:
        fixed_test = test_result["test_tuned"]
    selection.update({
        "test_auroc": float(fixed_test["test/subject/auroc"]),
        "test_balanced_accuracy": float(
            fixed_test["test/subject/balanced_accuracy"]
        ),
        "test_f1": float(fixed_test["test/subject/f1"]),
        "decision_threshold": float(
            test_result.get("decision_threshold", 0.5)
        ),
    })
    write_json(root / "final_selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
