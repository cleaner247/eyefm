"""EyeFM baseline runner: unified entry-point for ML + DL baselines.

For each task: extract 24-dim paper-features → run ML grid + run DL 4-arch →
write per-task test csvs and combined summary.

Usage:
    python -m baseline.run_baseline --task detox_binary --data-root /path/to/v2 --out-dir outputs/baseline
    python -m baseline.run_baseline --all --data-root /path/to/v2 --out-dir outputs/baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Ensure repo root on path so "from baseline.xxx import ..." works as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline.feature_extraction import (  # noqa: E402
    PAPER_FEATS_A_B,
    SACCADE_TASKS,
    build_subject_feature_table,
)
from baseline.ml_baseline import (  # noqa: E402
    CSV_FIELDS as ML_CSV_FIELDS,
    impute_and_standardize,
    run_grid_search,
    write_test_csv,
)
from baseline.dl_baseline import (  # noqa: E402
    CSV_FIELDS as DL_CSV_FIELDS,
    run_dl_task,
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )


def run_one_task(
    task: str,
    data_root: Path,
    out_dir: Path,
    skip_ml: bool = False,
    skip_dl: bool = False,
    dl_t_len: int = 1024,
    dl_epochs: int = 50,
    dl_archs: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run both ML grid + DL 4-arch baselines for a single task; return dict of results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {"ml": [], "dl": []}

    if not skip_ml:
        LOGGER.info("=== TASK %s: ML grid (LR/RF/SVM × 30 hp) ===", task)
        X_per_subj, y_per_subj, _, split_tags = build_subject_feature_table(
            task=task, data_root=data_root, splits=("train", "validation", "test")
        )
        from baseline.data_loader import get_n_classes
        n_classes = get_n_classes(task)
        X_per_subj = np.asarray(X_per_subj, dtype=np.float32)
        y_per_subj = np.asarray(y_per_subj, dtype=np.int64)
        split_tags = np.asarray(split_tags)
        tr_mask = split_tags == "train"
        va_mask = split_tags == "validation"
        te_mask = split_tags == "test"
        train_b, val_b, test_b = impute_and_standardize(
            X_per_subj[tr_mask], y_per_subj[tr_mask],
            X_per_subj[va_mask], y_per_subj[va_mask],
            X_per_subj[te_mask], y_per_subj[te_mask],
        )
        ml_results = run_grid_search(
            train_b.X, train_b.y, val_b.X, val_b.y, test_b.X, test_b.y, n_classes=n_classes
        )
        write_test_csv(out_dir / f"test_final_ml_{task}.csv", ml_results)
        results["ml"] = [asdict(r) for r in ml_results]
        for r in ml_results:
            LOGGER.info("  ML %s  val=%.4f  test=%.4f  hp=%s", r.model, r.val_score,
                        r.test_metrics.get("auroc", r.test_metrics.get("auroc_macro", float("nan"))),
                        r.hp_str)

    if not skip_dl:
        LOGGER.info("=== TASK %s: DL 4-arch (50 epoch val-pick) ===", task)
        archs = dl_archs or ["TCN", "TimesNet", "NST", "CNNTransformer"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dl_results = run_dl_task(
            task=task, data_root=data_root, out_dir=out_dir, archs=archs,
            t_len=dl_t_len, max_epochs=dl_epochs, device=device,
        )
        results["dl"] = dl_results
        for r in dl_results:
            LOGGER.info("  DL %s  best_val=%.4f  test=%.4f", r["arch"], r["best_val_score"],
                        r.get("auroc", r.get("auroc_macro", float("nan"))))
    return results


def write_combined_summary(out_dir: Path, all_results: dict[str, dict[str, list[dict]]]) -> None:
    """Merge all task-level csvs into a single combined baseline_summary.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task, mode_results in all_results.items():
        for r in mode_results.get("ml", []):
            rows.append({
                "task": task, "mode": "ml", "arch_or_model": r["model"],
                "val_score": r["val_score"],
                **r["test_metrics"],
            })
        for r in mode_results.get("dl", []):
            rows.append({
                "task": task, "mode": "dl", "arch_or_model": r["arch"],
                "val_score": r["best_val_score"],
                **r,
            })
    fields = sorted({k for row in rows for k in row.keys()})
    summary_csv = out_dir / "baseline_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    LOGGER.info("Wrote %s (%d rows)", summary_csv, len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EyeFM baseline runner (ML + DL).")
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to eyemae_fast_dataset root (must contain finetune/<task>/).")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output directory for csv + summary.")
    parser.add_argument("--task", action="append", default=None,
                        choices=["detox_binary", "pd_related_5class"],
                        help="Task to run. Repeatable. Default: both 2 task.")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 2 tasks (detox_binary, pd_related_5class).")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--dl-t-len", type=int, default=1024, help="T_LEN for DL archs.")
    parser.add_argument("--dl-epochs", type=int, default=50, help="Max epochs (val-pick best).")
    parser.add_argument("--dl-arch", action="append", default=None,
                        choices=["TCN", "TimesNet", "NST", "CNNTransformer"],
                        help="Restrict DL archs. Repeatable.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.all or not args.task:
        tasks = ["detox_binary", "pd_related_5class"]
    else:
        tasks = list(args.task)
    LOGGER.info("Tasks: %s", tasks)
    LOGGER.info("data_root=%s out_dir=%s", args.data_root, args.out_dir)
    if not args.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    all_results: dict[str, dict[str, list[dict]]] = {}
    for task in tasks:
        all_results[task] = run_one_task(
            task=task, data_root=args.data_root, out_dir=args.out_dir,
            skip_ml=args.skip_ml, skip_dl=args.skip_dl,
            dl_t_len=args.dl_t_len, dl_epochs=args.dl_epochs,
            dl_archs=args.dl_arch,
        )
    write_combined_summary(args.out_dir, all_results)
    LOGGER.info("All done.")


if __name__ == "__main__":
    main()
