"""
Aggregate per-seed predictions into per-arch mean ± std metrics.

Usage:
    python -m baseline.aggregate_per_seed \
        --out-dir <baseline_run_dir> \
        --task <task_name> \
        --arch <arch_name> \
        --csv-out <output.csv>           # optional, defaults to <out-dir>/per_seed_metrics.csv

Reads:
    <out-dir>/per_seed_preds_dl_<arch>_<task>_seed<seed>.npz  (one per seed)
Writes (if --csv-out):
    one row per seed + a "mean ± std" row, all metrics for the task

Why this exists:
    The main run writes `baseline_summary.csv` with ensemble metrics only
    (averaged predicted probabilities across seeds, then a single threshold
    from val). For paper-grade reporting we also want the per-seed mean ± std
    for every metric, so this script recomputes metrics on each seed's
    saved predictions independently.

This is a thin helper — it imports the same metric functions used during
training (from baseline.dl_baseline) so the numbers are guaranteed to match
the in-loop values.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

# Make `python -m baseline.aggregate_per_seed` work whether the script is
# invoked from the project root or from inside src/.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baseline.dl_baseline import (  # noqa: E402
    _compute_metrics_arr,
    _compute_metrics_arr_at_threshold,
)


PER_SEED_NPZ = "per_seed_preds_dl_{arch}_{task}_seed{seed}.npz"


def _load_seed(npz_path: Path) -> dict:
    """Load per-seed npz and compute all metrics on its test/val probs."""
    d = np.load(npz_path, allow_pickle=True)
    y_te = d["y_true"]
    p_te = d["probs"]
    y_va = d["y_val"]
    p_va = d["probs_val"]
    val_threshold = float(d["val_threshold"])
    n_classes = int(d["n_classes"])

    te = _compute_metrics_arr(y_te, p_te, n_classes)
    if n_classes == 2:
        te_tuned = _compute_metrics_arr_at_threshold(y_te, p_te, n_classes, val_threshold)
        te["balanced_accuracy_tuned"] = te_tuned["balanced_accuracy"]
    else:
        te["balanced_accuracy_tuned"] = float("nan")
    return {
        "seed": int(d["seed"]),
        "use_swa": bool(d["use_swa"]),
        "val_threshold": val_threshold,
        "n_classes": n_classes,
        **te,
    }


def _summarize(rows: list[dict], n_classes: int) -> list[dict]:
    """Append mean and std rows across all per-seed rows."""
    if not rows:
        return rows
    # Numeric columns to summarize
    keys = [k for k in rows[0].keys() if k not in ("seed", "use_swa")]
    keys = [k for k in keys if isinstance(rows[0][k], (int, float))]

    mean_row: dict = {"seed": "mean", "use_swa": ""}
    std_row: dict = {"seed": "std", "use_swa": ""}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        # skip NaN-only columns
        finite = [v for v in vals if v == v]
        if not finite:
            mean_row[k] = float("nan")
            std_row[k] = float("nan")
            continue
        mean_row[k] = statistics.mean(finite)
        std_row[k] = statistics.stdev(finite) if len(finite) > 1 else 0.0
    rows.append(mean_row)
    rows.append(std_row)
    return rows


def aggregate(out_dir: Path, task: str, arch: str, csv_out: Path | None = None) -> list[dict]:
    """Find all per-seed npz files and compute per-seed + mean ± std metrics."""
    seeds = sorted(int(p.stem.split("seed")[-1]) for p in out_dir.glob(PER_SEED_NPZ.format(arch=arch, task=task, seed="*")))
    if not seeds:
        raise FileNotFoundError(f"No per-seed npz for arch={arch} task={task} under {out_dir}")

    rows = [_load_seed(out_dir / PER_SEED_NPZ.format(arch=arch, task=task, seed=s)) for s in seeds]
    n_classes = rows[0]["n_classes"]
    rows = _summarize(rows, n_classes)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path, help="per-arch out_dir from run_baseline")
    p.add_argument("--task", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    rows = aggregate(args.out_dir, args.task, args.arch, args.csv_out)
    for r in rows:
        seed = r.get("seed", "?")
        if isinstance(seed, int):
            auroc = r.get("auroc", float("nan"))
            ba = r.get("balanced_accuracy", float("nan"))
            ba_t = r.get("balanced_accuracy_tuned", float("nan"))
            auprc = r.get("auprc", float("nan"))
            print(f"seed={seed}  auroc={auroc:.4f}  bal_acc={ba:.4f}  ba_tuned={ba_t:.4f}  auprc={auprc:.4f}")
        else:
            auroc = r.get("auroc", float("nan"))
            ba = r.get("balanced_accuracy", float("nan"))
            ba_t = r.get("balanced_accuracy_tuned", float("nan"))
            auprc = r.get("auprc", float("nan"))
            print(f"{seed:>4}: auroc={auroc:.4f}  bal_acc={ba:.4f}  ba_tuned={ba_t:.4f}  auprc={auprc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
