from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .config import load_config
from .manifest import TrialRecord, scan_trial_records, summarize_records
from .utils import ensure_dir, setup_matplotlib_cache, write_json
from .visualize import plot_eye_pattern_counts, plot_patch_length_histogram


def infer_n_samples(record: TrialRecord, cfg: dict) -> int:
    if record.n_samples is not None:
        return int(record.n_samples)
    z = np.load(record.path, allow_pickle=True)
    keys = cfg["data"]["npz_keys"]
    if keys.get("gaze", "gaze") in z.files:
        return int(z[keys.get("gaze", "gaze")].shape[0])
    if keys.get("eye", "eye") in z.files:
        return int(z[keys.get("eye", "eye")].shape[0])
    raise ValueError(f"{record.path}: cannot infer trial length from keys {z.files}")


def inspect_records(cfg: dict, min_patches: int) -> tuple[list[dict], dict]:
    all_records = scan_trial_records(cfg["data"]["data_dir"], exclude_no_eye_keep=False)
    excluded = sum(1 for r in all_records if not (r.left_final_keep or r.right_final_keep))
    records = [r for r in all_records if r.left_final_keep or r.right_final_keep]
    patch_samples = int(cfg["patch"]["samples"])
    rows: list[dict] = []
    long_rows: list[dict] = []
    for record in records:
        n_samples = infer_n_samples(record, cfg)
        n_patches = n_samples // patch_samples
        row = record.to_dict()
        row["n_samples"] = n_samples
        row["n_patches"] = n_patches
        rows.append(row)
        if n_patches >= min_patches:
            long_rows.append(row)
    lengths = [int(r["n_patches"]) for r in rows]
    max_patches = int(cfg["model"]["max_patches"])
    summary = summarize_records(records, excluded)
    summary.update(
        {
            "min_patches_filter": min_patches,
            "patch_samples": patch_samples,
            "model_max_patches": max_patches,
            "num_long_trials": len(long_rows),
            "num_trials_over_model_max_patches": sum(1 for x in lengths if x > max_patches),
            "max_observed_patches": max(lengths) if lengths else 0,
            "median_observed_patches": float(np.median(np.asarray(lengths))) if lengths else 0.0,
        }
    )
    return long_rows, {"summary": summary, "all_lengths": lengths}


def write_long_trials_csv(rows: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    ensure_dir(out.parent)
    fieldnames = [
        "rel_path",
        "subject_id",
        "base_subject_id",
        "task_name",
        "source_suffix",
        "usable_eye_pattern",
        "left_final_keep",
        "right_final_keep",
        "n_samples",
        "n_patches",
        "path",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: int(x["n_patches"]), reverse=True):
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--min_patches", type=int, default=257)
    parser.add_argument("--out_dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    setup_matplotlib_cache()
    out_dir = Path(args.out_dir or "outputs/eda_long_trials")
    ensure_dir(out_dir)
    long_rows, info = inspect_records(cfg, args.min_patches)
    summary = info["summary"]
    lengths = info["all_lengths"]
    write_long_trials_csv(long_rows, out_dir / "long_trials.csv")
    write_json(summary, out_dir / "summary.json")
    plot_patch_length_histogram(lengths, out_dir / "patch_length_histogram.png", int(cfg["model"]["max_patches"]))
    plot_eye_pattern_counts(Counter(row["usable_eye_pattern"] for row in long_rows), out_dir / "long_trial_eye_patterns.png")
    print(
        f"Inspected {summary['usable_trials']} usable trials; "
        f"{summary['num_long_trials']} have >= {args.min_patches} patches; "
        f"{summary['num_trials_over_model_max_patches']} exceed model.max_patches={summary['model_max_patches']}. "
        f"Wrote outputs to {out_dir}"
    )


if __name__ == "__main__":
    main()
