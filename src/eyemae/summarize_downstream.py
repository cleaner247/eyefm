from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .downstream_data import DEFAULT_DISEASES
from .utils import read_json, write_json


DEFAULT_MODES = ["scratch", "pretrained_linear_probe", "pretrained_partial", "pretrained_full"]


def _normalize_fold(fold: str | int) -> str:
    text = str(fold)
    if text.startswith("fold_"):
        return text
    return f"fold_{int(text)}"


def summarize_downstream_outputs(
    output_root: str | Path,
    *,
    diseases: list[str] | None = None,
    modes: list[str] | None = None,
    folds: list[str | int] | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    fold_roots: list[tuple[str | None, Path]]
    if folds is None:
        fold_roots = [(None, root)]
    else:
        fold_roots = [(fold_name, root / fold_name) for fold_name in (_normalize_fold(fold) for fold in folds)]
    for fold_name, fold_root in fold_roots:
        for disease in diseases or DEFAULT_DISEASES:
            for mode in modes or DEFAULT_MODES:
                metrics_path = fold_root / disease / mode / "metrics_final.json"
                row = {"disease": disease, "mode": mode, "status": "missing", "metrics_path": str(metrics_path)}
                if fold_name is not None:
                    row["fold"] = fold_name
                if not metrics_path.exists():
                    rows.append(row)
                    continue
                metrics = read_json(metrics_path)
                row["status"] = "ok"
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str)) or value is None:
                        row[key] = value
                rows.append(row)
    return rows


def aggregate_summary_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("disease", "mode"),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        row["n"] = len(group_rows)
        numeric_keys = sorted(
            {
                metric_key
                for group_row in group_rows
                for metric_key, value in group_row.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        )
        for metric_key in numeric_keys:
            values = [
                float(group_row[metric_key])
                for group_row in group_rows
                if isinstance(group_row.get(metric_key), (int, float)) and math.isfinite(float(group_row[metric_key]))
            ]
            if not values:
                continue
            mean = sum(values) / float(len(values))
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
                std = math.sqrt(variance)
            else:
                std = math.nan
            row[f"{metric_key}/mean"] = mean
            row[f"{metric_key}/std"] = std
            row[f"{metric_key}/n"] = len(values)
        out.append(row)
    return out


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/downstream_disease_binary_seed42")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--aggregate_csv", default=None)
    parser.add_argument("--aggregate_json", default=None)
    parser.add_argument("--disease", action="append", default=None)
    parser.add_argument("--mode", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    args = parser.parse_args()
    rows = summarize_downstream_outputs(args.output_root, diseases=args.disease, modes=args.mode, folds=args.fold)
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.output_root) / "summary.csv"
    out_json = Path(args.out_json) if args.out_json else Path(args.output_root) / "summary.json"
    write_summary_csv(out_csv, rows)
    write_json(out_json, {"rows": rows})
    result = {"csv": str(out_csv), "json": str(out_json), "num_rows": len(rows)}
    if args.fold is not None:
        aggregate_rows = aggregate_summary_rows(rows)
        aggregate_csv = Path(args.aggregate_csv) if args.aggregate_csv else Path(args.output_root) / "summary_aggregate.csv"
        aggregate_json = Path(args.aggregate_json) if args.aggregate_json else Path(args.output_root) / "summary_aggregate.json"
        write_summary_csv(aggregate_csv, aggregate_rows)
        write_json(aggregate_json, {"rows": aggregate_rows})
        result.update({"aggregate_csv": str(aggregate_csv), "aggregate_json": str(aggregate_json), "num_aggregate_rows": len(aggregate_rows)})
    print(result)


if __name__ == "__main__":
    main()
