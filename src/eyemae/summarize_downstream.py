from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .downstream_data import DEFAULT_DISEASES
from .utils import read_json, write_json


DEFAULT_MODES = ["scratch", "pretrained_linear_probe", "pretrained_partial", "pretrained_full"]


def summarize_downstream_outputs(
    output_root: str | Path,
    *,
    diseases: list[str] | None = None,
    modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for disease in diseases or DEFAULT_DISEASES:
        for mode in modes or DEFAULT_MODES:
            metrics_path = root / disease / mode / "metrics_final.json"
            if not metrics_path.exists():
                rows.append({"disease": disease, "mode": mode, "status": "missing", "metrics_path": str(metrics_path)})
                continue
            metrics = read_json(metrics_path)
            row = {"disease": disease, "mode": mode, "status": "ok", "metrics_path": str(metrics_path)}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str)) or value is None:
                    row[key] = value
            rows.append(row)
    return rows


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
    parser.add_argument("--disease", action="append", default=None)
    parser.add_argument("--mode", action="append", default=None)
    args = parser.parse_args()
    rows = summarize_downstream_outputs(args.output_root, diseases=args.disease, modes=args.mode)
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.output_root) / "summary.csv"
    out_json = Path(args.out_json) if args.out_json else Path(args.output_root) / "summary.json"
    write_summary_csv(out_csv, rows)
    write_json(out_json, {"rows": rows})
    print({"csv": str(out_csv), "json": str(out_json), "num_rows": len(rows)})


if __name__ == "__main__":
    main()
