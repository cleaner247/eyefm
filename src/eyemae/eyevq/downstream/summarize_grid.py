"""Summarize validation-only EyeVQ Subject-MIL grid runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"E(?P<epoch>\d+) train_loss=(?P<train_loss>[0-9.]+) .*?"
    r"val_auroc=(?P<val_auroc>[0-9.]+).*?lr_enc=(?P<lr_enc>[0-9.eE+-]+)"
)
STEP_PATTERN = re.compile(
    r"S(?P<step>\d+) train_loss=(?P<train_loss>[0-9.]+) .*?"
    r"val_auroc=(?P<val_auroc>[0-9.]+).*?lr_enc="
    r"(?P<lr_enc_min>[0-9.eE+-]+)\.\.(?P<lr_enc_max>[0-9.eE+-]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", required=True)
    args = parser.parse_args()
    root = Path(args.grid_root)
    rows: list[dict[str, object]] = []

    for result_path in sorted(root.glob("*/metrics_val_best.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cfg = result["cfg"]
        curve = []
        log_path = result_path.parent / "train.log"
        for match in EPOCH_PATTERN.finditer(log_path.read_text(encoding="utf-8")):
            curve.append({
                "epoch": int(match.group("epoch")),
                "val_auroc": float(match.group("val_auroc")),
            })
        for match in STEP_PATTERN.finditer(log_path.read_text(encoding="utf-8")):
            curve.append({
                "step": int(match.group("step")),
                "val_auroc": float(match.group("val_auroc")),
            })

        def best_through(max_epochs: int) -> float:
            values = [
                point["val_auroc"]
                for point in curve
                if point.get("epoch", max_epochs) < max_epochs
            ]
            return max(values) if values else float("nan")

        frozen = int(cfg["model"]["freeze_bottom_layers"])
        rows.append({
            "run": result_path.parent.name,
            "unfrozen_layers": 12 - frozen,
            "encoder_lr": float(cfg["train"]["encoder_lr"]),
            "head_lr": float(cfg["train"]["head_lr"]),
            "max_epochs": int(cfg["train"]["epochs"]),
            "best_epoch": result.get("best_epoch"),
            "best_step": int(result.get("best_step", -1)),
            "best_smoothed_val_auroc": result.get("best_smoothed_val_auroc"),
            "best_val_auroc": float(result["best_val_auroc"]),
            "best_val_auroc_e40": best_through(40),
            "best_val_auroc_e60": best_through(60),
            "best_val_auroc_e80": best_through(80),
            "best_val_auroc_e100": best_through(100),
            "best_val_auroc_e120": best_through(120),
        })

    rows.sort(key=lambda row: float(row["best_val_auroc"]), reverse=True)
    output = root / "grid_summary.csv"
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
