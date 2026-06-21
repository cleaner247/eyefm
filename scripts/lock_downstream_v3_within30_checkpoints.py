from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def read_config_list(path: Path) -> list[Path]:
    configs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(Path(line))
    return configs


def maybe_lock(cfg_path: Path, *, force: bool) -> dict[str, Any]:
    cfg = read_yaml(cfg_path)
    out_dir = Path(str(cfg.get("experiment", {}).get("output_dir", "")))
    result: dict[str, Any] = {
        "config": str(cfg_path),
        "output_dir": str(out_dir),
        "status": "skipped",
    }
    metrics_last = out_dir / "metrics_last.json"
    metrics_best = out_dir / "metrics_best.json"
    checkpoint_best = out_dir / "checkpoint_best.pt"
    target_checkpoint = out_dir / "checkpoint_best_within30.pt"
    target_metrics = out_dir / "metrics_best_within30.json"
    lock_info = out_dir / "within30_lock_info.json"
    if not metrics_last.exists() or not metrics_best.exists() or not checkpoint_best.exists():
        result["reason"] = "missing_last_best_or_checkpoint"
        return result
    last = read_json(metrics_last)
    best = read_json(metrics_best)
    last_epoch = int(last.get("epoch", -1))
    best_epoch = int(best.get("epoch", -1))
    result["last_epoch"] = last_epoch
    result["best_epoch"] = best_epoch
    if target_checkpoint.exists() and target_metrics.exists() and lock_info.exists() and not force:
        locked = read_json(target_metrics)
        result["status"] = "already_locked"
        result["locked_epoch"] = int(locked.get("epoch", -1))
        return result
    if last_epoch < 29:
        result["reason"] = "last_epoch_lt_29"
        return result
    if best_epoch >= 30:
        result["reason"] = "best_epoch_not_within30"
        return result
    shutil.copy2(checkpoint_best, target_checkpoint)
    shutil.copy2(metrics_best, target_metrics)
    write_json(
        lock_info,
        {
            "source_checkpoint": str(checkpoint_best),
            "source_metrics": str(metrics_best),
            "target_checkpoint": str(target_checkpoint),
            "target_metrics": str(target_metrics),
            "last_epoch_at_lock": last_epoch,
            "best_epoch_at_lock": best_epoch,
        },
    )
    result["status"] = "locked"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    rows = [maybe_lock(path, force=args.force) for path in read_config_list(Path(args.config_list_file))]
    payload = {
        "locked": sum(row["status"] == "locked" for row in rows),
        "already_locked": sum(row["status"] == "already_locked" for row in rows),
        "rows": rows,
    }
    if args.report_json:
        write_json(Path(args.report_json), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
