from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


TASK_ORDER = [
    "pd_related_5class",
    "pd_binary",
    "epilepsy_binary",
    "detox_binary",
    "migraine_binary",
    "ad_binary",
    "mci_binary",
    "mci_matched_binary",
]
MODE_ORDER = ["scratch", "linear_probe", "partial", "full"]
MODE_LABEL = {
    "scratch": "Scratch",
    "linear_probe": "Linear probe",
    "partial": "Partial fine-tune",
    "full": "Full fine-tune",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def read_config_list(path: Path) -> list[Path]:
    configs: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            configs.append(Path(line))
    return configs


def metric_key(label_type: str, split: str, name: str) -> str:
    prefix = f"{split}/subject"
    if label_type == "multiclass":
        mapping = {
            "main": f"{prefix}/macro_auroc_ovr",
            "auprc": f"{prefix}/macro_auprc_ovr",
            "balanced_accuracy": f"{prefix}/balanced_accuracy",
            "f1": f"{prefix}/macro_f1",
            "weighted_f1": f"{prefix}/weighted_f1",
            "cohen_kappa": f"{prefix}/cohen_kappa",
        }
    else:
        mapping = {
            "main": f"{prefix}/auroc",
            "auprc": f"{prefix}/auprc",
            "balanced_accuracy": f"{prefix}/balanced_accuracy",
            "f1": f"{prefix}/f1",
            "weighted_f1": f"{prefix}/weighted_f1",
            "cohen_kappa": f"{prefix}/cohen_kappa",
        }
    return mapping[name]


def scalar(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def task_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    task = str(row["task"])
    mode = str(row["mode"])
    return (
        TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER),
        MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER),
    )


def build_rows(config_list: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_path in read_config_list(config_list):
        cfg = read_yaml(cfg_path)
        task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
        mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
        label_type = str(cfg.get("label", {}).get("type", "binary"))
        output_dir = Path(str(cfg.get("experiment", {}).get("output_dir")))
        metrics_path = output_dir / "metrics_final.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        row = {
            "task": task,
            "mode": mode,
            "mode_label": MODE_LABEL.get(mode, mode),
            "status": "ok" if metrics_path.exists() else "missing",
            "max_epochs": cfg.get("downstream_train", {}).get("max_epochs"),
            "best_epoch": scalar(metrics, "best_epoch"),
            "best_metric": scalar(metrics, "best_metric"),
            "val_main": scalar(metrics, metric_key(label_type, "val", "main")),
            "test_main": scalar(metrics, metric_key(label_type, "test", "main")),
            "test_auprc": scalar(metrics, metric_key(label_type, "test", "auprc")),
            "test_balanced_accuracy": scalar(metrics, metric_key(label_type, "test", "balanced_accuracy")),
            "test_f1": scalar(metrics, metric_key(label_type, "test", "f1")),
            "test_weighted_f1": scalar(metrics, metric_key(label_type, "test", "weighted_f1")),
            "test_cohen_kappa": scalar(metrics, metric_key(label_type, "test", "cohen_kappa")),
            "metrics_path": str(metrics_path),
            "output_dir": str(output_dir),
            "config": str(cfg_path),
        }
        rows.append(row)
    rows.sort(key=task_sort_key)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "mode",
        "mode_label",
        "status",
        "max_epochs",
        "best_epoch",
        "best_metric",
        "val_main",
        "test_main",
        "test_auprc",
        "test_balanced_accuracy",
        "test_f1",
        "test_weighted_f1",
        "test_cohen_kappa",
        "metrics_path",
        "output_dir",
        "config",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task and isinstance(row.get("test_main"), (int, float))]
        if not task_rows:
            continue
        best.append(max(task_rows, key=lambda row: float(row["test_main"])))
    return best


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    status: dict[str, Any],
    config_list: Path,
    queue_status: Path,
    log_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(status.get("completed", []))
    failures = status.get("failures", [])
    lines = [
        "# Corrected v2 Downstream Fine-tune Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Run",
        "",
        "| item | value |",
        "| --- | --- |",
        "| dataset | `/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2` |",
        "| config queue | `" + str(config_list) + "` |",
        "| log dir | `" + str(log_dir) + "` |",
        "| queue status | `" + str(queue_status) + "` |",
        "| completed jobs | `" + str(completed) + "` |",
        "| failed jobs | `" + str(len(failures)) + "` |",
        "| max_epochs | `30` |",
        "| early stopping | `patience=10, min_epochs_before_early_stopping=0` |",
        "| pretrained checkpoint | `outputs/pretrain_v3_fast_cache44_nooffset/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt` |",
        "",
        "Test metrics are computed after loading the validation-selected `checkpoint_best.pt` for each run.",
        "",
        "## Best By Task",
        "",
        "| task | best mode by test main metric | best epoch | val main | test main | test AUPRC | balanced acc | weighted F1 | kappa |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows(rows):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["mode_label"]),
                    fmt(row["best_epoch"]),
                    fmt(row["val_main"]),
                    fmt(row["test_main"]),
                    fmt(row["test_auprc"]),
                    fmt(row["test_balanced_accuracy"]),
                    fmt(row["test_weighted_f1"]),
                    fmt(row["test_cohen_kappa"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| task | mode | status | best epoch | val main | test main | test AUPRC | balanced acc | F1 | weighted F1 | kappa |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["mode_label"]),
                    str(row["status"]),
                    fmt(row["best_epoch"]),
                    fmt(row["val_main"]),
                    fmt(row["test_main"]),
                    fmt(row["test_auprc"]),
                    fmt(row["test_balanced_accuracy"]),
                    fmt(row["test_f1"]),
                    fmt(row["test_weighted_f1"]),
                    fmt(row["test_cohen_kappa"]),
                ]
            )
            + " |"
        )
    if failures:
        lines.extend(["", "## Failures", "", "```json", json.dumps(failures, ensure_ascii=False, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def queue_done(status: dict[str, Any]) -> bool:
    return not status.get("running") and not status.get("pending") and "failures" in status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--config-list-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-hours", type=float, default=240.0)
    args = parser.parse_args()

    start = time.time()
    while True:
        if args.status.exists():
            status = read_json(args.status)
            if queue_done(status):
                break
        if time.time() - start > args.timeout_hours * 3600:
            raise TimeoutError(f"Timed out waiting for {args.status}")
        time.sleep(args.poll_seconds)

    rows = build_rows(args.config_list_file)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status_path": str(args.status),
        "config_list_file": str(args.config_list_file),
        "rows": rows,
        "missing_metrics": [row["config"] for row in rows if row["status"] != "ok"],
        "queue_status": status,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", rows)
    write_json(args.out_dir / "summary.json", payload)
    write_markdown(
        args.out_dir / "summary.md",
        rows,
        status=status,
        config_list=args.config_list_file,
        queue_status=args.status,
        log_dir=args.status.parent,
    )
    print(json.dumps({"summary_md": str(args.out_dir / "summary.md"), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
