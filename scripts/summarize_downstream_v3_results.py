from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


MODE_LABELS = {
    "scratch": "Scratch",
    "linear_probe": "Pretrained linear_probe",
    "partial": "Pretrained partial",
    "full": "Pretrained full",
}


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


def mode_output_name(mode: str) -> str:
    if mode == "scratch":
        return "scratch_full"
    if mode == "linear_probe":
        return "pretrained_linear_probe"
    if mode == "partial":
        return "pretrained_partial"
    if mode == "full":
        return "pretrained_full"
    return mode


def scalar(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return None


def metric_keys(label_type: str, split: str) -> dict[str, str]:
    prefix = f"{split}/subject"
    if label_type == "multiclass":
        return {
            "subject_auroc_or_macro_auroc": f"{prefix}/macro_auroc_ovr",
            "subject_auprc_or_macro_auprc": f"{prefix}/macro_auprc_ovr",
            "balanced_accuracy": f"{prefix}/balanced_accuracy",
            "f1": f"{prefix}/macro_f1",
            "weighted_f1": f"{prefix}/weighted_f1",
            "cohen_kappa": f"{prefix}/cohen_kappa",
        }
    return {
        "subject_auroc_or_macro_auroc": f"{prefix}/auroc",
        "subject_auprc_or_macro_auprc": f"{prefix}/auprc",
        "balanced_accuracy": f"{prefix}/balanced_accuracy",
        "f1": f"{prefix}/f1",
        "weighted_f1": f"{prefix}/weighted_f1",
        "cohen_kappa": f"{prefix}/cohen_kappa",
    }


def build_rows(config_list_file: Path, *, split: str = "test") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_path in read_config_list(config_list_file):
        cfg = read_yaml(cfg_path)
        task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
        mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
        out_dir = Path(str(cfg.get("experiment", {}).get("output_dir", "")))
        label_type = str(cfg.get("label", {}).get("type", "binary"))
        metrics_path = out_dir / "metrics_final.json"
        run_summary_path = out_dir / "run_summary.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        run_summary = read_json(run_summary_path) if run_summary_path.exists() else {}
        keys = metric_keys(label_type, split)

        encoder_init = "random" if mode == "scratch" else "pretrained"
        finetune_style = "full train from scratch" if mode == "scratch" else mode
        pretrain_exposure = run_summary.get(
            "pretraining_exposure_mode",
            cfg.get("pretraining_exposure", {}).get("mode", "all_unlabeled_or_unknown" if mode != "scratch" else "none"),
        )
        if mode == "scratch":
            pretrain_exposure = "none"

        row: dict[str, Any] = {
            "task": task,
            "model": MODE_LABELS.get(mode, mode),
            "mode": mode,
            "pretrain_exposure": pretrain_exposure,
            "encoder_initialization": encoder_init,
            "finetune_style": finetune_style,
            "status": "ok" if metrics_path.exists() else "missing",
            "best_epoch": scalar(metrics, "best_epoch"),
            "best_metric": scalar(metrics, "best_metric"),
            "validation_main_metric": scalar(
                metrics,
                "val/subject/macro_auroc_ovr" if label_type == "multiclass" else "val/subject/auroc",
            ),
            "test_main_metric": scalar(metrics, keys["subject_auroc_or_macro_auroc"]),
            "subject_auroc_or_macro_auroc": scalar(metrics, keys["subject_auroc_or_macro_auroc"]),
            "subject_auprc_or_macro_auprc": scalar(metrics, keys["subject_auprc_or_macro_auprc"]),
            "balanced_accuracy": scalar(metrics, keys["balanced_accuracy"]),
            "f1": scalar(metrics, keys["f1"]),
            "weighted_f1": scalar(metrics, keys["weighted_f1"]),
            "cohen_kappa": scalar(metrics, keys["cohen_kappa"]),
            "metrics_path": str(metrics_path),
            "output_dir": str(out_dir),
            "config": str(cfg_path),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "model",
        "mode",
        "pretrain_exposure",
        "encoder_initialization",
        "finetune_style",
        "status",
        "best_epoch",
        "best_metric",
        "validation_main_metric",
        "test_main_metric",
        "subject_auroc_or_macro_auroc",
        "subject_auprc_or_macro_auprc",
        "balanced_accuracy",
        "f1",
        "weighted_f1",
        "cohen_kappa",
        "metrics_path",
        "output_dir",
        "config",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Task",
        "Model",
        "Pretrain exposure",
        "Encoder init",
        "Fine-tune",
        "Subject AUROC / Macro AUROC",
        "Subject AUPRC / Macro AUPRC",
        "Balanced Acc",
        "F1",
        "Weighted F1",
        "Cohen Kappa",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["task"],
            row["model"],
            row["pretrain_exposure"],
            row["encoder_initialization"],
            row["finetune_style"],
            _fmt(row["subject_auroc_or_macro_auroc"]),
            _fmt(row["subject_auprc_or_macro_auprc"]),
            _fmt(row["balanced_accuracy"]),
            _fmt(row["f1"]),
            _fmt(row["weighted_f1"]),
            _fmt(row["cohen_kappa"]),
            row["status"],
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-csv", default="outputs/downstream_v3_fast/summary_first_version.csv")
    parser.add_argument("--out-json", default="outputs/downstream_v3_fast/summary_first_version.json")
    parser.add_argument("--out-md", default="outputs/downstream_v3_fast/summary_first_version.md")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows = build_rows(Path(args.config_list_file), split=args.split)
    missing = [row["config"] for row in rows if row["status"] != "ok"]
    write_csv(Path(args.out_csv), rows)
    write_json(Path(args.out_json), {"rows": rows, "split": args.split, "missing_count": len(missing)})
    write_markdown(Path(args.out_md), rows)
    result = {
        "csv": args.out_csv,
        "json": args.out_json,
        "markdown": args.out_md,
        "num_rows": len(rows),
        "missing_count": len(missing),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_complete and missing:
        raise SystemExit(f"Missing metrics_final.json for {len(missing)} configs")


if __name__ == "__main__":
    main()
