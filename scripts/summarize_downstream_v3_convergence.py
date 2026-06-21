from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from eyemae.evaluate_downstream import evaluate_downstream_checkpoint


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


def metric_keys(label_type: str, split: str) -> dict[str, str]:
    prefix = f"{split}/subject"
    if label_type == "multiclass":
        return {
            "main": f"{prefix}/macro_auroc_ovr",
            "auprc": f"{prefix}/macro_auprc_ovr",
            "balanced_accuracy": f"{prefix}/balanced_accuracy",
            "f1": f"{prefix}/macro_f1",
            "weighted_f1": f"{prefix}/weighted_f1",
            "cohen_kappa": f"{prefix}/cohen_kappa",
        }
    return {
        "main": f"{prefix}/auroc",
        "auprc": f"{prefix}/auprc",
        "balanced_accuracy": f"{prefix}/balanced_accuracy",
        "f1": f"{prefix}/f1",
        "weighted_f1": f"{prefix}/weighted_f1",
        "cohen_kappa": f"{prefix}/cohen_kappa",
    }


def task_mode(cfg: dict[str, Any]) -> tuple[str, str]:
    task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
    mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
    return task, mode


def choose_checkpoint(out_dir: Path, view: str, *, epoch1_train_dir: Path | None = None) -> tuple[Path | None, str, int | None]:
    if view == "epoch1":
        checkpoint = out_dir / "checkpoint_epoch_000.pt"
        metrics = out_dir / "metrics_epoch_000.json"
        epoch = None
        if metrics.exists():
            epoch = int(read_json(metrics).get("epoch", 0))
        if checkpoint.exists():
            return checkpoint, "checkpoint_epoch_000", epoch
        if epoch1_train_dir is not None:
            for candidate_name in ["checkpoint_epoch_000.pt", "checkpoint_best.pt"]:
                candidate = epoch1_train_dir / candidate_name
                if candidate.exists():
                    metrics_final = epoch1_train_dir / "metrics_final.json"
                    if metrics_final.exists():
                        payload = read_json(metrics_final)
                        best_epoch = payload.get("best_epoch")
                        epoch = int(best_epoch) if isinstance(best_epoch, int) else 0
                    return candidate, f"epoch1_train_{candidate_name.removesuffix('.pt')}", epoch
        return None, "checkpoint_missing", epoch

    if view != "within30":
        raise ValueError(f"Unknown convergence view: {view}")

    checkpoint = out_dir / "checkpoint_best_within30.pt"
    metrics = out_dir / "metrics_best_within30.json"
    if checkpoint.exists():
        epoch = None
        if metrics.exists():
            epoch = int(read_json(metrics).get("epoch", -1))
        return checkpoint, "checkpoint_best_within30", epoch

    final_metrics = out_dir / "metrics_final.json"
    best_checkpoint = out_dir / "checkpoint_best.pt"
    if final_metrics.exists() and best_checkpoint.exists():
        payload = read_json(final_metrics)
        epoch = payload.get("best_epoch")
        if isinstance(epoch, int) and epoch < 30:
            return best_checkpoint, "checkpoint_best_final_best_within30", epoch

    best_metrics = out_dir / "metrics_best.json"
    if best_metrics.exists() and best_checkpoint.exists():
        payload = read_json(best_metrics)
        epoch = payload.get("epoch")
        if isinstance(epoch, int) and epoch < 30:
            return best_checkpoint, "checkpoint_best_current_best_within30", epoch

    return None, "checkpoint_missing", None


def convergence_output_dir(root: Path, view: str, task: str, mode: str) -> Path:
    return root / view / task / mode


def load_existing_metrics(path: Path, view: str, *, epoch1_train_dir: Path | None = None) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    archived_within30 = path / "metrics_final_within30.json"
    if metrics_path.exists():
        return read_json(metrics_path)
    if view == "within30" and archived_within30.exists():
        return read_json(archived_within30)
    if view == "epoch1" and epoch1_train_dir is not None:
        epoch1_metrics = epoch1_train_dir / "metrics_final.json"
        if epoch1_metrics.exists():
            return read_json(epoch1_metrics)
    return {}


def build_rows(configs: list[Path], *, view: str, convergence_root: Path, evaluate: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_path in configs:
        cfg = read_yaml(cfg_path)
        task, mode = task_mode(cfg)
        label_type = str(cfg.get("label", {}).get("type", "binary"))
        out_dir = Path(str(cfg.get("experiment", {}).get("output_dir", "")))
        conv_dir = convergence_output_dir(convergence_root, view, task, mode)
        epoch1_train_dir = convergence_root / "epoch1_train" / task / mode
        checkpoint, checkpoint_source, checkpoint_epoch = choose_checkpoint(
            out_dir,
            view,
            epoch1_train_dir=epoch1_train_dir,
        )
        metrics = load_existing_metrics(conv_dir, view, epoch1_train_dir=epoch1_train_dir)
        status = "ok" if metrics else "metrics_missing"

        if evaluate and checkpoint is not None and not metrics:
            disease = cfg.get("downstream", {}).get("disease") or cfg.get("label", {}).get("task_name")
            metrics = evaluate_downstream_checkpoint(
                cfg_path,
                checkpoint,
                disease=str(disease),
                output_dir=conv_dir,
                splits=["val", "test"],
            )
            status = "ok"
        elif checkpoint is None and not metrics:
            status = "checkpoint_missing"

        val_keys = metric_keys(label_type, "val")
        test_keys = metric_keys(label_type, "test")
        row = {
            "task": task,
            "mode": mode,
            "view": view,
            "status": status,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_source": checkpoint_source,
            "checkpoint": str(checkpoint) if checkpoint is not None else "",
            "validation_main_metric": metrics.get(val_keys["main"]),
            "test_main_metric": metrics.get(test_keys["main"]),
            "test_auprc": metrics.get(test_keys["auprc"]),
            "test_balanced_accuracy": metrics.get(test_keys["balanced_accuracy"]),
            "test_f1": metrics.get(test_keys["f1"]),
            "test_weighted_f1": metrics.get(test_keys["weighted_f1"]),
            "test_cohen_kappa": metrics.get(test_keys["cohen_kappa"]),
            "metrics_path": str(conv_dir / "metrics.json"),
            "config": str(cfg_path),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "mode",
        "view",
        "status",
        "checkpoint_epoch",
        "checkpoint_source",
        "validation_main_metric",
        "test_main_metric",
        "test_auprc",
        "test_balanced_accuracy",
        "test_f1",
        "test_weighted_f1",
        "test_cohen_kappa",
        "checkpoint",
        "metrics_path",
        "config",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Task",
        "Mode",
        "View",
        "Epoch",
        "Val Main",
        "Test Main",
        "Test AUPRC",
        "Bal Acc",
        "F1",
        "Weighted F1",
        "Kappa",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["task"],
            row["mode"],
            row["view"],
            fmt(row["checkpoint_epoch"]),
            fmt(row["validation_main_metric"]),
            fmt(row["test_main_metric"]),
            fmt(row["test_auprc"]),
            fmt(row["test_balanced_accuracy"]),
            fmt(row["test_f1"]),
            fmt(row["test_weighted_f1"]),
            fmt(row["test_cohen_kappa"]),
            row["status"],
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--view", choices=["within30", "epoch1"], required=True)
    parser.add_argument("--convergence-root", default="outputs/downstream_v3_fast_convergence")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = Path(args.convergence_root)
    rows = build_rows(read_config_list(Path(args.config_list_file)), view=args.view, convergence_root=root, evaluate=args.evaluate)
    out_csv = Path(args.out_csv or root / f"summary_{args.view}.csv")
    out_json = Path(args.out_json or root / f"summary_{args.view}.json")
    out_md = Path(args.out_md or root / f"summary_{args.view}.md")
    write_csv(out_csv, rows)
    write_json(out_json, {"view": args.view, "rows": rows, "missing_count": sum(row["status"] != "ok" for row in rows)})
    write_markdown(out_md, rows)
    result = {
        "view": args.view,
        "csv": str(out_csv),
        "json": str(out_json),
        "markdown": str(out_md),
        "num_rows": len(rows),
        "missing_count": sum(row["status"] != "ok" for row in rows),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_complete and result["missing_count"]:
        raise SystemExit(f"{args.view} convergence summary has {result['missing_count']} incomplete rows")


if __name__ == "__main__":
    main()
