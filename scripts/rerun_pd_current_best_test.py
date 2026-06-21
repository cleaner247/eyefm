from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from eyemae.downstream_config import load_downstream_config
from eyemae.downstream_data import PackedDownstreamDataset
from eyemae.downstream_metrics import write_prediction_csv
from eyemae.evaluate_downstream import _plan_split_name
from eyemae.finetune import (
    DownstreamClassifier,
    _confusion_payload,
    evaluate_classifier,
    make_datasets,
    make_downstream_loader,
    train_subject_class_weight,
    train_subject_pos_weight,
)
from eyemae.utils import ensure_dir, setup_logging, write_json


MODE_SPECS = {
    "scratch": {
        "config": "configs/downstream/pd_related_5class_random_seed20260620_fast_scratch.yaml",
        "checkpoint": "outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/scratch_full/checkpoint_best.pt",
        "output_dir": "outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/scratch",
    },
    "linear_probe": {
        "config": "configs/downstream/pd_related_5class_random_seed20260620_fast_linear_probe.yaml",
        "checkpoint": "outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_linear_probe/checkpoint_best.pt",
        "output_dir": "outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/linear_probe",
    },
    "partial": {
        "config": "configs/downstream/pd_related_5class_random_seed20260620_fast_partial.yaml",
        "checkpoint": "outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_partial/checkpoint_best.pt",
        "output_dir": "outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/partial",
    },
    "full": {
        "config": "configs/downstream/pd_related_5class_random_seed20260620_fast_full.yaml",
        "checkpoint": "outputs/downstream_v3_fast/pd_related_5class_random_seed20260620/pretrained_full/checkpoint_best.pt",
        "output_dir": "outputs/downstream_v3_fast_current_best_test_eval/pd_related_5class_random_seed20260620/full",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def apply_eval_overrides(cfg: dict[str, Any], *, batch_size: int, num_workers: int) -> None:
    train_cfg = cfg.setdefault("downstream_train", {})
    train_cfg["batch_size"] = int(batch_size)
    train_cfg["num_workers"] = int(num_workers)
    train_cfg["persistent_workers"] = bool(num_workers > 0)
    if num_workers <= 0:
        train_cfg.pop("prefetch_factor", None)


def write_split_outputs(
    cfg: dict[str, Any],
    output_dir: Path,
    split: str,
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
) -> None:
    plan_split = _plan_split_name(split)
    write_json(output_dir / f"{plan_split}_metrics.json", metrics)
    write_prediction_csv(output_dir / f"{split}_predictions.csv", rows)
    write_prediction_csv(output_dir / f"{split}_subject_predictions.csv", subject_rows)
    write_prediction_csv(output_dir / f"trial_predictions_{split}.csv", rows)
    write_prediction_csv(output_dir / f"subject_predictions_{split}.csv", subject_rows)
    threshold = float(cfg["downstream_eval"].get("threshold", 0.5))
    write_json(output_dir / f"confusion_matrix_{split}.json", _confusion_payload(rows, subject_rows, cfg, threshold))
    write_json(output_dir / "metrics.json", metrics)


def evaluate_mode(mode: str, *, device: torch.device, batch_size: int, num_workers: int) -> dict[str, float]:
    spec = MODE_SPECS[mode]
    cfg_path = Path(spec["config"])
    checkpoint_path = Path(spec["checkpoint"])
    output_dir = ensure_dir(spec["output_dir"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {mode}: {checkpoint_path}")

    log(f"{mode}: load config {cfg_path}")
    cfg = load_downstream_config(cfg_path)
    apply_eval_overrides(cfg, batch_size=batch_size, num_workers=num_workers)
    disease = cfg.get("downstream", {}).get("disease") or cfg.get("label", {}).get("task_name")
    if not disease:
        raise ValueError(f"{mode}: missing downstream disease/task name")

    log(f"{mode}: build datasets")
    datasets = make_datasets(cfg, str(disease))
    log(f"{mode}: train_trials={len(datasets['train'])} test_trials={len(datasets['test'])}")

    log(f"{mode}: build model on {device}")
    model = DownstreamClassifier(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    log(
        f"{mode}: loaded checkpoint epoch={checkpoint.get('epoch', 'NA')} "
        f"global_step={checkpoint.get('global_step', 'NA')}"
    )

    label_type = str(cfg.get("label", {}).get("type", "binary"))
    if label_type == "multiclass":
        if not isinstance(datasets["train"], PackedDownstreamDataset):
            raise ValueError("Multiclass downstream currently requires packed_mmap dataset")
        pos_weight = train_subject_class_weight(
            datasets["train"],
            int(cfg.get("label", {}).get("num_classes", 5)),
            device,
        )
    else:
        pos_weight = train_subject_pos_weight(datasets["train"], device)

    log(f"{mode}: start test evaluation")
    loader = make_downstream_loader(datasets["test"], cfg, train=False)
    metrics, rows, subject_rows = evaluate_classifier(
        model,
        loader,
        cfg,
        device,
        split_name="test",
        pos_weight=pos_weight,
    )
    log(f"{mode}: write outputs to {output_dir}")
    write_split_outputs(cfg, output_dir, "test", metrics, rows, subject_rows)
    write_json(
        output_dir / "rerun_info.json",
        {
            "mode": mode,
            "config": str(cfg_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "device": str(device),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    log(
        f"{mode}: subject AUROC="
        f"{metrics.get('test/subject/macro_auroc_ovr', metrics.get('test/subject_auroc'))}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODE_SPECS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    device = torch.device(args.device)
    all_metrics: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        if mode not in MODE_SPECS:
            raise ValueError(f"Unknown mode {mode}; choices={sorted(MODE_SPECS)}")
        all_metrics[mode] = evaluate_mode(
            mode,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
