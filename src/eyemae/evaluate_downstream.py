from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .downstream_config import load_downstream_config
from .downstream_metrics import binary_confusion_matrix, write_prediction_csv
from .finetune import (
    DownstreamClassifier,
    _confusion_payload,
    evaluate_classifier,
    make_datasets,
    make_downstream_loader,
    train_subject_class_weight,
    train_subject_pos_weight,
)
from .downstream_data import PackedDownstreamDataset
from .utils import ensure_dir, setup_logging, write_json


LOGGER = logging.getLogger(__name__)


def _plan_split_name(split: str) -> str:
    return "validation" if split == "val" else split


def evaluate_downstream_checkpoint(
    cfg_path: str | Path,
    checkpoint_path: str | Path,
    *,
    disease: str,
    output_dir: str | Path | None,
    splits: list[str],
) -> dict[str, float]:
    cfg = load_downstream_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = make_datasets(cfg, disease)
    loaders = {split: make_downstream_loader(datasets[split], cfg, train=False) for split in datasets}
    model = DownstreamClassifier(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    label_type = str(cfg.get("label", {}).get("type", "binary"))
    if label_type == "multiclass":
        if not isinstance(datasets["train"], PackedDownstreamDataset):
            raise ValueError("multiclass downstream currently requires packed_mmap dataset")
        pos_weight = train_subject_class_weight(datasets["train"], int(cfg.get("label", {}).get("num_classes", 5)), device)
    else:
        pos_weight = train_subject_pos_weight(datasets["train"], device)
    out_dir = ensure_dir(output_dir or (Path(cfg["experiment"]["output_dir"]) / "evaluation"))
    all_metrics: dict[str, float] = {}
    for split in splits:
        metrics, rows, subject_rows = evaluate_classifier(
            model,
            loaders[split],
            cfg,
            device,
            split_name=split,
            pos_weight=pos_weight,
        )
        all_metrics.update(metrics)
        plan_split = _plan_split_name(split)
        write_json(out_dir / f"{plan_split}_metrics.json", metrics)
        write_prediction_csv(out_dir / f"{split}_predictions.csv", rows)
        write_prediction_csv(out_dir / f"{split}_subject_predictions.csv", subject_rows)
        write_prediction_csv(out_dir / f"trial_predictions_{split}.csv", rows)
        write_prediction_csv(out_dir / f"subject_predictions_{split}.csv", subject_rows)
        if plan_split != split:
            write_prediction_csv(out_dir / f"{plan_split}_predictions.csv", rows)
            write_prediction_csv(out_dir / f"{plan_split}_subject_predictions.csv", subject_rows)
            write_prediction_csv(out_dir / f"trial_predictions_{plan_split}.csv", rows)
            write_prediction_csv(out_dir / f"subject_predictions_{plan_split}.csv", subject_rows)
        threshold = float(cfg["downstream_eval"].get("threshold", 0.5))
        confusion_payload = _confusion_payload(rows, subject_rows, cfg, threshold)
        write_json(out_dir / f"confusion_matrix_{split}.json", confusion_payload)
        if plan_split != split:
            write_json(out_dir / f"confusion_matrix_{plan_split}.json", confusion_payload)
    write_json(out_dir / "metrics.json", all_metrics)
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", action="append", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_downstream_config(args.config)
    disease = args.disease or cfg.get("downstream", {}).get("disease") or cfg.get("label", {}).get("task_name")
    if disease is None:
        raise ValueError("Provide --disease or set downstream.disease / label.task_name in config")
    splits = args.split or ["test"]
    metrics = evaluate_downstream_checkpoint(args.config, args.checkpoint, disease=disease, output_dir=args.output_dir, splits=splits)
    LOGGER.info("%s", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
