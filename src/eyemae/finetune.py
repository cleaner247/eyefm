from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint_utils import extract_encoder_state_dict
from .downstream_config import load_downstream_config
from .downstream_data import DownstreamTrialDataset, collate_downstream_trials
from .downstream_metrics import (
    aggregate_subject_predictions,
    binary_confusion_matrix,
    compute_binary_metrics,
    sigmoid,
    write_prediction_csv,
)
from .manifest import build_record_index
from .model import build_model
from .pooling import eye_mean_pool
from .preprocess import load_area_stats
from .train import autocast_context, make_writer, move_batch_to_device
from .utils import atomic_torch_save, ensure_dir, set_seed, setup_logging, write_json


LOGGER = logging.getLogger(__name__)


class DownstreamClassifier(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = build_model(cfg)
        d_model = int(cfg["model"]["d_model"])
        head_cfg = cfg.get("downstream", {}).get("head", {})
        hidden = int(head_cfg.get("hidden_dim", 256))
        dropout = float(head_cfg.get("dropout", cfg["model"].get("dropout", 0.0)))
        if hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = self.encoder.forward_features(
            batch["content"],
            batch["quality"],
            batch["stim"],
            batch["task_id"],
            batch["pad_mask"],
            batch["eye_token_valid"],
            mae_mask=None,
        )
        pooled, has_valid_eye_token, valid_token_count = eye_mean_pool(
            features["hidden_eye"],
            batch["eye_token_valid"],
            batch["pad_mask"],
        )
        logits = self.head(pooled).squeeze(-1)
        return {
            "logit": logits,
            "embedding": pooled,
            "has_valid_eye_token": has_valid_eye_token,
            "valid_token_count": valid_token_count,
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    encoder_total = sum(param.numel() for param in model.encoder.parameters()) if hasattr(model, "encoder") else 0
    head_total = sum(param.numel() for param in model.head.parameters()) if hasattr(model, "head") else 0
    return {
        "total": int(total),
        "trainable": int(trainable),
        "encoder_total": int(encoder_total),
        "head_total": int(head_total),
    }


def load_pretrained_encoder(model: DownstreamClassifier, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, skipped = extract_encoder_state_dict(checkpoint)
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    allowed_missing = {key for key in model.encoder.state_dict() if key.startswith("pred_head.")}
    if unexpected:
        raise RuntimeError(f"Unexpected pretrained encoder keys: {unexpected[:20]}")
    unexpected_missing = [key for key in missing if key not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(f"Missing pretrained encoder keys: {unexpected_missing[:20]}")
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "global_step": int(checkpoint.get("global_step", -1)) if isinstance(checkpoint, dict) else -1,
        "skipped_keys": skipped[:20],
        "num_skipped_keys": len(skipped),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def apply_freeze_mode(model: DownstreamClassifier, mode: str, *, partial_last_n_layers: int = 4) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if mode == "scratch" or mode == "pretrained_full":
        return
    if mode == "pretrained_linear_probe":
        for param in model.encoder.parameters():
            param.requires_grad = False
        return
    if mode == "pretrained_partial":
        for param in model.encoder.parameters():
            param.requires_grad = False
        n_layers = len(model.encoder.blocks)
        for block in model.encoder.blocks[max(0, n_layers - int(partial_last_n_layers)) :]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.encoder.final_norm.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unknown downstream mode: {mode}")


def split_file_for(cfg: dict[str, Any], disease: str, split: str) -> Path:
    split_dir = Path(cfg["downstream"]["split_dir"])
    return split_dir / disease / f"{split}.txt"


def output_dir_for(cfg: dict[str, Any], disease: str, mode: str) -> Path:
    exp_cfg = cfg.get("experiment", {})
    if exp_cfg.get("output_dir"):
        return Path(str(exp_cfg["output_dir"]).format(disease=disease, mode=mode))
    root = Path(exp_cfg.get("output_root", "outputs/downstream_disease_binary"))
    return root / disease / mode


def make_downstream_loader(
    dataset: DownstreamTrialDataset,
    cfg: dict[str, Any],
    *,
    train: bool,
) -> DataLoader:
    train_cfg = cfg["downstream_train"]
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    if not torch.cuda.is_available():
        num_workers = 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        collate_fn=collate_downstream_trials,
        num_workers=num_workers,
        pin_memory=bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)) and num_workers > 0,
    )


def make_datasets(
    cfg: dict[str, Any],
    disease: str,
    *,
    max_train_trials: int | None = None,
    max_eval_trials: int | None = None,
) -> dict[str, DownstreamTrialDataset]:
    data_dir = cfg["data"]["data_dir"]
    record_index = build_record_index(data_dir, exclude_no_eye_keep=True)
    area_stats = load_area_stats(cfg["area"]["stats_path"])
    train_dataset = DownstreamTrialDataset(
        data_dir,
        split_file_for(cfg, disease, "train"),
        cfg,
        disease=disease,
        record_index=record_index,
        area_stats=area_stats,
        max_trials=max_train_trials,
    )
    train_counts = dict(Counter(record.base_subject_id for record in train_dataset.records))
    datasets = {
        "train": train_dataset,
        "val": DownstreamTrialDataset(
            data_dir,
            split_file_for(cfg, disease, "val"),
            cfg,
            disease=disease,
            record_index=record_index,
            area_stats=area_stats,
            max_trials=max_eval_trials,
            train_subject_trial_counts=train_counts,
        ),
        "test": DownstreamTrialDataset(
            data_dir,
            split_file_for(cfg, disease, "test"),
            cfg,
            disease=disease,
            record_index=record_index,
            area_stats=area_stats,
            max_trials=max_eval_trials,
            train_subject_trial_counts=train_counts,
        ),
    }
    return datasets


def downstream_dataset_summary(dataset: DownstreamTrialDataset) -> dict[str, Any]:
    subject_labels: dict[str, int] = {}
    trial_label_counts: Counter[str] = Counter()
    for record in dataset.records:
        label = 1 if record.group == "患病" else 0
        subject_labels[record.base_subject_id] = label
        trial_label_counts[str(label)] += 1
    subject_label_counts = Counter(str(label) for label in subject_labels.values())
    return {
        "num_subjects": len(subject_labels),
        "num_trials": len(dataset.records),
        "label_counts": {
            "0": int(trial_label_counts.get("0", 0)),
            "1": int(trial_label_counts.get("1", 0)),
        },
        "subject_label_counts": {
            "0": int(subject_label_counts.get("0", 0)),
            "1": int(subject_label_counts.get("1", 0)),
        },
    }


def make_run_summary(
    cfg: dict[str, Any],
    datasets: dict[str, DownstreamTrialDataset],
    *,
    disease: str,
    mode: str,
    pretrained_info: dict[str, Any] | None,
    param_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "disease": disease,
        "finetune_mode": mode,
        "pretrained_checkpoint": pretrained_info["checkpoint"] if pretrained_info else None,
        "pretraining_exposure_mode": cfg.get("pretraining_exposure", {}).get("mode", "all_unlabeled_or_unknown"),
        "pretrain_subject_manifest": cfg.get("pretraining_exposure", {}).get("pretrain_subject_manifest"),
        "param_counts": param_counts,
        "splits": {name: downstream_dataset_summary(dataset) for name, dataset in datasets.items()},
    }


def train_subject_pos_weight(dataset: DownstreamTrialDataset, device: torch.device) -> torch.Tensor:
    subject_labels: dict[str, int] = {}
    for record in dataset.records:
        label = 1 if record.group == "患病" else 0
        previous = subject_labels.get(record.base_subject_id)
        if previous is not None and previous != label:
            raise ValueError(f"Subject has conflicting train labels: {record.base_subject_id}")
        subject_labels[record.base_subject_id] = label
    counts = Counter(subject_labels.values())
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    if n_pos <= 0 or n_neg <= 0:
        LOGGER.warning("One-class train split; using pos_weight=1.0")
        value = 1.0
    else:
        value = n_neg / float(n_pos)
    return torch.tensor(value, dtype=torch.float32, device=device)


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(valid.any()):
        return logits.sum() * 0.0, torch.zeros((), dtype=torch.float32, device=logits.device)
    raw = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], pos_weight=pos_weight, reduction="none")
    weights = sample_weight[valid]
    denominator = weights.sum().clamp_min(1e-12)
    return (raw * weights).sum() / denominator, denominator


def build_optimizer(model: DownstreamClassifier, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["downstream_train"]
    head_lr = float(train_cfg.get("head_lr", train_cfg["lr"]))
    encoder_lr = float(train_cfg.get("encoder_lr", train_cfg["lr"]))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    head_params: list[nn.Parameter] = []
    encoder_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(param)
        else:
            encoder_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if not groups:
        raise RuntimeError("No trainable parameters after applying freeze mode")
    return torch.optim.AdamW(groups, betas=tuple(float(v) for v in train_cfg.get("betas", [0.9, 0.95])))


@torch.no_grad()
def evaluate_classifier(
    model: DownstreamClassifier,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    split_name: str,
    pos_weight: torch.Tensor,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total_weighted_loss = 0.0
    total_loss_den = 0.0
    precision = str(cfg["downstream_train"].get("precision", cfg["train"].get("precision", "fp32")))
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, precision):
            out = model(batch)
            loss, loss_den = weighted_bce_loss(
                out["logit"],
                batch["label"],
                batch["sample_weight"],
                out["has_valid_eye_token"],
                pos_weight,
            )
        total_weighted_loss += float(loss.detach().cpu()) * float(loss_den.detach().cpu())
        total_loss_den += float(loss_den.detach().cpu())
        logits = out["logit"].detach().float().cpu().tolist()
        labels = batch["label"].detach().float().cpu().tolist()
        valid = out["has_valid_eye_token"].detach().cpu().tolist()
        valid_counts = out["valid_token_count"].detach().float().cpu().tolist()
        task_ids = batch["task_id"].detach().cpu().tolist()
        for i, (label, logit, is_valid, valid_count) in enumerate(zip(labels, logits, valid, valid_counts)):
            if not is_valid:
                continue
            rows.append(
                {
                    "split": split_name,
                    "label": int(label),
                    "logit": float(logit),
                    "prob": sigmoid(float(logit)),
                    "base_subject_id": batch["base_subject_id"][i],
                    "subject_key": batch["subject_key"][i],
                    "subject_id": batch["record_subject_id"][i],
                    "record_subject_id": batch["record_subject_id"][i],
                    "trial_id": batch["trial_id"][i],
                    "task_id": int(task_ids[i]),
                    "path": batch["path"][i],
                    "disease": batch["disease"][i],
                    "group": batch["group"][i],
                    "usable_eye_pattern": batch["usable_eye_pattern"][i],
                    "valid_eye_token_count": float(valid_count),
                }
            )
    labels = [int(row["label"]) for row in rows]
    logits = [float(row["logit"]) for row in rows]
    threshold = float(cfg["downstream_eval"].get("threshold", 0.5))
    metrics = compute_binary_metrics(labels, logits, threshold=threshold, prefix=f"{split_name}/trial")
    metrics[f"{split_name}/weighted_loss"] = total_weighted_loss / total_loss_den if total_loss_den > 0 else math.nan
    subject_rows = aggregate_subject_predictions(rows)
    subject_metrics = compute_binary_metrics(
        [int(row["label"]) for row in subject_rows],
        [float(row["logit"]) for row in subject_rows],
        threshold=threshold,
        prefix=f"{split_name}/subject",
    )
    metrics.update(subject_metrics)
    model.train()
    return metrics, rows, subject_rows


def _metric_improved(value: float, best: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    return value < best if mode == "min" else value > best


def save_downstream_checkpoint(
    path: Path,
    model: DownstreamClassifier,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    mode: str,
    disease: str,
    pretrained_info: dict[str, Any] | None,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "mode": mode,
            "disease": disease,
            "pretrained": pretrained_info,
        },
        path,
    )


def _select_monitor_value(metrics: dict[str, Any], monitor: str) -> tuple[float, str]:
    value = float(metrics.get(monitor, math.nan))
    if math.isfinite(value):
        return value, monitor
    fallback = monitor.replace("auroc", "balanced_accuracy")
    fallback_value = float(metrics.get(fallback, math.nan))
    return fallback_value, fallback


def train_downstream(cfg: dict[str, Any], *, disease: str, mode: str, resume: str | Path | None = None) -> dict[str, Any]:
    set_seed(int(cfg["downstream_train"].get("seed", cfg["train"].get("seed", 42))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(output_dir_for(cfg, disease, mode))
    write_json(out_dir / "config.json", cfg)
    datasets = make_datasets(
        cfg,
        disease,
        max_train_trials=cfg.get("debug", {}).get("max_train_trials"),
        max_eval_trials=cfg.get("debug", {}).get("max_eval_trials"),
    )
    loaders = {
        split: make_downstream_loader(dataset, cfg, train=(split == "train"))
        for split, dataset in datasets.items()
    }
    model = DownstreamClassifier(cfg).to(device)
    pretrained_info = None
    if mode != "scratch":
        pretrained_info = load_pretrained_encoder(model, cfg["downstream"]["pretrained_checkpoint"])
    apply_freeze_mode(
        model,
        mode,
        partial_last_n_layers=int(cfg["downstream"].get("partial_last_n_layers", 4)),
    )
    param_counts = count_parameters(model)
    write_json(out_dir / "param_counts.json", param_counts)
    LOGGER.info("%s/%s parameters: %s", disease, mode, param_counts)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["downstream_train"].get("precision") == "fp16")
    pos_weight = train_subject_pos_weight(datasets["train"], device)
    write_json(out_dir / "class_weight.json", {"pos_weight": float(pos_weight.detach().cpu())})
    writer = make_writer(out_dir, 0)
    monitor = cfg["downstream_checkpoint"].get("monitor", "val/subject/auroc")
    monitor_mode = cfg["downstream_checkpoint"].get("mode", "max")
    best_metric = -math.inf if monitor_mode == "max" else math.inf
    best_epoch = -1
    start_epoch = 0
    if resume is not None:
        resume_checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(resume_checkpoint["model"])
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        best_metric = float(resume_checkpoint.get("best_metric", best_metric))
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        best_epoch = int(resume_checkpoint.get("best_epoch", -1))
        LOGGER.info("Resumed downstream checkpoint %s from epoch=%s", resume, start_epoch)
    patience = int(cfg["downstream_train"].get("early_stopping_patience", 10))
    epochs_without_improve = 0
    max_epochs = int(cfg["downstream_train"].get("max_epochs", 100))
    max_train_batches = cfg.get("debug", {}).get("max_train_batches")
    max_eval_batches = cfg.get("debug", {}).get("max_eval_batches")
    precision = str(cfg["downstream_train"].get("precision", cfg["train"].get("precision", "fp32")))
    grad_clip = float(cfg["downstream_train"].get("grad_clip", 1.0))
    global_step = 0
    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            loss_sum = 0.0
            loss_den_sum = 0.0
            for batch_index, batch in enumerate(loaders["train"]):
                if max_train_batches is not None and batch_index >= int(max_train_batches):
                    break
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    out = model(batch)
                    loss, loss_den = weighted_bce_loss(
                        out["logit"],
                        batch["label"],
                        batch["sample_weight"],
                        out["has_valid_eye_token"],
                        pos_weight,
                    )
                if float(loss_den.detach().cpu()) <= 0:
                    continue
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                loss_sum += float(loss.detach().cpu()) * float(loss_den.detach().cpu())
                loss_den_sum += float(loss_den.detach().cpu())
                if global_step % int(cfg["downstream_train"].get("log_every_steps", 50)) == 0:
                    LOGGER.info(
                        "%s/%s epoch=%s step=%s train_loss=%.5f",
                        disease,
                        mode,
                        epoch,
                        global_step,
                        loss_sum / max(1e-12, loss_den_sum),
                    )
                global_step += 1

            val_metrics, val_rows, val_subject_rows = evaluate_classifier(
                model,
                loaders["val"],
                cfg,
                device,
                split_name="val",
                pos_weight=pos_weight,
                max_batches=max_eval_batches,
            )
            train_loss = loss_sum / loss_den_sum if loss_den_sum > 0 else math.nan
            epoch_metrics = {"epoch": epoch, "train/weighted_loss": train_loss, **val_metrics}
            write_json(out_dir / "metrics_last.json", epoch_metrics)
            for key, value in epoch_metrics.items():
                if isinstance(value, (float, int)) and math.isfinite(float(value)):
                    writer.add_scalar(key, float(value), epoch)
            LOGGER.info(
                "%s/%s epoch=%s train_loss=%.5f val_subject_auroc=%.5g val_subject_f1=%.5g",
                disease,
                mode,
                epoch,
                train_loss,
                val_metrics.get("val/subject/auroc", math.nan),
                val_metrics.get("val/subject/f1", math.nan),
            )
            metric_value, used_monitor = _select_monitor_value(epoch_metrics, monitor)
            if used_monitor != monitor:
                LOGGER.warning("%s is NaN; using fallback monitor %s", monitor, used_monitor)
            if _metric_improved(metric_value, best_metric, monitor_mode):
                best_metric = metric_value
                best_epoch = epoch
                epochs_without_improve = 0
                save_downstream_checkpoint(
                    out_dir / "checkpoint_best.pt",
                    model,
                    optimizer,
                    cfg,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    mode=mode,
                    disease=disease,
                    pretrained_info=pretrained_info,
                )
                write_prediction_csv(out_dir / "val_predictions_best.csv", val_rows)
                write_prediction_csv(out_dir / "val_subject_predictions_best.csv", val_subject_rows)
                write_json(out_dir / "metrics_best.json", epoch_metrics)
            else:
                epochs_without_improve += 1
            save_downstream_checkpoint(
                out_dir / "checkpoint_last.pt",
                model,
                optimizer,
                cfg,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                mode=mode,
                disease=disease,
                pretrained_info=pretrained_info,
            )
            if epochs_without_improve >= patience:
                LOGGER.info("%s/%s early stopping at epoch=%s", disease, mode, epoch)
                break
        best_path = out_dir / "checkpoint_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        final_metrics: dict[str, Any] = {"best_epoch": best_epoch, "best_metric": best_metric}
        for split in cfg["downstream_eval"].get("evaluate_splits", ["train", "val", "test"]):
            metrics, rows, subject_rows = evaluate_classifier(
                model,
                loaders[split],
                cfg,
                device,
                split_name=split,
                pos_weight=pos_weight,
                max_batches=max_eval_batches,
            )
            final_metrics.update(metrics)
            write_prediction_csv(out_dir / f"{split}_predictions.csv", rows)
            write_prediction_csv(out_dir / f"{split}_subject_predictions.csv", subject_rows)
            write_prediction_csv(out_dir / f"trial_predictions_{split}.csv", rows)
            write_prediction_csv(out_dir / f"subject_predictions_{split}.csv", subject_rows)
            threshold = float(cfg["downstream_eval"].get("threshold", 0.5))
            write_json(
                out_dir / f"confusion_matrix_{split}.json",
                {
                    "trial": binary_confusion_matrix(
                        [int(row["label"]) for row in rows],
                        [float(row["logit"]) for row in rows],
                        threshold=threshold,
                    ),
                    "subject": binary_confusion_matrix(
                        [int(row["label"]) for row in subject_rows],
                        [float(row["logit"]) for row in subject_rows],
                        threshold=threshold,
                    ),
                },
            )
        write_json(out_dir / "metrics_final.json", final_metrics)
        write_json(
            out_dir / "run_summary.json",
            make_run_summary(
                cfg,
                datasets,
                disease=disease,
                mode=mode,
                pretrained_info=pretrained_info,
                param_counts=param_counts,
            ),
        )
        return final_metrics
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_downstream_config(args.config)
    if args.split_dir is not None:
        cfg.setdefault("downstream", {})["split_dir"] = args.split_dir
    if args.output_root is not None:
        cfg.setdefault("experiment", {})["output_root"] = args.output_root
        cfg.setdefault("experiment", {})["output_dir"] = None
    if args.output_dir is not None:
        cfg.setdefault("experiment", {})["output_dir"] = args.output_dir
    disease = args.disease or cfg.get("downstream", {}).get("disease")
    mode = args.mode or cfg.get("downstream", {}).get("mode")
    if disease is None:
        raise ValueError("Provide --disease or downstream.disease")
    if mode is None:
        raise ValueError("Provide --mode or downstream.mode")
    if args.max_epochs is not None:
        cfg["downstream_train"]["max_epochs"] = int(args.max_epochs)
    cfg.setdefault("debug", {})
    if args.max_train_batches is not None:
        cfg["debug"]["max_train_batches"] = int(args.max_train_batches)
    if args.max_eval_batches is not None:
        cfg["debug"]["max_eval_batches"] = int(args.max_eval_batches)
    cfg["downstream"]["mode"] = mode
    metrics = train_downstream(cfg, disease=disease, mode=mode, resume=args.resume)
    print(metrics)


if __name__ == "__main__":
    main()
