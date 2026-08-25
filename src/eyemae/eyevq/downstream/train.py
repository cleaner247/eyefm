#!/usr/bin/env python3
"""Fine-tune EyeVQ-BERT on downstream binary classification (e.g. MCI).

Aligns with EyeMAE finetune pattern:
  - PackedDownstreamDataset with subject-stratified splits
  - Subject-level pos_weight for balanced BCE loss
  - Subject-level AUROC / balanced accuracy / F1 evaluation
  - Best checkpoint on val/subject/auroc, early stopping

BERT ingests raw patches directly (no tokenizer needed for finetune).

Usage:
  CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 \
      -m eyemae.eyevq.downstream.train \
      --config configs/eyevq/downstream_mci.yaml --output_dir outputs/eyevq/downstream_mci
"""

from __future__ import annotations

import argparse
import atexit
import logging
import math
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from eyemae.data import load_area_stats
from eyemae.downstream_data import (
    PackedDownstreamDataset,
    collate_downstream_trials,
)
from eyemae.finetune import train_subject_pos_weight, train_subject_class_weight
from eyemae.downstream_metrics import (
    aggregate_subject_predictions,
    aggregate_subject_predictions_multiclass,
    compute_binary_metrics,
    compute_multiclass_metrics,
    sigmoid,
    softmax,
)
from eyemae.utils import atomic_torch_save, get_rank_world, set_seed, write_json
from eyemae.eyevq.config import CONFIG_VERSION, load_bert_checkpoint
from eyemae.eyevq.downstream.model import EyeVQForClassification

LOGGER = logging.getLogger(__name__)

def _ddp_cleanup():
    """Graceful DDP cleanup, called on exit or error."""
    if dist.is_initialized():
        dist.destroy_process_group()


_cleanup_registered = False


def register_cleanup():
    """Register atexit + signal handlers for cleanup."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True
    atexit.register(_ddp_cleanup)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: (atexit.unregister(_ddp_cleanup), _ddp_cleanup(), sys.exit(0)))
        except (ValueError, OSError):
            pass  # signal only works in main thread


# ═══════════════════════════════════════════════════════════════
# Distributed setup
# ═══════════════════════════════════════════════════════════════

def setup_distributed():
    rank, world_size, local_rank = get_rank_world()
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device("cuda", local_rank)
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def setup_logging(rank, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ft")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if rank == 0:
        for h in [logging.FileHandler(output_dir / "train.log"), logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


# ═══════════════════════════════════════════════════════════════
# DataLoader
# ═══════════════════════════════════════════════════════════════

def make_dataloader(dataset, cfg, *, train, rank, world_size):
    tc = cfg["train"]
    bsz = int(tc["batch_size"])
    num_workers = int(tc.get("num_workers", 4))
    if not torch.cuda.is_available():
        num_workers = 0

    sampler = None
    shuffle = train
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=int(tc.get("seed", 42)),
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_downstream_trials,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=int(tc.get("prefetch_factor", 4)) if num_workers > 0 else None,
    )


# ═══════════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════════

def get_lr(step, warmup, max_steps, base_lr, min_lr=0.0):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(progress * math.pi))


def get_encoder_head_lrs(
    step, warmup, max_steps, encoder_lr, head_lr, encoder_min_lr, head_min_lr
):
    """Return independent cosine schedules for encoder and classifier head."""
    return (
        get_lr(step, warmup, max_steps, encoder_lr, encoder_min_lr),
        get_lr(step, warmup, max_steps, head_lr, head_min_lr),
    )


# ═══════════════════════════════════════════════════════════════
# Evaluation (matches EyeMAE downstream_metrics)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, *, split_name, pos_weight, max_batches=None, threshold=0.5, label_type="binary", num_classes=2):
    """Return metrics dict + trial-level prediction rows.

    Supports binary (BCE) and multiclass (CE).
    """
    model.eval()
    all_rows = []
    total_loss = 0.0
    total_den = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        content = batch["content"].transpose(-1, -2).contiguous()
        stim = batch["stim"].transpose(-1, -2).contiguous()

        logits = model(
            stim_patches=stim, eye_patches=content,
            quality=batch["quality"], pad_mask=batch["pad_mask"],
            eye_nonmissing_frac=batch["eye_nonmissing_frac"],
            task_ids=batch["task_id"],
        )

        labels = batch["label"]
        sample_weight = batch["sample_weight"]

        if label_type == "multiclass":
            raw = F.cross_entropy(logits, labels.long(), reduction="none")
            weights = sample_weight.to(dtype=raw.dtype)
        else:
            labels_f = labels.float()
            raw = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels_f, reduction="none")
            class_weight = torch.where(
                labels_f > 0.5,
                pos_weight.to(device=device, dtype=raw.dtype),
                torch.ones((), dtype=raw.dtype, device=device),
            )
            weights = sample_weight.to(dtype=raw.dtype) * class_weight
        den = weights.sum().clamp_min(1e-12)
        total_loss += float((raw * weights).sum().item())
        total_den += float(den.item())

        logits_cpu = logits.detach().float().cpu()
        for i in range(len(labels)):
            row = {
                "split": split_name,
                "label": int(labels[i].item()),
                "base_subject_id": batch["base_subject_id"][i],
                "ml_subject_id": batch.get("ml_subject_id", batch["base_subject_id"])[i],
                "subject_key": batch["subject_key"][i],
                "global_trial_id": batch["global_trial_id"][i],
                "task_id": int(batch["task_id"][i].item()),
                "group": batch["group"][i],
                "usable_eye_pattern": batch["usable_eye_pattern"][i],
            }
            if label_type == "multiclass":
                logit_list = [float(v) for v in logits_cpu[i].tolist()]
                for c in range(num_classes):
                    row[f"logit_{c}"] = logit_list[c]
                probs = softmax(logit_list)
                row["prob"] = probs
                row["pred"] = int(max(range(num_classes), key=lambda c: probs[c]))
            else:
                row["logit"] = float(logits_cpu[i].squeeze().item())
                row["prob"] = sigmoid(row["logit"])
            all_rows.append(row)

    model.train()

    metrics = {f"{split_name}/weighted_loss": total_loss / max(total_den, 1e-12)}

    if label_type == "multiclass":
        trial_labels = [row["label"] for row in all_rows]
        trial_logits = [[row[f"logit_{c}"] for c in range(num_classes)] for row in all_rows]
        metrics.update(compute_multiclass_metrics(trial_labels, trial_logits, num_classes=num_classes, prefix=f"{split_name}/trial"))
        subject_rows = aggregate_subject_predictions_multiclass(all_rows, num_classes)
        subject_labels = [row["label"] for row in subject_rows]
        subject_logits = [[row[f"logit_{c}"] for c in range(num_classes)] for row in subject_rows]
        metrics.update(compute_multiclass_metrics(subject_labels, subject_logits, num_classes=num_classes, prefix=f"{split_name}/subject"))
    else:
        trial_labels = [row["label"] for row in all_rows]
        trial_logits = [row["logit"] for row in all_rows]
        metrics.update(compute_binary_metrics(trial_labels, trial_logits, threshold=threshold, prefix=f"{split_name}/trial"))
        subject_rows = aggregate_subject_predictions(all_rows)
        subject_labels = [row["label"] for row in subject_rows]
        subject_logits = [row["logit"] for row in subject_rows]
        metrics.update(compute_binary_metrics(subject_labels, subject_logits, threshold=threshold, prefix=f"{split_name}/subject"))

    return metrics, all_rows, subject_rows


def find_best_threshold(subject_rows, metric="balanced_accuracy"):
    """Find classification threshold that maximizes a metric on subject-level predictions.

    Sweeps 50 thresholds in [0.01, 0.99]. Returns (best_threshold, best_value).
    """
    labels = [row["label"] for row in subject_rows]
    logits = [row["logit"] for row in subject_rows]

    best_threshold = 0.5
    best_value = -1.0

    for threshold in [i / 100.0 for i in range(1, 100)]:
        metrics = compute_binary_metrics(labels, logits, threshold=threshold)
        value = metrics.get(metric, -1.0)
        if value > best_value:
            best_value = value
            best_threshold = threshold

    return best_threshold, best_value


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ft_mci")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(args.output_dir)

    rank, world_size, local_rank, device = setup_distributed()
    register_cleanup()
    logger = setup_logging(rank, output_dir)
    set_seed(int(cfg["train"].get("seed", 42)) + rank)

    mc = cfg["model"]
    tc = cfg["train"]
    dc = cfg["data"]
    lc = cfg.get("label", {})
    label_type = str(lc.get("type", "binary"))
    num_classes = int(mc.get("num_classes", 2 if label_type == "binary" else lc.get("num_classes", 5)))

    # ── 1. Load BERT pretrained weights ──
    bert_ckpt = mc["bert_checkpoint"]
    if is_rank0():
        logger.info(f"Loading BERT from {bert_ckpt}")

    bert, bert_cfg, ckpt = load_bert_checkpoint(bert_ckpt, torch.device("cpu"))
    if is_rank0():
        logger.info(
            "BERT loaded strictly from embedded config "
            f"(d={bert.d_model}, layers={len(bert.transformer)}, step={ckpt.get('step', '?')})"
        )

    model = EyeVQForClassification(
        bert, num_classes=num_classes, dropout=mc["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_rank0():
        logger.info(f"Model: {n_params:,} params ({n_trainable:,} trainable)")

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model

    # ── 2. Datasets (packed_mmap downstream format, aligned with EyeMAE) ──
    area_stats = load_area_stats(dc["area_stats_path"])
    data_dir = dc["data_dir"]

    # Build a minimal cfg for PackedDownstreamDataset (label/patch/preprocess)
    downstream_cfg = {
        "data": {
            "data_dir": data_dir,
            "format": "packed_mmap",
            "require_any_eye_keep": bool(dc.get("require_any_eye_keep", True)),
            "max_open_shards_per_worker": 44,
            "validate_offsets": False,
        },
        "patch": dict(bert_cfg["patch"]),
        "area": {
            "stats_path": dc["area_stats_path"],
            "use_log1p": bool(bert_cfg.get("area", {}).get("use_log1p", True)),
            "per_eye": bool(bert_cfg.get("area", {}).get("per_eye", False)),
            "clip": float(bert_cfg.get("area", {}).get("clip", 5.0)),
            "eps": float(bert_cfg.get("area", {}).get("eps", 1e-6)),
            "mad_scale": float(bert_cfg.get("area", {}).get("mad_scale", 1.4826)),
            "mad_floor": float(bert_cfg.get("area", {}).get("mad_floor", 0.0)),
            "min_subject_valid_frames": int(
                bert_cfg.get("area", {}).get("min_subject_valid_frames", 0)
            ),
        },
        "normalization": {"x_clip_deg": 30.0, "y_clip_deg": 20.0},
        "label": {"missing_value": 2, "blink_value": 1, "nonblink_value": 0,
                  "type": label_type, "num_classes": num_classes},
        "attention": {"min_nonmissing_frac_for_eye_token": 0.05},
        "downstream": {"split_dir": str(Path(dc.get("split_dir", "")))},
    }

    train_index = str(Path(data_dir) / dc["train_index"])
    val_index = str(Path(data_dir) / dc["val_index"])
    test_index = str(Path(data_dir) / dc["test_index"])

    train_ds = PackedDownstreamDataset(
        data_dir, train_index, downstream_cfg,
        area_stats=area_stats,
    )
    train_counts = dict(Counter(row["ml_subject_id"] for row in train_ds.rows))
    val_ds = PackedDownstreamDataset(
        data_dir, val_index, downstream_cfg,
        area_stats=area_stats,
        train_subject_trial_counts=train_counts,
    )
    test_ds = PackedDownstreamDataset(
        data_dir, test_index, downstream_cfg,
        area_stats=area_stats,
        train_subject_trial_counts=train_counts,
    )

    if is_rank0():
        logger.info(f"Train: {len(train_ds)} trials | Val: {len(val_ds)} trials | Test: {len(test_ds)} trials")

    # ── 3. DataLoaders ──
    train_loader = make_dataloader(train_ds, cfg, train=True, rank=rank, world_size=world_size)
    val_loader = make_dataloader(val_ds, cfg, train=False, rank=0, world_size=1) if is_rank0() else None
    test_loader = make_dataloader(test_ds, cfg, train=False, rank=0, world_size=1) if is_rank0() else None

    # ── 4. Class weight (subject-level) ──
    if label_type == "multiclass":
        class_weight = train_subject_class_weight(train_ds, num_classes, device)
        pos_weight = None  # not used for multiclass
        if is_rank0():
            logger.info(f"Class weight: {[float(v) for v in class_weight.cpu().tolist()]}")
            write_json(output_dir / "class_weight.json", {
                "class_weight": [float(v) for v in class_weight.cpu().tolist()],
                "source": "train_subject_class_weight",
            })
    else:
        class_weight = None
        pos_weight = train_subject_pos_weight(train_ds, device)
        if is_rank0():
            logger.info(f"Pos weight: {pos_weight.item():.4f} (neg/pos subject ratio in train)")
            write_json(output_dir / "class_weight.json", {
                "class_weight_for_label": {"0": 1.0, "1": float(pos_weight.item())},
                "source": "train_subject_pos_weight",
            })

    # ── 5. Optimizer (separate encoder/head LRs) ──
    head_params = []
    encoder_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            head_params.append(param)
        else:
            encoder_params.append(param)

    head_lr = float(tc.get("head_lr", tc["lr"]))
    encoder_lr = float(tc.get("encoder_lr", tc["lr"]))
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": tc["weight_decay"]},
        {"params": head_params, "lr": head_lr, "weight_decay": tc["weight_decay"]},
    ], betas=tuple(tc.get("betas", [0.9, 0.95])))

    steps_per_epoch = len(train_loader)
    total_steps = tc["epochs"] * steps_per_epoch
    if "warmup_epochs" in tc:
        warmup_steps = int(tc["warmup_epochs"]) * steps_per_epoch
    else:
        warmup_steps = int(total_steps * tc.get("warmup_ratio", 0.05))
    min_lr = float(tc.get("min_lr", tc["lr"] / 10))
    encoder_min_lr = float(tc.get("encoder_min_lr", min_lr))
    head_min_lr = float(tc.get("head_min_lr", min_lr))

    if is_rank0():
        logger.info(f"Epochs: {tc['epochs']} | Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
        logger.info(
            f"LR: enc={encoder_lr:.1e}->{encoder_min_lr:.1e} "
            f"head={head_lr:.1e}->{head_min_lr:.1e} | Warmup: {warmup_steps}"
        )
        logger.info("=" * 60)

    # ── 6. Training loop ──
    best_auroc = -1.0
    best_epoch = -1
    global_step = 0
    patience = int(tc.get("early_stopping_patience", 15))
    epochs_without_improve = 0

    for epoch in range(tc["epochs"]):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_den = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            content = batch["content"].transpose(-1, -2).contiguous()
            stim = batch["stim"].transpose(-1, -2).contiguous()

            encoder_step_lr, head_step_lr = get_encoder_head_lrs(
                global_step,
                warmup_steps,
                total_steps,
                encoder_lr,
                head_lr,
                encoder_min_lr,
                head_min_lr,
            )
            optimizer.param_groups[0]["lr"] = encoder_step_lr
            optimizer.param_groups[1]["lr"] = head_step_lr

            logits = model(
                stim_patches=stim, eye_patches=content,
                quality=batch["quality"], pad_mask=batch["pad_mask"],
                eye_nonmissing_frac=batch["eye_nonmissing_frac"],
                task_ids=batch["task_id"],
            )

            labels = batch["label"]
            sample_weight = batch["sample_weight"]

            if label_type == "multiclass":
                raw = F.cross_entropy(logits, labels.long(), reduction="none")
                weights = sample_weight.to(dtype=raw.dtype)
                # Apply class weight
                if class_weight is not None:
                    weights = weights * class_weight[labels.long()].to(device=device, dtype=raw.dtype)
            else:
                labels_f = labels.float()
                raw = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels_f, reduction="none")
                cw = torch.where(
                    labels_f > 0.5,
                    pos_weight.to(device=device, dtype=raw.dtype),
                    torch.ones((), dtype=raw.dtype, device=device),
                )
                weights = sample_weight.to(dtype=raw.dtype) * cw
            den = weights.sum().clamp_min(1e-12)
            loss = (raw * weights).sum() / den

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tc.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()

            epoch_loss += float(loss.item()) * float(den.item())
            epoch_den += float(den.item())

            if is_rank0() and global_step % 50 == 0:
                if label_type == "multiclass":
                    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                else:
                    acc = ((logits.squeeze(-1) > 0).float() == labels.float()).float().mean().item()
                logger.info(
                    f"E{epoch} step={global_step:05d} loss={loss.item():.4f} "
                    f"acc={acc:.4f} lr_enc={encoder_step_lr:.2e} "
                    f"lr_head={head_step_lr:.2e}"
                )

            global_step += 1

        avg_train_loss = epoch_loss / max(epoch_den, 1e-12)
        epoch_time = time.time() - t0

        # ── Validation ──
        should_stop = False
        if is_rank0():
            val_metrics, _, _ = evaluate(
                raw_model, val_loader, device,
                split_name="val", pos_weight=pos_weight,
                label_type=label_type, num_classes=num_classes,
            )
            # Monitor metric
            if label_type == "multiclass":
                monitor_val = val_metrics.get("val/subject/macro_auroc_ovr", float("nan"))
                monitor_name = "macro_auroc_ovr"
            else:
                monitor_val = val_metrics.get("val/subject/auroc", float("nan"))
                monitor_name = "auroc"
            bacc = val_metrics.get("val/subject/balanced_accuracy", float("nan"))

            logger.info(
                f"E{epoch} train_loss={avg_train_loss:.4f} "
                f"val_subj_{monitor_name}={monitor_val:.4f} val_subj_bacc={bacc:.4f} "
                f"time={epoch_time:.0f}s"
            )

            improved = monitor_val > best_auroc if math.isfinite(monitor_val) else False
            if improved:
                best_auroc = monitor_val
                best_epoch = epoch
                epochs_without_improve = 0
                atomic_torch_save({
                    "epoch": epoch, "step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auroc": best_auroc,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                    "bert_cfg": bert_cfg,
                    "config_version": CONFIG_VERSION,
                }, output_dir / "ckpt_best.pt")
                logger.info(f"  🏆 New best: subject/{monitor_name}={best_auroc:.4f}")
            else:
                epochs_without_improve += 1

            atomic_torch_save({
                "epoch": epoch, "step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "bert_cfg": bert_cfg,
                "config_version": CONFIG_VERSION,
            }, output_dir / "ckpt_last.pt")

            write_json(output_dir / "metrics_last.json",
                       {"epoch": epoch, "train/weighted_loss": avg_train_loss, **val_metrics})

            if epochs_without_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs w/o improvement (best epoch={best_epoch})")
                should_stop = True

        # Rank 0 performs validation; every worker must wait for its decision.
        # Otherwise rank 0 can stop while peers block forever in the next DDP forward.
        stop_tensor = torch.tensor(int(should_stop), device=device)
        if world_size > 1:
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
        if bool(stop_tensor.item()):
            break

    # ── 7. Test evaluation ──
    if is_rank0() and best_epoch >= 0:
        logger.info("=" * 60)
        logger.info("Loading best checkpoint for test evaluation...")
        best_ckpt = torch.load(output_dir / "ckpt_best.pt", map_location=device, weights_only=False)
        raw_model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Best checkpoint from epoch {best_ckpt['epoch']} ({'macro_auroc_ovr' if label_type == 'multiclass' else 'auroc'}={best_ckpt['best_auroc']:.4f})")

        if label_type == "multiclass":
            # No threshold tuning for multiclass (argmax decision)
            test_metrics, _, _ = evaluate(
                raw_model, test_loader, device,
                split_name="test", pos_weight=pos_weight,
                label_type=label_type, num_classes=num_classes,
            )
            logger.info(
                f"Test: subj_acc={test_metrics.get('test/subject/accuracy', float('nan')):.4f} "
                f"subj_macro_auroc={test_metrics.get('test/subject/macro_auroc_ovr', float('nan')):.4f} "
                f"subj_bacc={test_metrics.get('test/subject/balanced_accuracy', float('nan')):.4f}"
            )
            write_json(output_dir / "metrics_test.json", test_metrics)
        else:
            # Binary: threshold tuning on val
            _, _, val_subject_rows = evaluate(
                raw_model, val_loader, device,
                split_name="val", pos_weight=pos_weight,
                label_type=label_type, num_classes=num_classes,
            )
            best_thr, best_val_bacc = find_best_threshold(val_subject_rows, metric="balanced_accuracy")
            logger.info(f"Threshold tuned on val: thr={best_thr:.2f} (max val_subj_bacc={best_val_bacc:.4f})")

            test_metrics, _, _ = evaluate(
                raw_model, test_loader, device,
                split_name="test", pos_weight=pos_weight, threshold=best_thr,
            )
            logger.info(
                f"Test (thr={best_thr:.2f}): "
                f"subj_auroc={test_metrics.get('test/subject/auroc', float('nan')):.4f} "
                f"subj_bacc={test_metrics.get('test/subject/balanced_accuracy', float('nan')):.4f} "
                f"subj_f1={test_metrics.get('test/subject/f1', float('nan')):.4f}"
            )

            test_metrics_05, _, _ = evaluate(
                raw_model, test_loader, device,
                split_name="test", pos_weight=pos_weight, threshold=0.5,
            )
            logger.info(
                f"Test (thr=0.50): "
                f"subj_auroc={test_metrics_05.get('test/subject/auroc', float('nan')):.4f} "
                f"subj_bacc={test_metrics_05.get('test/subject/balanced_accuracy', float('nan')):.4f} "
                f"subj_f1={test_metrics_05.get('test/subject/f1', float('nan')):.4f}"
            )

            write_json(output_dir / "metrics_test.json", {
                "tuned_threshold": best_thr,
                "val_bacc_at_tuned_thr": best_val_bacc,
                "test_tuned": test_metrics,
                "test_default_05": test_metrics_05,
            })

    # Non-zero ranks wait while rank 0 evaluates the best checkpoint.
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if is_rank0():
        logger.info(f"Done. Best epoch={best_epoch} auroc={best_auroc:.4f}")


if __name__ == "__main__":
    main()
