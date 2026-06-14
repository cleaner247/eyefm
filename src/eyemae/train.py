from __future__ import annotations

import argparse
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .batching import TokenBatchSampler
from .config import load_config, split_path_for_name, validate_config
from .data import TrialDataset, collate_trials
from .losses import compute_reconstruction_loss
from .masking import generate_mae_mask
from .metrics import finalize_metric_sums, new_metric_sums, reduce_metric_sums, update_metric_sums
from .model import build_model
from .utils import atomic_torch_save, cosine_lr, ensure_dir, get_rank_world, is_rank0, set_seed, setup_logging, write_json
from .visualize import save_visualizations


LOGGER = logging.getLogger(__name__)


class NoOpWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_writer(out_dir: Path, rank: int):
    if rank != 0:
        return NoOpWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(out_dir / "tensorboard"))
    except Exception:
        return NoOpWriter()


def setup_distributed(cfg: dict[str, Any]) -> tuple[int, int, int, torch.device]:
    rank, world_size, local_rank = get_rank_world()
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device("cuda", local_rank)
        else:
            backend = "gloo"
            device = torch.device("cpu")
        torch.distributed.init_process_group(backend=backend)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_loader(
    dataset: TrialDataset,
    cfg: dict[str, Any],
    *,
    train: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, Any]:
    num_workers = int(cfg["train"].get("num_workers", 0)) if train else max(0, min(2, int(cfg["train"].get("num_workers", 0))))
    if world_size == 1 and not torch.cuda.is_available():
        num_workers = 0
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg["train"].get("pin_memory", True)) and torch.cuda.is_available(),
        "persistent_workers": bool(cfg["train"].get("persistent_workers", False)) and num_workers > 0,
        "collate_fn": collate_trials,
    }
    if train and cfg["train"].get("batch_trials_per_gpu") is None and cfg["train"].get("max_seq_tokens_per_gpu") is not None:
        max_seq_tokens = int(cfg["train"]["max_seq_tokens_per_gpu"])
        configured_max_trials = int(cfg["train"]["max_trials_per_gpu"])
        max_patches = int(cfg["model"]["max_patches"])
        max_trials = min(configured_max_trials, max(64, max_seq_tokens // max(1, 3 * max_patches)))
        if rank == 0 and max_trials < configured_max_trials:
            LOGGER.info(
                "Using effective max_trials_per_gpu=%s below configured upper bound=%s for bucketed dynamic batches",
                max_trials,
                configured_max_trials,
            )
        sampler = TokenBatchSampler(
            dataset,
            max_seq_tokens=max_seq_tokens,
            max_trials=max_trials,
            shuffle=True,
            seed=int(cfg["train"]["seed"]),
            bucket_by_length=bool(cfg["train"].get("bucket_by_length", False)),
            infinite=True,
            rank=rank,
            world_size=world_size,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common), sampler
    sampler = None
    shuffle = train and world_size == 1
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train)
        shuffle = False
    batch_size = int(cfg["train"].get("batch_trials_per_gpu") or min(16, int(cfg["train"].get("max_trials_per_gpu") or 16)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False, **common), sampler


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    cfg: dict[str, Any],
    best_metric: float,
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "config": cfg,
        "area_stats_path": cfg["area"]["stats_path"],
        "best_metric": float(best_metric),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    atomic_torch_save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("global_step", 0)), int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_metric", math.inf))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    global_step: int,
    rank: int,
    save_viz: bool = False,
) -> dict[str, float]:
    model.eval()
    sums = new_metric_sums(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["eval"]["seed"]) + int(global_step))
    first_batch = None
    first_pred = None
    first_mask = None
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        mae_mask, _ = generate_mae_mask(batch, cfg, generator=generator)
        out = model(
            batch["content"],
            batch["quality"],
            batch["stim"],
            batch["task_id"],
            batch["pad_mask"],
            batch["eye_token_valid"],
            mae_mask,
        )
        loss, stats = compute_reconstruction_loss(
            out["pred"], batch["content"], batch["quality"], mae_mask, batch["pad_mask"], batch["eye_token_valid"], cfg
        )
        update_metric_sums(sums, out["pred"], batch["content"], batch["quality"], mae_mask, batch["pad_mask"], batch["eye_token_valid"], stats, cfg)
        if first_batch is None:
            first_batch = batch
            first_pred = out["pred"]
            first_mask = mae_mask
    reduce_metric_sums(sums)
    metrics = finalize_metric_sums(sums, prefix="val", cfg=cfg)
    if save_viz and rank == 0 and first_batch is not None:
        save_visualizations(
            first_batch,
            first_pred,
            first_mask,
            Path(cfg["experiment"]["output_dir"]) / "visualizations" / f"step_{global_step}",
            cfg,
            max_trials=16,
        )
    model.train()
    return metrics


def train_main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.overfit_trials is not None:
        cfg.setdefault("debug", {})["overfit_trials"] = int(args.overfit_trials)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
        cfg["train"]["val_every_steps"] = min(int(cfg["train"]["val_every_steps"]), int(args.max_steps))
        cfg["train"]["save_every_steps"] = min(int(cfg["train"]["save_every_steps"]), int(args.max_steps))
    validate_config(cfg)
    rank, world_size, _local_rank, device = setup_distributed(cfg)
    setup_logging(rank)
    set_seed(int(cfg["train"]["seed"]) + rank)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    if rank == 0:
        write_json(out_dir / "config.json", cfg)

    max_trials = int(cfg.get("debug", {}).get("overfit_trials", 0) or 0) if args.overfit_trials is not None else None
    train_dataset = TrialDataset(cfg["data"]["data_dir"], split_path_for_name(cfg, "pretrain_train"), cfg, max_trials=max_trials)
    val_dataset = TrialDataset(cfg["data"]["data_dir"], split_path_for_name(cfg, "pretrain_val"), cfg, max_trials=max_trials)
    train_loader, train_sampler = make_loader(train_dataset, cfg, train=True, rank=rank, world_size=world_size)
    val_loader, _ = make_loader(val_dataset, cfg, train=False, rank=rank, world_size=world_size)

    model = build_model(cfg).to(device)
    if world_size > 1:
        ddp_kwargs = {"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(float(v) for v in cfg["train"]["betas"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["train"]["precision"] == "fp16")
    global_step = 0
    start_epoch = 0
    best_metric = math.inf
    if args.resume:
        global_step, start_epoch, best_metric = load_checkpoint(args.resume, model, optimizer)
        LOGGER.info("Resumed from %s at step=%s epoch=%s", args.resume, global_step, start_epoch)
    writer = make_writer(out_dir, rank)
    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    precision = str(cfg["train"]["precision"])
    epoch = start_epoch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < max_steps:
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(train_loader):
                if global_step >= max_steps:
                    break
                batch = move_batch_to_device(batch, device)
                generator = torch.Generator(device=device)
                generator.manual_seed(int(cfg["train"]["seed"]) * 1000003 + global_step * 31 + rank)
                mae_mask, _mask_type = generate_mae_mask(batch, cfg, generator=generator)
                lr = cosine_lr(
                    float(cfg["train"]["lr"]),
                    float(cfg["train"]["min_lr"]),
                    global_step,
                    max_steps,
                    int(cfg["train"]["warmup_steps"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with autocast_context(device, precision):
                    out = model(
                        batch["content"],
                        batch["quality"],
                        batch["stim"],
                        batch["task_id"],
                        batch["pad_mask"],
                        batch["eye_token_valid"],
                        mae_mask,
                    )
                    loss, stats = compute_reconstruction_loss(
                        out["pred"],
                        batch["content"],
                        batch["quality"],
                        mae_mask,
                        batch["pad_mask"],
                        batch["eye_token_valid"],
                        cfg,
                    )
                    loss = loss / grad_accum
                if float(stats["total_denominator"].detach().cpu()) <= 0:
                    if rank == 0:
                        LOGGER.warning("Skipping optimizer step with zero reconstruction denominator at step=%s", global_step)
                    global_step += 1
                    continue
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (global_step + 1) % grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if rank == 0 and global_step % int(cfg["train"]["log_every_steps"]) == 0:
                    LOGGER.info(
                        "step=%s loss=%.5f xy=%.5f area=%.5f blink=%.5f vel=%.5f lr=%.2e",
                        global_step,
                        float(stats["total_loss"].cpu()),
                        float(stats["xy_loss"].cpu()),
                        float(stats["area_loss"].cpu()),
                        float(stats["blink_loss"].cpu()),
                        float(stats["velocity_loss"].cpu()),
                        lr,
                    )
                    writer.add_scalar("train/total_loss", float(stats["total_loss"].cpu()), global_step)
                    writer.add_scalar("train/xy_loss", float(stats["xy_loss"].cpu()), global_step)
                    writer.add_scalar("train/blink_loss", float(stats["blink_loss"].cpu()), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                do_val = (global_step > 0 and global_step % int(cfg["train"]["val_every_steps"]) == 0) or global_step == max_steps - 1
                if do_val:
                    metrics = validate(model, val_loader, cfg, device, global_step=global_step, rank=rank, save_viz=True)
                    if rank == 0:
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, global_step)
                        write_json(out_dir / "metrics_last.json", metrics)
                        LOGGER.info(
                            "val step=%s %s",
                            global_step,
                            " ".join(f"{key}={value:.5g}" for key, value in sorted(metrics.items())),
                        )
                        monitor = cfg["checkpoint"]["monitor"]
                        value = metrics.get(monitor, metrics.get("val/total_loss", math.inf))
                        improved = value < best_metric if cfg["checkpoint"].get("mode", "min") == "min" else value > best_metric
                        if improved:
                            best_metric = value
                            save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, global_step, epoch, cfg, best_metric)

                if rank == 0 and (
                    (global_step > 0 and global_step % int(cfg["train"]["save_every_steps"]) == 0)
                    or global_step == max_steps - 1
                ):
                    save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, global_step, epoch, cfg, best_metric)
                    if global_step > 0:
                        save_checkpoint(out_dir / f"checkpoint_step_{global_step:08d}.pt", model, optimizer, global_step, epoch, cfg, best_metric)
                global_step += 1
            epoch += 1
    finally:
        writer.close()
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overfit_trials", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
