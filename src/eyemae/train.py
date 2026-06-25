from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .batching import TokenBatchSampler
from .config import load_config, split_path_for_name, validate_config
from .data import audit_packed_pretrain_splits, collate_trials, make_trial_dataset
from .eval_artifacts import (
    VizCollector,
    finalize_group_metrics,
    flatten_group_metrics,
    new_group_metric_sums,
    reduce_group_metric_sums,
    update_group_metrics,
)
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


def _timed_now(device: torch.device, enabled: bool) -> float:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class _DataPrefetcher:
    """Overlaps data loading with GPU compute: prefetches the next batch while
    the current one is being processed, using a dedicated CUDA stream so that
    CPU→GPU copies run concurrently with the main compute stream."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def __iter__(self):
        loader_iter = iter(self._loader)
        stream = self._stream
        device = self._device

        try:
            batch = next(loader_iter)
        except StopIteration:
            return
        if stream is not None:
            with torch.cuda.stream(stream):
                next_batch = move_batch_to_device(batch, device)
        else:
            next_batch = move_batch_to_device(batch, device)

        for batch in loader_iter:
            if stream is not None:
                torch.cuda.current_stream(device).wait_stream(stream)
            yield next_batch
            if stream is not None:
                with torch.cuda.stream(stream):
                    next_batch = move_batch_to_device(batch, device)
            else:
                next_batch = move_batch_to_device(batch, device)

        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        yield next_batch


def _interval_enabled(cfg_value: Any, global_step: int, max_steps: int, *, default: bool) -> bool:
    if cfg_value is None:
        return default
    interval = int(cfg_value)
    if interval <= 0:
        return False
    return global_step % interval == 0 or global_step == max_steps - 1


def _make_profiler_context(
    enabled: bool,
    device: torch.device,
    trace_dir: Path | None,
    global_step: int,
):
    if not enabled or trace_dir is None:
        return nullcontext()
    trace_step_dir = trace_dir / f"step_{global_step:08d}"
    trace_step_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # No schedule → records everything between __enter__ / __exit__,
    # capturing the full accumulation cycle (all micro-batches).
    prof = torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_step_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    return prof


def _compute_flops(model: torch.nn.Module, batch: dict[str, Any]) -> float:
    """Return approximate FLOPs for one micro-batch (forward + backward + optimizer)."""
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    content = batch["content"]
    bsz = int(content.shape[0])
    num_patches = int(content.shape[1])
    seq_len = num_patches * 3
    hidden = int(getattr(raw_model, "d_model", content.shape[-1]))
    num_layers = len(getattr(raw_model, "blocks", []))
    if num_layers == 0:
        return 0.0
    first_block = raw_model.blocks[0]
    ffn_hidden = int(getattr(first_block.ffn.w12, "out_features", hidden * 4) // 2)

    # Transformer: attention + FFN per layer
    attn_flops = 2.0 * bsz * seq_len * seq_len * hidden + 4.0 * bsz * seq_len * hidden * hidden
    ffn_flops = 8.0 * bsz * seq_len * hidden * ffn_hidden
    block_flops = float(num_layers) * (attn_flops + ffn_flops)

    # Tokenizers: 3 × Conv1d + Linear projection (small, ~2% of total)
    patch_samples = int(raw_model.patch) if hasattr(raw_model, "patch") else 20
    tok_flops = 3.0 * bsz * num_patches * hidden * patch_samples * 4  # rough conv estimate
    tok_flops += 3.0 * bsz * num_patches * hidden * hidden  # projection

    # Loss: MSE per predicted element (~negligible)
    loss_flops = bsz * num_patches * 2 * patch_samples * 4 * 2.0

    # Optimizer (AdamW): ~5 elementwise ops per parameter
    num_params = sum(p.numel() for p in raw_model.parameters())
    opt_flops = 5.0 * num_params

    total_flops = (block_flops + tok_flops + loss_flops) * 3.0 + opt_flops
    return total_flops


def _estimate_tflops(model: torch.nn.Module, batch: dict[str, Any], step_seconds: float) -> float:
    return _compute_flops(model, batch) / max(step_seconds, 1e-9) / 1e12


def make_loader(
    dataset: Dataset,
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
    if num_workers > 0 and cfg["train"].get("prefetch_factor") is not None:
        common["prefetch_factor"] = int(cfg["train"]["prefetch_factor"])
    if train and cfg["train"].get("batch_trials_per_gpu") is None and cfg["train"].get("max_seq_tokens_per_gpu") is not None:
        max_seq_tokens = int(cfg["train"]["max_seq_tokens_per_gpu"])
        configured_max_trials = int(cfg["train"]["max_trials_per_gpu"])
        max_trials = configured_max_trials
        if rank == 0:
            LOGGER.info(
                "Using token-based dynamic batches: max_seq_tokens_per_gpu=%s max_trials_per_gpu=%s",
                max_seq_tokens,
                max_trials,
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
            nmax_multiple=int(cfg["train"].get("pad_to_tokens", 0) or 0),
            drop_last=True,
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
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    atomic_torch_save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng_states = checkpoint.get("rng_states") or {}
    if "python" in rng_states:
        random.setstate(rng_states["python"])
    if "numpy" in rng_states:
        np.random.set_state(rng_states["numpy"])
    if "torch" in rng_states:
        torch.set_rng_state(rng_states["torch"])
    if torch.cuda.is_available() and rng_states.get("cuda"):
        torch.cuda.set_rng_state_all(rng_states["cuda"])
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
    compute_group_metrics: bool = True,
    log_timing: bool = False,
) -> dict[str, float]:
    model.eval()
    timing = {
        "data_wait": 0.0,
        "to_device": 0.0,
        "mask": 0.0,
        "forward_loss": 0.0,
        "metrics": 0.0,
        "reduce_finalize": 0.0,
        "viz": 0.0,
    }
    total_start = _timed_now(device, log_timing)
    prev_batch_end = total_start
    batches = 0
    sums = new_metric_sums(device)
    group_sums = new_group_metric_sums(device) if compute_group_metrics else None
    generator = torch.Generator(device=device)
    eval_seed = int(cfg["eval"]["seed"])
    if not bool(cfg["eval"].get("fixed_mask", True)):
        eval_seed += int(global_step)
    generator.manual_seed(eval_seed)
    viz_collector = VizCollector(max_trials=16, seed=eval_seed) if save_viz and rank == 0 else None
    for batch in loader:
        batch_start = _timed_now(device, log_timing)
        timing["data_wait"] += batch_start - prev_batch_end
        batch = move_batch_to_device(batch, device)
        after_move = _timed_now(device, log_timing)
        timing["to_device"] += after_move - batch_start
        mae_mask, mask_type = generate_mae_mask(batch, cfg, generator=generator)
        after_mask = _timed_now(device, log_timing)
        timing["mask"] += after_mask - after_move
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
        after_forward_loss = _timed_now(device, log_timing)
        timing["forward_loss"] += after_forward_loss - after_mask
        update_metric_sums(sums, out["pred"], batch["content"], batch["quality"], mae_mask, batch["pad_mask"], batch["eye_token_valid"], stats, cfg)
        if group_sums is not None:
            update_group_metrics(group_sums, batch, out["pred"], mae_mask, mask_type, cfg)
        if viz_collector is not None:
            viz_collector.observe(batch, out["pred"], mae_mask, mask_type)
        after_metrics = _timed_now(device, log_timing)
        timing["metrics"] += after_metrics - after_forward_loss
        prev_batch_end = after_metrics
        batches += 1
    reduce_start = _timed_now(device, log_timing)
    reduce_metric_sums(sums)
    if group_sums is not None:
        reduce_group_metric_sums(group_sums)
    metrics = finalize_metric_sums(sums, prefix="val", cfg=cfg)
    if group_sums is not None:
        group_metrics = finalize_group_metrics(group_sums, cfg)
        metrics.update(flatten_group_metrics(group_metrics))
        if rank == 0:
            write_json(Path(cfg["experiment"]["output_dir"]) / "metrics_last_by_group.json", group_metrics)
    reduce_end = _timed_now(device, log_timing)
    timing["reduce_finalize"] += reduce_end - reduce_start
    if viz_collector is not None:
        viz_start = _timed_now(device, log_timing)
        viz_batch = viz_collector.build_batch()
        if viz_batch is None:
            model.train()
            return metrics
        save_visualizations(
            viz_batch[0],
            viz_batch[1],
            viz_batch[2],
            Path(cfg["experiment"]["output_dir"]) / "visualizations" / f"step_{global_step}",
            cfg,
            max_trials=16,
        )
        viz_end = _timed_now(device, log_timing)
        timing["viz"] += viz_end - viz_start
    if log_timing and rank == 0:
        total = _timed_now(device, True) - total_start
        LOGGER.info(
            "val_timing step=%s batches=%s total=%.3fs data_wait=%.3fs to_device=%.3fs mask=%.3fs "
            "forward_loss=%.3fs metrics=%.3fs reduce_finalize=%.3fs viz=%.3fs group_metrics=%s save_viz=%s",
            global_step,
            batches,
            total,
            timing["data_wait"],
            timing["to_device"],
            timing["mask"],
            timing["forward_loss"],
            timing["metrics"],
            timing["reduce_finalize"],
            timing["viz"],
            compute_group_metrics,
            save_viz,
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
    packed_audit = audit_packed_pretrain_splits(cfg)
    rank, world_size, _local_rank, device = setup_distributed(cfg)
    setup_logging(rank)
    set_seed(int(cfg["train"]["seed"]) + rank)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])

    # ── TF32: A100 Tensor Cores can use TF32 for fp32 accumulations in matmuls.
    #     This is a free ~30-50% GEMM speedup with no model quality impact.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if rank == 0:
        write_json(out_dir / "config.json", cfg)
        if packed_audit:
            write_json(out_dir / "packed_pretrain_audit.json", packed_audit)

    max_trials = int(cfg.get("debug", {}).get("overfit_trials", 0) or 0) if args.overfit_trials is not None else None
    train_dataset = make_trial_dataset(cfg, split_path_for_name(cfg, "train"), max_trials=max_trials)
    val_dataset = make_trial_dataset(cfg, split_path_for_name(cfg, "validation"), max_trials=max_trials)
    train_loader, train_sampler = make_loader(train_dataset, cfg, train=True, rank=rank, world_size=world_size)
    train_loader = _DataPrefetcher(train_loader, device)
    val_loader, _ = make_loader(val_dataset, cfg, train=False, rank=rank, world_size=world_size)

    model = build_model(cfg).to(device)
    # Safety net — keep the filter for unexpected non‑canonical parameters.
    warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
    if world_size > 1:
        ddp_kwargs = {"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
        # ── DDP static graph: our computation graph is identical every
        #     micro-batch (same shapes, same ops).  Telling DDP this lets it
        #     pre-allocate gradient buckets and skip per-iteration bookkeeping,
        #     cutting the autograd-callback gap by ~30-50 %.
        model._set_static_graph()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(float(v) for v in cfg["train"]["betas"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        fused=True if device.type == "cuda" else False,  # fused kernel: fewer kernel launches
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["train"]["precision"] == "fp16")
    global_step = 0
    start_epoch = 0
    best_metric = math.inf
    # ── Early stopping ──
    _early_stop_metric = "val/masked_xy_rmse_deg"
    _early_stop_patience = int(cfg["train"].get("early_stop_patience", 2))
    _early_stop_count = 0
    _early_stop_best = math.inf
    if args.resume:
        global_step, start_epoch, best_metric = load_checkpoint(args.resume, model, optimizer)
        LOGGER.info("Resumed from %s at step=%s epoch=%s", args.resume, global_step, start_epoch)
    writer = make_writer(out_dir, rank)
    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    precision = str(cfg["train"]["precision"])
    timing_every = int(cfg["train"].get("timing_every_steps") or 0)
    profile_step = int(cfg["train"].get("profile_step", -1))
    profile_trace_dir = None
    if profile_step >= 0 and rank == 0:
        profile_trace_dir = out_dir / "profiler"
        profile_trace_dir.mkdir(parents=True, exist_ok=True)
    epoch = start_epoch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cycle_start = time.perf_counter()
    accumulated_flops = 0.0
    micro_step = 0
    profiling_now = False
    profiler_obj = None
    # Memory tracking — baseline after first few steps settle
    _mem_baseline_allocated: float | None = None
    _mem_warmup_steps = 5  # steps to skip before recording baseline
    _mem_leak_check_every = 200  # check for upward trend every N steps
    _mem_history: list[float] = []  # last few allocated-GB readings
    _seed = int(cfg["train"]["seed"])
    # Dedicated stream + generator for MAE mask GPU ops.  The .cpu() sync
    # only blocks this side-stream, not the default compute stream, so mask
    # generation overlaps with the previous micro-batch's backward pass.
    _mask_stream = torch.cuda.Stream() if device.type == "cuda" else None
    _mask_generator = torch.Generator(device=device) if device.type == "cuda" else None
    try:
        while global_step < max_steps:
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            prev_batch_end = time.perf_counter()
            for batch in train_loader:
                if global_step >= max_steps:
                    break
                # Timing is only measured on designated steps; skip the overhead otherwise.
                log_timing = timing_every > 0 and (
                    global_step % timing_every == 0 or global_step == max_steps - 1
                )
                step_start = _timed_now(device, log_timing)
                timing = (
                    {
                        "data_wait": step_start - prev_batch_end,
                        "to_device": 0.0,
                        "mask": 0.0,
                        "forward_loss": 0.0,
                        "denominator_cpu": 0.0,
                        "backward": 0.0,
                        "optimizer": 0.0,
                        "logging": 0.0,
                        "validation": 0.0,
                        "checkpoint": 0.0,
                    }
                    if log_timing
                    else None
                )

                # Batch is already on the correct device from _DataPrefetcher.
                if batch.get("content", torch.empty(0)).device != device:
                    batch = move_batch_to_device(batch, device)
                if log_timing:
                    after_move = _timed_now(device, True)
                    timing["to_device"] = after_move - step_start

                # ── MAE mask on dedicated stream → .cpu() only syncs side-stream,
                #     overlapping mask GPU work with the previous backward pass.
                if _mask_generator is not None:
                    _mask_generator.manual_seed(_seed * 1000003 + global_step * 31 + micro_step * 1009 + rank + 1)
                    mae_mask, _mask_type = generate_mae_mask(batch, cfg, generator=_mask_generator, stream=_mask_stream)
                else:
                    mae_mask, _mask_type = generate_mae_mask(batch, cfg, stream=_mask_stream)
                if log_timing:
                    after_mask = _timed_now(device, True)
                    timing["mask"] = after_mask - after_move

                # LR only changes per optimizer step; reuse across micro-batches.
                if micro_step == 0:
                    lr = cosine_lr(
                        float(cfg["train"]["lr"]),
                        float(cfg["train"]["min_lr"]),
                        global_step,
                        max_steps,
                        int(cfg["train"]["warmup_steps"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                # Profiler wraps the *entire* accumulation cycle (all micro-batches).
                if micro_step == 0:
                    should_profile = profile_step >= 0 and global_step == profile_step and rank == 0
                    if should_profile:
                        profiler_obj = _make_profiler_context(True, device, profile_trace_dir, global_step)
                        profiler_obj.__enter__()
                        profiling_now = True

                is_accum_step = (micro_step + 1) % grad_accum == 0
                sync_ctx = (
                    model.no_sync()
                    if world_size > 1 and not is_accum_step
                    else nullcontext()
                )
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
                with sync_ctx:
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                if is_accum_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if profiling_now:
                        profiler_obj.__exit__(None, None, None)
                        profiling_now = False
                after_forward_loss = _timed_now(device, log_timing)
                if log_timing:
                    timing["forward_loss"] = after_forward_loss - after_mask
                after_optimizer = _timed_now(device, log_timing)
                if log_timing:
                    timing["backward"] = after_optimizer - after_forward_loss
                    timing["optimizer"] = 0.0
                accumulated_flops += _compute_flops(model, batch)

                # Only proceed with logging / validation / checkpointing
                # after the accumulated optimizer step (one "real" step).
                if not is_accum_step:
                    micro_step += 1
                    prev_batch_end = time.perf_counter()
                    continue

                global_step += 1
                micro_step = 0
                step_wall = time.perf_counter() - cycle_start
                cycle_start = time.perf_counter()
                tflops_this_step = accumulated_flops / max(step_wall, 1e-9) / 1e12
                accumulated_flops = 0.0

                # ── Peak memory (GPU 0 only, after a full optimizer step) ──
                peak_allocated_gb: float = 0.0
                peak_reserved_gb: float = 0.0
                if device.type == "cuda":
                    peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
                    torch.cuda.reset_peak_memory_stats(device)

                # ── Memory leak detection ──
                mem_now_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                if global_step > _mem_warmup_steps:
                    if _mem_baseline_allocated is None:
                        _mem_baseline_allocated = mem_now_gb
                    _mem_history.append(mem_now_gb)
                    if len(_mem_history) > 50:
                        _mem_history.pop(0)
                    # Check every _mem_leak_check_every steps: is memory trending up?
                    if global_step % _mem_leak_check_every == 0 and len(_mem_history) >= 20:
                        recent_avg = sum(_mem_history[-20:]) / 20
                        older_avg = sum(_mem_history[:20]) / 20 if len(_mem_history) >= 40 else _mem_baseline_allocated
                        growth_pct = (recent_avg - older_avg) / max(older_avg, 0.001) * 100
                        if growth_pct > 5.0:
                            LOGGER.warning(
                                "⚠ MEMORY LEAK? allocated grew %.1f%% over last ~%d steps "
                                "(%.2f→%.2f GB).  step=%d",
                                growth_pct,
                                _mem_leak_check_every,
                                older_avg,
                                recent_avg,
                                global_step,
                            )

                if rank == 0 and global_step % int(cfg["train"]["log_every_steps"]) == 0:
                    log_start = _timed_now(device, log_timing)
                    total_den = float(stats["total_denominator"].cpu())
                    LOGGER.info(
                        "step=%s loss=%.5f xy=%.5f area=%.5f blink=%.5f vel=%.5f den=%.0f lr=%.2e tflops_per_gpu=%.1f wall=%.3fs mem_peak=%.1fGB mem_now=%.1fGB",
                        global_step,
                        float(stats["total_loss"].cpu()),
                        float(stats["xy_loss"].cpu()),
                        float(stats["area_loss"].cpu()),
                        float(stats["blink_loss"].cpu()),
                        float(stats["velocity_loss"].cpu()),
                        total_den,
                        lr,
                        tflops_this_step,
                        step_wall,
                        peak_allocated_gb,
                        mem_now_gb,
                    )
                    if total_den == 0:
                        LOGGER.warning("step=%s ALL loss denominators are 0 — MAE mask may have no eligible positions!", global_step)
                    writer.add_scalar("train/total_loss", float(stats["total_loss"].cpu()), global_step)
                    writer.add_scalar("train/xy_loss", float(stats["xy_loss"].cpu()), global_step)
                    writer.add_scalar("train/blink_loss", float(stats["blink_loss"].cpu()), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    if timing is not None:
                        timing["logging"] = _timed_now(device, True) - log_start

                do_val = (global_step > 0 and global_step % int(cfg["train"]["val_every_steps"]) == 0) or global_step == max_steps - 1
                if do_val:
                    val_start = _timed_now(device, log_timing)
                    save_viz = _interval_enabled(
                        cfg.get("eval", {}).get("visualization_every_steps"),
                        global_step,
                        max_steps,
                        default=True,
                    )
                    compute_group_metrics = _interval_enabled(
                        cfg.get("eval", {}).get("group_metrics_every_steps"),
                        global_step,
                        max_steps,
                        default=True,
                    )
                    metrics = validate(
                        model,
                        val_loader,
                        cfg,
                        device,
                        global_step=global_step,
                        rank=rank,
                        save_viz=save_viz,
                        compute_group_metrics=compute_group_metrics,
                        log_timing=log_timing,
                    )
                    if timing is not None:
                        timing["validation"] = _timed_now(device, True) - val_start
                    if rank == 0:
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, global_step)
                        write_json(out_dir / "metrics_last.json", metrics)
                        log_metrics = {key: value for key, value in metrics.items() if not key.startswith("val_group/")}
                        LOGGER.info(
                            "val step=%s %s",
                            global_step,
                            " ".join(f"{key}={value:.5g}" for key, value in sorted(log_metrics.items())),
                        )
                        monitor = cfg["checkpoint"]["monitor"]
                        value = metrics.get(monitor, metrics.get("val/total_loss", math.inf))
                        improved = value < best_metric if cfg["checkpoint"].get("mode", "min") == "min" else value > best_metric
                        if improved:
                            best_metric = value
                            save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, global_step, epoch, cfg, best_metric)
                        # ── Early stopping: if val_masked_xy_rmse_deg is
                        #     worse than best for *patience* consecutive
                        #     validations, training is considered converged.
                        _es_val = metrics.get(_early_stop_metric)
                        if _es_val is not None:
                            if _es_val < _early_stop_best:
                                _early_stop_best = _es_val
                                _early_stop_count = 0
                            else:
                                _early_stop_count += 1
                                if _early_stop_count >= _early_stop_patience:
                                    LOGGER.info(
                                        "Early stopping: %s=%.5g ≥ best=%.5g for %d consecutive validations — converged.",
                                        _early_stop_metric, _es_val, _early_stop_best, _early_stop_count,
                                    )
                                    global_step = max_steps  # break outer loop
                                    break

                if rank == 0 and (
                    (global_step > 0 and global_step % int(cfg["train"]["save_every_steps"]) == 0)
                    or global_step == max_steps - 1
                ):
                    checkpoint_start = _timed_now(device, log_timing)
                    save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, global_step, epoch, cfg, best_metric)
                    if global_step > 0:
                        save_checkpoint(out_dir / f"checkpoint_step_{global_step:08d}.pt", model, optimizer, global_step, epoch, cfg, best_metric)
                    if timing is not None:
                        timing["checkpoint"] = _timed_now(device, True) - checkpoint_start
                if log_timing and rank == 0:
                    total = _timed_now(device, True) - step_start
                    LOGGER.info(
                        "train_timing step=%s total=%.3fs data_wait=%.3fs to_device=%.3fs mask=%.3fs "
                        "forward_loss=%.3fs denominator_cpu=%.3fs backward=%.3fs optimizer=%.3fs "
                        "logging=%.3fs validation=%.3fs checkpoint=%.3fs tflops_per_gpu=%.1f wall=%.3fs",
                        global_step,
                        total,
                        timing["data_wait"],
                        timing["to_device"],
                        timing["mask"],
                        timing["forward_loss"],
                        timing["denominator_cpu"],
                        timing["backward"],
                        timing["optimizer"],
                        timing["logging"],
                        timing["validation"],
                        timing["checkpoint"],
                        tflops_this_step,
                        step_wall,
                    )
                prev_batch_end = time.perf_counter()
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
