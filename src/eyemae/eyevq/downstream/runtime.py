"""Shared distributed runtime and learning-rate schedules for subject MIL."""


from __future__ import annotations


import atexit
import logging
import math
import signal
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from eyemae.utils import get_rank_world


_cleanup_registered = False


def _ddp_cleanup():
    """Graceful DDP cleanup, called on exit or error."""
    if dist.is_initialized():
        dist.destroy_process_group()


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
            pass


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
