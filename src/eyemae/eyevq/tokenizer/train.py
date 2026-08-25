#!/usr/bin/env python3
"""
Train EyeVQ-MT tokenizer with 3-stage schedule + DDP.

Uses pretraining PackedPretrainDataset, collate_trials, TokenBatchSampler.
Padding: round Nmax up to nearest 64, cap at 384 (matches pretraining).

Usage:
  # 4-GPU DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      -m eyemae.eyevq.tokenizer.train --config configs/eyevq/tokenizer_joint.yaml

  # Single GPU
  CUDA_VISIBLE_DEVICES=2 python -m eyemae.eyevq.tokenizer.train \
      --config configs/eyevq/tokenizer_joint.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from eyemae.batching import TokenBatchSampler
from eyemae.data import (
    PackedPretrainDataset,
    PackedTrialStore,
    collate_trials,
    filter_packed_rows_with_usable_eye,
    load_area_stats,
    read_packed_index,
    validate_npz_trial,
    validate_area_normalization_contract,
)
from eyemae.patching import patchify_preprocessed_trial
from eyemae.preprocess import preprocess_trial
from eyemae.utils import get_rank_world, set_seed, atomic_torch_save, write_json
from eyemae.eyevq.config import (
    CONFIG_VERSION,
    build_tokenizer,
    normalized_state_dict,
    override_fsq_levels,
)
from eyemae.eyevq.data import collate_trials_fixed_nmax
from eyemae.eyevq.tokenizer.model import EyeVQTokenizer
from eyemae.eyevq.tokenizer.losses import compute_total_loss
from eyemae.eyevq.optim import build_adamw_param_groups
from eyemae.eyevq.artifacts import assert_run_identity, build_run_identity

LOGGER = logging.getLogger(__name__)


class RankStridedBatchSampler:
    """Assign complete pre-built batches to ranks without changing batching.

    Validation losses contain masked means whose denominators depend on batch
    contents.  Sharding indices before batching would therefore make the metric
    world-size dependent.  Striding complete deterministic batches preserves
    the single-rank metric exactly while keeping rank inputs disjoint.
    """

    def __init__(self, batch_sampler, *, rank: int, world_size: int) -> None:
        self.batch_sampler = batch_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        for batch_index, batch in enumerate(self.batch_sampler):
            if batch_index % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        total = len(self.batch_sampler)
        return max(0, (total + self.world_size - 1 - self.rank) // self.world_size)


# ──────────────────────────────────────────────
# Distributed setup (matches pretraining)
# ──────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank, world_size, local_rank = get_rank_world()
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device("cuda", local_rank)
        else:
            backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eyevq")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if rank == 0:
        fh = logging.FileHandler(output_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ──────────────────────────────────────────────
# LR Schedule
# ──────────────────────────────────────────────

def get_lr(
    step: int,
    warmup: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
    decay_start: int | None = None,
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    decay_start = warmup if decay_start is None else max(warmup, decay_start)
    if step < decay_start:
        return base_lr
    progress = (step - decay_start) / max(1, max_steps - decay_start)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(progress * math.pi))


def get_velocity_weight(step: int, loss_cfg: dict[str, Any]) -> float:
    """Delay velocity supervision, then cosine-ramp it to the target weight."""
    target = float(loss_cfg.get("eye_velocity_weight", 0.0))
    start = int(loss_cfg.get("eye_velocity_start_step", 0))
    ramp = int(loss_cfg.get("eye_velocity_ramp_steps", 0))
    if target < 0.0 or start < 0 or ramp < 0:
        raise ValueError("Velocity target weight/start/ramp must be non-negative")
    if step < start:
        return 0.0
    if ramp == 0 or step >= start + ramp:
        return target
    progress = (step - start) / ramp
    return target * 0.5 * (1.0 - math.cos(math.pi * progress))


# ──────────────────────────────────────────────
# Build pretraining-compatible dataset config
# ──────────────────────────────────────────────

def make_pretrain_cfg(
    data_path: str,
    area_stats_path: str,
    area_cfg: dict | None = None,
) -> dict:
    area = {
        "stats_path": area_stats_path,
        "use_log1p": True,
        "clip": 5.0,
        "eps": 1e-6,
        "mad_scale": 1.4826,
        "mad_floor": 0.02,
        "min_subject_valid_frames": 10_000,
    }
    area.update(area_cfg or {})
    area["stats_path"] = area_stats_path
    return {
        "data": {
            "data_dir": data_path,
            "format": "packed_mmap",
            "max_open_shards_per_worker": 16,
            "validate_offsets": True,
        },
        "patch": {"samples": 20, "stride": 20},
        "area": area,
        "preprocess": {
            "normalize": "area",
            "xy_scale": 1.0,
            "xy_offset": 0.0,
        },
        "normalization": {"x_clip_deg": 30.0, "y_clip_deg": 20.0},
        "label": {"missing_value": 2, "blink_value": 1, "nonblink_value": 0},
        "attention": {"min_nonmissing_frac_for_eye_token": 0.50},
    }


def build_manual_feature_loss_metadata(
    trial_norm_stats: dict[str, dict[str, Any]],
    num_features: int,
    device: torch.device,
    *,
    balanced_binary: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build type masks and train-prior-normalized BCE weights for 38D targets.

    For a binary feature with training positive fraction ``p``, balanced BCE
    uses ``pos_weight=(1-p)/p``.  Multiplying by
    ``1 / ((1-p) + p*pos_weight)`` keeps its expected class-weight mass at one,
    so class balancing does not silently inflate the manual-feature group.
    """
    indexed: list[tuple[int, str, dict[str, Any]]] = []
    for fallback_index, (name, values) in enumerate(trial_norm_stats.items()):
        feature_index = int(values.get("feature_idx", fallback_index))
        indexed.append((feature_index, name, values))
    indexed.sort(key=lambda item: item[0])
    observed_indices = [item[0] for item in indexed]
    if observed_indices != list(range(num_features)):
        raise ValueError(
            "manual feature stats must contain each feature_idx exactly once; "
            f"observed={observed_indices}"
        )

    binary_mask = torch.tensor(
        [bool(values.get("is_binary", 0)) for _, _, values in indexed],
        dtype=torch.bool,
        device=device,
    )
    count_mask = torch.tensor(
        [bool(values.get("is_count", 0)) for _, _, values in indexed],
        dtype=torch.bool,
        device=device,
    )
    if torch.any(binary_mask & count_mask):
        raise ValueError("manual feature stats mark a feature as both binary and count")

    pos_weight = torch.ones(num_features, dtype=torch.float32, device=device)
    bce_scale = torch.ones(num_features, dtype=torch.float32, device=device)
    if balanced_binary:
        for feature_index, name, values in indexed:
            if not bool(values.get("is_binary", 0)):
                continue
            positive_fraction = float(values.get("mean", math.nan))
            if not 0.0 < positive_fraction < 1.0:
                raise ValueError(
                    f"Binary feature {name} has invalid train positive fraction "
                    f"{positive_fraction}; balanced BCE requires both classes"
                )
            odds = (1.0 - positive_fraction) / positive_fraction
            pos_weight[feature_index] = odds
            bce_scale[feature_index] = 1.0 / (
                (1.0 - positive_fraction) + positive_fraction * odds
            )
    return binary_mask, count_mask, pos_weight, bce_scale


# ──────────────────────────────────────────────
# Raw trial loading (for k-means init)
# ──────────────────────────────────────────────

def load_trial_batch(store, rows, indices, pretrain_cfg, area_stats):
    """Load k-means samples through the production preprocessing path."""
    items = []
    for idx in indices:
        try:
            row = rows[idx]
            trial = store.read_trial(row)
            validate_npz_trial(trial, pretrain_cfg, path=trial["path"])
            processed = preprocess_trial(trial, pretrain_cfg, area_stats)
            patched = patchify_preprocessed_trial(processed, pretrain_cfg)
            if patched is None:
                continue
            items.append({
                "content": patched["content"].transpose(0, 1, 3, 2).copy(),
                "quality": patched["quality"],
                "stim": patched["stim"].transpose(0, 2, 1).copy(),
                "eye_nonmissing_frac": patched["eye_nonmissing_frac"],
            })
        except Exception:
            continue
    return items


def collate_vq_trials(items: list[dict]) -> dict:
    """Collate raw-loaded trials for k-means (channels-first format)."""
    batch_size = len(items)
    nmax = max(item["content"].shape[0] for item in items)
    content = torch.zeros(batch_size, nmax, 2, 4, 20)
    quality = torch.ones(batch_size, nmax, 2, 20, 1)
    stim = torch.zeros(batch_size, nmax, 4, 20)
    eye_nonmissing = torch.zeros(batch_size, nmax, 2)
    pad_mask = torch.ones(batch_size, nmax, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item["content"].shape[0]
        content[i, :n] = torch.from_numpy(item["content"])
        quality[i, :n] = torch.from_numpy(item["quality"])
        stim[i, :n] = torch.from_numpy(item["stim"])
        eye_nonmissing[i, :n] = torch.from_numpy(item["eye_nonmissing_frac"])
        pad_mask[i, :n] = False
    return {"content": content, "quality": quality, "stim": stim,
            "eye_nonmissing_frac": eye_nonmissing, "pad_mask": pad_mask}


# ──────────────────────────────────────────────
# K-Means Codebook Initialization
# ──────────────────────────────────────────────

def kmeans_init_codebook(
    model: EyeVQTokenizer,
    store,
    rows: list[dict],
    pretrain_cfg: dict,
    area_stats,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
    sample_size: int = 5000,
):
    """Initialize VQ codebooks via k-means on z_e from Stage A encoder.

    Samples z_e_stim and z_e_eye from training data, runs MiniBatchKMeans,
    and loads centroids into the codebooks.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        logger.warning("sklearn not available, skipping k-means init")
        return

    logger.info("=" * 60)
    logger.info("K-Means codebook initialization (Stage A → Stage B)")
    logger.info(f"Sampling {sample_size} trials for z_e collection...")

    enc_cfg = cfg["encoder"]
    vq_cfg = cfg["vq"]
    dec_cfg = cfg["decoder"]
    feat_cfg = cfg.get("manual_features", {})

    # Get K and D based on VQ type
    if vq_cfg.get("type") == "fsq":
        K_eye = 1
        fsq_L_val = vq_cfg["fsq_L"]
        for li in (fsq_L_val if isinstance(fsq_L_val, list) else [fsq_L_val]):
            K_eye *= int(li)
        D_eye = int(vq_cfg["fsq_d"])
    elif vq_cfg.get("type") in {"vq", "vqvae"}:
        K_eye = int(vq_cfg.get("codebook_size", 8192))
        D_eye = int(vq_cfg.get("code_dim", 128))
    else:
        K_eye = int(vq_cfg["eye"]["codebook_size"])
        D_eye = int(vq_cfg["eye"]["code_dim"])
    D_model = int(enc_cfg["d_model"])

    # Collect z_e from random trials (L/R only, after VQ projection)
    np.random.seed(42)
    indices = np.random.choice(len(rows), min(sample_size, len(rows)), replace=False)

    z_e_eye_list = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(indices), 32):
            batch_idx = indices[batch_start:batch_start+32]
            items = load_trial_batch(
                store, rows, batch_idx, pretrain_cfg, area_stats
            )
            if len(items) < 2:
                continue
            batch = collate_vq_trials(items)
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            content = batch["content"].transpose(-1, -2).contiguous()
            stim = batch["stim"].transpose(-1, -2).contiguous()
            eye_l = content[:, :, 0, :, :]
            eye_r = content[:, :, 1, :, :]

            # Encoder forward
            enc = model.encoder(stim, eye_l, eye_r,
                                batch["eye_nonmissing_frac"], batch["pad_mask"])
            # VQ projection
            lr_hidden = torch.stack(
                [enc["l_hidden"], enc["r_hidden"]], dim=2
            ).reshape(enc["l_hidden"].shape[0], -1, model.d_model)
            z_e = model.vq_proj(lr_hidden)
            z_e_eye_list.append(z_e.reshape(-1, D_eye).cpu().numpy())

    z_e_eye_all = np.concatenate(z_e_eye_list, axis=0)

    logger.info(f"Collected z_e_eye (for VQ): {z_e_eye_all.shape[0]:,} vectors")

    # Subsample to 200K for k-means
    max_samples = 200000
    if z_e_eye_all.shape[0] > max_samples:
        idx = np.random.choice(z_e_eye_all.shape[0], max_samples, replace=False)
        z_e_eye_sub = z_e_eye_all[idx]
    else:
        z_e_eye_sub = z_e_eye_all

    # Eye k-means
    logger.info(f"Running MiniBatchKMeans for eye (K={K_eye}, D={D_eye})...")
    km_eye = MiniBatchKMeans(n_clusters=K_eye, random_state=42, n_init=3,
                             batch_size=4096, max_iter=100)
    km_eye.fit(z_e_eye_sub)
    eye_centroids = torch.from_numpy(km_eye.cluster_centers_.astype(np.float32)).to(device)
    model.eye_codebook.init_from_kmeans(eye_centroids)
    logger.info(f"Eye k-means done. Inertia: {km_eye.inertia_:.2f}")

    # Verify
    z_norm = float(np.mean(np.linalg.norm(z_e_eye_sub, axis=1)))
    code_norm = eye_centroids.norm(dim=1).mean().item()
    logger.info(f"z_e_eye norm: {z_norm:.2f}  code norm: {code_norm:.2f}  ratio: {code_norm/max(z_norm,1e-8):.3f}")
    logger.info("=" * 60)

    model.train()


# ──────────────────────────────────────────────
# Kaiming (He) Initialization
# ──────────────────────────────────────────────

def kaiming_init_weights(model: nn.Module, logger: logging.Logger | None = None) -> None:
    """Apply Kaiming (He) normal initialization to all Linear and Conv layers.

    - Linear weight: kaiming_normal_(fan_in, nonlinearity='relu')
    - Conv1d / ConvTranspose1d weight: kaiming_normal_(fan_in, nonlinearity='relu')
    - All biases: zero
    - Embedding, LayerNorm, RMSNorm: kept as-is (small-scale normal init)
    - CLS tokens: kept as-is (normal std=0.02)
    """
    import torch.nn as nn

    def _init(module: nn.Module, prefix: str = "") -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if logger:
                logger.debug(f"  kaiming_init: {prefix}.weight shape={tuple(module.weight.shape)}")
        # Recursively apply to children
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _init(child, child_prefix)

    _init(model, "")
    if logger:
        logger.info("✅ Kaiming (He normal) initialization applied to all Linear/Conv layers")


def train_vq(
    cfg: dict,
    output_dir: Path,
    resume_from: str | None = None,
    reset_best_val_loss_on_resume: bool = False,
    allow_resume_identity_mismatch: bool = False,
):
    rank, world_size, local_rank, device = setup_distributed()
    logger = setup_logging(rank, output_dir)

    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 42)) + rank)
    data_path = train_cfg["data_path"]
    train_index = str(Path(data_path) / train_cfg.get("train_index", "pretrain/pretrain_train.csv"))
    area_stats_path = train_cfg.get("area_stats_path", "outputs/area_stats_v3_tokenizer.json")

    identity_dependencies: dict[str, str | Path] = {}
    manual_cfg = cfg.get("manual_features", {})
    if bool(manual_cfg.get("enabled", False)):
        for key in ("cache_path", "stats_path", "task_metric_weight_path"):
            if manual_cfg.get(key):
                identity_dependencies[f"manual_features.{key}"] = manual_cfg[key]
    identity_payload = (
        build_run_identity("tokenizer", cfg, dependencies=identity_dependencies)
        if rank == 0 else None
    )
    if world_size > 1:
        objects = [identity_payload]
        dist.broadcast_object_list(objects, src=0)
        identity_payload = objects[0]
    run_identity = identity_payload

    if is_rank0():
        logger.info(f"Config: {cfg.get('_config_path', 'N/A')}")
        logger.info(f"Device: {device} | GPUs: {world_size} | Rank: {rank}")

    # ── 1. Dataset (reuses pretraining preprocessing) ──
    pretrain_cfg = make_pretrain_cfg(data_path, area_stats_path, cfg.get("area"))
    # Override patch granularity from the YAML config (e.g. 100ms patches)
    if cfg.get("patch"):
        pretrain_cfg["patch"] = cfg["patch"]
    area_stats = load_area_stats(area_stats_path)
    validate_area_normalization_contract(area_stats, pretrain_cfg["area"])

    # Also keep store + rows for k-means init (uses raw data loading path)
    store = PackedTrialStore(data_path)
    rows = read_packed_index(train_index)
    raw_train_rows = len(rows)
    # Filter: only trials with ≤max_patches patches (counted at the CURRENT
    # patch granularity, e.g. 100ms → frame_length//100 ≤ 64)
    max_patches = int(cfg.get("encoder", {}).get("max_patches", 64))
    patch_samples = int(pretrain_cfg["patch"]["samples"])
    rows = [r for r in rows if int(r.get("frame_length", 0)) // patch_samples <= max_patches]
    length_filtered_rows = len(rows)
    if bool(train_cfg.get("require_any_eye_keep", True)):
        rows, excluded_no_eye_rows = filter_packed_rows_with_usable_eye(rows)
    else:
        excluded_no_eye_rows = []
    if is_rank0():
        logger.info(
            "Filtered rows (≤%d patches + any usable eye): %s "
            "(dropped_long=%s, dropped_both_eyes_invalid=%s)",
            max_patches,
            f"{len(rows):,}",
            f"{raw_train_rows - length_filtered_rows:,}",
            f"{len(excluded_no_eye_rows):,}",
        )

    max_trials_overfit = train_cfg.get("overfit_trials") or None
    if max_trials_overfit:
        rows = rows[:max_trials_overfit]

    dataset = PackedPretrainDataset(
        data_path, pretrain_cfg,
        rows=rows,
        area_stats=area_stats,
    )
    if is_rank0():
        logger.info(f"Training trials: {len(dataset)}")
        g = area_stats.get("global", {})
        transform = "log1p" if bool(pretrain_cfg["area"].get("use_log1p", True)) else "raw"
        logger.info(
            "Area stats: transform=%s mode=%s median=%.4f mad=%.4f "
            "left=(%.4f,%.4f) right=(%.4f,%.4f) mad_scale=%.4f mad_floor=%.4f",
            transform,
            "per_subject_eye" if bool(pretrain_cfg["area"].get("per_eye", False)) else "per_subject",
            float(g.get("median", 0.0)),
            float(g.get("mad", 0.0)),
            float(area_stats.get("global_by_eye", {}).get("left", {}).get("median", 0.0)),
            float(area_stats.get("global_by_eye", {}).get("left", {}).get("mad", 0.0)),
            float(area_stats.get("global_by_eye", {}).get("right", {}).get("median", 0.0)),
            float(area_stats.get("global_by_eye", {}).get("right", {}).get("mad", 0.0)),
            float(pretrain_cfg["area"].get("mad_scale", 1.4826)),
            float(pretrain_cfg["area"].get("mad_floor", 0.0)),
        )

    # ── 2. DataLoader with TokenBatchSampler + padding collate ──
    max_trials = int(train_cfg.get("max_trials_per_gpu", 512))
    num_workers = int(train_cfg.get("num_workers", 4))

    sampler = TokenBatchSampler(
        dataset,
        max_seq_tokens=9999999,  # disabled — fixed trial count
        max_trials=max_trials,
        shuffle=True,
        # All ranks must construct the same shuffled index stream. The sampler
        # then takes rank-strided disjoint subsets. Adding rank to this seed
        # creates different permutations and therefore overlapping data.
        seed=int(train_cfg.get("seed", 42)),
        bucket_by_length=True,
        infinite=True,
        rank=rank,
        world_size=world_size,
    )
    # Collate nmax = max_patches (hyperparameter: 20ms→256, 100ms→64)
    collate_nmax = int(cfg.get("encoder", {}).get("max_patches", 256))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=lambda items: collate_trials_fixed_nmax(items, nmax=collate_nmax),
    )

    val_loader = None
    val_index_name = train_cfg.get("val_index")
    if val_index_name:
        val_index = Path(data_path) / str(val_index_name)
        if not val_index.is_file():
            raise FileNotFoundError(f"Configured train.val_index does not exist: {val_index}")
        val_rows = read_packed_index(str(val_index))
        raw_val_rows = len(val_rows)
        val_rows = [
            row for row in val_rows
            if int(row.get("frame_length", 0)) // patch_samples <= max_patches
        ]
        length_filtered_val_rows = len(val_rows)
        if bool(train_cfg.get("require_any_eye_keep", True)):
            val_rows, excluded_val_no_eye_rows = filter_packed_rows_with_usable_eye(val_rows)
        else:
            excluded_val_no_eye_rows = []
        val_dataset = PackedPretrainDataset(
            data_path, pretrain_cfg, rows=val_rows, area_stats=area_stats
        )
        full_val_sampler = TokenBatchSampler(
            val_dataset,
            max_seq_tokens=9_999_999,
            max_trials=max_trials,
            shuffle=False,
            seed=int(train_cfg.get("seed", 42)),
            bucket_by_length=False,
            infinite=False,
        )
        val_sampler = RankStridedBatchSampler(
            full_val_sampler,
            rank=rank,
            world_size=world_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda items: collate_trials_fixed_nmax(items, nmax=collate_nmax),
        )
        if is_rank0():
            logger.info(
                "Validation trials: %s (dropped_long=%s, dropped_both_eyes_invalid=%s; ranks=%s)",
                f"{len(val_dataset):,}",
                f"{raw_val_rows - length_filtered_val_rows:,}",
                f"{len(excluded_val_no_eye_rows):,}",
                world_size,
            )

    # ── 3. Model ──
    enc_cfg = cfg["encoder"]
    vq_cfg = cfg["vq"]
    feat_cfg = cfg.get("manual_features", {})

    # ── Task metric weights [4, num_features] (per-task literature weights) ──
    # task_metric_weight[trial_task] reweights each feature's loss; each task
    # row sums to 1.0 and is non-zero only for that task's feature block.
    task_metric_weight: torch.Tensor | None = None
    tmw_path = feat_cfg.get("task_metric_weight_path")
    if features_enabled := bool(feat_cfg.get("enabled", True)):
        if not tmw_path or not Path(tmw_path).is_file():
            raise FileNotFoundError(
                "manual_features.task_metric_weight_path is required when manual features are enabled"
            )
        if is_rank0():
            logger.info(f"Loading task_metric_weight from: {tmw_path}")
        _tmw = torch.load(tmw_path, map_location="cpu", weights_only=False)
        if isinstance(_tmw, dict) and "task_metric_weight" in _tmw:
            task_metric_weight = _tmw["task_metric_weight"].float()  # [4, F]
        elif isinstance(_tmw, torch.Tensor):
            task_metric_weight = _tmw.float()
        else:
            raise TypeError("task metric weight file has no task_metric_weight tensor")
        expected_features = int(feat_cfg["num_features"])
        if task_metric_weight.ndim != 2 or task_metric_weight.shape[1] != expected_features:
            raise ValueError(
                f"task_metric_weight shape={tuple(task_metric_weight.shape)}, "
                f"expected [num_tasks, {expected_features}]"
            )
        if is_rank0():
            logger.info(f"task_metric_weight shape: {tuple(task_metric_weight.shape)}")

    model = build_tokenizer(cfg).to(device)

    # ── Kaiming (He) initialization ──
    if train_cfg.get("kaiming_init", False):
        if is_rank0():
            logger.info("Applying Kaiming (He normal) initialization...")
        kaiming_init_weights(model, logger if is_rank0() else None)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model

    if is_rank0():
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {n_params:,}")

    # torch.compile (with safety for DDP + dynamic batch shapes)
    compile_enabled = train_cfg.get("compile_model", True)
    if compile_enabled and hasattr(torch, "compile"):
        try:
            if is_rank0():
                logger.info("Applying torch.compile (mode=reduce-overhead, dynamic=False)...")
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            if is_rank0():
                logger.info("torch.compile applied successfully")
        except Exception as e:
            if is_rank0():
                logger.warning(f"torch.compile failed ({e}), falling back to eager mode")
            compile_enabled = False

    # ── 4. Optimizer ──
    optimizer_groups, optimizer_audit = build_adamw_param_groups(
        model, weight_decay=float(train_cfg["weight_decay"])
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(train_cfg["lr"]), betas=tuple(train_cfg["betas"]),
    )
    if is_rank0():
        write_json(output_dir / "optimizer_groups.json", optimizer_audit)
    use_bf16 = train_cfg.get("precision") == "bf16"
    grad_clip_val = float(train_cfg.get("grad_clip", 0))

    # ── Resume from checkpoint ──
    resume_step = 0
    resume_trials_seen = 0
    resume_best_val_loss = float("inf")
    if resume_from and Path(resume_from).exists():
        if is_rank0():
            logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if ckpt.get("config_version") != CONFIG_VERSION:
            raise ValueError("Refusing to resume a legacy tokenizer checkpoint without the current config schema")
        assert_run_identity(
            ckpt,
            run_identity,
            allow_mismatch=allow_resume_identity_mismatch,
        )
        raw_model.load_state_dict(normalized_state_dict(ckpt["model_state_dict"]), strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            if is_rank0():
                logger.info("No optimizer state in checkpoint — using fresh optimizer")
        resume_step = ckpt.get("step", 0)
        resume_trials_seen = ckpt.get("trials_seen", 0)
        resume_best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if reset_best_val_loss_on_resume:
            resume_best_val_loss = float("inf")
            if is_rank0():
                logger.info(
                    "Reset best_val_loss because the resumed run uses a new loss scale"
                )
        if is_rank0():
            logger.info(f"Resumed at step {resume_step}, trials_seen={resume_trials_seen}")

    # ── 5. Training loop (step-driven stages) ──
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg["min_lr"])
    warmup_steps = int(train_cfg["warmup_steps"])
    lr_decay_start_step = int(train_cfg.get("lr_decay_start_step", warmup_steps))

    stage_a_steps = int(train_cfg["stage_a_steps"])
    stage_b_steps = int(train_cfg["stage_b_steps"])
    max_steps = stage_a_steps + stage_b_steps
    stage_b_velocity_weight = float(cfg["loss"]["eye_velocity_weight"])
    stage_a_velocity_weight = float(
        cfg["loss"].get("eye_velocity_weight_stage_a", stage_b_velocity_weight)
    )
    if stage_a_velocity_weight < 0.0 or stage_b_velocity_weight < 0.0:
        raise ValueError("Eye velocity loss weights must be non-negative")
    stage_a_loss_cfg = {
        **cfg,
        "loss": {
            **cfg["loss"],
            "eye_velocity_weight": stage_a_velocity_weight,
        },
    }

    log_every = int(train_cfg["log_every_steps"])
    save_every = int(train_cfg["save_every_steps"])
    val_every = int(train_cfg.get("val_every_steps", save_every))

    if is_rank0():
        if stage_a_steps == 0:
            logger.info(f"Quantization enabled from step 0 | total steps: {max_steps}")
        else:
            logger.info(f"Stage A: 0–{stage_a_steps} | B: {stage_a_steps}–{max_steps}")
        logger.info(
            f"LR schedule: base={base_lr:.2e} | decay_start={lr_decay_start_step} "
            f"| end={max_steps} | min={min_lr:.2e}"
        )
        if vq_cfg.get("type") == "fsq":
            logger.info(
                "FSQ activation: "
                f"{vq_cfg.get('fsq_activation', 'tanh')} "
                f"| alpha={float(vq_cfg.get('ifsq_alpha', 1.6)):g}"
            )
        if stage_a_steps == 0:
            logger.info(
                f"Velocity schedule: 0 until step "
                f"{int(cfg['loss'].get('eye_velocity_start_step', 0))}, cosine ramp "
                f"{int(cfg['loss'].get('eye_velocity_ramp_steps', 0))} steps, "
                f"target={stage_b_velocity_weight:g}"
            )
        else:
            logger.info(
                f"Velocity weight: stage_A={stage_a_velocity_weight:g} "
                f"| stage_B={stage_b_velocity_weight:g}"
            )
        logger.info(f"Batch: max {max_trials} trials/gpu | BF16: {use_bf16}")
        logger.info("=" * 60)

    model.train()
    global_step = resume_step
    trials_seen = resume_trials_seen
    kmeans_done = global_step >= stage_a_steps
    sampler.set_start_batch(global_step)

    # ── Performance tracking accumulators ──
    total_fwd_time = 0.0
    total_bwd_time = 0.0
    steps_since_log = 0
    last_log_time = time.time()
    grad_raw_sum = 0.0
    grad_raw_max = 0.0
    grad_raw_last = 0.0
    grad_clip_count = 0

    # NaN safety: max consecutive NaN steps before abort
    nan_count = 0
    max_nan_steps = 5

    # ── Startup diagnostics ──
    if is_rank0():
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Rough memory estimate: params (bf16×2 + fp32) + grads (fp32) + optimizer (2×fp32) ≈ 10 bytes/param
        est_mem_gb = n_params * 10 / 1024**3
        logger.info(f"Trainable params: {n_trainable:,} | Est. model memory: {est_mem_gb:.2f} GB")
        if device.type == "cuda":
            free_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
            logger.info(f"GPU total memory: {free_gb:.1f} GB | GPU: {torch.cuda.get_device_name(device)}")
        logger.info("=" * 60)

    # ── Load precomputed 38d trial metrics (no-outlier version) ──
    metrics_cache_file = Path(str(feat_cfg.get("cache_path", "")))
    trial_stats_file = Path(str(feat_cfg.get("stats_path", "")))
    num_features = int(feat_cfg["num_features"])
    features_enabled = bool(feat_cfg.get("enabled", True))
    trial_metrics_cache: dict[str, tuple] = {}
    trial_norm_stats: dict = {}
    # Every rank must load targets. Loading them only on rank 0 changes the
    # computation graph/loss by rank and invalidates synchronized training.
    if features_enabled:
        if not str(metrics_cache_file) or not metrics_cache_file.is_file():
            raise FileNotFoundError(
                "manual_features.cache_path is required and must exist when manual features are enabled"
            )
        if is_rank0():
            logger.info(f"Loading {num_features}d trial metrics from cache: {metrics_cache_file}")
        trial_metrics_cache = torch.load(metrics_cache_file, map_location="cpu", weights_only=False)
        if is_rank0():
            logger.info(f"Trial metrics loaded: {len(trial_metrics_cache)} total")
        if str(trial_stats_file) and trial_stats_file.is_file():
            import json
            with open(trial_stats_file) as f:
                trial_norm_stats = json.load(f)
            if len(trial_norm_stats) != num_features:
                raise ValueError(
                    f"manual feature stats contain {len(trial_norm_stats)} features, expected {num_features}"
                )
            trial_norm_stats = dict(
                sorted(
                    trial_norm_stats.items(),
                    key=lambda item: int(item[1].get("feature_idx", 0)),
                )
            )
            if is_rank0():
                logger.info(f"Trial norm stats loaded from {trial_stats_file}")
        else:
            raise FileNotFoundError(
                "manual_features.stats_path is required: feature type metadata "
                f"controls SmoothL1/BCE/count exclusion ({trial_stats_file})"
            )
        if is_rank0():
            logger.info("=" * 60)

    # ── Helper: normalize 38d trial targets (robust z-score only) ──
    norm_eps = 1e-8
    norm_clip_lo, norm_clip_hi = -5.0, 5.0
    # Build feature-type masks and binary class weights from train-only stats.
    binary_mask = None
    count_mask = None
    binary_pos_weight = None
    binary_bce_scale = None
    if trial_norm_stats:
        balanced_binary = bool(
            cfg.get("loss", {}).get("manual_feature_balanced_bce", False)
        )
        binary_mask, count_mask, binary_pos_weight, binary_bce_scale = (
            build_manual_feature_loss_metadata(
                trial_norm_stats,
                num_features,
                device,
                balanced_binary=balanced_binary,
            )
        )
        if is_rank0():
            n_bin = binary_mask.sum().item()
            n_count = count_mask.sum().item()
            binary_mode = "train-prior balanced BCE" if balanced_binary else "BCE"
            logger.info(
                f"Trial features: {n_bin} binary({binary_mode}) | "
                f"{n_count} count(EXCLUDED) | "
                f"{num_features-n_bin-n_count} continuous(z-score SmoothL1)"
            )
            if balanced_binary and n_bin:
                selected = binary_pos_weight[binary_mask]
                logger.info(
                    "Manual binary pos_weight: min=%.4f max=%.4f "
                    "(expected loss scale normalized per feature)",
                    float(selected.min().item()),
                    float(selected.max().item()),
                )

    def normalize_trial_targets(continuous: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Robust z-score normalization per feature: (x - median) / (mad * 1.4826).

        - Binary flag features (is_binary=1): NOT z-scored, kept as 0/1 for BCE.
        - Count features (is_count=1): NOT z-scored and excluded from loss.
        - Continuous features: robust z-scored, clipped to [-5, 5].
        - Only normalizes positions where loss_mask=True.
        """
        if not trial_norm_stats:
            return continuous, loss_mask
        cont = continuous.clone()
        mask = loss_mask.clone()
        for j, (name, s) in enumerate(trial_norm_stats.items()):
            if not s or s.get("mad", 0) <= 0:
                continue
            # Binary (BCE) and excluded count features: skip z-score.
            if s.get("is_binary", 0) or s.get("is_count", 0):
                continue
            median = s["median"]
            robust_std = s["mad"] * 1.4826 + norm_eps
            m = mask[:, j].bool()
            if m.any():
                cont[m, j] = ((cont[m, j] - median) / robust_std).clamp(norm_clip_lo, norm_clip_hi)
        return cont, mask

    def lookup_trial_targets(batch: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not features_enabled:
            return None, None
        target_list, mask_list = [], []
        for gid in batch.get("global_trial_id", []):
            cached = trial_metrics_cache.get(gid)
            if cached is None:
                target_list.append(torch.zeros(num_features))
                mask_list.append(torch.zeros(num_features, dtype=torch.bool))
                continue
            target = torch.as_tensor(cached[0], dtype=torch.float32)
            target_mask = torch.as_tensor(cached[1], dtype=torch.bool)
            if target.numel() != num_features or target_mask.numel() != num_features:
                raise ValueError(
                    f"Trial {gid} has {target.numel()} metric values and "
                    f"{target_mask.numel()} mask values; expected {num_features}"
                )
            target_list.append(target)
            mask_list.append(target_mask)
        if not target_list:
            return None, None
        return normalize_trial_targets(
            torch.stack(target_list).to(device), torch.stack(mask_list).to(device)
        )

    @torch.no_grad()
    def run_validation(velocity_weight: float) -> dict[str, float]:
        if val_loader is None:
            return {}
        validation_cfg = {
            **cfg,
            "loss": {
                **cfg["loss"],
                "eye_velocity_weight": float(velocity_weight),
            },
        }
        raw_model.eval()
        metric_keys = (
            "L_eye",
            "L_feat",
            "L_feat_binary",
            "L_feat_continuous",
            "xy_loss",
            "area_loss",
            "blink_loss",
            "vel_loss",
            "xy_loss_weighted",
            "area_loss_weighted",
            "blink_loss_weighted",
            "vel_loss_weighted",
            "blink_positive_fraction",
        )
        sums: dict[str, float] = {"val/loss": 0.0}
        sums.update({f"val/{key}": 0.0 for key in metric_keys})
        n_batches = 0
        n_trials = 0
        code_counts = torch.zeros(
            raw_model.eye_codebook.codebook_size,
            dtype=torch.long,
            device=device,
        )
        max_val_batches_cfg = train_cfg.get("max_val_batches")
        max_val_batches = (
            None if max_val_batches_cfg is None else int(max_val_batches_cfg)
        )
        if max_val_batches is not None and max_val_batches <= 0:
            raise ValueError("train.max_val_batches must be positive or null")
        for val_batch in val_loader:
            val_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in val_batch.items()
            }
            v_content = val_batch["content"].transpose(-1, -2).contiguous()
            v_stim = val_batch["stim"].transpose(-1, -2).contiguous()
            v_quality = val_batch["quality"]
            v_pad = val_batch["pad_mask"]
            v_nm = val_batch["eye_nonmissing_frac"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                v_out = raw_model(
                    stim=v_stim,
                    content=v_content,
                    quality=v_quality,
                    pad_mask=v_pad,
                    eye_nonmissing_frac=v_nm,
                    quantize=True,
                )
                v_targets, v_loss_mask = lookup_trial_targets(val_batch)
                v_feature_weights = None
                if task_metric_weight is not None and v_targets is not None:
                    v_feature_weights = task_metric_weight.to(device)[val_batch["task_id"].long()]
                v_loss, v_stats = compute_total_loss(
                    pred_eye=v_out["eye_recon"],
                    target_eye=v_content,
                    pred_features=v_out["manual_feat_pred"],
                    target_features=v_targets,
                    quality=v_quality,
                    pad_mask=v_pad,
                    commit_eye=v_out["commit_eye"],
                    loss_mask=v_loss_mask,
                    binary_mask=binary_mask,
                    count_mask=count_mask,
                    binary_pos_weight=binary_pos_weight,
                    binary_bce_scale=binary_bce_scale,
                    feature_weights=v_feature_weights,
                    cfg=validation_cfg,
                )
            batch_trials = int(v_content.shape[0])
            sums["val/loss"] = sums.get("val/loss", 0.0) + float(v_loss.item()) * batch_trials
            for key in metric_keys:
                if key in v_stats:
                    sums[f"val/{key}"] = (
                        sums.get(f"val/{key}", 0.0)
                        + float(v_stats[key].item()) * batch_trials
                    )
            valid_eye = (v_nm >= float(vq_cfg.get("min_nonmissing_frac", 0.50))) & (~v_pad.unsqueeze(-1))
            valid_codes = v_out["code_ids"][valid_eye]
            if valid_codes.numel():
                code_counts += torch.bincount(
                    valid_codes.long(), minlength=code_counts.numel()
                )
            n_batches += 1
            n_trials += batch_trials
            if max_val_batches is not None and n_batches >= max_val_batches:
                break
        if world_size > 1:
            ordered_sum_keys = tuple(sums)
            reduced = torch.tensor(
                [*(sums[key] for key in ordered_sum_keys), n_batches, n_trials],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            for index, key in enumerate(ordered_sum_keys):
                sums[key] = float(reduced[index].item())
            n_batches = int(reduced[-2].item())
            n_trials = int(reduced[-1].item())
            dist.all_reduce(code_counts, op=dist.ReduceOp.SUM)
        raw_model.train()
        result = {key: value / max(n_trials, 1) for key, value in sums.items()}
        result["val/batches"] = float(n_batches)
        result["val/trials"] = float(n_trials)
        result["val/velocity_weight_effective"] = float(velocity_weight)
        if code_counts.sum().item() > 0:
            counts = code_counts.float()
            if hasattr(raw_model.eye_codebook, "usage_stats_from_counts"):
                usage = raw_model.eye_codebook.usage_stats_from_counts(counts)
            else:
                probs = counts[counts > 0] / counts.sum()
                usage = {
                    "code_perplexity": float(
                        torch.exp(-(probs * probs.log()).sum()).item()
                    ),
                    "active_code_fraction": float((counts > 0).float().mean().item()),
                    "top1_code_frequency": float(probs.max().item()),
                }
            result["val/code_perplexity"] = usage["code_perplexity"]
            result["val/active_codes"] = float((counts > 0).sum().item())
            result["val/top1_code_frequency"] = usage["top1_code_frequency"]
            for key in (
                "code_dim_perplexity_mean",
                "code_dim_perplexity_min",
                "code_dim_active_fraction_min",
                "code_dim_top1_frequency_max",
            ):
                if key in usage:
                    result[f"val/{key}"] = usage[key]
        return result

    best_val_loss = resume_best_val_loss

    for epoch in range(1000):
        if global_step >= max_steps:
            break
        sampler.set_epoch(epoch)

        # Accumulators for log-interval averaging
        acc_loss_stats: dict[str, float] = {}
        acc_vq_stats: dict[str, float] = {}

        for batch in loader:
            if global_step >= max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Stage control: A = AE only, B = VQ + trial head
            quantize = global_step >= stage_a_steps

            # K-means init at Stage A → B transition (VQ only, FSQ no-op)
            if quantize and not kmeans_done:
                if cfg.get("vq", {}).get("type") != "fsq":
                    if is_rank0():
                        kmeans_init_codebook(
                            raw_model, store, rows, pretrain_cfg, area_stats,
                            cfg, device, logger, sample_size=10000,
                        )
                    # Broadcast codebook weights + EMA buffers to all ranks
                    if world_size > 1:
                        dist.barrier()
                        for param in raw_model.eye_codebook.parameters():
                            dist.broadcast(param.data, src=0)
                        for buf in raw_model.eye_codebook.buffers():
                            dist.broadcast(buf.data, src=0)
                kmeans_done = True

            lr = get_lr(
                global_step, warmup_steps, max_steps, base_lr, min_lr,
                decay_start=lr_decay_start_step,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Transpose pretraining time-first format → tokenizer channels-first format
            # collate: content [B,N,2,20,4] → need [B,N,2,4,20]; stim [B,N,20,4] → [B,N,4,20]
            content = batch["content"].transpose(-1, -2).contiguous()
            stim = batch["stim"].transpose(-1, -2).contiguous()
            quality = batch["quality"]
            pad_mask = batch["pad_mask"]
            eye_nonmissing = batch["eye_nonmissing_frac"]
            eye_valid = eye_nonmissing >= float(vq_cfg.get("min_nonmissing_frac", 0.50))

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # Training forward must go through DDP; raw_model is reserved
                # for checkpoint/state access and rank-0 no-grad validation.
                out = model(
                    stim=stim, content=content, quality=quality,
                    pad_mask=pad_mask, eye_nonmissing_frac=eye_nonmissing,
                    quantize=quantize,
                )

                trial_targets, loss_mask_norm = lookup_trial_targets(batch)

                # ── Per-task feature weights (task_metric_weight[trial_task]) ──
                feature_weights = None
                if task_metric_weight is not None and trial_targets is not None:
                    tid = batch.get("task_id")
                    if tid is not None:
                        tid_t = torch.as_tensor(tid, dtype=torch.long, device=device)
                        feature_weights = task_metric_weight.to(device)[tid_t]

                base_step_cfg = cfg if quantize else stage_a_loss_cfg
                velocity_weight_now = get_velocity_weight(
                    global_step, base_step_cfg["loss"]
                )
                step_loss_cfg = {
                    **base_step_cfg,
                    "loss": {
                        **base_step_cfg["loss"],
                        "eye_velocity_weight": velocity_weight_now,
                    },
                }

                loss, loss_stats = compute_total_loss(
                    pred_eye=out["eye_recon"],
                    target_eye=content,
                    pred_features=out["manual_feat_pred"],
                    target_features=trial_targets,
                    quality=quality, pad_mask=pad_mask,
                    commit_eye=out["commit_eye"],
                    loss_mask=loss_mask_norm,
                    binary_mask=binary_mask,
                    count_mask=count_mask,
                    binary_pos_weight=binary_pos_weight,
                    binary_bce_scale=binary_bce_scale,
                    feature_weights=feature_weights,
                    cfg=step_loss_cfg,
                )

                # ── Accumulate stats over log interval ──
                if is_rank0():
                    for k, v in loss_stats.items():
                        acc_loss_stats[k] = acc_loss_stats.get(k, 0.0) + v.item()
                    # Recompute usage from valid eye tokens only. Quantizer
                    # diagnostics include padded slots and are misleading for
                    # short trials.
                    valid_codes = out["code_ids"][eye_valid & (~pad_mask.unsqueeze(-1))]
                    if valid_codes.numel():
                        counts = torch.bincount(
                            valid_codes.long(), minlength=raw_model.eye_codebook.codebook_size
                        ).float()
                        if hasattr(raw_model.eye_codebook, "usage_stats_from_counts"):
                            valid_stats = raw_model.eye_codebook.usage_stats_from_counts(counts)
                        else:
                            probs = counts[counts > 0] / counts.sum()
                            valid_stats = {
                                "active_code_fraction": float((counts > 0).float().mean().item()),
                                "code_perplexity": float(torch.exp(-(probs * probs.log()).sum()).item()),
                                "dead_code_count": float((counts == 0).sum().item()),
                                "top1_code_frequency": float(probs.max().item()),
                            }
                        for key, value in valid_stats.items():
                            acc_vq_stats[key] = acc_vq_stats.get(key, 0.0) + value
                    acc_vq_stats["commitment_loss"] = acc_vq_stats.get("commitment_loss", 0.0) + out["commit_eye"].item()

            # ── NaN / Inf detection ──
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nan_count += 1
                if is_rank0():
                    logger.warning(f"⚠ step={global_step}: loss={loss_val} (NaN/Inf #{nan_count}/{max_nan_steps})")
                    # ── Log per-component losses to identify the source ──
                    for k, v in loss_stats.items():
                        lv = v.item()
                        if not math.isfinite(lv):
                            logger.warning(f"  ⚡ NaN/Inf in loss component: {k}={lv}")
                    # Check encoder outputs for NaN
                    if "l_hidden" in out:
                        lh = out["l_hidden"]
                        logger.warning(f"  l_hidden: min={lh.min().item():.4f} max={lh.max().item():.4f} has_nan={torch.isnan(lh).any().item()}")
                    if "r_hidden" in out:
                        rh = out["r_hidden"]
                        logger.warning(f"  r_hidden: min={rh.min().item():.4f} max={rh.max().item():.4f} has_nan={torch.isnan(rh).any().item()}")
                if nan_count >= max_nan_steps:
                    if is_rank0():
                        logger.error(f"❌ Aborting: {nan_count} consecutive NaN/Inf steps")
                    break
                optimizer.zero_grad()
                global_step += 1
                continue
            nan_count = 0

            optimizer.zero_grad()
            loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.time()

            # ``clip_grad_norm_`` already computes and returns the pre-clipping
            # norm.  Reuse it every step so the log reports an interval
            # distribution rather than one potentially unrepresentative batch.
            if grad_clip_val > 0:
                norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_val
                )
            else:
                norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                )
            grad_norm_before = float(norm_tensor.item())
            if is_rank0():
                grad_raw_sum += grad_norm_before
                grad_raw_max = max(grad_raw_max, grad_norm_before)
                grad_raw_last = grad_norm_before
                if grad_clip_val > 0 and grad_norm_before > grad_clip_val:
                    grad_clip_count += 1

            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.time()

            # fwd_time = forward+loss+backward, opt_time = grad_norm+optimizer_step
            total_fwd_time += (t1 - t0)
            total_bwd_time += (t2 - t1)
            steps_since_log += 1

            # ── Track trials seen ──
            trials_seen += content.shape[0] * world_size

            # Log (rank 0 only)
            if is_rank0() and global_step % log_every == 0:
                stage = "A" if global_step < stage_a_steps else "B"

                # ── Timing & throughput ──
                elapsed = time.time() - last_log_time
                steps_per_sec = steps_since_log / elapsed if elapsed > 0 else 0
                trials_per_sec = (content.shape[0] * world_size * steps_since_log) / elapsed if elapsed > 0 else 0
                avg_fwd_ms = (total_fwd_time / steps_since_log) * 1000 if steps_since_log > 0 else 0
                avg_opt_ms = (total_bwd_time / steps_since_log) * 1000 if steps_since_log > 0 else 0
                tok_per_sec = (content.shape[0] * content.shape[1] * world_size * steps_since_log) / elapsed if elapsed > 0 else 0

                # ── GPU memory ──
                mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                # ── Estimated TFLOPS (rough: 2 * params * tokens / time) ──
                n_params = sum(p.numel() for p in model.parameters())
                est_tflops = (2 * n_params * steps_since_log) / (elapsed * 1e12) if elapsed > 0 else 0

                # ── Build log parts (averaged over log interval) ──
                n_avg = max(steps_since_log, 1)
                parts = [
                    f"step={global_step:06d}/{max_steps}", f"stage={stage}",
                    f"loss={loss_val:.5f}", f"lr={lr:.2e}", f"cb={cb_now:.4f}",
                    f"vel_w={velocity_weight_now:.4f}",
                ]
                # Only output summary losses, skip per-feature detail
                summary_keys = [k for k in sorted(acc_loss_stats.keys())
                                if k in ("L_feat", "L_eye", "L_total",
                                         "L_eye_weighted", "L_feat_weighted", "L_commit_weighted",
                                         "xy_loss", "area_loss", "blink_loss", "vel_loss",
                                         "commitment_loss")]
                for k in summary_keys:
                    parts.append(f"{k}={acc_loss_stats[k]/n_avg:.5f}")

                # VQ codebook usage (averaged)
                if quantize and acc_vq_stats:
                    nq = max(n_avg, 1)
                    parts.append(f"eye_act={acc_vq_stats.get('active_code_fraction',0)/nq:.3f}")
                    parts.append(f"eye_perp={acc_vq_stats.get('code_perplexity',0)/nq:.1f}")
                    parts.append(f"eye_dead={acc_vq_stats.get('dead_code_count',0)/nq:.0f}")
                    parts.append(f"eye_top1={acc_vq_stats.get('top1_code_frequency',0)/nq:.3f}")
                    if "code_dim_perplexity_min" in acc_vq_stats:
                        parts.append(
                            "eye_dim_perp_min="
                            f"{acc_vq_stats['code_dim_perplexity_min']/nq:.2f}"
                        )
                        parts.append(
                            "eye_dim_top1_max="
                            f"{acc_vq_stats['code_dim_top1_frequency_max']/nq:.3f}"
                        )

                # Gradient distribution over the full logging interval.
                grad_raw_mean = grad_raw_sum / n_avg
                parts.append(f"grad_raw={grad_raw_mean:.2f}")
                parts.append(f"grad_raw_last={grad_raw_last:.2f}")
                parts.append(f"grad_raw_max={grad_raw_max:.2f}")
                if grad_clip_val > 0:
                    parts.append(f"grad_clip_frac={grad_clip_count/n_avg:.3f}")

                # Performance metrics
                B, N = content.shape[:2]
                total_tok = B * N * world_size
                total_trials = B * world_size
                parts.append(f"step/s={steps_per_sec:.1f}")
                parts.append(f"tok/s={tok_per_sec:.0f}")
                parts.append(f"trial/s={trials_per_sec:.0f}")
                parts.append(f"fwd={avg_fwd_ms:.0f}ms")
                parts.append(f"opt={avg_opt_ms:.0f}ms")
                parts.append(f"tflops~{est_tflops:.3f}")
                parts.append(f"mem_alloc={mem_allocated:.2f}G")
                parts.append(f"mem_res={mem_reserved:.2f}G")
                parts.append(f"batch={total_trials}×{N}")

                logger.info(" | ".join(parts))

                # ── Reset accumulators ──
                acc_loss_stats = {}
                acc_vq_stats = {}
                total_fwd_time = 0.0
                total_bwd_time = 0.0
                steps_since_log = 0
                last_log_time = time.time()
                grad_raw_sum = 0.0
                grad_raw_max = 0.0
                grad_raw_last = 0.0
                grad_clip_count = 0

            # Every rank evaluates a disjoint validation shard; scalar sums and
            # code counts are reduced before rank 0 logs/saves the result.
            val_metrics = None
            should_validate = global_step > 0 and (
                global_step % val_every == 0 or global_step % save_every == 0
            )
            if should_validate:
                if world_size > 1:
                    dist.barrier()
                val_metrics = run_validation(
                    get_velocity_weight(global_step, cfg["loss"])
                )
                if is_rank0():
                    if val_metrics:
                        logger.info(" | ".join(f"{key}={value:.5f}" for key, value in val_metrics.items()))
                        if val_metrics["val/loss"] < best_val_loss:
                            best_val_loss = val_metrics["val/loss"]
                            atomic_torch_save({
                                "step": global_step,
                                "trials_seen": trials_seen,
                                "model_state_dict": raw_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "cfg": cfg,
                                "config_version": CONFIG_VERSION,
                                "run_identity": run_identity,
                                "val_metrics": val_metrics,
                                "best_val_loss": best_val_loss,
                            }, output_dir / "ckpt_best.pt")
                            logger.info(f"New best tokenizer val/loss={best_val_loss:.5f}")
                if world_size > 1:
                    dist.barrier()

            # Save (rank 0 only)
            if is_rank0() and global_step % save_every == 0 and global_step > 0:
                ckpt_path = output_dir / f"ckpt_step{global_step:06d}.pt"
                atomic_torch_save({
                    "step": global_step, "trials_seen": trials_seen,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "cfg": cfg,
                    "config_version": CONFIG_VERSION,
                    "run_identity": run_identity,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                logger.info(f"Saved: {ckpt_path}")

            global_step += 1

    # The loop checks periodic validation before incrementing its zero-based
    # counter, so a 50K run would otherwise leave its final segment unvalidated.
    if world_size > 1:
        dist.barrier()
    final_val_metrics = run_validation(
        get_velocity_weight(global_step, cfg["loss"])
    )
    if is_rank0():
        if final_val_metrics:
            logger.info(
                "Final validation | "
                + " | ".join(
                    f"{key}={value:.5f}"
                    for key, value in final_val_metrics.items()
                )
            )
            if final_val_metrics["val/loss"] < best_val_loss:
                best_val_loss = final_val_metrics["val/loss"]
                atomic_torch_save({
                    "step": global_step,
                    "trials_seen": trials_seen,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "config_version": CONFIG_VERSION,
                    "run_identity": run_identity,
                    "val_metrics": final_val_metrics,
                    "best_val_loss": best_val_loss,
                }, output_dir / "ckpt_best.pt")
        final_path = output_dir / "ckpt_final.pt"
        atomic_torch_save({
            "step": global_step, "trials_seen": trials_seen,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "cfg": cfg,
            "config_version": CONFIG_VERSION,
            "run_identity": run_identity,
            "val_metrics": final_val_metrics,
            "best_val_loss": best_val_loss,
        }, final_path)
        logger.info(f"Done. Steps: {global_step}. Final: {final_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/vq_tokenizer/base")
    parser.add_argument("--overfit_trials", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--allow-resume-identity-mismatch",
        action="store_true",
        help="Explicitly allow a deliberate schedule/objective extension from a non-identical checkpoint",
    )
    # ── CLI overrides for LR sweep (override YAML values) ──
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training learning rate")
    parser.add_argument("--min-lr", type=float, default=None,
                        help="Override min learning rate (set =lr for constant)")
    parser.add_argument("--lr-decay-start-step", type=int, default=None,
                        help="Start cosine decay at this global step")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override Stage-B steps; total = stage_a_steps + this value")
    parser.add_argument("--max-trials-per-gpu", type=int, default=None,
                        help="Override max trials per GPU (batch size)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override LR warmup steps")
    parser.add_argument("--fsq-levels", default=None,
                        help="Comma-separated FSQ levels, e.g. 9,7,7,5,5")
    parser.add_argument("--eye-xy-weight", type=float, default=None,
                        help="Override loss.eye_xy_weight")
    parser.add_argument("--eye-area-weight", type=float, default=None,
                        help="Override loss.eye_area_weight")
    parser.add_argument("--eye-blink-weight", type=float, default=None,
                        help="Override loss.eye_blink_weight")
    parser.add_argument("--eye-velocity-weight", type=float, default=None,
                        help="Override loss.eye_velocity_weight")
    parser.add_argument("--eye-velocity-start-step", type=int, default=None,
                        help="Keep velocity loss at zero before this step")
    parser.add_argument("--eye-velocity-ramp-steps", type=int, default=None,
                        help="Cosine-ramp velocity loss over this many steps")
    parser.add_argument("--manual-feature-group-weight", type=float, default=None,
                        help="Override loss.manual_feature_group_weight")
    parser.add_argument("--manual-feature-binary-weight", type=float, default=None,
                        help="Scale binary BCE inside the manual-feature objective")
    parser.add_argument("--manual-feature-continuous-weight", type=float, default=None,
                        help="Scale continuous SmoothL1 inside the manual-feature objective")
    parser.add_argument(
        "--reset-best-val-loss-on-resume",
        action="store_true",
        help="Reset checkpoint best loss when objective weights have changed",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config
    override_fsq_levels(cfg, args.fsq_levels)

    if args.overfit_trials:
        cfg["train"]["overfit_trials"] = args.overfit_trials

    # ── Apply CLI overrides ──
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)
    if args.min_lr is not None:
        cfg["train"]["min_lr"] = float(args.min_lr)
    if args.lr_decay_start_step is not None:
        cfg["train"]["lr_decay_start_step"] = int(args.lr_decay_start_step)
    if args.max_steps is not None:
        cfg["train"]["stage_b_steps"] = int(args.max_steps)
    if args.max_trials_per_gpu is not None:
        cfg["train"]["max_trials_per_gpu"] = int(args.max_trials_per_gpu)
    if args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = int(args.warmup_steps)
    loss_overrides = {
        "eye_xy_weight": args.eye_xy_weight,
        "eye_area_weight": args.eye_area_weight,
        "eye_blink_weight": args.eye_blink_weight,
        "eye_velocity_weight": args.eye_velocity_weight,
        "manual_feature_group_weight": args.manual_feature_group_weight,
        "manual_feature_binary_weight": args.manual_feature_binary_weight,
        "manual_feature_continuous_weight": args.manual_feature_continuous_weight,
    }
    for key, value in loss_overrides.items():
        if value is not None:
            if value < 0:
                raise ValueError(f"--{key.replace('_', '-')} must be non-negative")
            cfg["loss"][key] = float(value)
    if args.eye_velocity_start_step is not None:
        if args.eye_velocity_start_step < 0:
            raise ValueError("--eye-velocity-start-step must be non-negative")
        cfg["loss"]["eye_velocity_start_step"] = int(args.eye_velocity_start_step)
    if args.eye_velocity_ramp_steps is not None:
        if args.eye_velocity_ramp_steps < 0:
            raise ValueError("--eye-velocity-ramp-steps must be non-negative")
        cfg["loss"]["eye_velocity_ramp_steps"] = int(args.eye_velocity_ramp_steps)

    train_vq(
        cfg,
        Path(args.output_dir),
        resume_from=args.resume,
        reset_best_val_loss_on_resume=args.reset_best_val_loss_on_resume,
        allow_resume_identity_mismatch=args.allow_resume_identity_mismatch,
    )


if __name__ == "__main__":
    main()
