#!/usr/bin/env python3
"""
EyeVQ-BERT Masked Code Prediction Pretraining.

Usage:
  # 4-GPU DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      -m eyemae.eyevq.pretrain.train \
      --config configs/eyevq/pretrain_joint.yaml \
      --output_dir outputs/eyevq/pretrain_joint

  # Single GPU
  python -m eyemae.eyevq.pretrain.train \
      --config configs/eyevq/pretrain_joint.yaml \
      --output_dir outputs/eyevq/pretrain_joint
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from eyemae.batching import TokenBatchSampler
from eyemae.data import (
    PackedPretrainDataset,
    filter_packed_rows_with_usable_eye,
    load_area_stats,
    read_packed_index,
    validate_area_normalization_contract,
)
from eyemae.utils import atomic_torch_save, get_rank_world, set_seed, write_json
from eyemae.eyevq.config import (
    CONFIG_VERSION,
    build_bert,
    normalized_state_dict,
    override_fsq_levels,
)
from eyemae.eyevq.data import collate_trials_fixed_nmax
from eyemae.eyevq.pretrain.masking import (
    PAIRED_MULTIBLOCK,
    PAIRED_RANDOM,
    PAIRED_SPAN,
    generate_eyemae_mask,
    paired_time_eligibility,
)
from eyemae.eyevq.pretrain.model import TARGET_RAW_PATCH
from eyemae.eyevq.optim import build_adamw_param_groups
from eyemae.eyevq.artifacts import (
    CACHE_FORMAT_VERSION,
    assert_run_identity,
    build_run_identity,
    cache_contract,
    sha256_file,
    sha256_json,
    validate_cache_identity,
)

# ──────────────────────────────────────────────
# Distributed setup
# ──────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int, torch.device]:
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


def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eyevq_bert")
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

def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(progress * math.pi))


# ──────────────────────────────────────────────
# Pretraining config
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
            "max_open_shards_per_worker": 44,
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


# ──────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────

def load_code_ids_cache(
    path: str,
    *,
    patch_samples: int,
    tokenizer_checkpoint: str,
    tokenizer_sha256: str | None = None,
    contract_sha256: str | None = None,
    validate_identity: bool = True,
):
    """加载预计算的 code_ids 缓存 (npz)。

    返回 (gid_to_idx, codes_arr):
      gid_to_idx:    dict gid(str) → row index (codes_arr 中的行号)
      codes_arr:     np.ndarray [T, max_patches, 2] int16 (全量, 行序 = gids)
    用固定数组 + 行号索引, 支持向量化一次 gather (替代旧 dict-of-array 逐 trial 拷贝)。
    """
    import numpy as np
    if tokenizer_sha256 is None:
        tokenizer_sha256 = sha256_file(tokenizer_checkpoint)
    if contract_sha256 is None:
        raise ValueError("A code-ID cache contract SHA256 is required")
    if validate_identity:
        validate_cache_identity(
            path,
            tokenizer_checkpoint=tokenizer_checkpoint,
            contract_sha256=contract_sha256,
        )
    with np.load(path, allow_pickle=True) as d:
        version = int(d["format_version"]) if "format_version" in d else 0
        layout = str(d["lr_layout"]) if "lr_layout" in d else ""
        cached_patch = int(d["patch_samples"]) if "patch_samples" in d else -1
        cached_tokenizer = str(d["tokenizer_checkpoint"]) if "tokenizer_checkpoint" in d else ""
        invalid_eye_code_id = int(d["invalid_eye_code_id"]) if "invalid_eye_code_id" in d else -1
        cached_tokenizer_sha256 = str(d["tokenizer_sha256"]) if "tokenizer_sha256" in d else ""
        cached_contract_sha256 = str(d["cache_contract_sha256"]) if "cache_contract_sha256" in d else ""
        gids = [str(g) for g in d["gids"]]
        codes = np.asarray(d["code_ids"])
    if version not in {3, CACHE_FORMAT_VERSION} or layout != "time_major_lr_v1" or invalid_eye_code_id != 0:
        raise ValueError(
            "Legacy/misaligned code-ID cache rejected. Regenerate it with "
            "python -m eyemae.eyevq.precompute_codes "
            f"(format_version={CACHE_FORMAT_VERSION}, time-major L/R, invalid-eye sentinel 0)."
        )
    if cached_patch != patch_samples:
        raise ValueError(f"Code cache patch_samples={cached_patch}, expected {patch_samples}")
    if Path(cached_tokenizer).resolve() != Path(tokenizer_checkpoint).resolve():
        raise ValueError("Code cache was generated by a different tokenizer checkpoint")
    if version == CACHE_FORMAT_VERSION:
        if cached_tokenizer_sha256 != tokenizer_sha256:
            raise ValueError("Code cache tokenizer content SHA256 does not match")
        if cached_contract_sha256 != contract_sha256:
            raise ValueError("Code cache preprocessing/FSQ/data contract does not match")
    if codes.ndim != 3 or codes.shape[-1] != 2 or codes.shape[0] != len(gids):
        raise ValueError(f"Invalid code-ID cache shape: {codes.shape}")
    if len(set(gids)) != len(gids):
        raise ValueError("Code-ID cache contains duplicate global_trial_id values")
    gid_to_idx = {g: i for i, g in enumerate(gids)}
    return gid_to_idx, codes


def lookup_code_ids(batch, gid_to_idx, codes_arr, device, max_patches):
    """向量化查表: 一次 numpy fancy indexing + 单次 CPU→GPU 拷贝。

    cache 为固定 [T, max_patches, 2] 数组 → 按 gid 行号一次性 gather。
    相比旧的逐 trial torch.from_numpy(...).to(device) (B 次小拷贝 + 同步), 提速 ~100×。
    padding 部分 (num_patches 之后) 为 0, 由 pad_mask 排除, 不影响训练。

    缓存必须完整覆盖 batch；缺失任何 global_trial_id 都立即报错，禁止回退 tokenizer。
    """
    gids = batch.get("global_trial_id", [])
    if not gids:
        raise ValueError("Batch does not contain global_trial_id; code-ID cache lookup is impossible")
    missing = [str(g) for g in gids if str(g) not in gid_to_idx]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Code-ID cache misses {len(missing)} trial(s), e.g. {preview}. "
            "Regenerate the cache with --split all."
        )
    row_indices = np.fromiter(
        (gid_to_idx[str(g)] for g in gids),
        dtype=np.int64,
        count=len(gids),
    )
    if codes_arr.shape[1] < max_patches:
        raise ValueError(
            f"Code-ID cache max_patches={codes_arr.shape[1]}, expected at least {max_patches}"
        )
    arr = codes_arr[row_indices, :max_patches]
    return torch.from_numpy(arr).to(device=device, dtype=torch.long, non_blocking=True)


def train_bert(
    cfg: dict,
    output_dir: Path,
    resume_from: str | None = None,
    *,
    allow_resume_identity_mismatch: bool = False,
):
    rank, world_size, local_rank, device = setup_distributed()
    logger = setup_logging(rank, output_dir)

    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 42)) + rank)
    data_path = train_cfg["data_path"]
    train_index = str(Path(data_path) / train_cfg.get("train_index", "pretrain/pretrain_train.csv"))
    area_stats_path = train_cfg.get("area_stats_path", "outputs/area_stats_v3_tokenizer.json")
    target_type = str(
        cfg.get("bert", {}).get(
            "target_type",
            "factorized_code"
            if bool(cfg.get("bert", {}).get("factorized_fsq", False))
            else "joint_code",
        )
    )
    uses_code_targets = target_type != TARGET_RAW_PATCH
    tokenizer_ckpt = train_cfg.get("tokenizer_checkpoint")
    mask_cfg = cfg.get("mask", {})

    cache_split = str(train_cfg.get("code_ids_cache_split", "all"))
    cache_contract_sha256 = (
        sha256_json(cache_contract(cfg, split=cache_split)) if uses_code_targets else None
    )
    identity_payload = None
    if rank == 0:
        dependencies = {}
        if uses_code_targets:
            dependencies = {
                "tokenizer_checkpoint": tokenizer_ckpt,
                "code_ids_cache": train_cfg.get("code_ids_cache"),
            }
        identity_payload = build_run_identity("bert", cfg, dependencies=dependencies)
        if uses_code_targets:
            validate_cache_identity(
                train_cfg["code_ids_cache"],
                tokenizer_checkpoint=tokenizer_ckpt,
                contract_sha256=cache_contract_sha256,
            )
    if world_size > 1:
        objects = [identity_payload]
        dist.broadcast_object_list(objects, src=0)
        identity_payload = objects[0]
    run_identity = identity_payload
    tokenizer_sha256 = (
        run_identity["dependencies"]["tokenizer_checkpoint"]["sha256"]
        if uses_code_targets else None
    )

    if is_rank0():
        logger.info(f"Config: {cfg.get('_config_path', 'N/A')}")
        logger.info(f"Device: {device} | GPUs: {world_size} | Rank: {rank}")

    # ── 1. Dataset and cache metadata ──
    pretrain_cfg = make_pretrain_cfg(data_path, area_stats_path, cfg.get("area"))
    # Override patch granularity from YAML config (e.g. 40ms patches)
    if cfg.get("patch"):
        pretrain_cfg["patch"] = cfg["patch"]
    area_stats = load_area_stats(area_stats_path)
    validate_area_normalization_contract(area_stats, pretrain_cfg["area"])

    # Filter >max_patches patch trials (pass rows directly, no temp file)
    rows = read_packed_index(train_index)
    max_patches = int(cfg.get("bert", {}).get("max_patches", 256))
    patch_samples = int(pretrain_cfg["patch"]["samples"])
    rows_filtered = [r for r in rows if int(r.get("frame_length", 0)) // patch_samples <= max_patches]
    n_after_length_filter = len(rows_filtered)
    if bool(train_cfg.get("require_any_eye_keep", True)):
        rows_filtered, excluded_no_eye_rows = filter_packed_rows_with_usable_eye(rows_filtered)
    else:
        excluded_no_eye_rows = []
    if is_rank0():
        logger.info(
            "Filtered trials (≤%d patches @ %dms + any usable eye): %s "
            "(dropped_long=%s, dropped_both_eyes_invalid=%s)",
            max_patches,
            patch_samples,
            f"{len(rows_filtered):,}",
            f"{len(rows) - n_after_length_filter:,}",
            f"{len(excluded_no_eye_rows):,}",
        )

    dataset = PackedPretrainDataset(
        data_path, pretrain_cfg,
        rows=rows_filtered,
        area_stats=area_stats,
    )
    if is_rank0():
        logger.info(f"Training trials: {len(dataset)}")

    code_ids_cache = train_cfg.get("code_ids_cache")
    gid_to_idx, codes_arr = None, None
    if uses_code_targets:
        if not tokenizer_ckpt:
            raise ValueError("train.tokenizer_checkpoint is required for code targets")
        if not code_ids_cache:
            raise ValueError(
                "train.code_ids_cache is required for code targets. BERT never calls the tokenizer."
            )
        if not Path(code_ids_cache).is_file():
            raise FileNotFoundError(
                f"Code-ID cache does not exist: {code_ids_cache}. "
                "Run python -m eyemae.eyevq.precompute_codes before BERT training."
            )
        gid_to_idx, codes_arr = load_code_ids_cache(
            code_ids_cache,
            patch_samples=patch_samples,
            tokenizer_checkpoint=tokenizer_ckpt,
            tokenizer_sha256=tokenizer_sha256,
            contract_sha256=cache_contract_sha256,
            validate_identity=False,
        )
        if is_rank0():
            logger.info(f"Code-ID cache loaded: {len(gid_to_idx):,} trials from {code_ids_cache}")
            logger.info("Tokenizer is not constructed in the BERT process")
    elif is_rank0():
        logger.info("Target: raw_patch; tokenizer and code-ID cache are not used")

    # ── 2. DataLoader ──
    max_tokens = int(train_cfg.get("max_seq_tokens_per_gpu", 393216))
    max_trials = int(train_cfg.get("max_trials_per_gpu", 128))
    num_workers = int(train_cfg.get("num_workers", 4))

    sampler = TokenBatchSampler(
        dataset,
        max_seq_tokens=max_tokens,
        max_trials=max_trials,
        shuffle=True,
        seed=int(train_cfg.get("seed", 42)),
        bucket_by_length=False,   # 数据都 pad 到 max_patch, 分桶无收益; 纯随机打乱 (与 tokenizer 一致)
        infinite=True,
        rank=rank,
        world_size=world_size,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "collate_fn": lambda items: collate_trials_fixed_nmax(items, nmax=max_patches),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    loader = DataLoader(**loader_kwargs)

    # ── 3. Model: independent BERT weights, unweighted CE ──
    model = build_bert(cfg).to(device)

    if is_rank0():
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model params: {n_params:,} (trainable: {n_trainable:,})")

    # DDP (find_unused_parameters=False — all BERT params are reachable)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                        find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model

    # ── Resume from checkpoint (必须在 torch.compile 之前加载, 避免 _orig_mod. key 不匹配) ──
    resume_step = 0
    resume_best_val_acc = -1.0
    resume_best_val_loss = float("inf")
    if resume_from is not None:
        if is_rank0():
            logger.info(f"Resuming from {resume_from}")
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        if ck.get("config_version") != CONFIG_VERSION:
            raise ValueError("Refusing to resume a legacy pretrain checkpoint without the current config schema")
        assert_run_identity(
            ck,
            run_identity,
            allow_mismatch=allow_resume_identity_mismatch,
        )
        ck_sd = ck["model_state_dict"]
        # 剥离 module. / _orig_mod. 前缀 (ckpt 由 DDP + torch.compile 保存)
        if any(k.startswith("module.") for k in ck_sd):
            ck_sd = {k[len("module."):]: v for k, v in ck_sd.items()}
        if any("_orig_mod." in k for k in ck_sd):
            ck_sd = {k.replace("_orig_mod.", ""): v for k, v in ck_sd.items()}
        raw_model.load_state_dict(normalized_state_dict(ck_sd), strict=True)
        if is_rank0():
            logger.info("Resume: 全部权重加载成功")
        resume_step = int(ck.get("step", 0))
        resume_best_val_acc = float(ck.get("best_val_acc", -1.0))
        resume_best_val_loss = float(ck.get("best_val_loss", float("inf")))
        if is_rank0():
            logger.info(f"Resumed at step={resume_step}")
    # torch.compile — disabled for compatibility
    compile_enabled = train_cfg.get("compile_model", False)
    if compile_enabled and hasattr(torch, "compile"):
        try:
            # Persist compiled kernels across runs to avoid recompilation
            _cache_dir = str(output_dir / "torchinductor_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cache_dir)

            if is_rank0():
                logger.info(f"Compiling transformer per-layer (dynamic=True, like EyeFM) + pred_head...")
                logger.info(f"  Inductor cache: {_cache_dir}")

            # Per-layer compile: same strategy as EyeFM — each TransformerBlock individually.
            # Compiling the entire nn.TransformerEncoder as one giant graph is 10× slower
            # because Inductor tries to fuse across all 12 layers.
            n_layers = len(raw_model.transformer)
            for i in range(n_layers):
                raw_model.transformer[i] = torch.compile(
                    raw_model.transformer[i], dynamic=True
                )
            raw_model.pred_head = torch.compile(raw_model.pred_head, dynamic=True)
            if is_rank0():
                logger.info(f"torch.compile applied ({n_layers} layers individually + pred_head)")
            if world_size > 1:
                dist.barrier()  # sync after compile applied
        except Exception as e:
            if is_rank0():
                logger.warning(f"torch.compile failed ({e}), falling back to eager")

    # ── 5. Optimizer ──
    optimizer_groups, optimizer_audit = build_adamw_param_groups(
        model, weight_decay=float(train_cfg.get("weight_decay", 0.05))
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
    )
    if is_rank0():
        write_json(output_dir / "optimizer_groups.json", optimizer_audit)

    # 恢复优化器状态 (resume)
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])

    use_bf16 = train_cfg.get("precision") == "bf16"
    grad_clip_val = float(train_cfg.get("grad_clip", 1.0))

    # ── 6. Training loop ──
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", base_lr / 10))
    warmup_steps = int(train_cfg.get("warmup_steps", 2000))
    max_steps = int(train_cfg["total_steps"])
    lr_schedule_steps = int(train_cfg.get("lr_schedule_total_steps", max_steps))
    if lr_schedule_steps < max_steps:
        raise ValueError(
            "train.lr_schedule_total_steps must be >= train.total_steps; "
            f"got {lr_schedule_steps} < {max_steps}"
        )
    if lr_schedule_steps <= warmup_steps:
        raise ValueError(
            "train.lr_schedule_total_steps must be greater than warmup_steps"
        )
    log_every = int(train_cfg.get("log_every_steps", 100))
    val_every = int(train_cfg.get("val_every_steps", 500))
    save_every = int(train_cfg.get("save_every_steps", 10000))

    # ── Validation dataset ──
    val_index_path = train_cfg.get("val_index", None)
    val_loader = None
    val_mask_generator = None
    if val_index_path and is_rank0():
        val_index = str(Path(data_path) / val_index_path)
        if Path(val_index).exists():
            val_rows = read_packed_index(val_index)
            val_rows = [r for r in val_rows if int(r.get("frame_length", 0)) // patch_samples <= max_patches]
            if bool(train_cfg.get("require_any_eye_keep", True)):
                val_rows, excluded_val_no_eye_rows = filter_packed_rows_with_usable_eye(val_rows)
            else:
                excluded_val_no_eye_rows = []
            val_dataset = PackedPretrainDataset(data_path, pretrain_cfg, rows=val_rows, area_stats=area_stats)
            val_sampler = TokenBatchSampler(val_dataset, max_seq_tokens=max_tokens, max_trials=max_trials,
                                            shuffle=False, seed=int(train_cfg["seed"]),
                                            bucket_by_length=False, infinite=False, rank=0, world_size=1)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0,
                                    pin_memory=True, collate_fn=lambda items: collate_trials_fixed_nmax(items, nmax=max_patches))
            val_mask_generator = torch.Generator(device).manual_seed(42)
            logger.info(
                "Validation trials: %s (dropped_both_eyes_invalid=%s)",
                f"{len(val_dataset):,}",
                f"{len(excluded_val_no_eye_rows):,}",
            )

    @torch.no_grad()
    def run_validation(model_eval, step):
        if val_loader is None:
            return {}
        # 固定验证 seed: 每次验证用完全相同的样本 + mask, 结果严格可比
        val_mask_generator.manual_seed(42)
        model_eval.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        n_batches = 0
        acc_top5_sum = 0.0
        acc_top10_sum = 0.0
        per_dim_acc_sum = None
        per_dim_ce_sum = None
        dim_dist_sum = None
        total_masked = 0
        total_supervised_trials = 0
        total_masked_pairs = 0
        total_eligible_pairs = 0
        n_trials = 0
        max_val_batches_cfg = train_cfg.get("max_val_batches")
        max_val_batches = (
            None if max_val_batches_cfg is None else int(max_val_batches_cfg)
        )
        if max_val_batches is not None and max_val_batches <= 0:
            raise ValueError("train.max_val_batches must be positive or null")
        for val_batch in val_loader:
            val_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in val_batch.items()}
            v_content = val_batch["content"].transpose(-1, -2).contiguous()
            v_stim = val_batch["stim"].transpose(-1, -2).contiguous()
            v_quality = val_batch["quality"]
            v_pad = val_batch["pad_mask"]
            v_nm = val_batch["eye_nonmissing_frac"]
            v_task = val_batch["task_id"]
            v_valid = v_nm >= min_nonmissing
            v_code_ids = (
                lookup_code_ids(val_batch, gid_to_idx, codes_arr, device, max_patches)
                if uses_code_targets else None
            )
            v_mask_output = generate_eyemae_mask(
                v_valid,
                v_pad,
                mask_ratio,
                mode=mask_mode,
                generator=val_mask_generator,
                span_min=mask_span_min,
                span_max=mask_span_max,
                span_length_distribution=mask_span_length_distribution,
                span_length_power=mask_span_length_power,
                span_length_probabilities_values=mask_span_length_probabilities,
                multiblock_num_blocks=mask_multiblock_num_blocks,
                multiblock_target_scale_min=mask_multiblock_target_scale_min,
                multiblock_target_scale_max=mask_multiblock_target_scale_max,
                multiblock_min_gap=mask_multiblock_min_gap,
                return_span_lengths=use_span_length_embedding,
            )
            if use_span_length_embedding:
                v_mask, _, v_span_lengths = v_mask_output
            else:
                v_mask, _ = v_mask_output
                v_span_lengths = None
            v_masked_pairs = int(v_mask[:, 2::3].sum().item())
            v_eligible_pairs = int(paired_time_eligibility(v_valid, v_pad).sum().item())

            v_loss, v_stats = model_eval(stim_patches=v_stim, eye_patches=v_content,
                                          quality=v_quality, pad_mask=v_pad,
                                          eye_nonmissing_frac=v_nm, task_ids=v_task,
                                          eye_code_ids=v_code_ids, bert_mask=v_mask,
                                          mask_span_lengths=v_span_lengths)
            n_masked = int(torch.as_tensor(v_stats.get("n_masked", 0)).item())
            n_supervised_trials = int(
                torch.as_tensor(v_stats.get("n_supervised_trials", 0)).item()
            )
            batch_trials = int(v_content.shape[0])
            total_val_loss += v_loss.item() * n_supervised_trials
            # Some length buckets can contain no eligible eye token at all.
            # The model then quite correctly reports n_masked=0, but accuracy is
            # the mean of an empty tensor (NaN).  NaN * 0 is still NaN, so only
            # accumulate token metrics for batches that actually supervise a
            # token.  Loss is trial-weighted separately and remains well-defined.
            if n_masked > 0:
                total_val_acc += (
                    float(torch.as_tensor(v_stats["bert_acc"]).item()) * n_masked
                )
            total_masked += n_masked
            total_supervised_trials += n_supervised_trials
            total_masked_pairs += v_masked_pairs
            total_eligible_pairs += v_eligible_pairs
            n_trials += batch_trials
            if n_masked > 0 and "acc_top5" in v_stats:
                acc_top5_sum += float(torch.as_tensor(v_stats["acc_top5"]).item()) * n_masked
                acc_top10_sum += float(torch.as_tensor(v_stats["acc_top10"]).item()) * n_masked
            if "per_dim_acc" in v_stats:
                if per_dim_acc_sum is None:
                    per_dim_acc_sum = [0.0] * len(v_stats["per_dim_acc"])
                    per_dim_ce_sum = [0.0] * len(v_stats["per_dim_ce"])
                    dim_dist_sum = [0.0] * len(v_stats["dim_dist"])
                for i, value in enumerate(v_stats["per_dim_ce"].tolist()):
                    per_dim_ce_sum[i] += value * n_supervised_trials
                if n_masked > 0:
                    for i, a in enumerate(v_stats["per_dim_acc"].tolist()):
                        per_dim_acc_sum[i] += a * n_masked
                    for i, d in enumerate(v_stats["dim_dist"].tolist()):
                        dim_dist_sum[i] += d * n_masked
            n_batches += 1
            if max_val_batches is not None and n_batches >= max_val_batches:
                break
        model_eval.train()
        avg_loss = total_val_loss / max(total_supervised_trials, 1)
        avg_acc = total_val_acc / max(total_masked, 1)
        res = {"val/loss": avg_loss,
               "val/acc": avg_acc,
               "val/perplexity": math.exp(avg_loss),
               "val/batches": float(n_batches),
               "val/trials": float(n_trials),
               "val/masked_tokens": float(total_masked),
               "val/supervised_trials": float(total_supervised_trials),
               "val/masked_pairs": float(total_masked_pairs),
               "val/eligible_pairs": float(total_eligible_pairs),
               "val/realized_pair_mask_ratio": (
                   total_masked_pairs / max(total_eligible_pairs, 1)
               )}
        if total_masked > 0 and acc_top5_sum > 0:
            res["val/acc_top5"] = acc_top5_sum / total_masked
            res["val/acc_top10"] = acc_top10_sum / total_masked
        if per_dim_acc_sum is not None:
            res["val/per_dim_ce"] = [
                value / max(total_supervised_trials, 1) for value in per_dim_ce_sum
            ]
            res["val/per_dim_acc"] = [a / max(total_masked, 1) for a in per_dim_acc_sum]
            res["val/dim_dist"] = [d / max(total_masked, 1) for d in dim_dist_sum]
        return res


    mask_ratio = float(mask_cfg.get("eye_masking_ratio", 0.25))
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask.eye_masking_ratio must be in (0, 1), got {mask_ratio}")
    mask_mode = str(mask_cfg.get("mode", PAIRED_RANDOM))
    if mask_mode == "uniform":
        mask_mode = PAIRED_RANDOM
    if mask_mode not in {PAIRED_RANDOM, PAIRED_SPAN, PAIRED_MULTIBLOCK}:
        raise ValueError(f"Unsupported paired mask mode: {mask_mode!r}")
    mask_span_min = int(mask_cfg.get("span_min_patches", 2))
    mask_span_max = int(mask_cfg.get("span_max_patches", 6))
    mask_span_length_distribution = str(
        mask_cfg.get("span_length_distribution", "uniform")
    )
    mask_span_length_power = float(mask_cfg.get("span_length_power", 1.0))
    raw_span_probabilities = mask_cfg.get("span_length_probabilities")
    mask_span_length_probabilities = (
        None
        if raw_span_probabilities is None
        else [float(value) for value in raw_span_probabilities]
    )
    use_span_length_embedding = bool(
        cfg.get("bert", {}).get("predictor_span_length_embedding", False)
    )
    mask_multiblock_num_blocks = int(mask_cfg.get("multiblock_num_blocks", 4))
    mask_multiblock_target_scale_min = float(
        mask_cfg.get("multiblock_target_scale_min", 0.12)
    )
    mask_multiblock_target_scale_max = float(
        mask_cfg.get("multiblock_target_scale_max", 0.18)
    )
    mask_multiblock_min_gap = int(mask_cfg.get("multiblock_min_gap", 1))
    if mask_span_min < 1 or mask_span_max < mask_span_min:
        raise ValueError(
            "mask span bounds must satisfy 1 <= span_min_patches <= "
            f"span_max_patches, got [{mask_span_min}, {mask_span_max}]"
        )
    min_nonmissing = float(cfg.get("vq", {}).get("min_nonmissing_frac", 0.85))

    if is_rank0():
        logger.info(
            f"Total steps: {max_steps} | LR schedule steps: {lr_schedule_steps} "
            f"| Warmup: {warmup_steps} | BF16: {use_bf16}"
        )
        logger.info(f"Loss: label_smoothing={float(cfg.get('loss', {}).get('label_smoothing', 0.0))} (0=关闭)")
        span_text = (
            f"; span=[{mask_span_min},{mask_span_max}]"
            f"; length_distribution={mask_span_length_distribution}"
            f"; length_power={mask_span_length_power:g}"
            f"; length_probabilities={mask_span_length_probabilities}"
            if mask_mode == PAIRED_SPAN else ""
        )
        if mask_mode == PAIRED_MULTIBLOCK:
            span_text = (
                f"; blocks={mask_multiblock_num_blocks}"
                f"; target_scale=[{mask_multiblock_target_scale_min:g},"
                f"{mask_multiblock_target_scale_max:g}]"
                f"; overlap=false; min_gap={mask_multiblock_min_gap}"
                "; one primary length per trial"
            )
        logger.info(
            f"Mask: mode={mask_mode}; reference_target={mask_ratio:.0%}{span_text}; "
            "L/R are masked together at the same time patch; eligibility requires "
            "both eyes valid and non-padding"
        )
        logger.info(f"Min nonmissing: {min_nonmissing} | Fixed Nmax: {max_patches}")
        logger.info(
            "Predictor span-length embedding: %s (predictor-only; absent from encoder/downstream)",
            use_span_length_embedding,
        )
        logger.info("=" * 60)

    model.train()

    # Sync all ranks before starting the training loop.
    if world_size > 1:
        dist.barrier()

    # 恢复步数 (resume) 或从 0 开始
    if resume_from is not None:
        # 从加载的 ckpt 恢复步数 (已在 optimizer 加载时设置 resume_step)
        global_step = resume_step
        best_val_acc = resume_best_val_acc
        best_val_loss = resume_best_val_loss
    else:
        global_step = 0
        best_val_acc = -1.0
        best_val_loss = float("inf")
    sampler.set_start_batch(global_step)
    total_loss = torch.zeros((), device=device)
    total_acc = torch.zeros((), device=device)
    steps_since_log = 0
    last_log_time = time.time()
    nan_count = 0
    max_nan_steps = 5
    total_data_wait = 0.0
    total_gpu_time = 0.0

    for epoch in range(1000):
        if global_step >= max_steps:
            break
        sampler.set_epoch(epoch)

        prev_iter_end = time.time()
        for batch in loader:
            if global_step >= max_steps:
                break

            data_wait = time.time() - prev_iter_end

            _t0 = time.time()
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Eye validity for masking (lightweight, before step timer)
            eye_valid = (batch["eye_nonmissing_frac"] >= min_nonmissing)  # [B,N,2]
            pad_mask = batch["pad_mask"]

            # Prepare inputs
            content = batch["content"].transpose(-1, -2).contiguous()  # [B,N,2,4,20]
            stim = batch["stim"].transpose(-1, -2).contiguous()        # [B,N,4,20]
            quality = batch["quality"]                                  # [B,N,2,20,1]
            eye_nonmissing = batch["eye_nonmissing_frac"]               # [B,N,2]
            task_ids = batch["task_id"]                                 # [B]

            # ── 1. Read precomputed labels; no tokenizer exists in this process ──
            eye_code_ids = (
                lookup_code_ids(batch, gid_to_idx, codes_arr, device, max_patches)
                if uses_code_targets else None
            )

            # ── 2. Mask jointly valid L/R tokens at the same time patches ──
            mask_output = generate_eyemae_mask(
                eye_valid=eye_valid,
                pad_mask=pad_mask,
                mask_ratio=mask_ratio,
                mode=mask_mode,
                span_min=mask_span_min,
                span_max=mask_span_max,
                span_length_distribution=mask_span_length_distribution,
                span_length_power=mask_span_length_power,
                span_length_probabilities_values=mask_span_length_probabilities,
                multiblock_num_blocks=mask_multiblock_num_blocks,
                multiblock_target_scale_min=mask_multiblock_target_scale_min,
                multiblock_target_scale_max=mask_multiblock_target_scale_max,
                multiblock_min_gap=mask_multiblock_min_gap,
                return_span_lengths=use_span_length_embedding,
            )
            if use_span_length_embedding:
                bert_mask, _, mask_span_lengths = mask_output
            else:
                bert_mask, _ = mask_output
                mask_span_lengths = None
            should_log = global_step % log_every == 0
            masked_pairs_tensor = bert_mask[:, 2::3].sum()
            eligible_pairs_tensor = (
                paired_time_eligibility(eye_valid, pad_mask).sum()
                if should_log else None
            )
            local_supervised_trials = bert_mask.any(dim=1).sum()
            global_supervised_trials = None
            denominator_work = None
            if world_size > 1:
                global_supervised_trials = local_supervised_trials.detach().clone()
                denominator_work = dist.all_reduce(
                    global_supervised_trials,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
            _t_mask_done = time.time()

            step_start = time.time()

            lr = get_lr(
                global_step, warmup_steps, lr_schedule_steps, base_lr, min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            _t_prep_done = time.time()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # ── 3. BERT forward (raw patches → predict code IDs) ──
                loss, stats = model(
                    stim_patches=stim,
                    eye_patches=content,
                    quality=quality,
                    pad_mask=pad_mask,
                    eye_nonmissing_frac=eye_nonmissing,
                    task_ids=task_ids,
                    eye_code_ids=eye_code_ids,
                    bert_mask=bert_mask,
                    mask_span_lengths=mask_span_lengths,
                )
            _t_fwd_done = time.time()

            backward_loss = loss
            if denominator_work is not None:
                # The tiny denominator collective was launched before the
                # forward pass, so its communication is hidden under compute.
                denominator_work.wait()
                backward_loss = loss * (
                    world_size * local_supervised_trials
                    / global_supervised_trials.clamp_min(1)
                )


            # Device-side assertion is enqueued on the CUDA stream and does
            # not force the host to wait before backward.  A non-finite loss
            # still fails fast instead of contaminating optimizer state.
            torch._assert_async(torch.isfinite(loss), "non-finite BERT loss")

            optimizer.zero_grad(set_to_none=True)
            backward_loss.backward()
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

            total_loss.add_(loss.detach())
            total_acc.add_(stats["bert_acc"])
            steps_since_log += 1

            gpu_time = time.time() - step_start
            total_data_wait += data_wait
            total_gpu_time += gpu_time

            # Detailed timing breakdown (log every 500 steps)
            _t_bwd_done = time.time()
            _time_mask = (_t_mask_done - _t0) * 1000
            _time_prep = (_t_prep_done - _t_mask_done) * 1000
            _time_fwd  = (_t_fwd_done - _t_prep_done) * 1000
            _time_bwd  = (_t_bwd_done - _t_fwd_done) * 1000

            # Log
            # Per-rank batch sizes for imbalance debugging (log every 500 steps).
            # dist.all_gather is a collective op — ALL ranks must participate.
            _batch_sizes_for_log: list[int] | None = None
            if world_size > 1 and global_step % 500 == 0:
                _t = torch.tensor(content.shape[0], device=device)
                _all_sizes = [torch.zeros_like(_t) for _ in range(world_size)]
                dist.all_gather(_all_sizes, _t)
                _batch_sizes_for_log = [int(s.item()) for s in _all_sizes]

            if is_rank0() and should_log:
                masked_pairs, eligible_pairs, loss_sum, acc_sum = torch.stack([
                    masked_pairs_tensor.to(total_loss.dtype),
                    eligible_pairs_tensor.to(total_loss.dtype),
                    total_loss,
                    total_acc,
                ]).tolist()
                avg_loss = loss_sum / steps_since_log
                avg_acc = acc_sum / steps_since_log
                elapsed = time.time() - last_log_time
                steps_per_sec = steps_since_log / elapsed if elapsed > 0 else 0
                avg_data_wait = total_data_wait / steps_since_log
                avg_gpu = total_gpu_time / steps_since_log
                parts = [
                    f"step={global_step:06d}/{max_steps}",
                    f"loss={avg_loss:.5f}",
                    f"acc={avg_acc:.3f}",
                    f"lr={lr:.2e}",
                    f"masked={int(masked_pairs * 2)}",
                    f"masked_pairs={int(masked_pairs)}",
                    f"pair_ratio={masked_pairs / max(eligible_pairs, 1):.3f}",
                    f"batch={content.shape[0]}",
                    f"step/s={steps_per_sec:.1f}",
                    f"gpu={avg_gpu*1000:.0f}ms",
                    f"mask={_time_mask:.0f}ms",
                    f"prep={_time_prep:.0f}ms",
                    f"fwd={_time_fwd:.0f}ms",
                    f"bwd={_time_bwd:.0f}ms",
                ]
                if _batch_sizes_for_log is not None:
                    parts.append(f"batch_dist={_batch_sizes_for_log}")
                logger.info(" | ".join(parts))

                total_loss.zero_()
                total_acc.zero_()
                steps_since_log = 0
                last_log_time = time.time()
                total_data_wait = 0.0
                total_gpu_time = 0.0

            # Rank-0 validation is bracketed by collectives. Peers cannot enter
            # the next DDP forward while rank 0 is still evaluating/saving.
            val_metrics = None
            should_validate = global_step > 0 and (
                global_step % val_every == 0 or global_step % save_every == 0
            )
            if should_validate:
                if world_size > 1:
                    dist.barrier()
                if is_rank0():
                    val_metrics = run_validation(raw_model, global_step)
                    if val_metrics:
                        _line = (f"  val/loss={val_metrics['val/loss']:.5f} "
                                 f"val/acc={val_metrics['val/acc']:.4f} val/ppl={val_metrics['val/perplexity']:.1f}")
                        if "val/acc_top5" in val_metrics:
                            _line += (f" top5={val_metrics['val/acc_top5']:.4f} "
                                      f"top10={val_metrics['val/acc_top10']:.4f}")
                        if "val/per_dim_acc" in val_metrics:
                            _pdce = ",".join(
                                f"{value:.3f}" for value in val_metrics["val/per_dim_ce"]
                            )
                            _pda = ",".join(f"{a:.3f}" for a in val_metrics["val/per_dim_acc"])
                            _dd = ",".join(f"{d:.2f}" for d in val_metrics["val/dim_dist"])
                            _line += (
                                f" | per-dim_ce=[{_pdce}] per-dim_acc=[{_pda}] "
                                f"dim_dist=[{_dd}]"
                            )
                        logger.info(_line)
                if world_size > 1:
                    dist.barrier()

            # ── Save checkpoint (every save_every steps) ──
            if is_rank0() and global_step % save_every == 0 and global_step > 0:
                ckpt_path = output_dir / f"ckpt_step{global_step:06d}.pt"
                improved = bool(
                    val_metrics
                    and (
                        val_metrics["val/loss"] < best_val_loss
                        if target_type == TARGET_RAW_PATCH
                        else val_metrics["val/acc"] > best_val_acc
                    )
                )
                if improved:
                    best_val_acc = val_metrics["val/acc"]
                    best_val_loss = val_metrics["val/loss"]
                state = {
                    "step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "config_version": CONFIG_VERSION,
                    "run_identity": run_identity,
                    "val_metrics": val_metrics,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "rng_states": {
                        "python": __import__("random").getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    },
                }
                atomic_torch_save(state, ckpt_path)
                logger.info(f"Saved: {ckpt_path}")

                # Best checkpoint
                if improved:
                    best_path = output_dir / "ckpt_best.pt"
                    atomic_torch_save(state, best_path)
                    selection = (
                        f"val/loss={best_val_loss:.5f}"
                        if target_type == TARGET_RAW_PATCH
                        else f"val/acc={best_val_acc:.4f}"
                    )
                    logger.info(f"🏆 New best: {selection}")

            if world_size > 1 and global_step % save_every == 0 and global_step > 0:
                dist.barrier()

            global_step += 1
            prev_iter_end = time.time()

    # Always evaluate the actual final weights; periodic validation is checked
    # before incrementing the zero-based step counter.
    if world_size > 1:
        dist.barrier()
    final_val_metrics = None
    if is_rank0():
        final_val_metrics = run_validation(raw_model, global_step)
        if final_val_metrics:
            logger.info(
                "Final validation | "
                + " | ".join(
                    f"{key}={value:.5f}"
                    for key, value in final_val_metrics.items()
                    if isinstance(value, (int, float))
                )
            )
            final_improved = (
                final_val_metrics["val/loss"] < best_val_loss
                if target_type == TARGET_RAW_PATCH
                else final_val_metrics["val/acc"] > best_val_acc
            )
            if final_improved:
                best_val_acc = final_val_metrics["val/acc"]
                best_val_loss = final_val_metrics["val/loss"]
                atomic_torch_save({
                    "step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "config_version": CONFIG_VERSION,
                    "run_identity": run_identity,
                    "val_metrics": final_val_metrics,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                }, output_dir / "ckpt_best.pt")
        final_path = output_dir / "ckpt_final.pt"
        atomic_torch_save({
            "step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            "config_version": CONFIG_VERSION,
            "run_identity": run_identity,
            "val_metrics": final_val_metrics,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }, final_path)
        logger.info(f"Done. Steps: {global_step}. Final: {final_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eyevq_bert_base")
    parser.add_argument("--resume", type=str, default=None, help="续训: 从指定 ckpt 恢复 (模型+优化器+步数)")
    parser.add_argument(
        "--allow-resume-identity-mismatch",
        action="store_true",
        help="Explicitly allow a deliberate schedule/config extension from a non-identical checkpoint",
    )
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None,
                        help="Tokenizer checkpoint identity recorded in the code-ID cache")
    parser.add_argument("--code-ids-cache", type=str, default=None,
                        help="Required precomputed code-ID cache")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override train.total_steps")
    parser.add_argument("--mask-ratio", type=float, default=None,
                        help="Override the paired L/R time-patch mask ratio")
    parser.add_argument(
        "--mask-mode",
        choices=(PAIRED_RANDOM, PAIRED_SPAN, PAIRED_MULTIBLOCK),
        default=None,
                        help="Override paired masking policy")
    parser.add_argument("--mask-span-min", type=int, default=None,
                        help="Minimum paired-span length in patches")
    parser.add_argument("--mask-span-max", type=int, default=None,
                        help="Maximum paired-span length in patches")
    parser.add_argument(
        "--mask-span-length-distribution",
        choices=("uniform", "symmetric_power", "token_balanced", "explicit"),
        default=None,
        help="Categorical prior over paired-span lengths",
    )
    parser.add_argument("--mask-span-power", type=float, default=None,
                        help="Positive exponent for symmetric_power span lengths")
    parser.add_argument(
        "--mask-span-probabilities",
        default=None,
        help="Comma-separated explicit probabilities for span_min..span_max",
    )
    parser.add_argument("--mask-multiblock-num-blocks", type=int, default=None)
    parser.add_argument("--mask-multiblock-scale-min", type=float, default=None)
    parser.add_argument("--mask-multiblock-scale-max", type=float, default=None)
    parser.add_argument("--mask-multiblock-min-gap", type=int, default=None)
    parser.add_argument("--fsq-levels", default=None,
                        help="Comma-separated FSQ levels, e.g. 9,7,7,5,5")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config
    override_fsq_levels(cfg, args.fsq_levels)
    if args.tokenizer_checkpoint is not None:
        cfg["train"]["tokenizer_checkpoint"] = args.tokenizer_checkpoint
    if args.code_ids_cache is not None:
        cfg["train"]["code_ids_cache"] = args.code_ids_cache
    if args.max_steps is not None:
        cfg["train"]["total_steps"] = int(args.max_steps)
    if args.mask_ratio is not None:
        cfg["mask"]["eye_masking_ratio"] = float(args.mask_ratio)
    if args.mask_mode is not None:
        cfg["mask"]["mode"] = str(args.mask_mode)
    if args.mask_span_min is not None:
        cfg["mask"]["span_min_patches"] = int(args.mask_span_min)
    if args.mask_span_max is not None:
        cfg["mask"]["span_max_patches"] = int(args.mask_span_max)
    if args.mask_span_length_distribution is not None:
        cfg["mask"]["span_length_distribution"] = str(
            args.mask_span_length_distribution
        )
    if args.mask_span_power is not None:
        cfg["mask"]["span_length_power"] = float(args.mask_span_power)
    if args.mask_span_probabilities is not None:
        cfg["mask"]["span_length_probabilities"] = [
            float(value.strip())
            for value in args.mask_span_probabilities.split(",")
        ]
    if args.mask_multiblock_num_blocks is not None:
        cfg["mask"]["multiblock_num_blocks"] = int(args.mask_multiblock_num_blocks)
    if args.mask_multiblock_scale_min is not None:
        cfg["mask"]["multiblock_target_scale_min"] = float(
            args.mask_multiblock_scale_min
        )
    if args.mask_multiblock_scale_max is not None:
        cfg["mask"]["multiblock_target_scale_max"] = float(
            args.mask_multiblock_scale_max
        )
    if args.mask_multiblock_min_gap is not None:
        cfg["mask"]["multiblock_min_gap"] = int(args.mask_multiblock_min_gap)

    output_dir = Path(args.output_dir)
    train_bert(
        cfg,
        output_dir,
        resume_from=args.resume,
        allow_resume_identity_mismatch=args.allow_resume_identity_mismatch,
    )


if __name__ == "__main__":
    main()
