#!/usr/bin/env python3
"""使用训练完成的 tokenizer 预计算 code IDs，供 BERT 全程只读缓存。

code IDs 只依赖输入和固定 tokenizer，因此这里只编码一次；正式 BERT
训练进程不会构造 tokenizer，也没有动态回退路径。

批量编码: 每批取 n_batch 个 trial, padding 到 max_patches, 一次 forward 编码,
GPU 利用率高。DDP 4 卡并行: 数据按 trial 切分到各 rank, 每卡 batch-per-gpu=512,
各自编码后保存分片, rank0 合并。

缓存格式 (npz):
  gids:      [N] object (global_trial_id)
  code_ids:  [N, max_patches, 2] int16 (无效位置填 0)
  num_patches: [N] int16 (每个 trial 实际 patch 数)
用法 (4卡):
  torchrun --nproc_per_node=4 --master_port=29657 \
      -m eyemae.eyevq.precompute_codes \
      --config configs/eyevq/pretrain_joint.yaml \
      --tokenizer-checkpoint outputs/eyevq/tokenizer_joint_fsq9755_50k/ckpt_final.pt \
      --out outputs/eyevq/code_ids_joint_fsq9755_50k_v2.npz --split all --n-batch 512
"""
import argparse, os, time
from pathlib import Path
import numpy as np
import torch

from eyemae.data import (
    PackedPretrainDataset,
    filter_packed_rows_with_usable_eye,
    load_area_stats,
    read_packed_index,
    validate_area_normalization_contract,
)
from eyemae.eyevq.config import (
    _quantizer_spec,
    load_tokenizer_checkpoint,
    override_fsq_levels,
)
from eyemae.eyevq.artifacts import (
    CACHE_FORMAT_VERSION,
    cache_contract,
    sha256_file,
    sha256_json,
    write_cache_manifest,
)
from eyemae.eyevq.pretrain.train import make_pretrain_cfg


def mask_invalid_eye_codes(
    code_ids: torch.Tensor,
    eye_nonmissing_frac: torch.Tensor,
    pad_mask: torch.Tensor,
    min_nonmissing_frac: float,
) -> torch.Tensor:
    """Set every non-trainable eye/padding position to the fixed sentinel 0."""
    valid_eye = (eye_nonmissing_frac >= float(min_nonmissing_frac)) & (~pad_mask.unsqueeze(-1))
    return code_ids.masked_fill(~valid_eye, 0)


def load_tokenizer_for_cache(checkpoint_path, cfg, device, pretrain_cfg):
    """Load and validate the tokenizer only in the offline cache stage."""
    tokenizer, tokenizer_cfg, checkpoint = load_tokenizer_checkpoint(checkpoint_path, device)
    expected_patch = int(pretrain_cfg["patch"]["samples"])
    actual_patch = int(tokenizer_cfg["patch"]["samples"])
    if expected_patch != actual_patch:
        raise ValueError(
            f"Tokenizer patch.samples={actual_patch} does not match pretrain "
            f"patch.samples={expected_patch}"
        )
    tokenizer_spec = _quantizer_spec(tokenizer_cfg)
    pretrain_spec = _quantizer_spec(cfg)
    if tokenizer_spec != pretrain_spec:
        raise ValueError(
            "Pretrain quantizer does not match the tokenizer checkpoint: "
            f"{pretrain_spec} != {tokenizer_spec}"
        )
    if int(checkpoint.get("step", -1)) < 0:
        raise ValueError("Tokenizer checkpoint has no valid training step")
    return tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-checkpoint", default=None,
                    help="Override train.tokenizer_checkpoint")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    ap.add_argument("--device", default=None)  # torchrun 自动设 cuda:local_rank
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--fsq-levels", default=None,
                    help="Comma-separated FSQ levels matching the tokenizer")
    args = ap.parse_args()

    # DDP 设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device("cuda", local_rank)
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        dist = None

    import yaml
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    override_fsq_levels(cfg, args.fsq_levels)
    train_cfg = cfg["train"]
    tokenizer_checkpoint = args.tokenizer_checkpoint or train_cfg["tokenizer_checkpoint"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data_path = train_cfg["data_path"]
    area_stats_path = train_cfg.get("area_stats_path")
    pretrain_cfg = make_pretrain_cfg(data_path, area_stats_path, cfg.get("area"))
    if cfg.get("patch"):
        pretrain_cfg["patch"] = cfg["patch"]
    area_stats = load_area_stats(area_stats_path)
    validate_area_normalization_contract(area_stats, pretrain_cfg["area"])

    tokenizer = load_tokenizer_for_cache(
        tokenizer_checkpoint, cfg, device, pretrain_cfg
    )
    contract_sha256 = sha256_json(cache_contract(cfg, split=args.split))
    tokenizer_sha256 = sha256_file(tokenizer_checkpoint) if rank == 0 else ""
    max_patches = int(cfg.get("bert", {}).get("max_patches", 128))
    patch_samples = int(pretrain_cfg["patch"]["samples"])

    # 收集 rows (train / val / both)
    all_rows = []
    if args.split in ("train", "all"):
        idx = str(Path(data_path) / train_cfg["train_index"])
        rows = read_packed_index(idx)
        rows = [r for r in rows if int(r.get("frame_length", 0)) // patch_samples <= max_patches]
        if bool(train_cfg.get("require_any_eye_keep", True)):
            rows, excluded = filter_packed_rows_with_usable_eye(rows)
        else:
            excluded = []
        all_rows += rows
        if rank == 0:
            print(
                f"[train] {len(rows):,} "
                f"(dropped_both_eyes_invalid={len(excluded):,})",
                flush=True,
            )
    if args.split in ("val", "all"):
        idx = str(Path(data_path) / train_cfg["val_index"])
        if Path(idx).exists():
            rows = read_packed_index(idx)
            rows = [r for r in rows if int(r.get("frame_length", 0)) // patch_samples <= max_patches]
            if bool(train_cfg.get("require_any_eye_keep", True)):
                rows, excluded = filter_packed_rows_with_usable_eye(rows)
            else:
                excluded = []
            all_rows += rows
            if rank == 0:
                print(
                    f"[val] {len(rows):,} "
                    f"(dropped_both_eyes_invalid={len(excluded):,})",
                    flush=True,
                )
    if not all_rows:
        if rank == 0:
            print("无数据!")
        return

    ds = PackedPretrainDataset(data_path, pretrain_cfg, rows=all_rows, area_stats=area_stats)
    n_total = len(ds)

    # ── 按 rank 切分数据: 每个 rank 处理 [start:end) ──
    chunk = (n_total + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n_total)
    n_local = end - start
    if rank == 0:
        print(f"总 trial: {n_total:,} | world_size={world_size} | 每卡≈{n_local:,} | "
              f"max_patches={max_patches} | n_batch/卡={args.n_batch}", flush=True)
    if dist is not None:
        # Make the rank-to-device contract explicit.  Without device_ids NCCL
        # warns that it may choose an unknown GPU for a barrier and can hang on
        # systems whose visible-device order differs from physical numbering.
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    # 本地编码
    gids = []
    code_arr = np.zeros((n_local, max_patches, 2), dtype=np.int16)
    npatch_arr = np.zeros(n_local, dtype=np.int16)
    t0 = time.time()
    done = 0
    while done < n_local:
        hi = min(done + args.n_batch, n_local)
        items = [ds[start + i] for i in range(done, hi)]
        B = len(items)
        content = torch.zeros(B, max_patches, 2, 4, patch_samples, device=device)
        stim = torch.zeros(B, max_patches, 4, patch_samples, device=device)
        nm = torch.zeros(B, max_patches, 2, device=device)
        quality = torch.ones(B, max_patches, 2, patch_samples, 1, device=device)
        pad = torch.ones(B, max_patches, dtype=torch.bool, device=device)
        for b, item in enumerate(items):
            n = min(item["content"].shape[0], max_patches)
            content[b, :n] = torch.from_numpy(item["content"][:n]).permute(0, 1, 3, 2)  # [n,2,4,20]
            stim[b, :n] = torch.from_numpy(item["stim"][:n]).permute(0, 2, 1)            # [n,4,20]
            nm[b, :n] = torch.from_numpy(item["eye_nonmissing_frac"][:n])
            quality[b, :n] = torch.from_numpy(item["quality"][:n])
            pad[b, :n] = False
            gids.append(str(item["global_trial_id"]))
            npatch_arr[done + b] = n
        with torch.no_grad():
            cids = tokenizer.encode_codes(stim, content, quality, pad, nm)
            cids = mask_invalid_eye_codes(
                cids, nm, pad, float(cfg["vq"].get("min_nonmissing_frac", 0.50))
            )
            cids = cids.cpu().numpy().astype(np.int16)
        code_arr[done:hi] = cids
        done = hi
        if rank == 0 and done % 10000 < args.n_batch:
            el = time.time() - t0
            print(f"  rank0: {done:,}/{n_local:,}  ({el:.0f}s, {el/max(done,1)*1000:.2f}ms/trial)", flush=True)

    # 保存分片
    shard = f"{args.out}.rank{rank}.npz"
    np.savez_compressed(shard, gids=np.array(gids, dtype=object),
                        code_ids=code_arr, num_patches=npatch_arr)
    if rank == 0:
        el = time.time() - t0
        print(f"rank0 完成 {n_local:,} trials in {el:.0f}s | 分片 → {shard}", flush=True)

    # 合并 (rank0)
    if dist is not None:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
    if rank == 0:
        all_gids, all_code, all_np = [], [], []
        for r in range(world_size):
            s = np.load(f"{args.out}.rank{r}.npz", allow_pickle=True)
            all_gids.extend(s["gids"].tolist())
            all_code.append(s["code_ids"])
            all_np.append(s["num_patches"])
        code_ids = np.concatenate(all_code, axis=0)
        npatch = np.concatenate(all_np, axis=0)
        gids_arr = np.array(all_gids, dtype=object)
        # 按 gid 排序保持一致
        order = np.argsort(gids_arr, kind="stable")
        gids_arr = gids_arr[order]
        code_ids = code_ids[order]
        npatch = npatch[order]
        output_path = Path(args.out)
        temporary_path = output_path.with_name(output_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            format_version=np.int64(CACHE_FORMAT_VERSION),
            lr_layout=np.array("time_major_lr_v1"),
            invalid_eye_code_id=np.int64(0),
            min_nonmissing_frac=np.float64(cfg["vq"].get("min_nonmissing_frac", 0.50)),
            patch_samples=np.int64(patch_samples),
            tokenizer_checkpoint=np.array(str(Path(tokenizer_checkpoint).resolve())),
            tokenizer_sha256=np.array(tokenizer_sha256),
            cache_contract_sha256=np.array(contract_sha256),
            gids=gids_arr,
            code_ids=code_ids,
            num_patches=npatch,
        )
        temporary_path.replace(output_path)
        write_cache_manifest(
            output_path,
            tokenizer_checkpoint=tokenizer_checkpoint,
            contract_sha256=contract_sha256,
            num_trials=len(gids_arr),
        )
        print(f"合并完成 → {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB, "
              f"{len(gids_arr):,} trials)", flush=True)
        # 清理分片
        for r in range(world_size):
            p = Path(f"{args.out}.rank{r}.npz")
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
