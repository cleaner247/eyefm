#!/usr/bin/env python3
"""Build EyeMAE packed dataset from V5 (75Hz guarded low-pass) data using V4's index.

V5 is the 75Hz-filtered re-clean of the V3/V4 data: trial identity, splits and
per-view layouts are identical to the previous ML-ready dataset, so this builder
reuses the V4 index (shard assignment, trial order, frame offsets) unchanged and
only swaps the payload (X_data / y_frame) for the V5 filtered data.

Payload sources, in priority order:
  1. The 12 packed disease-view npz files (STORED, mmap-able).
  2. Per-trial npz files under V5 `_raw_structured_tmp/matched_groups_full`
     (covers trials outside the 12 views, e.g. 对照组数据汇总).

Output layout mirrors V4 exactly:
  <out>/pretrain/{shards,subjects.csv,trials.csv,pretrain/*.csv,columns.json,...}
  <out>/finetune/<task>/{shards,subjects.csv,trials.csv,train.csv,...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

V4_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v4")
V5_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v5")
V5_RAW = Path(
    "/mnt/disk_sde/data-260606/extracted/"
    "cd_speed4_hard_blink_fullfilter75_ml_ready_subjectkey_20260820"
)
OUT_ROOT = V5_ROOT  # pack pretrain/ and finetune/ next to the 12 views
UNPACKED_DIR = V5_ROOT / "_unpacked_npy"  # 12 视图 npz 解压后的裸 npy（mmap 用）

TASKS = (
    "ad_binary",
    "detox_binary",
    "epilepsy_binary",
    "mci_binary",
    "mci_matched_binary",
    "migraine_binary",
    "pd_binary",
    "pd_related_5class",
)
VIEWS = (
    "AD",
    "AD匹配后",
    "MCI",
    "MCI匹配后",
    "PD相关",
    "PD相关_帕金森病匹配后",
    "PD相关_特发性震颤匹配后",
    "PD相关_运动障碍匹配后",
    "PD相关_震颤匹配后",
    "偏头痛",
    "戒毒所",
    "癫痫",
)
SPLITS = ("train", "validation", "test")

X_COLUMNS = 10
Y_COLUMNS = 2

# Per-trial npz layout: gaze = [Lx, Ly, Larea, Rx, Ry, Rarea, Llabel, Rlabel]
# stimulus = [stim_x, stim_y, stim_on, cross_on]
# packed X_data = [Lx, Ly, Larea, Rx, Ry, Rarea, stim_x, stim_y, stim_on, fix_on]
# packed y_frame = [Llabel, Rlabel]

# Shared lookup tables (set in main, read-only in workers after fork).
_VIEW_MAP: dict[str, tuple[str, int]] = {}
_PER_TRIAL_MAP: dict[str, str] = {}


def stem_key(row: dict[str, str]) -> str:
    return f"{row['source_stem']}|{row['original_trial_index']}"


def build_view_map() -> dict[str, tuple[str, str, int]]:
    """(source_stem|original_trial_index) -> (view, split, packed_trial_index)."""
    result: dict[str, tuple[str, str, int]] = {}
    collisions: set[str] = set()
    for view in VIEWS:
        for split in SPLITS:
            manifest = V5_ROOT / view / f"manifest_{split}.csv"
            npz_path = V5_ROOT / view / f"{split}.npz"
            if not manifest.is_file() or not npz_path.is_file():
                continue
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = stem_key(row)
                    entry = (view, split, int(row["packed_trial_index"]))
                    if key in result and result[key] != entry:
                        collisions.add(key)
                        continue
                    result[key] = entry
    if collisions:
        # Never let filesystem/view iteration order silently select a payload.
        # Ambiguous keys must use the authoritative per-trial export fallback.
        for key in collisions:
            result.pop(key, None)
        print(
            f"[view-map] {len(collisions)} keys collide across views "
            "(excluded from views; per-trial fallback required)"
        )
    return result


def build_per_trial_map() -> dict[str, str]:
    """(source_stem|original_trial_index) -> per-trial npz path (fallback)."""
    manifest_all = V5_ROOT / "_raw_structured_tmp" / "matched_groups_full" / "manifest_all.csv"
    result: dict[str, str] = {}
    with manifest_all.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = stem_key(row)
            result.setdefault(key, row["trial_npz"])
    return result


def load_source_map() -> tuple[dict[str, tuple[str, str, int]], dict[str, str]]:
    view_map = build_view_map()
    per_trial_map = build_per_trial_map()
    print(f"[source-map] view entries={len(view_map)} per-trial entries={len(per_trial_map)}")
    return view_map, per_trial_map


def dataset_bases(dataset: str) -> tuple[Path, Path]:
    """(v4 source base, v5 output base) for a dataset name."""
    if dataset == "pretrain":
        return V4_ROOT / "pretrain", OUT_ROOT / "pretrain"
    return V4_ROOT / "finetune" / dataset, OUT_ROOT / "finetune" / dataset


def read_shard_rows(source_base: Path, shard_id: str) -> list[dict[str, str]]:
    """Read trial rows of one shard from the dataset-level trials.csv.

    trials.csv (unlike shard trial_index.csv) carries `source_stem`, which is
    the stable identity shared with the V5 manifests.
    """
    rows: list[dict[str, str]] = []
    with (source_base / "trials.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["shard_id"] == shard_id:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"{source_base}: shard {shard_id} has no rows in trials.csv")
    return rows


def unpack_view_npy(view: str, split: str) -> Path:
    """解压一个视图 npz 的 npy 条目到裸文件（STORED → 直接拷贝），返回目录。"""
    source = V5_ROOT / view / f"{split}.npz"
    destination = UNPACKED_DIR / view / split
    marker = destination / ".complete"
    if marker.is_file():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".partial"
    temporary.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for name in ("X_data.npy", "y_frame.npy", "X_offsets.npy", "X_lengths.npy"):
            with archive.open(name) as source_handle, \
                    (temporary / name).open("wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=64 * 1024 * 1024)
    for name in ("X_data.npy", "y_frame.npy", "X_offsets.npy", "X_lengths.npy"):
        os.replace(temporary / name, destination / name)
    temporary.rmdir()
    marker.write_text("ok")
    return destination


def unpack_all_view_npy() -> None:
    """解压 12 个视图的全部 split 到裸 npy（幂等）。"""
    started = time.time()
    jobs = [
        (view, split)
        for view in VIEWS
        for split in SPLITS
        if (V5_ROOT / view / f"{split}.npz").is_file()
    ]
    done = 0
    for view, split in jobs:
        unpack_view_npy(view, split)
        done += 1
        print(f"[unpack] {done}/{len(jobs)} {view}/{split} ({time.time() - started:.0f}s)", flush=True)
    print(f"[unpack] complete: {done} npz files, {time.time() - started:.0f}s")


def open_view_arrays(view: str, split: str) -> dict[str, Any]:
    """打开解压后的裸 npy（mmap）。"""
    directory = UNPACKED_DIR / view / split
    return {
        "X_data": np.load(directory / "X_data.npy", mmap_mode="r"),
        "y_frame": np.load(directory / "y_frame.npy", mmap_mode="r"),
        "X_offsets": np.load(directory / "X_offsets.npy", mmap_mode="r"),
    }


def copy_view_payload(
    arrays: dict[str, Any],
    packed_index: int,
    frame_length: int,
    out_x: np.ndarray,
    out_y: np.ndarray,
    out_pos: int,
) -> None:
    """Copy one trial's payload from opened view arrays into output arrays."""
    x = arrays["X_data"]
    y = arrays["y_frame"]
    start = int(arrays["X_offsets"][packed_index])
    end = start + frame_length
    out_x[out_pos : out_pos + frame_length] = x[start:end]
    out_y[out_pos : out_pos + frame_length] = y[start:end]


def copy_per_trial_payload(
    npz_path: str,
    frame_length: int,
    out_x: np.ndarray,
    out_y: np.ndarray,
    out_pos: int,
) -> None:
    """Copy one trial's payload from a per-trial npz into output arrays."""
    payload = np.load(npz_path, allow_pickle=False)
    gaze = payload["gaze"]
    stimulus = payload["stimulus"]
    if gaze.shape[0] != frame_length:
        raise RuntimeError(
            f"per-trial frame mismatch: {npz_path} has {gaze.shape[0]} != {frame_length}"
        )
    out_x[out_pos : out_pos + frame_length, :6] = gaze[:, :6]
    out_x[out_pos : out_pos + frame_length, 6:] = stimulus[:, :4]
    out_y[out_pos : out_pos + frame_length] = gaze[:, 6:8].astype(np.int8)


def process_shard(args: tuple[str, str, str]) -> dict[str, Any]:
    dataset, shard_id, _ = args
    source_base, destination_base = dataset_bases(dataset)
    rows = read_shard_rows(source_base, shard_id)
    total_frames = sum(int(row["frame_length"]) for row in rows)

    destination = destination_base / "shards" / shard_id
    destination.mkdir(parents=True, exist_ok=True)
    out_x = np.empty((total_frames, X_COLUMNS), dtype=np.float32)
    out_y = np.zeros((total_frames, Y_COLUMNS), dtype=np.int8)
    offsets = np.empty(len(rows), dtype=np.int64)
    lengths = np.empty(len(rows), dtype=np.int32)

    per_trial_hits = 0
    missing: list[str] = []
    pos = 0
    arrays_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for local_index, row in enumerate(rows):
        key = stem_key(row)
        frame_length = int(row["frame_length"])
        offsets[local_index] = pos
        lengths[local_index] = frame_length
        if key in _VIEW_MAP:
            view, split, packed_index = _VIEW_MAP[key]
            arrays = arrays_cache.get((view, split))
            if arrays is None:
                arrays = open_view_arrays(view, split)
                arrays_cache[(view, split)] = arrays
            copy_view_payload(arrays, packed_index, frame_length, out_x, out_y, pos)
        elif key in _PER_TRIAL_MAP:
            copy_per_trial_payload(_PER_TRIAL_MAP[key], frame_length, out_x, out_y, pos)
            per_trial_hits += 1
        else:
            missing.append(key)
        pos += frame_length
    if missing:
        raise RuntimeError(
            f"{dataset}/{shard_id}: {len(missing)} trials have no V5 payload "
            f"(first: {missing[:3]})"
        )
    if pos != total_frames:
        raise RuntimeError(f"{dataset}/{shard_id}: frame accounting mismatch")

    # v5 导出把 MISSING/BLINK 帧的坐标写成 NaN，而 v4 packed 格式要求无 NaN
    #（训练时通过 QC label 屏蔽无效帧，坐标值本身不参与计算）。统一置 0。
    nan_count = int(np.isnan(out_x).sum())
    out_x = np.nan_to_num(out_x, nan=0.0)

    np.save(destination / "X_data.npy", out_x)
    np.save(destination / "y_frame.npy", out_y)
    np.save(destination / "X_offsets.npy", offsets)
    np.save(destination / "X_lengths.npy", lengths)
    shutil.copy2(source_base / "shards" / shard_id / "trial_index.csv", destination / "trial_index.csv")

    bytes_written = out_x.nbytes + out_y.nbytes
    return {
        "dataset": dataset,
        "shard_id": shard_id,
        "trials": len(rows),
        "total_frames": total_frames,
        "per_trial_hits": per_trial_hits,
        "nan_replaced": nan_count,
        "bytes_written": bytes_written,
    }


def copy_index_and_metadata(source_base: Path, destination_base: Path, dataset: str) -> None:
    destination_base.mkdir(parents=True, exist_ok=True)
    for name in ("columns.json", "label_maps.json", "audit_summary.json", "split_summary.json"):
        source = source_base / name
        if source.is_file():
            shutil.copy2(source, destination_base / name)
    for name in ("trials.csv", "subjects.csv"):
        shutil.copy2(source_base / name, destination_base / name)
    if dataset == "pretrain":
        sub = source_base / "pretrain"
        dest_sub = destination_base / "pretrain"
        dest_sub.mkdir(parents=True, exist_ok=True)
        for name in (
            "pretrain_train.csv",
            "pretrain_validation.csv",
            "pretrain_test.csv",
            "pretrain_all_unique.csv",
            "pretrain_split_summary.json",
        ):
            shutil.copy2(sub / name, dest_sub / name)
    else:
        for name in ("train.csv", "validation.csv", "test.csv"):
            shutil.copy2(source_base / name, destination_base / name)

    manifest = json.loads((source_base / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_version": "eyemae_fast_dataset_v5",
            "format": "packed_mmap",
            "derived_from_dataset": str(V4_ROOT),
            "index_derived_from_dataset": str(V4_ROOT),
            "signal_derived_from_dataset": str(V5_ROOT),
            "signal_filter_policy": (
                "guarded_75hz_butterworth_lowpass_zero_phase_on_stable_valid_segments;"
                " blink_missing_and_saccade_protected; qc_labels_frozen_before_filter"
            ),
            "frame_label_policy": (
                "no_filter_qc_labels_from_speed4_hard_blink_pipeline;"
                " unchanged_from_v5_views"
            ),
            "payload_source": str(V5_ROOT),
        }
    )
    (destination_base / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_dataset(dataset: str, workers: int) -> dict[str, Any]:
    source_base, destination_base = dataset_bases(dataset)
    if not source_base.is_dir():
        raise FileNotFoundError(source_base)
    shard_ids = sorted(p.name for p in (source_base / "shards").iterdir() if p.is_dir())
    print(f"[{dataset}] shards={len(shard_ids)} workers={workers}", flush=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_shard, (dataset, shard_id, "")): shard_id
            for shard_id in shard_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            gib = sum(item["bytes_written"] for item in results) / 2**30
            print(
                f"[{dataset}] {completed}/{len(shard_ids)} {result['shard_id']}; "
                f"{gib:.1f} GiB written, {time.time() - started:.0f}s",
                flush=True,
            )
    copy_index_and_metadata(source_base, destination_base, dataset)
    return {
        "dataset": dataset,
        "num_shards": len(results),
        "num_trials": sum(item["trials"] for item in results),
        "num_frames": sum(item["total_frames"] for item in results),
        "per_trial_payloads": sum(item["per_trial_hits"] for item in results),
        "elapsed_seconds": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack V5 (75Hz filtered) data into V4-style packed_mmap layout."
    )
    parser.add_argument("--only-dataset", choices=("all", "pretrain", *TASKS), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global _VIEW_MAP, _PER_TRIAL_MAP
    _VIEW_MAP, _PER_TRIAL_MAP = load_source_map()

    if args.dry_run:
        for dataset in ("pretrain", *TASKS):
            source_base, _ = dataset_bases(dataset)
            shard_ids = sorted(p.name for p in (source_base / "shards").iterdir() if p.is_dir())
            total = 0
            for shard_id in shard_ids:
                total += len(read_shard_rows(source_base, shard_id))
            print(f"[dry-run] {dataset}: {len(shard_ids)} shards, {total} trials")
        return 0

    unpack_all_view_npy()

    datasets = ("pretrain", *TASKS) if args.only_dataset == "all" else (args.only_dataset,)
    summaries = []
    for dataset in datasets:
        summaries.append(build_dataset(dataset, args.workers))

    summary = {
        "dataset_version": "eyemae_fast_dataset_v5",
        "source_v4_root": str(V4_ROOT),
        "payload_v5_root": str(V5_ROOT),
        "output_root": str(OUT_ROOT),
        "datasets": summaries,
    }
    (OUT_ROOT / "v5_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
