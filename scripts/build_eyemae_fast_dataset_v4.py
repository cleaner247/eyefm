from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v3")
DEFAULT_V4_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v4")
DEFAULT_PROVENANCE_ZIP = REPO_ROOT / "eye_qc_cd_rust_speed4_hard_blink_20260616.zip"
DEFAULT_SUMMARY_NAME = "v4_all_highfreq_area_filter_summary.json"

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
EYE_SPECS = (
    ("left", (0, 1, 2), 0),
    ("right", (3, 4, 5), 1),
)
CHANNEL_NAMES = (
    "left_x",
    "left_y",
    "left_area",
    "right_x",
    "right_y",
    "right_area",
)


@dataclass(frozen=True)
class ShardJob:
    dataset_name: str
    shard_id: str
    source_x: str
    source_y: str
    destination_x: str
    marker_path: str
    trials: tuple[tuple[int, int, str], ...]
    window_frames: int


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-v4-write")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def base_roots(root: Path) -> list[tuple[str, Path]]:
    items = [("pretrain", root / "pretrain")]
    items.extend((task, root / "finetune" / task) for task in TASKS)
    return items


def stage_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[v4-copy] stage metadata/labels {source} -> {destination}", flush=True)

    def stage_file(source_file: str, destination_file: str) -> str:
        source_path = Path(source_file)
        destination_path = Path(destination_file)
        if source_path.name == "X_data.npy":
            # The worker atomically replaces this staging link with an
            # independent filtered file.  Avoid copying 97 GiB only to
            # overwrite it immediately.
            os.symlink(source_path, destination_path)
            return str(destination_path)
        try:
            os.link(source_path, destination_path)
        except PermissionError:
            # Files imported from the original shared dataset may be owned by
            # another account; Linux protected_hardlinks correctly rejects
            # linking them.  Labels and indexes total only about 5 GiB.
            shutil.copy2(source_path, destination_path)
        return str(destination_path)

    shutil.copytree(source, destination, copy_function=stage_file, symlinks=True)


def retarget_symlinks(source_root: Path, destination_root: Path) -> dict[str, str]:
    changed: dict[str, str] = {}
    source_text = str(source_root)
    destination_text = str(destination_root)
    for path in sorted(destination_root.rglob("*")):
        if not path.is_symlink():
            continue
        if path.name == "X_data.npy":
            # This is a temporary source-data staging link and must continue
            # pointing at v3 until the worker replaces the directory entry.
            continue
        target = os.readlink(path)
        new_target = target.replace(source_text, destination_text)
        if new_target == target:
            # Some v3 resplit directories were created against an older
            # installation root (for example /mnt/disk_sde/data-260606/...).
            # The dataset-version directory is the stable anchor; preserve
            # everything below it and make the v4 tree self-contained.
            target_parts = Path(target).parts
            try:
                version_index = target_parts.index(source_root.name)
            except ValueError:
                pass
            else:
                new_target = str(
                    destination_root.joinpath(*target_parts[version_index + 1 :])
                )
        if new_target == target:
            continue
        path.unlink()
        os.symlink(new_target, path, target_is_directory=True)
        changed[str(path)] = new_target
    return changed


def read_trials_by_shard(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["shard_id"]].append(
                (
                    int(row["frame_offset"]),
                    int(row["frame_length"]),
                    str(row["global_trial_id"]),
                )
            )
    for trials in grouped.values():
        trials.sort(key=lambda item: item[0])
    return grouped


def centered_masked_mean(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """Match the package's centered NaN-aware moving average in O(T*C)."""
    array = np.asarray(values)
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if array.ndim != 2 or array.shape[0] != mask.size:
        raise ValueError("values must be [T,C] and valid_mask must be [T]")
    if window_frames <= 0 or window_frames % 2 == 0:
        raise ValueError("window_frames must be a positive odd integer")
    length = array.shape[0]
    if length == 0:
        return array.astype(np.float64, copy=True)

    half = window_frames // 2
    indices = np.arange(length, dtype=np.int64)
    lower = np.maximum(0, indices - half)
    upper = np.minimum(length, indices + half + 1)

    count_prefix = np.empty(length + 1, dtype=np.int64)
    count_prefix[0] = 0
    np.cumsum(mask, dtype=np.int64, out=count_prefix[1:])
    counts = count_prefix[upper] - count_prefix[lower]

    work = np.where(mask[:, None], array, 0.0).astype(np.float64, copy=False)
    value_prefix = np.empty((length + 1, array.shape[1]), dtype=np.float64)
    value_prefix[0, :] = 0.0
    np.cumsum(work, axis=0, dtype=np.float64, out=value_prefix[1:, :])
    sums = value_prefix[upper, :] - value_prefix[lower, :]

    output = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts[:, None], out=output, where=counts[:, None] > 0)
    return output


def update_channel_stats(
    stats: dict[str, dict[str, float | int]],
    channel_name: str,
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    channel = stats[channel_name]
    valid_before = np.asarray(before)[valid_mask].astype(np.float64, copy=False)
    valid_after = np.asarray(after)[valid_mask].astype(np.float64, copy=False)
    delta = valid_after - valid_before
    channel["valid_values"] += int(delta.size)
    channel["changed_values_gt_1e-6"] += int(np.sum(np.abs(delta) > 1e-6))
    channel["delta_abs_sum"] += float(np.sum(np.abs(delta), dtype=np.float64))
    channel["delta_square_sum"] += float(np.sum(delta * delta, dtype=np.float64))
    if delta.size:
        channel["delta_abs_max"] = max(
            float(channel["delta_abs_max"]), float(np.max(np.abs(delta)))
        )

    if before.size > 1:
        adjacent = valid_mask[:-1] & valid_mask[1:]
        if np.any(adjacent):
            before_diff = np.diff(np.asarray(before, dtype=np.float64))[adjacent]
            after_diff = np.diff(np.asarray(after, dtype=np.float64))[adjacent]
            channel["adjacent_valid_differences"] += int(before_diff.size)
            channel["first_diff_square_before_sum"] += float(
                np.sum(before_diff * before_diff, dtype=np.float64)
            )
            channel["first_diff_square_after_sum"] += float(
                np.sum(after_diff * after_diff, dtype=np.float64)
            )


def empty_channel_stats() -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "valid_values": 0,
            "changed_values_gt_1e-6": 0,
            "delta_abs_sum": 0.0,
            "delta_square_sum": 0.0,
            "delta_abs_max": 0.0,
            "adjacent_valid_differences": 0,
            "first_diff_square_before_sum": 0.0,
            "first_diff_square_after_sum": 0.0,
        }
        for name in CHANNEL_NAMES
    }


def filter_trial_chunk(
    x_chunk: np.ndarray,
    labels: np.ndarray,
    window_frames: int,
    stats: dict[str, dict[str, float | int]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter all VALID x/y/area samples, preserving stimulus and QC labels."""
    output = np.asarray(x_chunk).copy()
    label_array = np.asarray(labels)
    if output.ndim != 2 or output.shape[1] != 10:
        raise ValueError(f"expected X_data trial [T,10], got {output.shape}")
    if label_array.shape != (output.shape[0], 2):
        raise ValueError(
            f"expected y_frame trial {(output.shape[0], 2)}, got {label_array.shape}"
        )

    counters: Counter[str] = Counter()
    for eye_name, columns, label_column in EYE_SPECS:
        raw = np.asarray(output[:, columns])
        valid = (
            (label_array[:, label_column] == 0)
            & np.isfinite(raw).all(axis=1)
            & (raw[:, 2] > 0)
        )
        counters[f"{eye_name}_valid_frames"] = int(valid.sum())
        counters[f"{eye_name}_protected_frames"] = int(valid.size - valid.sum())
        if not np.any(valid):
            continue
        smoothed = centered_masked_mean(raw, valid, window_frames)
        for local_column, global_column in enumerate(columns):
            before = raw[:, local_column].copy()
            after = before.copy()
            finite_smooth = valid & np.isfinite(smoothed[:, local_column])
            after[finite_smooth] = smoothed[finite_smooth, local_column]
            output[:, global_column] = after.astype(output.dtype, copy=False)
            if stats is not None:
                update_channel_stats(
                    stats,
                    CHANNEL_NAMES[global_column],
                    before,
                    output[:, global_column],
                    valid,
                )
    return output, dict(counters)


def validate_trial_coverage(
    trials: Iterable[tuple[int, int, str]],
    num_frames: int,
) -> None:
    expected_offset = 0
    for offset, length, trial_id in trials:
        if length <= 0:
            raise ValueError(f"{trial_id}: non-positive frame length {length}")
        if offset != expected_offset:
            raise ValueError(
                f"{trial_id}: non-contiguous frame offset {offset}; "
                f"expected {expected_offset}"
            )
        expected_offset = offset + length
    if expected_offset != num_frames:
        raise ValueError(
            f"trial coverage ends at {expected_offset}, X_data contains {num_frames} frames"
        )


def completed_marker_is_valid(job: ShardJob) -> bool:
    marker = Path(job.marker_path)
    destination = Path(job.destination_x)
    source = Path(job.source_x)
    if not marker.is_file() or not destination.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not bool(payload.get("completed", False)):
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size == destination_stat.st_size
        and source_stat.st_ino != destination_stat.st_ino
        and int(payload.get("destination_bytes", -1)) == destination_stat.st_size
    )


def process_shard(job: ShardJob) -> dict[str, Any]:
    if completed_marker_is_valid(job):
        payload = json.loads(Path(job.marker_path).read_text(encoding="utf-8"))
        payload["resumed_from_marker"] = True
        return payload

    source_x_path = Path(job.source_x)
    source_y_path = Path(job.source_y)
    destination_x_path = Path(job.destination_x)
    marker_path = Path(job.marker_path)
    temporary_path = destination_x_path.with_name("X_data.v4-filtering.tmp.npy")
    if temporary_path.exists():
        temporary_path.unlink()

    source_stat_before = source_x_path.stat()
    source_x = np.load(source_x_path, mmap_mode="r")
    source_y = np.load(source_y_path, mmap_mode="r")
    if source_x.ndim != 2 or source_x.shape[1] != 10:
        raise ValueError(f"{source_x_path}: expected [N,10], got {source_x.shape}")
    if source_y.shape != (source_x.shape[0], 2):
        raise ValueError(
            f"{source_y_path}: expected {(source_x.shape[0], 2)}, got {source_y.shape}"
        )
    validate_trial_coverage(job.trials, int(source_x.shape[0]))

    destination = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=source_x.dtype,
        shape=source_x.shape,
    )
    channel_stats = empty_channel_stats()
    counters: Counter[str] = Counter()
    started = time.time()
    try:
        for trial_index, (offset, length, trial_id) in enumerate(job.trials):
            end = offset + length
            raw_chunk = np.asarray(source_x[offset:end])
            labels = np.asarray(source_y[offset:end])
            filtered, trial_counts = filter_trial_chunk(
                raw_chunk,
                labels,
                job.window_frames,
                channel_stats,
            )
            if not np.array_equal(filtered[:, 6:10], raw_chunk[:, 6:10]):
                raise RuntimeError(f"{trial_id}: stimulus columns changed")
            if not np.isfinite(filtered[:, :6]).all():
                counters["trials_with_nonfinite_eye_output"] += 1
            if np.any(filtered[:, (2, 5)] < 0):
                counters["trials_with_negative_area_output"] += 1
            destination[offset:end] = filtered
            counters.update(trial_counts)
            counters["trials"] += 1
            counters["frames"] += length
            if trial_index == 0 or (trial_index + 1) % 10000 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"[v4-filter] {job.dataset_name}/{job.shard_id}: "
                    f"{trial_index + 1}/{len(job.trials)} trials "
                    f"({(trial_index + 1) / elapsed:.1f} trials/s)",
                    file=sys.stderr,
                    flush=True,
                )
        destination.flush()
    finally:
        del destination
        del source_x
        del source_y

    if counters["trials_with_nonfinite_eye_output"]:
        raise RuntimeError(
            f"{job.dataset_name}/{job.shard_id}: non-finite eye outputs in "
            f"{counters['trials_with_nonfinite_eye_output']} trials"
        )
    if counters["trials_with_negative_area_output"]:
        raise RuntimeError(
            f"{job.dataset_name}/{job.shard_id}: negative area outputs in "
            f"{counters['trials_with_negative_area_output']} trials"
        )

    os.replace(temporary_path, destination_x_path)
    source_stat_after = source_x_path.stat()
    destination_stat = destination_x_path.stat()
    if (
        source_stat_after.st_ino != source_stat_before.st_ino
        or source_stat_after.st_size != source_stat_before.st_size
        or source_stat_after.st_mtime_ns != source_stat_before.st_mtime_ns
    ):
        raise RuntimeError(f"source X_data changed while filtering: {source_x_path}")
    if destination_stat.st_ino == source_stat_after.st_ino:
        raise RuntimeError(f"destination still hardlinked to source: {destination_x_path}")

    result = {
        "completed": True,
        "resumed_from_marker": False,
        "dataset_name": job.dataset_name,
        "shard_id": job.shard_id,
        "source_x": str(source_x_path),
        "source_y": str(source_y_path),
        "destination_x": str(destination_x_path),
        "destination_bytes": destination_stat.st_size,
        "window_frames": job.window_frames,
        "elapsed_seconds": time.time() - started,
        "counters": dict(counters),
        "channels": channel_stats,
    }
    atomic_write_json(marker_path, result)
    return result


def create_jobs(
    source_root: Path,
    destination_root: Path,
    window_frames: int,
) -> list[ShardJob]:
    jobs: list[ShardJob] = []
    for dataset_name, source_base in base_roots(source_root):
        destination_base = (
            destination_root / "pretrain"
            if dataset_name == "pretrain"
            else destination_root / "finetune" / dataset_name
        )
        trials_by_shard = read_trials_by_shard(source_base / "trials.csv")
        for shard_id, trials in sorted(trials_by_shard.items()):
            source_shard = source_base / "shards" / shard_id
            destination_shard = destination_base / "shards" / shard_id
            jobs.append(
                ShardJob(
                    dataset_name=dataset_name,
                    shard_id=shard_id,
                    source_x=str(source_shard / "X_data.npy"),
                    source_y=str(source_shard / "y_frame.npy"),
                    destination_x=str(destination_shard / "X_data.npy"),
                    marker_path=str(destination_shard / "v4_filter_summary.json"),
                    trials=tuple(trials),
                    window_frames=window_frames,
                )
            )
    return jobs


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    channels = empty_channel_stats()
    total_bytes = 0
    elapsed_sum = 0.0
    resumed = 0
    for result in results:
        counters.update(result["counters"])
        total_bytes += int(result["destination_bytes"])
        elapsed_sum += float(result.get("elapsed_seconds", 0.0))
        resumed += int(bool(result.get("resumed_from_marker", False)))
        for name, values in result["channels"].items():
            for key, value in values.items():
                if key == "delta_abs_max":
                    channels[name][key] = max(
                        float(channels[name][key]), float(value)
                    )
                else:
                    channels[name][key] += value

    for values in channels.values():
        valid = int(values["valid_values"])
        differences = int(values["adjacent_valid_differences"])
        values["mean_abs_delta"] = (
            float(values["delta_abs_sum"]) / valid if valid else math.nan
        )
        values["rmse_delta"] = (
            math.sqrt(float(values["delta_square_sum"]) / valid)
            if valid
            else math.nan
        )
        values["first_diff_rms_before"] = (
            math.sqrt(float(values["first_diff_square_before_sum"]) / differences)
            if differences
            else math.nan
        )
        values["first_diff_rms_after"] = (
            math.sqrt(float(values["first_diff_square_after_sum"]) / differences)
            if differences
            else math.nan
        )
        before_rms = float(values["first_diff_rms_before"])
        after_rms = float(values["first_diff_rms_after"])
        values["first_diff_rms_reduction_fraction"] = (
            1.0 - after_rms / before_rms
            if math.isfinite(before_rms) and before_rms > 0
            else math.nan
        )
    return {
        "num_shards": len(results),
        "num_resumed_shards": resumed,
        "destination_x_bytes": total_bytes,
        "worker_elapsed_seconds_sum": elapsed_sum,
        "counters": dict(counters),
        "channels": channels,
    }


def update_metadata(
    source_root: Path,
    destination_root: Path,
    provenance_zip: Path,
    provenance_sha256: str,
    window_frames: int,
    summary_path: Path,
) -> None:
    policy = (
        "all_valid_eye_x_y_area_centered_nan_aware_moving_average_"
        f"window_{window_frames}_frames; blink_missing_and_invalid_samples_preserved"
    )
    for manifest in sorted(destination_root.glob("pretrain/dataset_manifest.json")) + sorted(
        destination_root.glob("finetune/*/dataset_manifest.json")
    ):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update(
            {
                "dataset_version": "eyemae_fast_dataset_v4",
                "derived_from_dataset": str(source_root),
                "signal_filter_policy": policy,
                "signal_filter_provenance_zip": str(provenance_zip),
                "signal_filter_provenance_sha256": provenance_sha256,
                "frame_label_policy": (
                    "unchanged_from_v3_source_npz_gaze_columns_6_7; "
                    "filter_applied_only_to_X_data_valid_eye_signals"
                ),
            }
        )
        atomic_write_json(manifest, payload)

    readme = f"""# EyeMAE Fast Dataset v4

This dataset is an independent signal-filtered derivative of:

`{source_root}`

The selected trials, subjects, splits, stimulus columns, QC labels, and
left/right keep metadata are unchanged from v3.  For every trial and eye, all
VALID samples of x, y, and pupil area are filtered with the
{window_frames}-frame centered NaN-aware moving average inherited from the
supplied high-frequency QC package.
Unlike the supplied conditional implementation, filtering is always applied;
pupil area is filtered as well.  BLINK, MISSING, non-finite, and non-positive
area samples are protected and retain their v3 values.

The raw pupil/glint a/b/c/d channels required to rerun QC labels are not present
in the packed v3 dataset, so v4 intentionally preserves v3 `y_frame` labels
instead of fabricating a second QC pass from incomplete inputs.

Provenance package: `{provenance_zip}`

SHA256: `{provenance_sha256}`

Build summary: `{summary_path}`
"""
    atomic_write_text(destination_root / "README.md", readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build EyeMAE packed dataset v4 by unconditionally filtering all "
            "VALID x/y/area channels from v3."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--provenance-zip", type=Path, default=DEFAULT_PROVENANCE_ZIP)
    parser.add_argument("--window-frames", type=int, default=41)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-dataset", choices=("all", "pretrain", *TASKS), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    destination_root = args.output_root.resolve()
    provenance_zip = args.provenance_zip.resolve()
    if source_root == destination_root:
        raise ValueError("source-root and output-root must differ")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not provenance_zip.is_file():
        raise FileNotFoundError(provenance_zip)
    if args.window_frames <= 0 or args.window_frames % 2 == 0:
        raise ValueError("--window-frames must be a positive odd integer")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    provenance_sha256 = sha256_file(provenance_zip)
    print(f"[v4-provenance] {provenance_zip} sha256={provenance_sha256}", flush=True)
    if args.dry_run:
        jobs = create_jobs(source_root, source_root, args.window_frames)
        print(
            json.dumps(
                {
                    "source_root": str(source_root),
                    "would_output_root": str(destination_root),
                    "num_shards": len(jobs),
                    "num_trials": sum(len(job.trials) for job in jobs),
                    "window_frames": args.window_frames,
                },
                indent=2,
            )
        )
        return 0

    stage_copy_tree(source_root, destination_root)
    changed_symlinks = retarget_symlinks(source_root, destination_root)
    jobs = create_jobs(source_root, destination_root, args.window_frames)
    if args.only_dataset != "all":
        jobs = [job for job in jobs if job.dataset_name == args.only_dataset]
    if not jobs:
        raise ValueError(f"no shard jobs selected for {args.only_dataset}")

    print(
        f"[v4-build] jobs={len(jobs)} workers={args.workers} "
        f"window={args.window_frames}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_jobs = {executor.submit(process_shard, job): job for job in jobs}
        for completed, future in enumerate(as_completed(future_jobs), start=1):
            job = future_jobs[future]
            result = future.result()
            results.append(result)
            elapsed = max(time.time() - started, 1e-9)
            processed_gib = sum(int(item["destination_bytes"]) for item in results) / 2**30
            print(
                f"[v4-build] completed {completed}/{len(jobs)} "
                f"{job.dataset_name}/{job.shard_id}; {processed_gib:.2f} GiB, "
                f"{processed_gib / elapsed * 3600:.2f} GiB/h",
                flush=True,
            )

    aggregate = aggregate_results(results)
    summary_path = destination_root / DEFAULT_SUMMARY_NAME
    summary = {
        "completed": args.only_dataset == "all",
        "dataset_version": "eyemae_fast_dataset_v4",
        "source_root": str(source_root),
        "output_root": str(destination_root),
        "provenance_zip": str(provenance_zip),
        "provenance_sha256": provenance_sha256,
        "window_frames": args.window_frames,
        "filter_scope": "all VALID left/right x, y, and area samples",
        "protected_scope": "BLINK, MISSING, invalid, and non-positive-area samples",
        "labels_and_splits": "unchanged from v3",
        "selected_dataset": args.only_dataset,
        "elapsed_seconds": time.time() - started,
        "retargeted_symlinks": changed_symlinks,
        "aggregate": aggregate,
        "shards": sorted(results, key=lambda item: (item["dataset_name"], item["shard_id"])),
    }
    atomic_write_json(summary_path, summary)
    if args.only_dataset == "all":
        update_metadata(
            source_root,
            destination_root,
            provenance_zip,
            provenance_sha256,
            args.window_frames,
            summary_path,
        )
    print(f"[v4-build] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
