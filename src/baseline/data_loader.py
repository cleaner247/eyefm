"""EyeFM v2 packed-mmap data loader for DL baselines.

Loads per-trial windows from `shards/shard_xxxxxx/{X_data,y_frame}.npy`
into PyTorch Dataset for subject-level batched training of the 4 DL
baselines (TCN / NST / CNNTransformer / TimesNet).

Reference:  docs/eyemae_baseline.md
"""
from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


# ==== v2 layout constants ====
COL_LEFT_X, COL_LEFT_Y, COL_LEFT_AREA = 0, 1, 2
COL_RIGHT_X, COL_RIGHT_Y, COL_RIGHT_AREA = 3, 4, 5
COL_STIM_X, COL_STIM_Y, COL_STIM_ON, COL_FIX_ON = 6, 7, 8, 9
N_X_COLS = 10

COL_LEFT_QC, COL_RIGHT_QC = 0, 1
N_Y_COLS = 2


@dataclass(frozen=True)
class SplitData:
    """Per-split view of the v2 dataset, packed in memory for fast access."""

    rows: list[dict]                  # raw csv rows (trials)
    labels: list[int]                 # per-row integer label
    subj_ids: list[str]               # per-row ml_subject_id
    tasks: list[str]                  # per-row task name (ProSaccade / AntiSaccade / ...)
    frame_offsets: list[int]
    frame_lengths: list[int]
    shard_ids: list[str]


def _shard_dir_name(shard_id: str) -> str:
    """Normalize csv shard_id to directory name (handle 'shard_' prefix)."""
    return shard_id if shard_id.startswith("shard_") else f"shard_{shard_id}"


def _filter_task_rows(rows: Sequence[dict], saccade_tasks: Sequence[str]) -> list[dict]:
    """Keep only trials whose 'task' field matches one of saccade_tasks."""
    if not saccade_tasks:
        return list(rows)
    keep = set(saccade_tasks)
    return [r for r in rows if r.get("task") in keep]


def get_label_map(task: str) -> dict[str, int]:
    """String label → int map per task."""
    if task == "pd_related_5class":
        return {"-1": 0, "0": 1, "1": 2, "2": 3, "3": 4}
    if task in ("detox_binary", "pd_binary"):
        return {"0": 0, "1": 1}
    raise ValueError(f"Unknown task: {task}")


def get_n_classes(task: str) -> int:
    if task == "pd_related_5class":
        return 5
    return 2


def load_split(task: str, split: str, data_root: Path) -> list[dict]:
    """Load v2 split csv (rows = trials)."""
    csv_path = data_root / "finetune" / task / f"{split}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_split_data(
    task: str,
    data_root: Path,
    saccade_tasks: Sequence[str] = ("ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade"),
    splits: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, SplitData]:
    """Build per-split SplitData dict for a downstream task."""
    label_map = get_label_map(task)
    # `pd_related_5class` carries the 5-class label in `pd_disease_label`
    # (-1 healthy, 0..3 disease grade); `detox_binary` / `pd_binary` use
    # `health_label` (0 healthy, 1 disease). Reading the wrong field would
    # silently collapse the label distribution to 2 classes.
    label_field = "pd_disease_label" if task == "pd_related_5class" else "health_label"
    out: dict[str, SplitData] = {}
    for split in splits:
        rows = load_split(task, split, data_root)
        rows = _filter_task_rows(rows, saccade_tasks)
        out[split] = SplitData(
            rows=rows,
            labels=[int(label_map.get(r.get(label_field, r.get("health_label", "0")), 0)) for r in rows],
            subj_ids=[r.get("ml_subject_id", "") for r in rows],
            tasks=[r.get("task", "") for r in rows],
            frame_offsets=[int(r["frame_offset"]) for r in rows],
            frame_lengths=[int(r["frame_length"]) for r in rows],
            shard_ids=[r["shard_id"] for r in rows],
        )
        LOGGER.info("split=%s task=%s rows=%d subj=%d", split, task, len(rows), len(set(out[split].subj_ids)))
    return out


def _open_shard(shard_dir: Path) -> tuple[np.memmap, np.memmap]:
    return (
        np.load(shard_dir / "X_data.npy", mmap_mode="r"),
        np.load(shard_dir / "y_frame.npy", mmap_mode="r"),
    )


class ShardCache:
    """Lazy mmap handle cache for shard files (avoids re-opening for every trial)."""

    def __init__(self, data_root: Path, task: str) -> None:
        self.data_root = data_root
        self.task = task
        self._cache: dict[str, tuple[np.memmap, np.memmap]] = {}

    def get(self, shard_id: str) -> tuple[np.memmap, np.memmap]:
        cached = self._cache.get(shard_id)
        if cached is not None:
            return cached
        shard_dir = self.data_root / "finetune" / self.task / "shards" / _shard_dir_name(shard_id)
        cached = _open_shard(shard_dir)
        self._cache[shard_id] = cached
        return cached


class TrialDataset(Dataset):
    """Per-trial PyTorch Dataset returning (X[T,10], label[int]) windows.

    X is right-padded / truncated to a fixed T_LEN to allow batching.
    """

    def __init__(
        self,
        split: SplitData,
        shard_cache: ShardCache,
        t_len: int = 1024,
    ) -> None:
        self.split = split
        self.shard_cache = shard_cache
        self.t_len = t_len
        # pre-group trial indices by shard to avoid repeated mmap opens
        self._by_shard: dict[str, list[int]] = defaultdict(list)
        for i, shard_id in enumerate(split.shard_ids):
            self._by_shard[shard_id].append(i)

    def __len__(self) -> int:
        return len(self.split.rows)

    def _load_window(self, idx: int) -> np.ndarray:
        shard_id = self.split.shard_ids[idx]
        offset = self.split.frame_offsets[idx]
        length = self.split.frame_lengths[idx]
        X_mmap, _ = self.shard_cache.get(shard_id)
        n_avail = min(length, len(X_mmap) - offset)
        if n_avail <= 0:
            return np.zeros((self.t_len, N_X_COLS), dtype=np.float32)
        window = np.array(X_mmap[offset:offset + n_avail], dtype=np.float32, copy=True)
        if window.shape[0] >= self.t_len:
            return window[: self.t_len]
        pad = np.zeros((self.t_len - window.shape[0], N_X_COLS), dtype=np.float32)
        return np.concatenate([window, pad], axis=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self._load_window(idx)
        return torch.from_numpy(x), self.split.labels[idx]


def make_subject_batches(
    split: SplitData,
    seed: int = 42,
) -> list[list[int]]:
    """Group trials by subject → list of batched trial-index lists.

    Each subject's trials are processed in one forward pass (inter-subject batching).
    """
    rng = np.random.default_rng(seed)
    by_subj: dict[str, list[int]] = defaultdict(list)
    for i, subj in enumerate(split.subj_ids):
        by_subj[subj].append(i)
    batches: list[list[int]] = []
    for subj, idxs in by_subj.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        batches.append(idxs)
    rng.shuffle(batches)
    return batches


def collate_subjects(batch_idxs: list[int], dataset: TrialDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack one subject's trials into (K, T, 10) + (K,) label tensor."""
    xs, ys = [], []
    for i in batch_idxs:
        x, y = dataset[i]
        xs.append(torch.from_numpy(x))
        ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def class_weights_from_labels(labels: Sequence[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for balanced loss (normalized to mean=1)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    inv = inv / inv.mean()  # normalize so mean weight = 1
    return torch.tensor(inv, dtype=torch.float32)
