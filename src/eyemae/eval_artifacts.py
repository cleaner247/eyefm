from __future__ import annotations

import random
from typing import Any

import torch

from .data import ID_TO_TASK
from .metrics import finalize_metric_sums, new_metric_sums, reduce_metric_sums, update_metric_sums


MASK_TYPE_NAMES = {1: "random", 2: "short_span", 3: "long_span"}

GROUP_BUCKETS = {
    "task_id": tuple(ID_TO_TASK.values()),
    "eye": ("left", "right"),
    "trial_length": ("lt_64", "64_127", "128_255", "ge_256"),
    "missing_fraction": ("zero", "le_0.25", "0.25_0.50", "gt_0.50"),
    "subject_eye_availability": ("both_eye_subject", "left_only_subject", "right_only_subject"),
    "eye_token_valid_fraction": ("all", "0.75_1.00", "0.25_0.75", "lt_0.25"),
    "mask_type": tuple(MASK_TYPE_NAMES.values()),
}


def new_group_metric_sums(device: torch.device) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    return {
        group: {bucket: new_metric_sums(device) for bucket in buckets}
        for group, buckets in GROUP_BUCKETS.items()
    }


def reduce_group_metric_sums(group_sums: dict[str, dict[str, dict[str, torch.Tensor]]]) -> None:
    for buckets in group_sums.values():
        for sums in buckets.values():
            reduce_metric_sums(sums)


def finalize_group_metrics(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    cfg: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        group: {bucket: finalize_metric_sums(sums, cfg=cfg) for bucket, sums in sorted(buckets.items())}
        for group, buckets in sorted(group_sums.items())
    }


def flatten_group_metrics(
    group_metrics: dict[str, dict[str, dict[str, float]]],
    *,
    prefix: str = "val_group",
) -> dict[str, float]:
    return {
        f"{prefix}/{group}/{bucket}/{metric}": value
        for group, buckets in group_metrics.items()
        for bucket, metrics in buckets.items()
        for metric, value in metrics.items()
    }


def update_group_metrics(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    batch: dict[str, Any],
    pred: torch.Tensor,
    mae_mask: torch.Tensor,
    mask_type: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    bsz, n_patches, eyes = batch["eye_token_valid"].shape

    def trial_tokens(trial_filter: torch.Tensor) -> torch.Tensor:
        return trial_filter[:, None, None].expand(bsz, n_patches, eyes)

    for task_id, task_name in ID_TO_TASK.items():
        _update_group(
            group_sums,
            "task_id",
            task_name,
            trial_tokens(batch["task_id"] == int(task_id)),
            pred,
            batch,
            mae_mask,
            cfg,
        )

    left_filter = torch.zeros_like(batch["eye_token_valid"], dtype=torch.bool)
    left_filter[:, :, 0] = True
    right_filter = torch.zeros_like(batch["eye_token_valid"], dtype=torch.bool)
    right_filter[:, :, 1] = True
    _update_group(group_sums, "eye", "left", left_filter, pred, batch, mae_mask, cfg)
    _update_group(group_sums, "eye", "right", right_filter, pred, batch, mae_mask, cfg)

    n_real = (~batch["pad_mask"]).sum(dim=1)
    length_buckets = {
        "lt_64": n_real < 64,
        "64_127": (n_real >= 64) & (n_real < 128),
        "128_255": (n_real >= 128) & (n_real < 256),
        "ge_256": n_real >= 256,
    }
    for name, filt in length_buckets.items():
        _update_group(group_sums, "trial_length", name, trial_tokens(filt), pred, batch, mae_mask, cfg)

    missing_frac = _trial_missing_fraction(batch)
    missing_buckets = {
        "zero": missing_frac == 0,
        "le_0.25": (missing_frac > 0) & (missing_frac <= 0.25),
        "0.25_0.50": (missing_frac > 0.25) & (missing_frac <= 0.50),
        "gt_0.50": missing_frac > 0.50,
    }
    for name, filt in missing_buckets.items():
        _update_group(group_sums, "missing_fraction", name, trial_tokens(filt), pred, batch, mae_mask, cfg)

    subject_suffix = torch.tensor([_subject_suffix_id(subject_id) for subject_id in batch["subject_id"]], device=pred.device)
    subject_buckets = {
        "both_eye_subject": subject_suffix == 0,
        "left_only_subject": subject_suffix == 1,
        "right_only_subject": subject_suffix == 2,
    }
    for name, filt in subject_buckets.items():
        _update_group(group_sums, "subject_eye_availability", name, trial_tokens(filt), pred, batch, mae_mask, cfg)

    valid_frac = batch["eye_token_valid"].float().sum(dim=(1, 2)) / ((~batch["pad_mask"]).sum(dim=1).float() * 2.0).clamp_min(1.0)
    valid_buckets = {
        "all": valid_frac >= 0.999,
        "0.75_1.00": (valid_frac >= 0.75) & (valid_frac < 0.999),
        "0.25_0.75": (valid_frac >= 0.25) & (valid_frac < 0.75),
        "lt_0.25": valid_frac < 0.25,
    }
    for name, filt in valid_buckets.items():
        _update_group(group_sums, "eye_token_valid_fraction", name, trial_tokens(filt), pred, batch, mae_mask, cfg)

    for mask_id, mask_name in MASK_TYPE_NAMES.items():
        _update_group(group_sums, "mask_type", mask_name, mask_type == mask_id, pred, batch, mae_mask, cfg)


def _update_group(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    group: str,
    bucket: str,
    token_filter: torch.Tensor,
    pred: torch.Tensor,
    batch: dict[str, Any],
    mae_mask: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    sums = group_sums[group][bucket]
    if not bool(token_filter.any()):
        return
    update_metric_sums(
        sums,
        pred,
        batch["content"],
        batch["quality"],
        mae_mask,
        batch["pad_mask"],
        batch["eye_token_valid"],
        None,
        cfg,
        token_filter=token_filter,
    )


def trial_missing_fraction(batch: dict[str, Any]) -> torch.Tensor:
    return _trial_missing_fraction(batch)


def subject_suffix_id(subject_id: str) -> int:
    return _subject_suffix_id(subject_id)


def _trial_missing_fraction(batch: dict[str, Any]) -> torch.Tensor:
    missing = batch["quality"][..., 0].bool()
    real = (~batch["pad_mask"])[:, :, None, None].expand_as(missing)
    denom = real.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    return (missing & real).sum(dim=(1, 2, 3)).float() / denom


def _subject_suffix_id(subject_id: str) -> int:
    if subject_id.endswith("D"):
        return 0
    if subject_id.endswith("L"):
        return 1
    if subject_id.endswith("R"):
        return 2
    return -1


class VizCollector:
    def __init__(self, *, max_trials: int, seed: int) -> None:
        self.max_trials = int(max_trials)
        self.rng = random.Random(seed)
        self.seen = 0
        self.random_pool: list[dict[str, Any]] = []
        self.task_pool: dict[int, list[dict[str, Any]]] = {task_id: [] for task_id in ID_TO_TASK}
        self.long_span_pool: list[dict[str, Any]] = []
        self.single_eye_pool: list[dict[str, Any]] = []
        self.all_missing_stim_pool: list[dict[str, Any]] = []

    def observe(
        self,
        batch: dict[str, Any],
        pred: torch.Tensor,
        mae_mask: torch.Tensor,
        mask_type: torch.Tensor,
    ) -> None:
        for b in range(len(batch["subject_id"])):
            self.seen += 1
            n = int((~batch["pad_mask"][b]).sum().item())
            if n <= 0:
                continue
            task_id = int(batch["task_id"][b].detach().cpu())
            subject_id = str(batch["subject_id"][b])
            has_long_span = bool((mask_type[b, :n] == 3).any().detach().cpu())
            has_single_eye = subject_id.endswith(("L", "R"))
            stim_on = batch["stim"][b, :n, :, 1].amax(dim=-1) > 0
            all_missing_eye = batch["eye_nonmissing_frac"][b, :n] <= 0
            has_all_missing_stim = bool((all_missing_eye & stim_on[:, None]).any().detach().cpu())

            store_random = len(self.random_pool) < 64 or self.rng.randrange(self.seen) < 64
            store_category = (
                len(self.task_pool.get(task_id, [])) < 32
                or (has_long_span and len(self.long_span_pool) < 32)
                or (has_single_eye and len(self.single_eye_pool) < 32)
                or (has_all_missing_stim and len(self.all_missing_stim_pool) < 32)
            )
            if not store_random and not store_category:
                continue
            example = _slice_viz_example(batch, pred, mae_mask, mask_type, b, n)
            if store_random:
                if len(self.random_pool) < 64:
                    self.random_pool.append(example)
                else:
                    self.random_pool[self.rng.randrange(64)] = example
            if len(self.task_pool.get(task_id, [])) < 32:
                self.task_pool.setdefault(task_id, []).append(example)
            if has_long_span and len(self.long_span_pool) < 32:
                self.long_span_pool.append(example)
            if has_single_eye and len(self.single_eye_pool) < 32:
                self.single_eye_pool.append(example)
            if has_all_missing_stim and len(self.all_missing_stim_pool) < 32:
                self.all_missing_stim_pool.append(example)

    def build_batch(self) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None:
        selected: list[dict[str, Any]] = []
        keys: set[tuple[str, str]] = set()

        def add(example: dict[str, Any]) -> bool:
            if len(selected) >= self.max_trials:
                return False
            key = (str(example["subject_id"]), str(example["trial_id"]))
            if key in keys:
                return False
            keys.add(key)
            selected.append(example)
            return True

        def count_where(flag: str) -> int:
            return sum(1 for example in selected if bool(example.get(flag, False)))

        for task_id in sorted(self.task_pool):
            for example in self.task_pool[task_id]:
                if sum(1 for item in selected if int(item["task_id"]) == task_id) >= 2:
                    break
                add(example)
        for pool, flag, target in (
            (self.long_span_pool, "has_long_span", 2),
            (self.single_eye_pool, "has_single_eye", 2),
            (self.all_missing_stim_pool, "has_all_missing_stim", 2),
        ):
            for example in pool:
                if count_where(flag) >= target:
                    break
                add(example)
        random_order = list(self.random_pool)
        self.rng.shuffle(random_order)
        for example in random_order:
            if len(selected) >= self.max_trials:
                break
            add(example)
        if not selected:
            return None
        return _collate_viz_examples(selected)


def _slice_viz_example(
    batch: dict[str, Any],
    pred: torch.Tensor,
    mae_mask: torch.Tensor,
    mask_type: torch.Tensor,
    b: int,
    n: int,
) -> dict[str, Any]:
    task_id = int(batch["task_id"][b].detach().cpu())
    subject_id = str(batch["subject_id"][b])
    stim_on = batch["stim"][b, :n, :, 1].amax(dim=-1) > 0
    all_missing_eye = batch["eye_nonmissing_frac"][b, :n] <= 0
    return {
        "content": batch["content"][b, :n].detach().cpu().clone(),
        "quality": batch["quality"][b, :n].detach().cpu().clone(),
        "stim": batch["stim"][b, :n].detach().cpu().clone(),
        "task_id": task_id,
        "eye_nonmissing_frac": batch["eye_nonmissing_frac"][b, :n].detach().cpu().clone(),
        "eye_token_valid": batch["eye_token_valid"][b, :n].detach().cpu().clone(),
        "subject_id": subject_id,
        "trial_id": str(batch["trial_id"][b]),
        "path": str(batch["path"][b]) if "path" in batch else "",
        "pred": pred[b, :n].detach().cpu().clone(),
        "mae_mask": mae_mask[b, :n].detach().cpu().clone(),
        "mask_type": mask_type[b, :n].detach().cpu().clone(),
        "has_long_span": bool((mask_type[b, :n] == 3).any().detach().cpu()),
        "has_single_eye": subject_id.endswith(("L", "R")),
        "has_all_missing_stim": bool((all_missing_eye & stim_on[:, None]).any().detach().cpu()),
    }


def _collate_viz_examples(examples: list[dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    batch_size = len(examples)
    nmax = max(int(example["content"].shape[0]) for example in examples)
    patch = int(examples[0]["content"].shape[2])
    content = torch.zeros(batch_size, nmax, 2, patch, 4, dtype=torch.float32)
    quality = torch.ones(batch_size, nmax, 2, patch, 1, dtype=torch.float32)
    stim = torch.zeros(batch_size, nmax, patch, 4, dtype=torch.float32)
    task_id = torch.zeros(batch_size, dtype=torch.long)
    pad_mask = torch.ones(batch_size, nmax, dtype=torch.bool)
    eye_nonmissing_frac = torch.zeros(batch_size, nmax, 2, dtype=torch.float32)
    eye_token_valid = torch.zeros(batch_size, nmax, 2, dtype=torch.bool)
    pred = torch.zeros(batch_size, nmax, 2, patch, 4, dtype=torch.float32)
    mae_mask = torch.zeros(batch_size, nmax, 2, dtype=torch.bool)
    subject_id: list[str] = []
    trial_id: list[str] = []
    paths: list[str] = []
    for b, example in enumerate(examples):
        n = int(example["content"].shape[0])
        content[b, :n] = example["content"]
        quality[b, :n] = example["quality"]
        stim[b, :n] = example["stim"]
        task_id[b] = int(example["task_id"])
        pad_mask[b, :n] = False
        eye_nonmissing_frac[b, :n] = example["eye_nonmissing_frac"]
        eye_token_valid[b, :n] = example["eye_token_valid"]
        pred[b, :n] = example["pred"]
        mae_mask[b, :n] = example["mae_mask"]
        subject_id.append(str(example["subject_id"]))
        trial_id.append(str(example["trial_id"]))
        paths.append(str(example.get("path", "")))
    return (
        {
            "content": content,
            "quality": quality,
            "stim": stim,
            "task_id": task_id,
            "pad_mask": pad_mask,
            "eye_nonmissing_frac": eye_nonmissing_frac,
            "eye_token_valid": eye_token_valid,
            "subject_id": subject_id,
            "trial_id": trial_id,
            "path": paths,
        },
        pred,
        mae_mask,
    )
