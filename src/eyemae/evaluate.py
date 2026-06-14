from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .baselines import finalize_baseline_sums, new_baseline_sums, update_baseline_sums
from .batching import TokenBatchSampler
from .config import load_config, split_path_for_name, validate_config
from .data import ID_TO_TASK, TrialDataset, collate_trials
from .losses import compute_reconstruction_loss
from .masking import generate_mae_mask
from .metrics import finalize_metric_sums, new_metric_sums, update_metric_sums
from .model import build_model
from .train import move_batch_to_device
from .utils import ensure_dir, set_seed, setup_logging, write_json
from .visualize import save_visualizations


LOGGER = logging.getLogger(__name__)
MASK_TYPE_NAMES = {1: "random", 2: "short_span", 3: "long_span"}


def make_eval_loader(dataset: TrialDataset, cfg: dict[str, Any]) -> DataLoader:
    num_workers = 0 if not torch.cuda.is_available() else max(0, min(2, int(cfg["train"].get("num_workers", 0))))
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(cfg["train"].get("persistent_workers", False)) and num_workers > 0,
        "collate_fn": collate_trials,
    }
    if cfg["train"].get("batch_trials_per_gpu") is None and cfg["train"].get("max_seq_tokens_per_gpu") is not None:
        max_seq_tokens = int(cfg["train"]["max_seq_tokens_per_gpu"])
        configured_max_trials = int(cfg["train"].get("max_trials_per_gpu") or 256)
        max_patches = int(cfg["model"]["max_patches"])
        max_trials = min(configured_max_trials, max(64, max_seq_tokens // max(1, 3 * max_patches)))
        sampler = TokenBatchSampler(
            dataset,
            max_seq_tokens=max_seq_tokens,
            max_trials=max_trials,
            shuffle=False,
            seed=int(cfg["eval"]["seed"]),
            bucket_by_length=bool(cfg["train"].get("bucket_by_length", False)),
            infinite=False,
        )
        LOGGER.info(
            "Evaluation uses token batches: max_seq_tokens=%s max_trials=%s bucket_by_length=%s",
            max_seq_tokens,
            max_trials,
            bool(cfg["train"].get("bucket_by_length", False)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    batch_size = int(cfg["train"].get("batch_trials_per_gpu") or min(16, int(cfg["train"].get("max_trials_per_gpu") or 16)))
    LOGGER.info("Evaluation uses fixed trial batches: batch_size=%s", batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common)


@torch.no_grad()
def evaluate(cfg, checkpoint_path: str | Path, split: str) -> dict[str, float]:
    validate_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["eval"]["seed"]))
    dataset = TrialDataset(cfg["data"]["data_dir"], split_path_for_name(cfg, split), cfg)
    loader = make_eval_loader(dataset, cfg)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sums = new_metric_sums(device)
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    baseline_sums = new_baseline_sums(device)
    baseline_by_mask_type: dict[str, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["eval"]["seed"]))
    viz_collector = _VizCollector(max_trials=16, seed=int(cfg["eval"]["seed"]))
    seen_trials = 0
    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        mae_mask, mask_type = generate_mae_mask(batch, cfg, generator=generator)
        out = model(
            batch["content"],
            batch["quality"],
            batch["stim"],
            batch["task_id"],
            batch["pad_mask"],
            batch["eye_token_valid"],
            mae_mask,
        )
        _loss, stats = compute_reconstruction_loss(
            out["pred"], batch["content"], batch["quality"], mae_mask, batch["pad_mask"], batch["eye_token_valid"], cfg
        )
        update_metric_sums(
            sums,
            out["pred"],
            batch["content"],
            batch["quality"],
            mae_mask,
            batch["pad_mask"],
            batch["eye_token_valid"],
            stats,
            cfg,
        )
        update_baseline_sums(baseline_sums, batch, mae_mask, cfg)
        _update_group_metrics(group_sums, batch, out["pred"], mae_mask, mask_type, cfg, device)
        _update_baseline_groups(baseline_by_mask_type, batch, mae_mask, mask_type, cfg, device)
        viz_collector.observe(batch, out["pred"], mae_mask, mask_type)
        seen_trials += len(batch["subject_id"])
        if (batch_idx + 1) % 100 == 0:
            LOGGER.info("Evaluated %s batches / %s trials", batch_idx + 1, seen_trials)

    metrics = finalize_metric_sums(sums, prefix=split, cfg=cfg)
    group_metrics = _finalize_group_metrics(group_sums, cfg)
    baseline_metrics = {
        "overall": finalize_baseline_sums(baseline_sums),
        "by_mask_type": {name: finalize_baseline_sums(s) for name, s in sorted(baseline_by_mask_type.items())},
    }

    out_dir = ensure_dir(Path(cfg["experiment"]["output_dir"]) / "evaluation" / split)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "metrics_by_group.json", group_metrics)
    write_json(out_dir / "baselines.json", baseline_metrics)
    viz_batch = viz_collector.build_batch()
    if viz_batch is not None:
        save_visualizations(viz_batch[0], viz_batch[1], viz_batch[2], out_dir / "visualizations", cfg, max_trials=16)
    return metrics


def _update_group_metrics(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    batch: dict[str, Any],
    pred: torch.Tensor,
    mae_mask: torch.Tensor,
    mask_type: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
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
            device,
        )

    left_filter = torch.zeros_like(batch["eye_token_valid"], dtype=torch.bool)
    left_filter[:, :, 0] = True
    right_filter = torch.zeros_like(batch["eye_token_valid"], dtype=torch.bool)
    right_filter[:, :, 1] = True
    _update_group(group_sums, "eye", "left", left_filter, pred, batch, mae_mask, cfg, device)
    _update_group(group_sums, "eye", "right", right_filter, pred, batch, mae_mask, cfg, device)

    n_real = (~batch["pad_mask"]).sum(dim=1)
    length_buckets = {
        "lt_64": n_real < 64,
        "64_127": (n_real >= 64) & (n_real < 128),
        "128_255": (n_real >= 128) & (n_real < 256),
        "ge_256": n_real >= 256,
    }
    for name, filt in length_buckets.items():
        _update_group(group_sums, "trial_length", name, trial_tokens(filt), pred, batch, mae_mask, cfg, device)

    missing_frac = _trial_missing_fraction(batch)
    missing_buckets = {
        "zero": missing_frac == 0,
        "le_0.25": (missing_frac > 0) & (missing_frac <= 0.25),
        "0.25_0.50": (missing_frac > 0.25) & (missing_frac <= 0.50),
        "gt_0.50": missing_frac > 0.50,
    }
    for name, filt in missing_buckets.items():
        _update_group(group_sums, "missing_fraction", name, trial_tokens(filt), pred, batch, mae_mask, cfg, device)

    subject_suffix = torch.tensor([_subject_suffix_id(subject_id) for subject_id in batch["subject_id"]], device=device)
    subject_buckets = {
        "both_eye_subject": subject_suffix == 0,
        "left_only_subject": subject_suffix == 1,
        "right_only_subject": subject_suffix == 2,
    }
    for name, filt in subject_buckets.items():
        _update_group(group_sums, "subject_eye_availability", name, trial_tokens(filt), pred, batch, mae_mask, cfg, device)

    valid_frac = batch["eye_token_valid"].float().sum(dim=(1, 2)) / ((~batch["pad_mask"]).sum(dim=1).float() * 2.0).clamp_min(1.0)
    valid_buckets = {
        "all": valid_frac >= 0.999,
        "0.75_1.00": (valid_frac >= 0.75) & (valid_frac < 0.999),
        "0.25_0.75": (valid_frac >= 0.25) & (valid_frac < 0.75),
        "lt_0.25": valid_frac < 0.25,
    }
    for name, filt in valid_buckets.items():
        _update_group(group_sums, "eye_token_valid_fraction", name, trial_tokens(filt), pred, batch, mae_mask, cfg, device)

    for mask_id, mask_name in MASK_TYPE_NAMES.items():
        _update_group(group_sums, "mask_type", mask_name, mask_type == mask_id, pred, batch, mae_mask, cfg, device)


def _update_group(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    group: str,
    bucket: str,
    token_filter: torch.Tensor,
    pred: torch.Tensor,
    batch: dict[str, Any],
    mae_mask: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    if not bool(token_filter.any()):
        return
    sums = group_sums.setdefault(group, {}).setdefault(bucket, new_metric_sums(device))
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


def _update_baseline_groups(
    baseline_by_mask_type: dict[str, dict[str, torch.Tensor]],
    batch: dict[str, Any],
    mae_mask: torch.Tensor,
    mask_type: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    for mask_id, mask_name in MASK_TYPE_NAMES.items():
        token_filter = mask_type == mask_id
        if not bool(token_filter.any()):
            continue
        sums = baseline_by_mask_type.setdefault(mask_name, new_baseline_sums(device))
        update_baseline_sums(sums, batch, mae_mask, cfg, token_filter=token_filter)


def _finalize_group_metrics(
    group_sums: dict[str, dict[str, dict[str, torch.Tensor]]],
    cfg: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        group: {bucket: finalize_metric_sums(sums, cfg=cfg) for bucket, sums in sorted(buckets.items())}
        for group, buckets in sorted(group_sums.items())
    }


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


class _VizCollector:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="pretrain_val", choices=["pretrain_train", "pretrain_val", "pretrain_test"])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    metrics = evaluate(cfg, args.checkpoint, args.split)
    print(metrics)


if __name__ == "__main__":
    main()
