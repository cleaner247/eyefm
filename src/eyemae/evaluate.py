from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .baselines import finalize_baseline_sums, new_baseline_sums, update_baseline_sums
from .batching import TokenBatchSampler
from .config import load_config, split_path_for_name, validate_config
from .data import collate_trials, make_trial_dataset
from .eval_artifacts import MASK_TYPE_NAMES, VizCollector, finalize_group_metrics, new_group_metric_sums, update_group_metrics
from .losses import compute_reconstruction_loss
from .masking import generate_mae_mask
from .metrics import finalize_metric_sums, new_metric_sums, update_metric_sums
from .model import build_model
from .train import move_batch_to_device
from .utils import ensure_dir, set_seed, setup_logging, write_json
from .visualize import save_visualizations


LOGGER = logging.getLogger(__name__)


def make_eval_loader(dataset, cfg: dict[str, Any]) -> DataLoader:
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
        max_trials = configured_max_trials
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
    dataset = make_trial_dataset(cfg, split_path_for_name(cfg, split))
    loader = make_eval_loader(dataset, cfg)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sums = new_metric_sums(device)
    group_sums = new_group_metric_sums(device)
    baseline_sums = new_baseline_sums(device)
    baseline_by_mask_type: dict[str, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["eval"]["seed"]))
    viz_collector = VizCollector(max_trials=16, seed=int(cfg["eval"]["seed"]))
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
        update_group_metrics(group_sums, batch, out["pred"], mae_mask, mask_type, cfg)
        _update_baseline_groups(baseline_by_mask_type, batch, mae_mask, mask_type, cfg, device)
        viz_collector.observe(batch, out["pred"], mae_mask, mask_type)
        seen_trials += len(batch["subject_id"])
        if (batch_idx + 1) % 100 == 0:
            LOGGER.info("Evaluated %s batches / %s trials", batch_idx + 1, seen_trials)

    metrics = finalize_metric_sums(sums, prefix=split, cfg=cfg)
    group_metrics = finalize_group_metrics(group_sums, cfg)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test", "pretrain_train", "pretrain_val", "pretrain_validation", "pretrain_test"],
    )
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    metrics = evaluate(cfg, args.checkpoint, args.split)
    print(metrics)


if __name__ == "__main__":
    main()
