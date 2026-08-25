#!/usr/bin/env python3
"""Cache trial-level frozen EyeVQ-BERT CLS and nuisance metadata for Stage A.

The cache is deliberately label-preserving but optimizer-free: downstream
ablations may be screened with Train-only subject folds without repeatedly
running BERT.  Test features are cached for one final evaluation, but no Stage
A selector is allowed to read them.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from eyemae.data import load_area_stats
from eyemae.downstream_data import PackedDownstreamDataset
from eyemae.eyevq.config import load_bert_checkpoint
from eyemae.eyevq.downstream.demographics import (
    encode_subject_demographics,
    fit_demographic_spec,
    read_demographic_index,
)
from eyemae.eyevq.downstream.mil_data import TASK_IDS
from eyemae.eyevq.downstream.mil_model import EyeVQSubjectMIL
from eyemae.eyevq.downstream.split_audit import audit_split_rows, assert_clean_splits
from eyemae.eyevq.downstream.train_mil import (
    _downstream_cfg,
    _trial_loader,
    filter_subjects_below_task_minimum,
    validate_mci_area_stats,
)
from eyemae.utils import atomic_torch_save, write_json


QC_NAMES = (
    "one_eye",
    "blink_fraction",
    "missing_fraction",
    "patch_count",
    "valid_eye_token_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_qc_from_row(
    row: dict[str, Any], *, valid_eye_token_fraction: float
) -> list[float]:
    """Return label-independent trial quality covariates from one packed row."""
    left = str(row.get("left_final_keep", "")).strip().lower() in {
        "1", "true", "yes"
    }
    right = str(row.get("right_final_keep", "")).strip().lower() in {
        "1", "true", "yes"
    }
    frame_length = max(int(row.get("frame_length") or 0), 1)
    blink = int(row.get("left_blink_points") or 0) + int(
        row.get("right_blink_points") or 0
    )
    missing = int(row.get("left_missing_points") or 0) + int(
        row.get("right_missing_points") or 0
    )
    return [
        float(left ^ right),
        float(blink) / float(2 * frame_length),
        float(missing) / float(2 * frame_length),
        float(row.get("num_patches_20ms") or 0),
        float(valid_eye_token_fraction),
    ]


@torch.inference_mode()
def extract_split(
    model: EyeVQSubjectMIL,
    loader,
    *,
    row_by_trial: dict[str, dict[str, Any]],
    device: torch.device,
    bf16: bool,
) -> dict[str, Any]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    tasks: list[torch.Tensor] = []
    quality: list[torch.Tensor] = []
    subjects: list[str] = []
    trial_ids: list[str] = []
    seen: set[str] = set()
    for batch in loader:
        content = batch["content"].to(device, non_blocking=True).transpose(
            -1, -2
        ).contiguous()
        stim = batch["stim"].to(device, non_blocking=True).transpose(
            -1, -2
        ).contiguous()
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        nonmissing = batch["eye_nonmissing_frac"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=bool(bf16 and device.type == "cuda"),
        ):
            cls = model.encode_trials(
                stim, content, pad_mask, nonmissing, task_ids
            )
        valid_eye = (
            (nonmissing >= model.bert.min_nonmissing_frac)
            & ~pad_mask.unsqueeze(-1)
        )
        valid_fraction = valid_eye.float().sum(dim=(1, 2)) / (
            (~pad_mask).sum(dim=1).clamp_min(1).float() * 2.0
        )
        global_ids = [str(value) for value in batch["global_trial_id"]]
        for trial_id in global_ids:
            if trial_id in seen:
                raise ValueError(f"Duplicate trial in CLS cache: {trial_id}")
            if trial_id not in row_by_trial:
                raise KeyError(f"Packed index row missing for trial: {trial_id}")
            seen.add(trial_id)
        qc_rows = [
            trial_qc_from_row(
                row_by_trial[trial_id],
                valid_eye_token_fraction=float(valid_fraction[index].item()),
            )
            for index, trial_id in enumerate(global_ids)
        ]
        features.append(cls.detach().float().cpu().to(torch.float16))
        labels.append(batch["label"].detach().long().cpu())
        tasks.append(batch["task_id"].detach().long().cpu())
        quality.append(torch.tensor(qc_rows, dtype=torch.float32))
        subjects.extend(str(value) for value in batch["subject_key"])
        trial_ids.extend(global_ids)
    if not features:
        raise ValueError("Trial CLS cache extraction produced no batches")
    result = {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "task_ids": torch.cat(tasks, dim=0),
        "quality": torch.cat(quality, dim=0),
        "quality_names": list(QC_NAMES),
        "subject_keys": subjects,
        "global_trial_ids": trial_ids,
    }
    count = result["features"].shape[0]
    for name in ("labels", "task_ids", "quality"):
        if result[name].shape[0] != count:
            raise RuntimeError(f"Cache axis mismatch for {name}")
    if len(subjects) != count or len(trial_ids) != count:
        raise RuntimeError("Cache metadata axis mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-trials-per-task", type=int, default=4)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Cache only Train for representation screening; implies --no-test.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["train"]
    label_cfg = cfg.get("label", {"type": "binary", "num_classes": 2})
    num_classes = int(label_cfg.get("num_classes", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = Path(model_cfg["bert_checkpoint"]).resolve()
    bert, bert_cfg, _ = load_bert_checkpoint(bert_path, torch.device("cpu"))
    model = EyeVQSubjectMIL(
        bert,
        num_tasks=len(TASK_IDS),
        num_classes=num_classes,
        task_pooling="shared_head_mean",
        trial_pooling="logit_mean",
        freeze_embedding=True,
        freeze_bottom_layers=len(bert.transformer),
        demographic_dim=0,
    ).to(device)
    model.requires_grad_(False)

    area_path = Path(data_cfg["area_stats_path"])
    area_stats = load_area_stats(area_path)
    validate_mci_area_stats(
        area_stats,
        mode=str(data_cfg.get("area_stats_mode", "bert_pretraining_global")),
        area_stats_path=area_path,
        bert_area_stats_path=bert_cfg.get("train", {}).get("area_stats_path"),
    )
    downstream_cfg = _downstream_cfg(data_cfg, bert_cfg, label_cfg)
    data_dir = Path(data_cfg["data_dir"])
    split_files = {"train": data_cfg["train_index"]}
    if not args.train_only:
        split_files["val"] = data_cfg["val_index"]
    if not args.no_test and not args.train_only:
        split_files["test"] = data_cfg["test_index"]
    datasets: dict[str, PackedDownstreamDataset] = {}
    train_counts = None
    for split, name in split_files.items():
        dataset = PackedDownstreamDataset(
            data_dir,
            data_dir / name,
            downstream_cfg,
            area_stats=area_stats,
            train_subject_trial_counts=train_counts,
        )
        filter_subjects_below_task_minimum(
            dataset,
            task_ids=TASK_IDS,
            min_trials_per_task=int(args.min_trials_per_task),
        )
        datasets[split] = dataset
        if split == "train":
            train_counts = dict(
                Counter(str(row["ml_subject_id"]) for row in dataset.rows)
            )
    if set(datasets) == {"train", "val", "test"}:
        assert_clean_splits(audit_split_rows({k: v.rows for k, v in datasets.items()}))
    elif args.train_only:
        train_trial_ids = [
            str(row.get("global_trial_id", "")).strip()
            for row in datasets["train"].rows
        ]
        if any(not value for value in train_trial_ids):
            raise ValueError("Train-only cache contains an empty global_trial_id")
        if len(train_trial_ids) != len(set(train_trial_ids)):
            raise ValueError("Train-only cache contains duplicate global_trial_id rows")
    else:
        raise ValueError(
            "Cross-split cache audit requires train/val/test; use --train-only "
            "for an explicitly Train-only screening cache"
        )

    train_subjects = {
        str(row["ml_subject_id"]) for row in datasets["train"].rows
    }
    train_metadata = read_demographic_index(
        data_dir / data_cfg["train_index"], subjects=train_subjects
    )
    demographic_spec = fit_demographic_spec(train_metadata, age_encoding="zscore")

    cache: dict[str, Any] = {
        "schema_version": 1,
        "selection_source": "train_only_subject_folds",
        "test_used_for_selection": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "bert_checkpoint": str(bert_path),
        "bert_sha256": _sha256(bert_path),
        "area_stats_path": str(area_path.resolve()),
        "area_stats_sha256": _sha256(area_path.resolve()),
        "demographic_spec": demographic_spec,
        "splits": {},
    }
    bf16 = bool(train_cfg.get("bf16", True))
    for split, dataset in datasets.items():
        rows = {str(row["global_trial_id"]): row for row in dataset.rows}
        split_cache = extract_split(
            model,
            _trial_loader(dataset, train_cfg),
            row_by_trial=rows,
            device=device,
            bf16=bf16,
        )
        subjects = set(split_cache["subject_keys"])
        metadata = read_demographic_index(
            data_dir / split_files[split], subjects=subjects
        )
        encoded = encode_subject_demographics(metadata, demographic_spec)
        subject_order = sorted(subjects)
        split_cache["demographic_subject_keys"] = subject_order
        split_cache["demographics"] = torch.stack(
            [encoded[subject] for subject in subject_order]
        )
        cache["splits"][split] = split_cache

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, output)
    manifest = {
        key: value for key, value in cache.items() if key != "splits"
    }
    manifest["split_counts"] = {
        split: {
            "trials": int(value["features"].shape[0]),
            "subjects": len(set(value["subject_keys"])),
            "labels": dict(Counter(int(x) for x in value["labels"].tolist())),
        }
        for split, value in cache["splits"].items()
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == "__main__":
    main()
