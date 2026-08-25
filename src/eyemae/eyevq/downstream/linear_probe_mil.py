#!/usr/bin/env python3
"""Low-capacity subject classifier on cached all-trial EyeVQ-BERT features."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eyemae.data import load_area_stats
from eyemae.downstream_data import PackedDownstreamDataset
from eyemae.eyevq.config import load_bert_checkpoint
from eyemae.eyevq.downstream.mil_data import TASK_IDS
from eyemae.eyevq.downstream.demographics import (
    EDUCATION_CATEGORIES,
    SEX_CATEGORIES,
)
from eyemae.eyevq.downstream.mil_model import EyeVQSubjectMIL
from eyemae.eyevq.downstream.split_audit import audit_split_rows, assert_clean_splits
from eyemae.eyevq.downstream.train_mil import (
    _downstream_cfg,
    _trial_loader,
    filter_subjects_below_task_minimum,
    validate_mci_area_stats,
)
from eyemae.utils import write_json


@torch.inference_mode()
def extract_subject_features(
    model: EyeVQSubjectMIL,
    loader,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Accumulate exact per-task first and second moments over all valid trials."""
    model.eval()
    subjects: dict[str, dict[str, Any]] = {}
    for batch in loader:
        content = batch["content"].to(device).transpose(-1, -2).contiguous()
        stim = batch["stim"].to(device).transpose(-1, -2).contiguous()
        pad_mask = batch["pad_mask"].to(device)
        nonmissing = batch["eye_nonmissing_frac"].to(device)
        task_ids = batch["task_id"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cls = model.encode_trials(stim, content, pad_mask, nonmissing, task_ids)
        cls = cls.float().cpu()
        for index, key in enumerate(batch["subject_key"]):
            key = str(key)
            task_id = int(batch["task_id"][index])
            label = int(batch["label"][index])
            entry = subjects.setdefault(key, {
                "label": label,
                "sum": [None for _ in TASK_IDS],
                "square_sum": [None for _ in TASK_IDS],
                "count": [0 for _ in TASK_IDS],
            })
            if entry["label"] != label:
                raise ValueError(f"Inconsistent labels for subject {key}")
            value = cls[index]
            entry["sum"][task_id] = (
                value.clone() if entry["sum"][task_id] is None
                else entry["sum"][task_id] + value
            )
            value_square = value.square()
            entry["square_sum"][task_id] = (
                value_square.clone() if entry["square_sum"][task_id] is None
                else entry["square_sum"][task_id] + value_square
            )
            entry["count"][task_id] += 1
    return subjects


def feature_matrix(
    subjects: dict[str, dict[str, Any]], *, include_std: bool
) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = sorted(subjects)
    rows = []
    labels = []
    for key in keys:
        entry = subjects[key]
        task_features = []
        for task_id in TASK_IDS:
            count = int(entry["count"][task_id])
            if count <= 0:
                raise ValueError(f"Missing task {task_id} for subject {key}")
            mean = entry["sum"][task_id] / count
            task_features.append(mean)
            if include_std:
                mean_square = entry["square_sum"][task_id] / count
                task_features.append((mean_square - mean.square()).clamp_min(0).sqrt())
        rows.append(torch.cat(task_features).numpy())
        labels.append(int(entry["label"]))
    return keys, np.stack(rows), np.asarray(labels, dtype=np.int64)


def write_predictions(
    path: Path, keys: list[str], labels: np.ndarray, scores: np.ndarray, split: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "subject_key", "label", "logit", "prob"])
        writer.writeheader()
        for key, label, score in zip(keys, labels, scores):
            writer.writerow({
                "split": split,
                "subject_key": key,
                "label": int(label),
                "logit": float(score),
                "prob": float(1.0 / (1.0 + np.exp(-score))),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-trials-per-task", type=int, default=4)
    parser.add_argument("--downstream-checkpoint", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    downstream_checkpoint = None
    if args.downstream_checkpoint:
        downstream_checkpoint = torch.load(
            args.downstream_checkpoint, map_location="cpu", weights_only=False
        )
        cfg = downstream_checkpoint["cfg"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["train"]
    label_cfg = cfg.get("label", {"type": "binary", "num_classes": 2})
    demographics_cfg = cfg.get("demographics", {})
    demographic_spec = demographics_cfg.get("fitted_spec")
    demographic_dim = (
        int(demographic_spec["feature_dim"])
        if demographic_spec is not None
        else (
            2 + len(SEX_CATEGORIES) + len(EDUCATION_CATEGORIES)
            if bool(demographics_cfg.get("enabled", False))
            else 0
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert, bert_cfg, _ = load_bert_checkpoint(model_cfg["bert_checkpoint"], torch.device("cpu"))
    model = EyeVQSubjectMIL(
        bert,
        num_tasks=len(TASK_IDS),
        num_classes=int(label_cfg.get("num_classes", 2)),
        classifier_hidden=int(model_cfg.get("classifier_hidden", 32)),
        task_bottleneck_dim=int(model_cfg.get("task_bottleneck_dim", 16)),
        residual_hidden=int(model_cfg.get("residual_hidden", 16)),
        residual_include_task_mask=bool(
            model_cfg.get("residual_include_task_mask", True)
        ),
        dropout=float(model_cfg.get("dropout", 0.3)),
        task_pooling=str(cfg["mil"].get("task_pooling", "concat_task_cls")),
        trial_pooling=str(cfg["mil"].get("trial_pooling", "feature_mean")),
        cartesian_trials_per_task=int(cfg["mil"].get("trials_per_task", 4)),
        missing_task_embedding=str(
            cfg["mil"].get("missing_task_embedding", "none")
        ),
        demographic_dim=demographic_dim,
        demographic_projection_dim=int(
            demographics_cfg.get("projection_dim", 128)
        ),
        demographic_fusion=str(
            demographics_cfg.get("fusion", "additive_logit")
        ),
        classifier_head=str(model_cfg.get("classifier_head", "mlp")),
        freeze_embedding=bool(model_cfg.get("freeze_embedding", True)),
        freeze_bottom_layers=int(model_cfg.get("freeze_bottom_layers", len(bert.transformer))),
    ).to(device)
    if downstream_checkpoint is not None:
        model.load_state_dict(downstream_checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    area_stats_path = Path(data_cfg["area_stats_path"])
    area_stats = load_area_stats(area_stats_path)
    # Match the production Subject-MIL preflight exactly.  In particular, the
    # current V5 checkpoints use the same per-subject/per-eye transductive
    # statistics as BERT; treating every probe as the legacy global-stat mode
    # rejects a valid configuration and can tempt callers to bypass the audit.
    validate_mci_area_stats(
        area_stats,
        mode=str(data_cfg.get("area_stats_mode", "bert_pretraining_global")),
        area_stats_path=area_stats_path,
        bert_area_stats_path=bert_cfg.get("train", {}).get("area_stats_path"),
    )
    downstream_cfg = _downstream_cfg(data_cfg, bert_cfg)
    data_dir = Path(data_cfg["data_dir"])
    datasets = {}
    train_counts = None
    for split, index_name in (
        ("train", data_cfg["train_index"]),
        ("val", data_cfg["val_index"]),
        ("test", data_cfg["test_index"]),
    ):
        dataset = PackedDownstreamDataset(
            data_dir,
            data_dir / index_name,
            downstream_cfg,
            area_stats=area_stats,
            train_subject_trial_counts=train_counts,
        )
        filter_subjects_below_task_minimum(
            dataset,
            task_ids=TASK_IDS,
            min_trials_per_task=args.min_trials_per_task,
        )
        datasets[split] = dataset
        if split == "train":
            train_counts = dict(Counter(row["ml_subject_id"] for row in dataset.rows))
    assert_clean_splits(audit_split_rows({k: v.rows for k, v in datasets.items()}))

    aggregates = {
        split: extract_subject_features(model, _trial_loader(dataset, train_cfg), device)
        for split, dataset in datasets.items()
    }
    torch.save(aggregates, output_dir / "all_trial_cls_moments.pt")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []
    fitted = {}
    for feature_kind, include_std in (("mean", False), ("mean_std", True)):
        matrices = {
            split: feature_matrix(subjects, include_std=include_std)
            for split, subjects in aggregates.items()
        }
        train_keys, x_train, y_train = matrices["train"]
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=42, svd_solver="randomized")),
            ("classifier", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ])
        search = GridSearchCV(
            pipeline,
            param_grid={
                "pca__n_components": [8, 16, 32, 64, 128],
                "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        result = {
            "feature_kind": feature_kind,
            "inner_cv_auc_mean": float(search.best_score_),
            "inner_cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "inner_cv_train_auc": float(search.cv_results_["mean_train_score"][search.best_index_]),
            "best_params": search.best_params_,
        }
        for split in ("train", "val", "test"):
            keys, x, y = matrices[split]
            scores = search.best_estimator_.decision_function(x)
            result[f"{split}_auc"] = float(roc_auc_score(y, scores))
            result[f"{split}_accuracy_05"] = float(accuracy_score(y, scores >= 0))
            write_predictions(
                output_dir / f"predictions_{feature_kind}_{split}.csv",
                keys, y, scores, split,
            )
        candidates.append(result)
        fitted[feature_kind] = search.best_estimator_

    # Architecture is selected exclusively by train-fold CV, never by val/test.
    selected = max(candidates, key=lambda row: row["inner_cv_auc_mean"])
    write_json(output_dir / "metrics.json", {
        "selection_source": "train_only_stratified_5fold_cv",
        "test_used_for_selection": False,
        "encoder": (
            "frozen_downstream_checkpoint"
            if downstream_checkpoint is not None
            else "frozen_pretrained_ckpt_final"
        ),
        "downstream_checkpoint": args.downstream_checkpoint,
        "min_trials_per_task": args.min_trials_per_task,
        "candidates": candidates,
        "selected": selected,
        "split_subject_counts": {k: len(v) for k, v in aggregates.items()},
    })
    print(yaml.safe_dump(selected, sort_keys=False))


if __name__ == "__main__":
    main()
