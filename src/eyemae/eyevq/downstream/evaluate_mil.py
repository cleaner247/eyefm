"""Evaluate one selected Subject-MIL checkpoint on validation and test."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from eyemae.data import load_area_stats
from eyemae.downstream_data import PackedDownstreamDataset
from eyemae.eyevq.config import load_bert_checkpoint
from eyemae.eyevq.downstream.mil_model import EyeVQSubjectMIL
from eyemae.eyevq.downstream.demographics import (
    encode_subject_demographics,
    read_demographic_index,
)
from eyemae.eyevq.downstream.split_audit import audit_split_rows, assert_clean_splits
from eyemae.eyevq.downstream.train_mil import (
    _downstream_cfg,
    _trial_loader,
    apply_task_availability_policy,
    evaluate_subjects,
)
from eyemae.downstream_metrics import write_prediction_csv
from eyemae.utils import set_seed, write_json


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: torch.device | None = None,
    include_train: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    label_cfg = cfg.get("label", {"type": "binary", "num_classes": 2})
    label_type = str(label_cfg.get("type", "binary"))
    num_classes = int(label_cfg.get("num_classes", 2))
    demographics_cfg = cfg.get("demographics", {})
    demographics_enabled = bool(demographics_cfg.get("enabled", False))
    demographic_spec = demographics_cfg.get("fitted_spec")
    if demographics_enabled and not demographic_spec:
        raise ValueError("Checkpoint is missing its fitted demographic spec")
    set_seed(int(train_cfg.get("seed", 42)))

    bert, bert_cfg, _ = load_bert_checkpoint(
        model_cfg["bert_checkpoint"], torch.device("cpu")
    )
    model = EyeVQSubjectMIL(
        bert,
        num_tasks=4,
        num_classes=num_classes,
        output_dim=2 if label_type == "hierarchical_pd3" else None,
        classifier_hidden=int(model_cfg.get("classifier_hidden", 32)),
        task_bottleneck_dim=int(model_cfg.get("task_bottleneck_dim", 16)),
        residual_hidden=int(model_cfg.get("residual_hidden", 16)),
        residual_include_task_mask=bool(
            model_cfg.get("residual_include_task_mask", True)
        ),
        dropout=float(model_cfg.get("dropout", 0.3)),
        task_pooling=str(cfg["mil"].get("task_pooling", "concat_task_cls")),
        trial_pooling=str(cfg["mil"].get("trial_pooling", "feature_mean")),
        freeze_embedding=bool(model_cfg.get("freeze_embedding", True)),
        freeze_bottom_layers=int(model_cfg.get("freeze_bottom_layers", 4)),
        demographic_dim=(
            int(demographic_spec["feature_dim"])
            if demographics_enabled
            else 0
        ),
        demographic_projection_dim=int(
            demographics_cfg.get("projection_dim", 128)
        ),
        demographic_fusion=str(
            demographics_cfg.get("fusion", "additive_logit")
        ),
        demographic_max_alpha=float(
            demographics_cfg.get("max_alpha", 0.3)
        ),
        task_residual_logit_scale=float(
            cfg["mil"].get("task_residual_logit_scale", 0.2)
        ),
        classifier_head=str(model_cfg.get("classifier_head", "mlp")),
        cartesian_trials_per_task=int(cfg["mil"].get("trials_per_task", 4)),
        missing_task_embedding=str(
            cfg["mil"].get("missing_task_embedding", "none")
        ),
    )
    model.label_type = label_type
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    downstream_cfg = _downstream_cfg(data_cfg, bert_cfg, label_cfg)
    area_stats_path = Path(data_cfg["area_stats_path"])
    if not area_stats_path.is_file():
        raise FileNotFoundError(
            f"EyeVQ MIL evaluation requires the configured area statistics file: {area_stats_path}"
        )
    area_stats = load_area_stats(area_stats_path)
    data_dir = Path(data_cfg["data_dir"])
    train_trials = PackedDownstreamDataset(
        data_dir,
        data_dir / data_cfg["train_index"],
        downstream_cfg,
        area_stats=area_stats,
    )
    trials_per_task = int(cfg["mil"]["trials_per_task"])
    eligibility_min_trials = int(
        cfg["mil"].get("eligibility_min_trials_per_task", trials_per_task)
    )
    evaluation_min_trials = int(
        cfg["mil"].get("evaluation_min_trials_per_task", trials_per_task)
    )
    require_all_tasks = bool(cfg["mil"].get("train_require_all_tasks", True))
    sample_all_available_below_k = bool(
        cfg["mil"].get("sample_all_available_below_k", False)
    )
    task_coverage_loss_weighting = str(
        train_cfg.get("task_coverage_loss_weighting", "none")
    )
    apply_task_availability_policy(
        train_trials,
        task_ids=(0, 1, 2, 3),
        min_trials_per_task=eligibility_min_trials,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
    )
    train_counts = dict(Counter(row["ml_subject_id"] for row in train_trials.rows))
    val_trials = PackedDownstreamDataset(
        data_dir,
        data_dir / data_cfg["val_index"],
        downstream_cfg,
        area_stats=area_stats,
        train_subject_trial_counts=train_counts,
    )
    apply_task_availability_policy(
        val_trials,
        task_ids=(0, 1, 2, 3),
        min_trials_per_task=evaluation_min_trials,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
    )
    test_trials = PackedDownstreamDataset(
        data_dir,
        data_dir / data_cfg["test_index"],
        downstream_cfg,
        area_stats=area_stats,
        train_subject_trial_counts=train_counts,
    )
    apply_task_availability_policy(
        test_trials,
        task_ids=(0, 1, 2, 3),
        min_trials_per_task=evaluation_min_trials,
        require_all_tasks=require_all_tasks,
        sample_all_available_below_k=sample_all_available_below_k,
    )
    split_audit = audit_split_rows({
        "train": train_trials.rows,
        "val": val_trials.rows,
        "test": test_trials.rows,
    })
    assert_clean_splits(split_audit)
    train_loader: DataLoader | None = (
        _trial_loader(train_trials, train_cfg) if include_train else None
    )
    val_loader: DataLoader = _trial_loader(val_trials, train_cfg)
    test_loader: DataLoader = _trial_loader(test_trials, train_cfg)
    bf16 = bool(train_cfg.get("bf16", True))
    train_demographics = None
    val_demographics = None
    test_demographics = None
    if demographics_enabled:
        train_subjects = {str(row["ml_subject_id"]) for row in train_trials.rows}
        val_subjects = {str(row["ml_subject_id"]) for row in val_trials.rows}
        test_subjects = {str(row["ml_subject_id"]) for row in test_trials.rows}
        if include_train:
            train_demographics = encode_subject_demographics(
                read_demographic_index(
                    data_dir / data_cfg["train_index"], subjects=train_subjects
                ),
                demographic_spec,
            )
        val_demographics = encode_subject_demographics(
            read_demographic_index(
                data_dir / data_cfg["val_index"], subjects=val_subjects
            ),
            demographic_spec,
        )
        test_demographics = encode_subject_demographics(
            read_demographic_index(
                data_dir / data_cfg["test_index"], subjects=test_subjects
            ),
            demographic_spec,
        )

    train_metrics = None
    train_rows = None
    if include_train:
        if train_loader is None:
            raise RuntimeError("Train evaluation requested without a train loader")
        train_metrics, train_rows = evaluate_subjects(
            model,
            train_loader,
            device,
            split_name="train",
            bf16=bf16,
            subject_demographics=train_demographics,
            task_coverage_loss_weighting=task_coverage_loss_weighting,
        )

    val_metrics, val_rows = evaluate_subjects(
        model,
        val_loader,
        device,
        split_name="val",
        bf16=bf16,
        subject_demographics=val_demographics,
        task_coverage_loss_weighting=task_coverage_loss_weighting,
    )
    if label_type == "binary":
        threshold = 0.5
        val_bacc = float(val_metrics["val/subject/balanced_accuracy"])
    elif label_type == "hierarchical_pd3":
        threshold = 0.5
        val_bacc = float(val_metrics["val/subject/balanced_accuracy"])
    else:
        threshold = None
        val_bacc = float(val_metrics["val/subject/balanced_accuracy"])
    test_metrics, test_rows = evaluate_subjects(
        model,
        test_loader,
        device,
        split_name="test",
        threshold=threshold if threshold is not None else 0.5,
        bf16=bf16,
        subject_demographics=test_demographics,
        task_coverage_loss_weighting=task_coverage_loss_weighting,
    )
    declared_selection_metric = checkpoint.get("selection_metric")
    if declared_selection_metric is None:
        declared_selection_metric = train_cfg.get("selection_metric")
    if declared_selection_metric is None:
        declared_selection_metric = train_cfg.get("early_stopping_metric")
    if declared_selection_metric is None:
        declared_selection_metric = (
            "auroc" if label_type == "binary" else "macro_auroc_ovr"
        )
    selection_metric = str(declared_selection_metric).split("/")[-1]
    selection_key = f"val/subject/{selection_metric}"
    if selection_key not in val_metrics:
        raise KeyError(
            f"Checkpoint selection metric {selection_key!r} is absent from "
            "validation metrics"
        )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "best_step": int(checkpoint.get("step", -1)),
        # Keep best_val_auroc for compatibility with historical consumers;
        # best_val_selection_score is the semantically correct field when a
        # thresholded metric such as balanced accuracy selected the checkpoint.
        "best_val_auroc": float(val_metrics[selection_key]),
        "best_val_selection_score": float(val_metrics[selection_key]),
        "checkpoint_selection_metric": selection_key,
        "label_type": label_type,
        "num_classes": num_classes,
        "val": val_metrics,
        "test_evaluated": True,
        "split_audit": split_audit,
        "cfg": cfg,
    }
    if train_metrics is not None:
        result["train"] = train_metrics
    if label_type == "binary":
        result.update({
            "decision_rule": "fixed_threshold",
            "decision_threshold": float(threshold),
            "threshold_policy": "fixed_0.5",
            "val_balanced_accuracy_at_fixed_threshold": float(val_bacc),
            "val_default_05": val_metrics,
            "test": test_metrics,
            "test_default_05": test_metrics,
        })
    elif label_type == "hierarchical_pd3":
        result.update({
            "decision_rule": "disease_sigmoid_then_subtype_sigmoid",
            "disease_threshold": 0.5,
            "subtype_threshold": 0.5,
            "subtype_positive_class": "tremor_spectrum",
            "val_balanced_accuracy": float(val_bacc),
            "test": test_metrics,
        })
    else:
        result.update({
            "decision_rule": "argmax",
            "val_balanced_accuracy": float(val_bacc),
            "test": test_metrics,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_rows is not None:
        write_prediction_csv(output_dir / "predictions_train.csv", train_rows)
    write_prediction_csv(output_dir / "predictions_val_best.csv", val_rows)
    write_prediction_csv(output_dir / "predictions_test.csv", test_rows)
    write_json(output_dir / "metrics_test.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Evaluate the selected checkpoint on the full eligible training split too.",
    )
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        output_dir=args.output_dir,
        include_train=args.include_train,
    )
    test_metrics = (
        result["test"]
    )
    if result["label_type"] == "binary":
        print({
            "best_step": result["best_step"],
            "val_auroc": result["best_val_selection_score"],
            "test_auroc": test_metrics["test/subject/auroc"],
            "test_balanced_accuracy": test_metrics["test/subject/balanced_accuracy"],
        })
    else:
        print({
            "best_step": result["best_step"],
            "val_selection_score": result["best_val_selection_score"],
            "test_balanced_accuracy": test_metrics["test/subject/balanced_accuracy"],
            "test_weighted_f1": test_metrics["test/subject/weighted_f1"],
            "test_cohen_kappa": test_metrics["test/subject/cohen_kappa"],
        })


if __name__ == "__main__":
    main()
