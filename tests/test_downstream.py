from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from eyemae.config import load_config
from eyemae.downstream_data import DownstreamTrialDataset, collate_downstream_trials
from eyemae.downstream_metrics import (
    aggregate_subject_predictions,
    binary_auroc,
    compute_binary_metrics,
    compute_multiclass_metrics,
)
from eyemae.finetune import (
    DownstreamClassifier,
    apply_freeze_mode,
    load_pretrained_encoder,
    should_stop_early,
    train_downstream,
    weighted_bce_loss,
)
from eyemae.make_downstream_splits import make_downstream_splits
from eyemae.pooling import eye_mean_pool


def _write_raw_npz(path: Path, n: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y = np.linspace(1.0, -1.0, n, dtype=np.float32)
    gaze = np.stack(
        [
            x,
            y,
            np.full(n, 10.0, dtype=np.float32),
            x + 0.5,
            y - 0.5,
            np.full(n, 20.0, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    )
    stimulus = np.stack(
        [
            np.full(n, 3.0, dtype=np.float32),
            np.full(n, -2.0, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    )
    np.savez_compressed(path, gaze=gaze, stimulus=stimulus)


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "info_id",
        "info_id_with_suffix",
        "task",
        "source_suffix",
        "relative_npz",
        "trial_npz",
        "n_samples",
        "left_final_keep",
        "right_final_keep",
        "disease",
        "group",
        "success",
    ]
    with (manifest_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cfg(tmp_path: Path, data_dir: Path) -> dict:
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(data_dir)
    cfg["data"]["npz_schema"] = "cd_no_cond2_gaze_stimulus"
    cfg["data"]["nan_policy"] = "mark_missing"
    cfg["data"]["npz_keys"] = {"eye": "gaze", "stimulus": "stimulus"}
    cfg["data"]["gaze_columns"] = {
        "left_x": 0,
        "left_y": 1,
        "left_area": 2,
        "right_x": 3,
        "right_y": 4,
        "right_area": 5,
        "left_label": 6,
        "right_label": 7,
    }
    cfg["data"]["stimulus_columns"] = {"stim_x": 0, "stim_y": 1, "stim_on": 2, "fix_on": 3}
    cfg["area"]["stats_path"] = str(tmp_path / "missing_area_stats.json")
    cfg["model"].update({"d_model": 32, "n_layers": 4, "n_heads": 4, "ffn_hidden": 64, "max_patches": 64})
    cfg["downstream"] = {
        "split_dir": str(tmp_path / "splits"),
        "pretrained_checkpoint": str(tmp_path / "pretrained.pt"),
        "partial_last_n_layers": 2,
        "head": {"hidden_dim": 16, "dropout": 0.0},
    }
    cfg["downstream_split"] = {
        "out_dir": str(tmp_path / "splits"),
        "seed": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "diseases": ["MCI"],
    }
    cfg["downstream_train"] = {
        "seed": 42,
        "precision": "fp32",
        "lr": 1e-3,
        "encoder_lr": 1e-4,
        "head_lr": 1e-3,
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "batch_size": 2,
        "num_workers": 0,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "log_every_steps": 1000,
    }
    cfg["downstream_eval"] = {"threshold": 0.5, "evaluate_splits": ["val", "test"]}
    cfg["downstream_checkpoint"] = {"monitor": "val/subject/auroc", "mode": "max"}
    return cfg


def _make_downstream_fixture(root: Path, num_subjects_per_class: int = 10) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, group in [(0, "健康"), (1, "患病")]:
        for i in range(num_subjects_per_class):
            subject = f"S{label}{i:03d}"
            path = root / "trials" / subject / "ProSaccade" / "trial_0000.npz"
            _write_raw_npz(path)
            rows.append(
                {
                    "subject": subject,
                    "info_id": subject,
                    "info_id_with_suffix": f"{subject}_D",
                    "task": "ProSaccade",
                    "source_suffix": "D",
                    "relative_npz": path.relative_to(root).as_posix(),
                    "trial_npz": str(path),
                    "n_samples": "40",
                    "left_final_keep": "1",
                    "right_final_keep": "1",
                    "disease": "MCI",
                    "group": group,
                    "success": "0",
                }
            )
    _write_manifest(root, rows)
    return rows


def test_downstream_splits_are_subject_stratified(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_downstream_fixture(root)
    cfg = _cfg(tmp_path, root)
    summary = make_downstream_splits(cfg)
    split_dir = tmp_path / "splits" / "MCI"
    assert (split_dir / "train.txt").exists()
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_summary = summary["diseases"]["MCI"]["splits"][split]
        assert split_summary["subject_label_counts"]["0"] > 0
        assert split_summary["subject_label_counts"]["1"] > 0
        for row in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            subject = Path(row).parts[1]
            assert subject not in seen
            seen[subject] = split


def test_downstream_kfold_splits_are_subject_stratified(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_downstream_fixture(root, num_subjects_per_class=10)
    cfg = _cfg(tmp_path, root)
    cfg["downstream_split"].update({"strategy": "subject_stratified_kfold", "num_folds": 5})
    summary = make_downstream_splits(cfg)
    assert summary["strategy"] == "subject_stratified_kfold"
    test_seen: dict[str, int] = {}
    for fold_index in range(5):
        split_dir = tmp_path / "splits" / f"fold_{fold_index}" / "MCI"
        split_subjects: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            split_summary = summary["diseases"]["MCI"]["folds"][fold_index]["splits"][split]
            assert split_summary["subject_label_counts"]["0"] > 0
            assert split_summary["subject_label_counts"]["1"] > 0
            subjects = {Path(row).parts[1] for row in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()}
            split_subjects[split] = subjects
        assert not split_subjects["train"] & split_subjects["val"]
        assert not split_subjects["train"] & split_subjects["test"]
        assert not split_subjects["val"] & split_subjects["test"]
        for subject in split_subjects["test"]:
            assert subject not in test_seen
            test_seen[subject] = fold_index
    assert len(test_seen) == 20


def test_downstream_dataset_applies_manifest_final_keep(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = root / "trials" / "S001" / "ProSaccade" / "trial_0000.npz"
    _write_raw_npz(path)
    _write_manifest(
        root,
        [
            {
                "subject": "S001",
                "info_id": "S001",
                "info_id_with_suffix": "S001_D",
                "task": "ProSaccade",
                "source_suffix": "D",
                "relative_npz": path.relative_to(root).as_posix(),
                "trial_npz": str(path),
                "n_samples": "40",
                "left_final_keep": "0",
                "right_final_keep": "1",
                "disease": "MCI",
                "group": "患病",
                "success": "0",
            }
        ],
    )
    split = tmp_path / "split.txt"
    split.write_text(path.relative_to(root).as_posix() + "\n", encoding="utf-8")
    cfg = _cfg(tmp_path, root)
    dataset = DownstreamTrialDataset(root, split, cfg, disease="MCI")
    item = dataset[0]
    assert item["label"] == 1.0
    assert not item["eye_token_valid"][:, 0].any()
    assert item["eye_token_valid"][:, 1].all()
    batch = collate_downstream_trials([item])
    assert batch["label"].shape == (1,)
    assert batch["subject_key"] == ["S001"]


def test_eye_mean_pool_masks_padding_and_invalid_eye_tokens() -> None:
    hidden = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    eye_valid = torch.tensor([[[True, False], [True, True]]])
    pad_mask = torch.tensor([[False, True]])
    pooled, has_valid, denom = eye_mean_pool(hidden, eye_valid, pad_mask)
    assert has_valid.tolist() == [True]
    assert denom.tolist() == [1.0]
    assert torch.equal(pooled, hidden[:, 0, 0])


def test_weighted_bce_uses_manual_label_class_weight() -> None:
    logits = torch.zeros(2)
    labels = torch.tensor([0.0, 1.0])
    sample_weight = torch.ones(2)
    valid = torch.ones(2, dtype=torch.bool)
    loss, denominator = weighted_bce_loss(logits, labels, sample_weight, valid, torch.tensor(3.0))
    assert denominator.item() == 4.0
    assert torch.allclose(loss, torch.tensor(float(np.log(2.0))))


def test_downstream_freeze_modes_and_pretrained_loading(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, tmp_path)
    model = DownstreamClassifier(cfg)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({"model": model.encoder.state_dict(), "epoch": 3, "global_step": 7}, checkpoint)
    other = DownstreamClassifier(cfg)
    info = load_pretrained_encoder(other, checkpoint)
    assert info["epoch"] == 3
    assert info["num_skipped_keys"] > 0
    prefixed_checkpoint = tmp_path / "pretrained_module.pt"
    torch.save({"model": {f"module.{key}": value for key, value in model.encoder.state_dict().items()}}, prefixed_checkpoint)
    load_pretrained_encoder(DownstreamClassifier(cfg), prefixed_checkpoint)
    apply_freeze_mode(other, "pretrained_linear_probe")
    assert not any(param.requires_grad for param in other.encoder.parameters())
    assert all(param.requires_grad for param in other.head.parameters())
    apply_freeze_mode(other, "pretrained_partial", partial_last_n_layers=2)
    assert not any(param.requires_grad for param in other.encoder.content_tokenizer.parameters())
    assert not any(param.requires_grad for param in other.encoder.blocks[0].parameters())
    assert all(param.requires_grad for param in other.encoder.blocks[-1].parameters())
    assert all(param.requires_grad for param in other.encoder.final_norm.parameters())


def test_downstream_metrics_and_subject_aggregation() -> None:
    labels = [0, 0, 1, 1]
    logits = [-4.0, -2.0, 2.0, 4.0]
    metrics = compute_binary_metrics(labels, logits, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert binary_auroc(labels, logits) == 1.0
    rows = [
        {"subject_key": "A", "label": 0, "logit": -2.0},
        {"subject_key": "A", "label": 0, "logit": -4.0},
        {"subject_key": "B", "label": 1, "logit": 4.0},
    ]
    subject_rows = aggregate_subject_predictions(rows)
    assert len(subject_rows) == 2
    assert subject_rows[0]["num_trials"] == 2


def test_downstream_multiclass_weighted_f1_and_kappa() -> None:
    labels = [0, 0, 1, 1, 2]
    logits = [
        [4.0, 1.0, 0.0],
        [0.0, 4.0, 1.0],
        [0.0, 4.0, 1.0],
        [0.0, 1.0, 4.0],
        [0.0, 1.0, 4.0],
    ]
    metrics = compute_multiclass_metrics(labels, logits, num_classes=3)
    assert np.isclose(metrics["weighted_f1"], 0.6)
    assert np.isclose(metrics["cohen_kappa"], 0.4117647058823529)


def test_should_stop_early_respects_min_epochs() -> None:
    assert not should_stop_early(
        epoch=48,
        epochs_without_improve=20,
        patience=20,
        min_epochs_before_early_stopping=50,
    )
    assert should_stop_early(
        epoch=49,
        epochs_without_improve=20,
        patience=20,
        min_epochs_before_early_stopping=50,
    )
    assert not should_stop_early(
        epoch=60,
        epochs_without_improve=19,
        patience=20,
        min_epochs_before_early_stopping=50,
    )


def test_train_downstream_tiny_smoke(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_downstream_fixture(root, num_subjects_per_class=4)
    cfg = _cfg(tmp_path, root)
    cfg["experiment"]["output_dir"] = str(tmp_path / "out")
    cfg["downstream_train"]["max_epochs"] = 1
    cfg["downstream_eval"]["evaluate_splits"] = ["val"]
    cfg["debug"] = {"max_train_batches": 1, "max_eval_batches": 1}
    make_downstream_splits(cfg)
    metrics = train_downstream(cfg, disease="MCI", mode="scratch")
    assert (tmp_path / "out" / "checkpoint_last.pt").exists()
    assert (tmp_path / "out" / "metrics_final.json").exists()
    assert (tmp_path / "out" / "resolved_config.yaml").exists()
    assert (tmp_path / "out" / "validation_metrics.json").exists()
    assert (tmp_path / "out" / "trial_predictions_val.csv").exists()
    assert (tmp_path / "out" / "trial_predictions_validation.csv").exists()
    assert (tmp_path / "out" / "subject_predictions_validation.csv").exists()
    assert (tmp_path / "out" / "confusion_matrix_validation.json").exists()
    assert "val/subject/accuracy" in metrics
