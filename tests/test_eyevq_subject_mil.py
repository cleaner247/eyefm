from __future__ import annotations

from collections import Counter
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import yaml

from eyemae.eyevq.downstream.mil_data import SubjectBagDataset
from eyemae.eyevq.downstream.mil_model import EyeVQSubjectMIL
from eyemae.eyevq.downstream.train_mil import (
    aggregate_trial_view_task_logits,
    deranged_permutation,
    make_contiguous_trial_views,
    mask_tasks_below_task_minimum,
    retain_all_available_partial_tasks,
    task_coverage_weighted_supervised_loss,
    task_coverage_weights,
)


def test_trial_view_task_mean_and_trimmed_mean() -> None:
    logits = torch.tensor([
        [[[0.0], [10.0]]],
        [[[1.0], [11.0]]],
        [[[2.0], [12.0]]],
        [[[100.0], [13.0]]],
    ])
    torch.testing.assert_close(
        aggregate_trial_view_task_logits(logits, mode="task_mean"),
        logits.mean(dim=0),
    )
    expected = torch.tensor([[[1.5], [11.5]]])
    torch.testing.assert_close(
        aggregate_trial_view_task_logits(logits, mode="task_trimmed_mean"),
        expected,
    )


def test_make_contiguous_trial_views_is_view_major_and_disjoint():
    # B=2, T=2, K=4: each task's first/last two trials form the two views.
    source = torch.arange(16)
    result = make_contiguous_trial_views(
        source,
        num_subjects=2,
        num_tasks=2,
        trials_per_task=4,
        num_views=2,
    )
    assert result.tolist() == [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
    assert sorted(result.tolist()) == list(range(16))


def test_make_contiguous_trial_views_rejects_nondivisible_k():
    with pytest.raises(ValueError, match="divisible"):
        make_contiguous_trial_views(
            torch.arange(6),
            num_subjects=1,
            num_tasks=2,
            trials_per_task=3,
            num_views=2,
        )


def test_deranged_permutation_has_no_fixed_points() -> None:
    torch.manual_seed(3)
    permutation = deranged_permutation(8, torch.device("cpu"))
    assert sorted(permutation.tolist()) == list(range(8))
    assert torch.all(permutation != torch.arange(8))
from eyemae.eyevq.downstream.mil_sampler import (
    DistributedSubjectEpochSampler,
)
from eyemae.eyevq.downstream.split_audit import audit_split_rows, assert_clean_splits
from eyemae.eyevq.downstream.train_mil import (
    _downstream_cfg,
    build_optimizer_param_groups,
    filter_subjects_below_task_minimum,
    metrics_from_subject_rows,
    update_early_stopping_counter,
)
from eyemae.eyevq.downstream.evaluate_all_grid import make_summary_row
from eyemae.eyevq.downstream.finalize_grid import select_best_run
from eyemae.eyevq.pretrain.model import EyeVQBERT


def test_subject_mil_config_uses_epoch_training_and_all_transformer_layers() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/eyevq/downstream_mci_subject_mil.yaml").open(
        encoding="utf-8"
    ) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["mil"]["subjects_per_gpu"] == 4
    assert cfg["mil"]["class_sampling"] == "epoch_shuffle_no_class_quota"
    assert "negative_per_global_step" not in cfg["mil"]
    assert "positive_per_global_step" not in cfg["mil"]
    assert cfg["mil"]["trials_per_task"] == 2
    assert cfg["model"]["classifier_hidden"] == 32
    assert cfg["mil"]["epoch_random_trial_sampling"] is True
    assert "step_random_trial_sampling" not in cfg["mil"]
    assert "cyclic_trial_sampling" not in cfg["mil"]
    assert cfg["model"]["freeze_bottom_layers"] == 4
    assert cfg["model"]["freeze_embedding"] is True
    assert cfg["mil"]["task_pooling"] == "concat_task_cls"
    assert cfg["train"]["epochs"] == 100
    assert "max_steps" not in cfg["train"]
    assert "val_every_steps" not in cfg["train"]
    assert cfg["train"]["warmup_epochs"] == 4
    assert cfg["train"]["layer_decay"] == 1.0
    assert cfg["train"]["encoder_lr"] == 1e-5
    assert cfg["train"]["head_lr"] == 1e-5
    assert cfg["train"]["encoder_min_lr"] == 1e-6
    assert cfg["train"]["head_min_lr"] == 1e-6
    assert cfg["train"]["early_stopping_metric"] == "val/subject/auroc"
    assert cfg["train"]["early_stopping_min_epochs"] == 27
    assert cfg["train"]["early_stopping_patience_epochs"] == 10
    assert cfg["data"]["area_stats_path"].endswith(
        "/outputs/eyevq/mci_downstream/area_stats_bert_frozen.json"
    )
    assert cfg["data"]["require_any_eye_keep"] is True


def test_cartesian_mci_config_uses_strict_k4_and_subject_level_logit_mean() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root / "configs/eyevq/downstream_mci_subject_mil_cartesian_k4_h64.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["model"]["classifier_hidden"] == 64
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["train_require_all_tasks"] is True
    assert cfg["mil"]["task_pooling"] == "cartesian_task_cls"
    assert cfg["mil"]["trial_pooling"] == "cartesian_logit_mean"
    assert cfg["mil"]["eval_use_all_trials"] is True
    assert cfg["train"]["num_trial_views"] == 1
    assert cfg["train"]["trial_view_consistency_weight"] == 0.0
    assert cfg["demographics"]["fusion"] == "concat_task_cls"
    assert cfg["data"]["pretraining_overlap_policy"] == "allow_unlabeled_target_trials"


def test_cartesian_partial_mci_config_uses_mask_cls_and_all_short_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root
        / "configs/eyevq/downstream_mci_subject_mil_cartesian_partial_maskcls_k4_h64.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["model"]["classifier_hidden"] == 64
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["train_require_all_tasks"] is False
    assert cfg["mil"]["sample_all_available_below_k"] is True
    assert cfg["mil"]["missing_task_embedding"] == "learned_per_task"
    assert cfg["mil"]["task_pooling"] == "cartesian_task_cls"
    assert cfg["mil"]["trial_pooling"] == "cartesian_logit_mean"
    assert cfg["mil"]["eval_use_all_trials"] is True
    assert cfg["train"]["task_coverage_loss_weighting"] == "none"


def test_shared_logit_residual_config_uses_partial_k4_late_fusion() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root
        / "configs/eyevq/downstream_mci_subject_mil_shared_logit_residual_h16_partial_k4.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["model"]["classifier_hidden"] == 128
    assert cfg["model"]["residual_hidden"] == 16
    assert cfg["model"]["freeze_bottom_layers"] == 4
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["train_require_all_tasks"] is False
    assert cfg["mil"]["sample_all_available_below_k"] is True
    assert cfg["mil"]["missing_task_embedding"] == "none"
    assert cfg["mil"]["task_pooling"] == "shared_head_residual"
    assert cfg["mil"]["trial_pooling"] == "logit_mean"
    assert cfg["mil"]["eval_use_all_trials"] is True
    assert cfg["train"]["task_coverage_loss_weighting"] == "none"
    assert cfg["demographics"]["fusion"] == "late_residual"


def test_shared_logit_residual_strict_config_removes_partial_subjects() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root
        / "configs/eyevq/downstream_mci_subject_mil_shared_logit_residual_h16_strict_k4.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["model"]["classifier_hidden"] == 128
    assert cfg["model"]["residual_hidden"] == 16
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["train_require_all_tasks"] is True
    assert cfg["mil"]["sample_all_available_below_k"] is False
    assert cfg["mil"]["missing_task_embedding"] == "none"
    assert cfg["mil"]["task_pooling"] == "shared_head_residual"
    assert cfg["mil"]["trial_pooling"] == "logit_mean"
    assert cfg["train"]["task_coverage_loss_weighting"] == "none"
    assert cfg["demographics"]["fusion"] == "late_residual"


def test_pd5_final_config_uses_selected_logit_mean_scheme() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root / "configs/eyevq/downstream_pd5_subject_mil_logit_mean_final.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["label"]["type"] == "multiclass"
    assert cfg["label"]["num_classes"] == 5
    assert cfg["model"]["freeze_bottom_layers"] == 4
    assert cfg["model"]["freeze_embedding"] is True
    assert cfg["model"]["classifier_hidden"] == 128
    assert cfg["model"]["dropout"] == 0.3
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["trial_pooling"] == "logit_mean"
    assert cfg["mil"]["task_pooling"] == "shared_head_mean"
    assert cfg["mil"]["train_require_all_tasks"] is True
    assert cfg["train"]["num_trial_views"] == 1
    assert cfg["train"]["trial_view_consistency_weight"] == 0.0
    assert cfg["train"].get("task_coverage_loss_weighting", "none") == "none"
    assert cfg["train"]["class_weighting"] == "subject_inverse_frequency"
    assert cfg["demographics"]["age_encoding"] == "zscore"
    assert cfg["demographics"]["projection_dim"] == 128


def test_pd5_residual_config_is_strict_and_omits_task_mask_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    with (
        root
        / "configs/eyevq/downstream_pd5_subject_mil_shared_logit_residual_h16_nomask_strict_k4.yaml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["label"]["type"] == "multiclass"
    assert cfg["label"]["num_classes"] == 5
    assert cfg["model"]["classifier_hidden"] == 128
    assert cfg["model"]["residual_hidden"] == 16
    assert cfg["model"]["residual_include_task_mask"] is False
    assert cfg["mil"]["trials_per_task"] == 4
    assert cfg["mil"]["train_require_all_tasks"] is True
    assert cfg["mil"]["sample_all_available_below_k"] is False
    assert cfg["mil"]["task_pooling"] == "shared_head_residual"
    assert cfg["mil"]["trial_pooling"] == "logit_mean"
    assert cfg["demographics"]["fusion"] == "late_residual"


def test_subject_mil_forwards_required_eye_filter_to_packed_dataset() -> None:
    data_cfg = {
        "data_dir": "/tmp/mci",
        "area_stats_path": "/tmp/area.json",
        "require_any_eye_keep": True,
    }
    downstream_cfg = _downstream_cfg(
        data_cfg, {"patch": {"samples": 40, "stride": 40}}
    )
    assert downstream_cfg["data"]["require_any_eye_keep"] is True


def test_subject_mil_forwards_multiclass_label_schema() -> None:
    downstream_cfg = _downstream_cfg(
        {
            "data_dir": "/tmp/pd5",
            "area_stats_path": "/tmp/area.json",
            "require_any_eye_keep": True,
        },
        {"patch": {"samples": 40, "stride": 40}},
        {"type": "multiclass", "num_classes": 5},
    )
    assert downstream_cfg["label"]["type"] == "multiclass"
    assert downstream_cfg["label"]["num_classes"] == 5


def test_subject_filter_removes_whole_subject_when_any_task_has_fewer_than_k() -> None:
    rows = []
    labels = []
    for subject, task_counts, label in (
        ("complete", [2, 2, 2, 2], 0),
        ("short", [2, 2, 2, 1], 1),
    ):
        for task_id, count in enumerate(task_counts):
            for _ in range(count):
                rows.append({"ml_subject_id": subject, "task_id": str(task_id)})
                labels.append(label)
    dataset = SimpleNamespace(rows=rows, labels=labels)
    excluded = filter_subjects_below_task_minimum(
        dataset, task_ids=(0, 1, 2, 3), min_trials_per_task=2
    )
    assert excluded == ("short",)
    assert {row["ml_subject_id"] for row in dataset.rows} == {"complete"}
    assert dataset.labels == [0] * 8


def test_partial_task_policy_masks_short_tasks_and_drops_only_all_short_subjects() -> None:
    rows = []
    labels = []
    for subject, task_counts, label in (
        ("complete", [2, 2, 2, 2], 0),
        ("partial", [2, 1, 2, 0], 1),
        ("all_short", [1, 1, 1, 1], 2),
    ):
        for task_id, count in enumerate(task_counts):
            for _ in range(count):
                rows.append({"ml_subject_id": subject, "task_id": str(task_id)})
                labels.append(label)
    dataset = SimpleNamespace(rows=rows, labels=labels)

    excluded = mask_tasks_below_task_minimum(
        dataset, task_ids=(0, 1, 2, 3), min_trials_per_task=2
    )

    assert excluded == ("all_short",)
    assert {row["ml_subject_id"] for row in dataset.rows} == {"complete", "partial"}
    partial_tasks = {
        int(row["task_id"])
        for row in dataset.rows
        if row["ml_subject_id"] == "partial"
    }
    assert partial_tasks == {0, 2}
    assert dataset.subjects_with_missing_tasks == ("partial",)
    assert {
        (group["subject_key"], group["task_id"], group["valid_trials"])
        for group in dataset.masked_subject_task_groups
    } == {("partial", 1, 1), ("partial", 3, 0)}


def test_partial_all_available_policy_keeps_one_to_k_minus_one_trial_tasks() -> None:
    rows = []
    labels = []
    for subject, task_counts, label in (
        ("complete", [4, 4, 4, 4], 0),
        ("partial", [4, 2, 0, 1], 1),
    ):
        for task_id, count in enumerate(task_counts):
            for _ in range(count):
                rows.append({"ml_subject_id": subject, "task_id": str(task_id)})
                labels.append(label)
    dataset = SimpleNamespace(rows=rows, labels=labels)

    excluded = retain_all_available_partial_tasks(
        dataset, task_ids=(0, 1, 2, 3), target_trials_per_task=4
    )

    assert excluded == ()
    assert len(dataset.rows) == 23
    assert dataset.subjects_with_missing_tasks == ("partial",)
    assert dataset.masked_subject_task_groups == ({
        "subject_key": "partial", "task_id": 2, "valid_trials": 0,
    },)
    assert {
        (group["subject_key"], group["task_id"], group["valid_trials"])
        for group in dataset.subject_task_groups_below_k_used_all
    } == {("partial", 1, 2), ("partial", 3, 1)}


def test_present_fraction_coverage_weights_and_multiclass_loss() -> None:
    present = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ])
    coverage = task_coverage_weights(present, mode="present_fraction")
    torch.testing.assert_close(coverage, torch.tensor([1.0, 0.75, 0.5, 0.25]))

    logits = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    loss = task_coverage_weighted_supervised_loss(
        logits,
        labels,
        present,
        label_type="multiclass",
        coverage_mode="present_fraction",
        pos_weight=None,
        class_weights=class_weights,
    )
    per_subject = torch.nn.functional.cross_entropy(
        logits[0], labels, reduction="none"
    )
    effective = coverage * class_weights.index_select(0, labels)
    expected = (per_subject * effective).sum() / effective.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad.abs().sum(dim=-1) > 0)


def test_epoch100_runner_uses_eight_layers_and_runs_final_test() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/run_eyevq_mci_mil_k2_8l_epoch100.sh"
    ).read_text(encoding="utf-8")
    assert "--freeze-bottom-layers 4" in source
    assert "--epochs 100" in source
    assert "--encoder-lr 1e-5" in source
    assert "--skip-test" not in source
    trainer_source = (
        root / "src/eyemae/eyevq/downstream/train_mil.py"
    ).read_text(encoding="utf-8")
    assert 'training_stop_reason = "max_epochs"' in trainer_source
    assert 'training_stop_reason = "early_stopping"' in trainer_source
    assert '"test_evaluated": True' in trainer_source


def test_k4_runner_and_serial_comparison_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    k4 = (root / "scripts/run_eyevq_mci_mil_k4_8l_epoch100.sh").read_text(
        encoding="utf-8"
    )
    assert "--trials-per-task 4" in k4
    assert "--epochs 100" in k4
    assert "--skip-test" not in k4
    pipeline = (root / "scripts/wait_k2_then_run_k4_compare.sh").read_text(
        encoding="utf-8"
    )
    assert 'K2_DIR/metrics_test.json' in pipeline
    assert "run_eyevq_mci_mil_k4_8l_epoch100.sh" in pipeline
    assert "compare_eyevq_mci_k2_k4.py" in pipeline


def test_grid_runner_is_validation_only_and_keeps_seed_fixed() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_eyevq_mci_mil_grid.sh").read_text(encoding="utf-8")
    assert "--skip-test" in source
    assert "seed42" in source
    assert "--seed" not in source
    assert 'PATIENCE="${PATIENCE:-81}"' in source
    assert "for unfrozen_layers in 6 8 10 12" in source
    assert "for encoder_lr in 2e-6 5e-6 1e-5" in source


def test_120_epoch_layer_runner_has_fixed_lr_and_skips_test() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/run_eyevq_mci_mil_layers_lr1e5_e120.sh"
    ).read_text(encoding="utf-8")
    assert 'ENCODER_LR="1e-5"' in source
    assert 'EPOCHS="120"' in source
    assert 'PATIENCE="121"' in source
    assert "for unfrozen_layers in 6 8 10 12" in source
    assert "--skip-test" in source
    assert "--seed" not in source


def test_grid_finalizer_selects_only_highest_validation_auroc(tmp_path: Path) -> None:
    paths = []
    for index, auroc in enumerate((0.81, 0.90, 0.86, 0.84)):
        path = tmp_path / f"run{index}.json"
        path.write_text(json.dumps({
            "best_val_auroc": auroc,
            "best_epoch": 10 + index,
        }), encoding="utf-8")
        paths.append(path)
    best_path, best_result = select_best_run(paths)
    assert best_path.name == "run1.json"
    assert best_result["best_val_auroc"] == 0.90


def test_all_test_finalizer_evaluates_five_runs_without_test_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/finalize_eyevq_mci_mil_e120.sh"
    ).read_text(encoding="utf-8")
    assert "eyemae.eyevq.downstream.evaluate_all_grid" in source
    assert "--expected-runs 5" in source
    module_source = (
        root / "src/eyemae/eyevq/downstream/evaluate_all_grid.py"
    ).read_text(encoding="utf-8")
    assert '"test_evaluated_for_all_runs": True' in module_source
    assert '"test_results_used_for_selection": False' in module_source


def test_all_test_summary_marks_full_finetune() -> None:
    validation = {
        "best_epoch": 71,
        "best_val_auroc": 0.91,
        "cfg": {
            "model": {"freeze_bottom_layers": 0, "freeze_embedding": False},
            "train": {"encoder_lr": 1e-5, "head_lr": 5e-5},
        },
    }
    test_result = {
        "tuned_threshold": 0.43,
        "test_tuned": {
            "test/subject/auroc": 0.88,
            "test/subject/balanced_accuracy": 0.81,
            "test/subject/f1": 0.79,
            "test/subject/accuracy": 0.82,
        },
        "test_default_05": {
            "test/subject/balanced_accuracy": 0.78,
            "test/subject/f1": 0.75,
            "test/subject/accuracy": 0.80,
        },
    }
    row = make_summary_row(
        Path("full_finetune/metrics_val_best.json"), validation, test_result
    )
    assert row["full_finetune"] is True
    assert row["embedding_unfrozen"] is True
    assert row["test_auroc"] == 0.88


def test_full_finetune_runner_unfreezes_embedding_and_skips_test() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/run_eyevq_mci_mil_full_finetune_e120.sh"
    ).read_text(encoding="utf-8")
    assert "--freeze-bottom-layers 0" in source
    assert "--unfreeze-embedding" in source
    assert "--encoder-lr 1e-5" in source
    assert "--epochs 120" in source
    assert "--skip-test" in source


class _FakeTrialDataset:
    def __init__(self) -> None:
        self.rows = []
        self.labels = []
        for subject, label, tasks in (
            ("complete_0", 0, range(4)),
            ("complete_1", 1, range(4)),
            ("incomplete", 0, range(3)),
        ):
            for task_id in tasks:
                for trial_index in range(6):
                    self.rows.append({
                        "ml_subject_id": subject,
                        "task_id": str(task_id),
                        "global_trial_id": f"{subject}-{task_id}-{trial_index}",
                    })
                    self.labels.append(label)


def _tiny_mil(
    *, freeze_bottom_layers: int = 1, task_pooling: str = "concat_task_cls",
    trial_pooling: str = "feature_mean",
    demographic_dim: int = 0,
    demographic_projection_dim: int = 128,
    demographic_fusion: str = "additive_logit",
    classifier_head: str = "mlp",
    classifier_hidden: int = 8,
    residual_hidden: int = 4,
    residual_include_task_mask: bool = True,
    num_classes: int = 2,
    missing_task_embedding: str = "none",
) -> EyeVQSubjectMIL:
    bert = EyeVQBERT(
        K_e=15,
        d_model=16,
        n_layers=2,
        n_heads=4,
        dim_ff=32,
        dropout=0.0,
        max_time=4,
        patch_samples=20,
        fsq_L=[3, 5],
    )
    return EyeVQSubjectMIL(
        bert,
        num_tasks=4,
        num_classes=num_classes,
        classifier_hidden=classifier_hidden,
        task_bottleneck_dim=4,
        residual_hidden=residual_hidden,
        residual_include_task_mask=residual_include_task_mask,
        dropout=0.0,
        task_pooling=task_pooling,
        trial_pooling=trial_pooling,
        freeze_embedding=True,
        freeze_bottom_layers=freeze_bottom_layers,
        demographic_dim=demographic_dim,
        demographic_projection_dim=demographic_projection_dim,
        demographic_fusion=demographic_fusion,
        classifier_head=classifier_head,
        missing_task_embedding=missing_task_embedding,
    )


def test_subject_bags_exclude_incomplete_and_sample_four_unique_trials_per_task() -> None:
    dataset = SubjectBagDataset(
        _FakeTrialDataset(), trials_per_task=4, require_all_tasks=True, seed=7
    )
    assert dataset.subject_keys == ["complete_0", "complete_1"]
    assert dataset.excluded_subjects == ("incomplete",)
    assert dataset.labels == [0, 1]

    epoch_0 = dataset.sampled_trial_indices(0, 0)
    epoch_1 = dataset.sampled_trial_indices(0, 1)
    assert len(epoch_0) == 4
    assert all(len(indices) == 4 and len(set(indices)) == 4 for indices in epoch_0)
    assert any(a != b for a, b in zip(epoch_0, epoch_1))
    assert dataset.sampled_trial_indices(0, 1) == epoch_1


def test_subject_bags_retain_partial_subject_and_leave_missing_task_unsampled() -> None:
    dataset = SubjectBagDataset(
        _FakeTrialDataset(), trials_per_task=4, require_all_tasks=False, seed=7
    )
    assert dataset.subject_keys == ["complete_0", "complete_1", "incomplete"]
    assert dataset.excluded_subjects == ()
    incomplete_index = dataset.subject_keys.index("incomplete")
    record = dataset.records[incomplete_index]
    assert record.task_present_mask == (True, True, True, False)
    sampled = dataset.sampled_trial_indices(incomplete_index, 0)
    assert all(len(indices) == 4 for indices in sampled[:3])
    assert sampled[3] == ()
    assert dataset.task_count_by_subject["incomplete"] == 3


def test_subject_bags_sample_all_available_below_k_and_pad_only_tensor_slots() -> None:
    class VariableTrialDataset:
        def __init__(self) -> None:
            self.rows = []
            self.labels = []
            for task_id, count in enumerate((6, 3, 0, 1)):
                for trial_index in range(count):
                    self.rows.append({
                        "ml_subject_id": "partial",
                        "task_id": str(task_id),
                        "global_trial_id": f"partial-{task_id}-{trial_index}",
                    })
                    self.labels.append(1)

        def __getitem__(self, index: int):
            item = dict(self.rows[index])
            item["task_id"] = int(item["task_id"])
            return item

    dataset = SubjectBagDataset(
        VariableTrialDataset(),
        trials_per_task=4,
        require_all_tasks=False,
        sample_all_available_below_k=True,
        seed=7,
    )
    sampled = dataset.sampled_trial_indices(0, 3)
    assert [len(indices) for indices in sampled] == [4, 3, 0, 1]
    assert all(len(indices) == len(set(indices)) for indices in sampled)
    item = dataset[(0, 3)]
    assert len(item["trials"]) == 16
    assert item["task_present_mask"] == (True, True, False, True)
    assert [sum(mask) for mask in item["trial_slot_mask"]] == [4, 3, 0, 1]
    assert [trial["task_id"] for trial in item["trials"]] == (
        [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    )


def test_epoch_random_trial_sampling_explores_noncyclic_pairs() -> None:
    dataset = SubjectBagDataset(
        _FakeTrialDataset(), trials_per_task=2, require_all_tasks=True, seed=7
    )
    task_zero_pairs = []
    for epoch in range(30):
        sampled = dataset.sampled_trial_indices(0, epoch)
        assert all(len(indices) == 2 and len(set(indices)) == 2 for indices in sampled)
        assert dataset.sampled_trial_indices(0, epoch) == sampled
        task_zero_pairs.append(tuple(sorted(sampled[0])))
    # The removed cyclic-window implementation over six trials could expose
    # only three disjoint pairs when K=2.
    assert len(set(task_zero_pairs)) > 3


def test_epoch_sampler_has_no_class_quota_or_within_epoch_repeats() -> None:
    labels = [0] * 145 + [1] * 97
    samplers = [
        DistributedSubjectEpochSampler(
            labels,
            num_replicas=4,
            rank=rank,
            subjects_per_rank=4,
            epochs=100,
            seed=42,
        )
        for rank in range(4)
    ]
    assert all(len(sampler) == 15 for sampler in samplers)
    epoch_label_counts = []
    batch_label_counts = []
    for epoch in (0, 1, 37, 99):
        selected = samplers[0].epoch_indices(epoch)
        assert len(selected) == len(set(selected)) == 240
        assert len(samplers[0].dropped_indices(epoch)) == 2
        epoch_label_counts.append(Counter(labels[index] for index in selected))
        for batch_index in range(15):
            rank_batches = samplers[0].all_rank_indices(epoch, batch_index)
            flat = [index for batch in rank_batches for index in batch]
            assert len(flat) == len(set(flat)) == 16
            batch_label_counts.append(Counter(labels[index] for index in flat))
            assert all(len(batch) == 4 for batch in rank_batches)
            for rank in range(4):
                assert samplers[rank].all_rank_indices(epoch, batch_index) == rank_batches
    # The epoch composition follows shuffled data; no 10/6 per-step constraint exists.
    assert all(sum(counts.values()) == 240 for counts in epoch_label_counts)
    assert any(counts != Counter({0: 10, 1: 6}) for counts in batch_label_counts)
    exposure = samplers[0].exposure_audit()
    assert exposure["min_subject_exposure"] == 99
    assert exposure["max_subject_exposure"] == 100


def test_epoch_sampler_audit_reports_every_multiclass_label() -> None:
    sampler = DistributedSubjectEpochSampler(
        [0, 0, 1, 2, 3, 4, 4, 4],
        num_replicas=2,
        rank=0,
        subjects_per_rank=2,
        epochs=2,
        seed=9,
    )
    audit = sampler.audit_epoch(0)
    assert set(audit["selected_label_counts"]) == {"0", "1", "2", "3", "4"}
    assert set(audit["dropped_label_counts"]) == {"0", "1", "2", "3", "4"}


def test_layerwise_optimizer_groups_exclude_norm_and_bias_from_decay() -> None:
    model = _tiny_mil(freeze_bottom_layers=0)
    groups = build_optimizer_param_groups(
        model,
        encoder_lr=1e-5,
        head_lr=5e-5,
        layer_decay=0.8,
        weight_decay=0.05,
    )
    grouped = {
        name: group
        for group in groups
        for name in group["parameter_names"]
    }
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(grouped) == trainable
    assert grouped["bert.transformer.0.qkv.weight"]["lr_scale"] == 0.8
    assert grouped["bert.transformer.1.qkv.weight"]["lr_scale"] == 1.0
    assert grouped["bert.transformer.0.qkv.bias"]["weight_decay"] == 0.0
    assert grouped["subject_head.0.weight"]["weight_decay"] == 0.0
    assert grouped["subject_head.1.weight"]["weight_decay"] == 0.05


def test_demographic_branch_is_a_head_optimizer_group() -> None:
    model = _tiny_mil(freeze_bottom_layers=0, demographic_dim=16)
    groups = build_optimizer_param_groups(
        model,
        encoder_lr=1e-5,
        head_lr=5e-5,
        layer_decay=1.0,
        weight_decay=0.05,
    )
    grouped = {
        name: group for group in groups for name in group["parameter_names"]
    }
    assert grouped["demographic_head.weight"]["schedule"] == "head"
    assert grouped["demographic_head.bias"]["schedule"] == "head"


def test_demographic_projection_is_a_head_optimizer_group() -> None:
    model = _tiny_mil(
        freeze_bottom_layers=0,
        task_pooling="shared_head_mean",
        demographic_dim=16,
        demographic_fusion="concat_task_cls",
    )
    groups = build_optimizer_param_groups(
        model,
        encoder_lr=1e-5,
        head_lr=5e-5,
        layer_decay=0.8,
        weight_decay=0.05,
    )
    grouped = {
        name: group for group in groups for name in group["parameter_names"]
    }
    assert grouped["demographic_projection.weight"]["schedule"] == "head"
    assert grouped["demographic_projection.weight"]["weight_decay"] == 0.05
    assert grouped["demographic_projection.bias"]["schedule"] == "head"
    assert grouped["demographic_projection.bias"]["weight_decay"] == 0.0
    assert grouped["task_feature_norm.weight"]["schedule"] == "head"
    assert grouped["task_feature_norm.weight"]["weight_decay"] == 0.0


def test_cartesian_modules_are_head_optimizer_groups_with_correct_decay() -> None:
    model = _tiny_mil(
        freeze_bottom_layers=0,
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
        classifier_hidden=7,
        missing_task_embedding="learned_per_task",
    )
    groups = build_optimizer_param_groups(
        model,
        encoder_lr=1e-5,
        head_lr=2e-5,
        layer_decay=1.0,
        weight_decay=0.05,
    )
    grouped = {
        name: group for group in groups for name in group["parameter_names"]
    }
    cartesian_names = {
        name for name in grouped if name.startswith("cartesian_")
    }
    assert cartesian_names
    assert all(grouped[name]["schedule"] == "head" for name in cartesian_names)
    assert grouped["cartesian_feature_norm.weight"]["weight_decay"] == 0.0
    assert grouped["cartesian_mask_cls_embeddings"]["weight_decay"] == 0.0
    assert grouped["cartesian_hidden.weight"]["weight_decay"] == 0.05
    assert grouped["cartesian_output.weight"]["weight_decay"] == 0.05


def test_cross_task_residual_modules_are_head_groups_with_correct_decay() -> None:
    model = _tiny_mil(
        freeze_bottom_layers=0,
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
    )
    groups = build_optimizer_param_groups(
        model,
        encoder_lr=1e-5,
        head_lr=2e-5,
        layer_decay=1.0,
        weight_decay=0.05,
    )
    grouped = {
        name: group for group in groups for name in group["parameter_names"]
    }
    residual_names = {
        name for name in grouped if name.startswith("cross_task_residual_head.")
    }
    assert residual_names
    assert all(grouped[name]["schedule"] == "head" for name in residual_names)
    assert grouped["cross_task_residual_head.0.weight"]["weight_decay"] == 0.05
    assert grouped["cross_task_residual_head.0.bias"]["weight_decay"] == 0.0
    assert grouped["cross_task_residual_head.3.weight"]["weight_decay"] == 0.05
    assert grouped["cross_task_residual_head.3.bias"]["weight_decay"] == 0.0


def test_early_stopping_new_best_resets_patience_counter() -> None:
    assert update_early_stopping_counter(
        4,
        val_improved=True,
        global_step=500,
        min_steps=400,
    ) == 0
    assert update_early_stopping_counter(
        0,
        val_improved=False,
        global_step=520,
        min_steps=400,
    ) == 1
    assert update_early_stopping_counter(
        4,
        val_improved=False,
        global_step=380,
        min_steps=400,
    ) == 4


def test_split_audit_always_checks_subject_and_trial_identity() -> None:
    clean = {
        "train": [{"ml_subject_id": "a", "source_file_uid": "1", "original_trial_index": "0"}],
        "val": [{"ml_subject_id": "b", "source_file_uid": "2", "original_trial_index": "0"}],
        "test": [{"ml_subject_id": "c", "source_file_uid": "3", "original_trial_index": "0"}],
    }
    audit = audit_split_rows(clean)
    assert audit["passed"] is True
    assert_clean_splits(audit)
    leaked = dict(clean)
    leaked["test"] = [
        {"ml_subject_id": "c", "source_file_uid": "1", "original_trial_index": "0"}
    ]
    assert audit_split_rows(leaked)["passed"] is False


def test_mil_concatenates_task_features_in_fixed_order_and_zero_fills_missing() -> None:
    model = _tiny_mil()
    features = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
    mask = torch.tensor([[True, True, True, True], [True, False, True, False]])
    output = model.classify_task_features(features, mask)
    expected = features.clone()
    expected[1, 1] = 0.0
    expected[1, 3] = 0.0
    torch.testing.assert_close(output["concatenated_features"], expected.flatten(1))
    assert output["subject_logits"].shape == (2,)


def test_mil_trial_mean_is_order_invariant_within_each_task() -> None:
    model = _tiny_mil()
    features = torch.randn(2, 4, 4, 16)
    pooled = model.pool_trial_features(features.reshape(-1, 16), num_subjects=2, trials_per_task=4)
    reversed_trials = features.flip(dims=(2,))
    pooled_reversed = model.pool_trial_features(
        reversed_trials.reshape(-1, 16), num_subjects=2, trials_per_task=4
    )
    torch.testing.assert_close(pooled, pooled_reversed)


def test_mean_std_pooling_preserves_trial_variability_and_is_order_invariant() -> None:
    model = _tiny_mil(task_pooling="concat_task_mean_std")
    features = torch.randn(2, 4, 4, 16)
    pooled = model.pool_trial_features(
        features.reshape(-1, 16), num_subjects=2, trials_per_task=4
    )
    pooled_reversed = model.pool_trial_features(
        features.flip(dims=(2,)).reshape(-1, 16),
        num_subjects=2,
        trials_per_task=4,
    )
    assert pooled.shape == (2, 4, 32)
    torch.testing.assert_close(pooled, pooled_reversed)
    torch.testing.assert_close(pooled[..., :16], features.mean(dim=2))
    torch.testing.assert_close(pooled[..., 16:], features.std(dim=2, unbiased=False))
    output = model.classify_task_features(
        pooled, torch.ones(2, 4, dtype=torch.bool)
    )
    assert output["concatenated_features"].shape == (2, 128)


def test_shared_head_fuses_task_logits_with_learned_softmax_weights() -> None:
    model = _tiny_mil(task_pooling="shared_head_softmax")
    features = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, True], [True, False, True, True]])
    output = model.classify_task_features(features, mask)
    assert output["task_logits"].shape == (2, 4)
    assert output["task_weights"].shape == (2, 4)
    torch.testing.assert_close(output["task_weights"][0], torch.full((4,), 0.25))
    assert output["task_weights"][1, 1].item() == 0.0
    torch.testing.assert_close(output["task_weights"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        output["subject_logits"],
        (output["task_logits"] * output["task_weights"]).sum(dim=1),
    )


def test_shared_head_mean_uses_fixed_uniform_present_task_weights() -> None:
    model = _tiny_mil(task_pooling="shared_head_mean")
    assert not hasattr(model, "task_weight_logits")
    features = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, True], [True, False, True, True]])
    output = model.classify_task_features(features, mask)
    torch.testing.assert_close(output["task_weights"][0], torch.full((4,), 0.25))
    torch.testing.assert_close(
        output["task_weights"][1],
        torch.tensor([1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]),
    )
    torch.testing.assert_close(
        output["subject_logits"],
        (output["task_logits"] * output["task_weights"]).sum(dim=1),
    )


def test_shared_head_constrained_weights_start_equal_and_remain_bounded() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_constrained", trial_pooling="logit_mean"
    )
    present = torch.ones(2, 4, dtype=torch.bool)
    initial = model.classify_task_features(torch.randn(2, 4, 16), present)
    torch.testing.assert_close(initial["task_weights"], torch.full((2, 4), 0.25))
    with torch.no_grad():
        model.task_weight_residual.copy_(
            torch.tensor([-100.0, -1.0, 1.0, 100.0])
        )
    output = model.classify_task_features(torch.randn(2, 4, 16), present)
    assert output["task_weights"].min().item() >= 0.15
    assert output["task_weights"].max().item() <= 0.35
    torch.testing.assert_close(output["task_weights"].sum(dim=1), torch.ones(2))


def test_trial_logit_mean_classifies_k_trials_before_uniform_aggregation() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean", trial_pooling="logit_mean"
    ).train()
    features = torch.randn(2, 4, 4, 16, requires_grad=True)
    present = torch.ones(2, 4, dtype=torch.bool)
    output = model.classify_trial_features(features, present)
    manual_trial_logits = model.classify_individual_trial_features(
        features.reshape(-1, 16)
    ).reshape(2, 4, 4)
    manual_task_logits = manual_trial_logits.mean(dim=2)
    torch.testing.assert_close(output["trial_logits"], manual_trial_logits)
    torch.testing.assert_close(output["task_logits"], manual_task_logits)
    torch.testing.assert_close(
        output["subject_logits"], manual_task_logits.mean(dim=1)
    )
    reversed_output = model.classify_trial_features(features.flip(2), present)
    torch.testing.assert_close(
        output["subject_logits"], reversed_output["subject_logits"]
    )
    output["subject_logits"].sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad.abs().sum(dim=-1) > 0)


def test_shared_logit_residual_masks_short_task_slots_and_starts_as_mean() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
        residual_hidden=5,
    ).train()
    features = torch.randn(1, 4, 4, 16, requires_grad=True)
    demographics = torch.randn(1, 16, requires_grad=True)
    slot_mask = torch.tensor([
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ]
    ])
    present = slot_mask.any(dim=2)
    output = model.classify_trial_features(
        features,
        present,
        demographics,
        trial_slot_mask=slot_mask,
    )
    manual_trial_logits = model.classify_individual_trial_features(
        features.reshape(-1, 16)
    ).reshape(1, 4, 4)
    torch.testing.assert_close(
        output["task_logits"][0, 0], manual_trial_logits[0, 0].mean()
    )
    torch.testing.assert_close(
        output["task_logits"][0, 1], manual_trial_logits[0, 1, :2].mean()
    )
    torch.testing.assert_close(
        output["task_logits"][0, 2], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        output["task_logits"][0, 3], manual_trial_logits[0, 3, 0]
    )
    torch.testing.assert_close(
        output["base_subject_logits"],
        output["task_logits"][:, [0, 1, 3]].mean(dim=1),
    )
    torch.testing.assert_close(
        output["subject_logits"], output["base_subject_logits"]
    )
    torch.testing.assert_close(
        output["residual_coverage_gate"], torch.tensor([2.0 / 3.0])
    )
    assert output["residual_inputs"].shape == (1, 24)
    output["subject_logits"].sum().backward()
    assert features.grad is not None
    assert features.grad[slot_mask].abs().sum().item() > 0.0
    assert features.grad[~slot_mask].abs().sum().item() == 0.0
    assert demographics.grad is not None
    assert demographics.grad.abs().sum().item() == 0.0
    assert model.cross_task_residual_head[-1].weight.grad is not None
    assert model.cross_task_residual_head[-1].weight.grad.abs().sum().item() > 0.0


def test_shared_logit_residual_backpropagates_after_zero_initialized_output_moves() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
        residual_hidden=5,
    ).train()
    with torch.no_grad():
        model.cross_task_residual_head[-1].weight.fill_(0.1)
    features = torch.randn(3, 4, 2, 16, requires_grad=True)
    demographics = torch.randn(3, 16, requires_grad=True)
    output = model.classify_trial_features(
        features,
        torch.ones(3, 4, dtype=torch.bool),
        demographics,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["subject_logits"], torch.tensor([0.0, 1.0, 0.0])
    )
    loss.backward()
    assert features.grad is not None and features.grad.abs().sum().item() > 0.0
    assert demographics.grad is not None and demographics.grad.abs().sum().item() > 0.0
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.cross_task_residual_head.parameters()
    )
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.task_head.parameters()
        if parameter.requires_grad
    )


def test_shared_logit_residual_coverage_gate_tracks_present_tasks() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
    ).eval()
    with torch.no_grad():
        model.cross_task_residual_head[-1].bias.fill_(2.0)
    task_logits = torch.randn(4, 4)
    present = torch.tensor([
        [True, True, True, True],
        [True, True, False, True],
        [True, False, False, True],
        [False, True, False, False],
    ])
    output = model.aggregate_task_logits(
        task_logits, present, torch.randn(4, 16)
    )
    expected_gate = torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])
    torch.testing.assert_close(output["residual_coverage_gate"], expected_gate)
    torch.testing.assert_close(
        output["subject_logits"] - output["base_subject_logits"],
        2.0 * expected_gate,
    )


def test_multiclass_residual_without_task_mask_has_36_inputs_and_requires_complete_tasks() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
        residual_hidden=5,
        residual_include_task_mask=False,
        num_classes=5,
    ).eval()
    assert model.cross_task_residual_head[0].in_features == 36
    task_logits = torch.randn(2, 4, 5)
    demographics = torch.randn(2, 16)
    present = torch.ones(2, 4, dtype=torch.bool)
    output = model.aggregate_task_logits(task_logits, present, demographics)
    assert output["residual_inputs"].shape == (2, 36)
    torch.testing.assert_close(
        output["residual_inputs"][:, :20], task_logits.reshape(2, 20)
    )
    torch.testing.assert_close(output["residual_inputs"][:, 20:], demographics)
    torch.testing.assert_close(
        output["subject_logits"], output["base_subject_logits"]
    )
    present[0, 3] = False
    with pytest.raises(ValueError, match="requires every task"):
        model.aggregate_task_logits(task_logits, present, demographics)


def test_trial_logit_mean_repeats_demographics_to_every_trial() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="concat_task_cls",
    ).train()
    features = torch.randn(2, 4, 4, 16, requires_grad=True)
    demographics = torch.randn(2, 16, requires_grad=True)
    output = model.classify_trial_features(
        features, torch.ones(2, 4, dtype=torch.bool), demographics
    )
    repeated = (
        demographics[:, None, None, :]
        .expand(-1, 4, 4, -1)
        .reshape(-1, 16)
    )
    manual = model.classify_individual_trial_features(
        features.reshape(-1, 16), repeated
    ).reshape(2, 4, 4)
    torch.testing.assert_close(output["trial_logits"], manual)
    output["subject_logits"].sum().backward()
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0


def test_cartesian_task_cls_matches_explicit_concatenation_and_logit_mean() -> None:
    model = _tiny_mil(
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
        classifier_hidden=7,
    ).train()
    features = torch.randn(2, 4, 2, 16, requires_grad=True)
    demographics = torch.randn(2, 3, requires_grad=True)
    present = torch.ones(2, 4, dtype=torch.bool)
    output = model.classify_cartesian_trial_features(
        features, present, demographics
    )

    normalized = model.cartesian_feature_norm(features)
    explicit_inputs = []
    for task_indices in product(range(2), repeat=4):
        task_slots = [
            normalized[:, task_id, trial_id]
            for task_id, trial_id in enumerate(task_indices)
        ]
        explicit_inputs.append(torch.cat((*task_slots, demographics), dim=-1))
    explicit_inputs = torch.stack(explicit_inputs, dim=1)
    explicit_logits = model.cartesian_output(
        model.cartesian_dropout(
            model.cartesian_activation(model.cartesian_hidden(explicit_inputs))
        )
    ).squeeze(-1)

    assert output["combination_logits"].shape == (2, 2, 2, 2, 2)
    torch.testing.assert_close(
        output["combination_logits"], explicit_logits.reshape(2, 2, 2, 2, 2)
    )
    torch.testing.assert_close(
        output["subject_logits"], explicit_logits.mean(dim=1)
    )
    reversed_output = model.classify_cartesian_trial_features(
        features.flip(2), present, demographics
    )
    torch.testing.assert_close(
        output["subject_logits"], reversed_output["subject_logits"]
    )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["subject_logits"], torch.tensor([0.0, 1.0])
    )
    loss.backward()
    assert features.grad is not None
    assert torch.all(features.grad.abs().sum(dim=-1) > 0)
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0


def test_cartesian_task_cls_rejects_missing_task() -> None:
    model = _tiny_mil(
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
    )
    features = torch.randn(2, 4, 2, 16)
    demographics = torch.randn(2, 3)
    present = torch.tensor([
        [True, True, True, True],
        [True, True, False, True],
    ])
    with pytest.raises(ValueError, match="every task"):
        model.classify_cartesian_trial_features(
            features, present, demographics
        )


def test_all_trial_cartesian_evaluation_matches_dense_variable_product() -> None:
    model = _tiny_mil(
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
        classifier_hidden=7,
    ).eval()
    task_features = [
        torch.randn(count, 16) for count in (2, 3, 4, 2)
    ]
    demographics = torch.randn(1, 3)
    output = model.classify_all_cartesian_task_features(
        task_features, demographics, pair_chunk_size=2
    )

    normalized = [model.cartesian_feature_norm(x) for x in task_features]
    explicit_inputs = []
    for task_indices in product(*(range(x.shape[0]) for x in task_features)):
        task_slots = [
            normalized[task_id][trial_id].unsqueeze(0)
            for task_id, trial_id in enumerate(task_indices)
        ]
        explicit_inputs.append(
            torch.cat((*task_slots, demographics), dim=-1)
        )
    explicit_inputs = torch.cat(explicit_inputs, dim=0)
    explicit_logits = model.cartesian_output(
        model.cartesian_dropout(
            model.cartesian_activation(model.cartesian_hidden(explicit_inputs))
        )
    ).squeeze(-1)
    assert output["combination_count"] == 2 * 3 * 4 * 2
    torch.testing.assert_close(
        output["subject_logits"], explicit_logits.mean().reshape(1)
    )


def test_partial_cartesian_uses_only_valid_slots_and_task_mask_cls() -> None:
    model = _tiny_mil(
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
        classifier_hidden=7,
        missing_task_embedding="learned_per_task",
    ).train()
    features = torch.randn(2, 4, 4, 16, requires_grad=True)
    demographics = torch.randn(2, 3, requires_grad=True)
    slot_mask = torch.tensor([
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ],
        [[True, True, True, True]] * 4,
    ])
    present = slot_mask.any(dim=2)
    output = model.classify_cartesian_trial_features(
        features,
        present,
        demographics,
        trial_slot_mask=slot_mask,
    )

    normalized_real = [
        model.cartesian_feature_norm(features[0, task_id][slot_mask[0, task_id]])
        if present[0, task_id]
        else model.cartesian_feature_norm(
            model.cartesian_mask_cls_embeddings[task_id : task_id + 1]
        )
        for task_id in range(4)
    ]
    explicit_inputs = []
    for task_indices in product(
        *(range(task_features.shape[0]) for task_features in normalized_real)
    ):
        explicit_inputs.append(torch.cat((
            *(normalized_real[task_id][trial_id] for task_id, trial_id in enumerate(task_indices)),
            demographics[0],
        )))
    explicit_logits = model.cartesian_output(model.cartesian_activation(
        model.cartesian_hidden(torch.stack(explicit_inputs))
    )).squeeze(-1)
    assert output["combination_counts"].tolist() == [8, 256]
    torch.testing.assert_close(
        output["subject_logits"][0], explicit_logits.mean()
    )

    output["subject_logits"].sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad[slot_mask].abs().sum(dim=-1) > 0)
    assert torch.all(features.grad[~slot_mask] == 0)
    assert model.cartesian_mask_cls_embeddings.grad is not None
    assert model.cartesian_mask_cls_embeddings.grad[2].abs().sum() > 0
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0


def test_all_trial_cartesian_missing_task_matches_explicit_mask_cls_product() -> None:
    model = _tiny_mil(
        task_pooling="cartesian_task_cls",
        trial_pooling="cartesian_logit_mean",
        demographic_dim=3,
        demographic_fusion="concat_task_cls",
        classifier_hidden=7,
        missing_task_embedding="learned_per_task",
    ).eval()
    task_features = [
        torch.randn(2, 16),
        torch.empty(0, 16),
        torch.randn(3, 16),
        torch.randn(2, 16),
    ]
    demographics = torch.randn(1, 3)
    output = model.classify_all_cartesian_task_features(
        task_features, demographics, pair_chunk_size=2
    )
    resolved = [
        features
        if features.shape[0]
        else model.cartesian_mask_cls_embeddings[index : index + 1]
        for index, features in enumerate(task_features)
    ]
    normalized = [model.cartesian_feature_norm(x) for x in resolved]
    explicit_inputs = []
    for task_indices in product(*(range(x.shape[0]) for x in normalized)):
        explicit_inputs.append(torch.cat((
            *(normalized[task_id][trial_id] for task_id, trial_id in enumerate(task_indices)),
            demographics[0],
        )))
    explicit_logits = model.cartesian_output(model.cartesian_activation(
        model.cartesian_hidden(torch.stack(explicit_inputs))
    )).squeeze(-1)
    assert output["combination_count"] == 2 * 1 * 3 * 2
    torch.testing.assert_close(
        output["subject_logits"], explicit_logits.mean().reshape(1)
    )


def test_multiclass_trial_logit_mean_preserves_class_axis_and_gradients() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="concat_task_cls",
        num_classes=5,
    ).train()
    features = torch.randn(3, 4, 4, 16, requires_grad=True)
    demographics = torch.randn(3, 16, requires_grad=True)
    present = torch.ones(3, 4, dtype=torch.bool)
    output = model.classify_trial_features(features, present, demographics)

    assert output["trial_logits"].shape == (3, 4, 4, 5)
    assert output["task_logits"].shape == (3, 4, 5)
    assert output["subject_logits"].shape == (3, 5)
    torch.testing.assert_close(
        output["subject_logits"], output["trial_logits"].mean(dim=(1, 2))
    )
    loss = torch.nn.functional.cross_entropy(
        output["subject_logits"], torch.tensor([0, 2, 4])
    )
    loss.backward()
    assert features.grad is not None
    assert torch.all(features.grad.abs().sum(dim=-1) > 0)
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0


def test_multiclass_logit_mean_masks_missing_tasks_and_their_gradients() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        trial_pooling="logit_mean",
        num_classes=5,
    ).train()
    features = torch.randn(2, 4, 4, 16, requires_grad=True)
    present = torch.tensor([
        [True, True, True, True],
        [True, False, True, False],
    ])
    output = model.classify_trial_features(features, present)
    expected_second = output["task_logits"][1, [0, 2]].mean(dim=0)
    torch.testing.assert_close(output["subject_logits"][1], expected_second)
    torch.testing.assert_close(
        output["task_weights"][1], torch.tensor([0.5, 0.0, 0.5, 0.0])
    )

    loss = torch.nn.functional.cross_entropy(
        output["subject_logits"], torch.tensor([1, 3])
    )
    loss.backward()
    assert features.grad is not None
    assert features.grad[1, [1, 3]].abs().sum().item() == 0.0
    assert features.grad[1, [0, 2]].abs().sum().item() > 0.0


def test_multiclass_subject_rows_use_macro_ovr_metrics() -> None:
    rows = []
    for label in range(5):
        logits = [-2.0] * 5
        logits[label] = 2.0
        row = {"label": label}
        row.update({f"logit_{class_id}": value for class_id, value in enumerate(logits)})
        rows.append(row)
    metrics = metrics_from_subject_rows(
        rows, split_name="val", num_classes=5
    )
    assert metrics["val/subject/macro_auroc_ovr"] == 1.0
    assert metrics["val/subject/balanced_accuracy"] == 1.0


def test_demographic_additive_logit_backpropagates_to_eye_and_metadata() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean", demographic_dim=16
    ).train()
    features = torch.randn(3, 4, 16)
    demographics = torch.randn(3, 16)
    output = model.classify_task_features(
        features, torch.ones(3, 4, dtype=torch.bool), demographics
    )
    torch.testing.assert_close(
        output["subject_logits"],
        output["eye_subject_logits"] + output["demographic_logits"],
    )
    output["subject_logits"].sum().backward()
    assert model.demographic_head.weight.grad is not None
    assert model.demographic_head.weight.grad.abs().sum().item() > 0
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in model.task_head.parameters()
        if parameter.requires_grad
    )


def test_bounded_demographic_additive_is_nested_and_alpha_limited() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        demographic_dim=16,
        demographic_fusion="bounded_additive_logit",
    ).train()
    output = model.classify_task_features(
        torch.randn(3, 4, 16),
        torch.ones(3, 4, dtype=torch.bool),
        torch.randn(3, 16),
    )
    assert 0.0 <= output["demographic_alpha"].item() <= 0.3
    assert output["demographic_alpha"].item() < 0.01
    torch.testing.assert_close(
        output["subject_logits"],
        output["eye_subject_logits"] + output["demographic_logits"],
    )
    output["subject_logits"].sum().backward()
    assert model.demographic_alpha_logit.grad is not None


@pytest.mark.parametrize(
    ("classifier_head", "hidden", "expected_parameters"),
    (("mlp", 32, 705), ("mlp", 128, 2817), ("linear", 32, 21)),
)
def test_cls_is_normalized_before_compact_demographics_are_concatenated(
    classifier_head: str, hidden: int, expected_parameters: int
) -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        demographic_dim=16,
        demographic_projection_dim=4,
        demographic_fusion="concat_task_cls",
        classifier_head=classifier_head,
        classifier_hidden=hidden,
    ).train()
    features = torch.randn(3, 4, 16, requires_grad=True)
    demographics = torch.randn(3, 16, requires_grad=True)
    output = model.classify_task_features(
        features, torch.ones(3, 4, dtype=torch.bool), demographics
    )
    assert model.demographic_projection.in_features == 16
    assert model.demographic_projection.out_features == 4
    assert model.task_feature_norm.normalized_shape == (16,)
    assert output["task_head_inputs"].shape == (3, 4, 20)
    with torch.no_grad():
        projected_demographics = model.demographic_activation(
            model.demographic_projection(demographics)
        )
        expected_inputs = torch.cat((
            model.task_feature_norm(features),
            projected_demographics.unsqueeze(1).expand(-1, 4, -1),
        ), dim=-1)
    torch.testing.assert_close(output["task_head_inputs"], expected_inputs)
    assert sum(parameter.numel() for parameter in model.task_head.parameters()) == expected_parameters
    output["subject_logits"].sum().backward()
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert model.demographic_projection.weight.grad is not None
    assert model.demographic_projection.weight.grad.abs().sum() > 0
    assert model.task_feature_norm.weight.grad is not None
    assert model.task_feature_norm.weight.grad.abs().sum() > 0


def test_raw_demographics_are_directly_appended_after_cls_normalization() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_mean",
        demographic_dim=16,
        demographic_projection_dim=0,
        demographic_fusion="concat_task_cls",
        classifier_hidden=32,
    ).train()
    features = torch.randn(3, 4, 16, requires_grad=True)
    demographics = torch.randn(3, 16, requires_grad=True)
    output = model.classify_task_features(
        features, torch.ones(3, 4, dtype=torch.bool), demographics
    )
    assert isinstance(model.demographic_projection, torch.nn.Identity)
    assert isinstance(model.demographic_activation, torch.nn.Identity)
    assert model.task_feature_norm.normalized_shape == (16,)
    assert output["task_head_inputs"].shape == (3, 4, 32)
    expected = torch.cat((
        model.task_feature_norm(features),
        demographics.unsqueeze(1).expand(-1, 4, -1),
    ), dim=-1)
    torch.testing.assert_close(output["task_head_inputs"], expected)
    output["subject_logits"].sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert demographics.grad is not None and demographics.grad.abs().sum() > 0


def test_shared_head_subject_loss_reaches_all_fusion_parameters() -> None:
    model = _tiny_mil(task_pooling="shared_head_softmax").train()
    output = model.classify_task_features(
        torch.randn(3, 4, 16), torch.ones(3, 4, dtype=torch.bool)
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["subject_logits"], torch.tensor([0.0, 1.0, 0.0])
    )
    loss.backward()
    assert model.task_weight_logits.grad is not None
    assert model.task_weight_logits.grad.abs().sum().item() > 0
    assert all(
        parameter.grad is not None
        for parameter in model.task_head.parameters()
        if parameter.requires_grad
    )


def test_concat_auxiliary_loss_reaches_subject_and_task_heads() -> None:
    model = _tiny_mil(task_pooling="concat_task_cls_aux").train()
    output = model.classify_task_features(
        torch.randn(3, 4, 16), torch.ones(3, 4, dtype=torch.bool)
    )
    labels = torch.tensor([0.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["subject_logits"], labels
    ) + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
        output["task_logits"], labels[:, None].expand_as(output["task_logits"])
    )
    loss.backward()
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for module in (model.subject_head, model.task_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_task_bottleneck_preserves_interactions_and_backpropagates() -> None:
    model = _tiny_mil(task_pooling="task_bottleneck_concat").train()
    features = torch.randn(3, 4, 16)
    output = model.classify_task_features(
        features, torch.ones(3, 4, dtype=torch.bool)
    )
    assert output["compact_task_features"].shape == (3, 4, 4)
    assert output["compact_features"].shape == (3, 16)
    assert output["subject_logits"].shape == (3,)
    output["subject_logits"].sum().backward()
    assert all(
        parameter.grad is not None
        for module in (model.task_projector, model.subject_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_mil_forward_has_one_logit_per_subject_and_no_unused_trainable_parameters() -> None:
    model = _tiny_mil().train()
    subjects, tasks, trials, time = 2, 4, 2, 3
    flat = subjects * tasks * trials
    output = model(
        stim_patches=torch.randn(flat, time, 4, 20),
        eye_patches=torch.randn(flat, time, 2, 4, 20),
        pad_mask=torch.zeros(flat, time, dtype=torch.bool),
        eye_nonmissing_frac=torch.ones(flat, time, 2),
        task_ids=torch.arange(tasks).view(1, tasks, 1).expand(subjects, tasks, trials).reshape(-1),
        num_subjects=subjects,
        trials_per_task=trials,
        task_present_mask=torch.ones(subjects, tasks, dtype=torch.bool),
    )
    assert output["subject_logits"].shape == (subjects,)
    assert output["concatenated_features"].shape == (subjects, tasks * 16)
    output["subject_logits"].sum().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_shared_logit_residual_forward_has_no_unused_trainable_parameters() -> None:
    model = _tiny_mil(
        task_pooling="shared_head_residual",
        trial_pooling="logit_mean",
        demographic_dim=16,
        demographic_fusion="late_residual",
    ).train()
    subjects, tasks, trials, time = 2, 4, 2, 3
    flat = subjects * tasks * trials
    output = model(
        stim_patches=torch.randn(flat, time, 4, 20),
        eye_patches=torch.randn(flat, time, 2, 4, 20),
        pad_mask=torch.zeros(flat, time, dtype=torch.bool),
        eye_nonmissing_frac=torch.ones(flat, time, 2),
        task_ids=torch.arange(tasks).view(1, tasks, 1).expand(
            subjects, tasks, trials
        ).reshape(-1),
        num_subjects=subjects,
        trials_per_task=trials,
        task_present_mask=torch.ones(subjects, tasks, dtype=torch.bool),
        trial_slot_mask=torch.ones(subjects, tasks, trials, dtype=torch.bool),
        demographic_features=torch.randn(subjects, 16),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["subject_logits"], torch.tensor([0.0, 1.0])
    )
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_subject_feature_mixup_backpropagates_through_encoder_and_head() -> None:
    model = _tiny_mil().train()
    subjects, tasks, trials, time = 3, 4, 2, 3
    flat = subjects * tasks * trials
    permutation = torch.tensor([1, 2, 0])
    output = model(
        stim_patches=torch.randn(flat, time, 4, 20),
        eye_patches=torch.randn(flat, time, 2, 4, 20),
        pad_mask=torch.zeros(flat, time, dtype=torch.bool),
        eye_nonmissing_frac=torch.ones(flat, time, 2),
        task_ids=torch.arange(tasks).view(1, tasks, 1).expand(
            subjects, tasks, trials
        ).reshape(-1),
        num_subjects=subjects,
        trials_per_task=trials,
        task_present_mask=torch.ones(subjects, tasks, dtype=torch.bool),
        subject_mixup_permutation=permutation,
        subject_mixup_lambda=0.7,
    )
    assert output["subject_logits"].shape == (subjects,)
    output["subject_logits"].sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.subject_head.parameters()
        if parameter.requires_grad
    )
    assert any(
        parameter.grad is not None
        for parameter in model.bert.transformer[-1].parameters()
        if parameter.requires_grad
    )
    assert all(not parameter.requires_grad for parameter in model.bert.embed.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.bert.transformer[0].parameters()
    )
