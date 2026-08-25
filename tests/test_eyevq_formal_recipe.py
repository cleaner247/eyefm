from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from eyemae.eyevq.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "configs/eyevq/final"


def load(name: str) -> dict:
    return yaml.safe_load((FINAL / name).read_text(encoding="utf-8"))


def test_formal_attention_and_mask_are_explicit() -> None:
    tokenizer = load("tokenizer.yaml")
    bert = load("bert.yaml")
    assert tokenizer["model"]["architecture"] == "joint"
    for cfg in (tokenizer, bert):
        assert cfg["attention"] == {
            "stim_isolated": True,
            "stim_attend_cls": False,
        }
    assert bert["bert"]["factorized_fsq"] is True
    assert bert["mask"] == {
        "mode": "paired_span",
        "eye_masking_ratio": 0.60,
        "span_min_patches": 1,
        "span_max_patches": 5,
        "span_length_distribution": "uniform",
    }


def test_formal_recipe_uses_only_validation_supported_task_bags() -> None:
    recipe = load("recipe.yaml")
    mci = load("mci.yaml")
    pd5 = load("pd5.yaml")
    assert recipe["selection_policy"] == "validation_only"
    assert recipe["downstream"]["test_used_for_selection"] is False
    assert mci["mil"]["trials_per_task"] == 16
    assert mci["mil"]["evaluation_min_trials_per_task"] == 4
    assert pd5["mil"]["trials_per_task"] == 4
    assert pd5["mil"]["evaluation_min_trials_per_task"] == 4


def test_formal_fsq_has_no_fake_commitment_objective() -> None:
    tokenizer = load("tokenizer.yaml")
    assert "commitment_beta" not in tokenizer["vq"]
    assert tokenizer["loss"]["eye_commit_group_weight"] == pytest.approx(0.0)


def test_quality_gate_fails_closed(tmp_path: Path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.recipe = load("recipe.yaml")
    checkpoint = tmp_path / "weak_tokenizer.pt"
    torch.save(
        {
            "val_metrics": {
                "val/L_eye": 1.0,
                "val/L_feat": 1.0,
                "val/active_codes": 1,
                "val/code_perplexity": 1.0,
                "val/top1_code_frequency": 1.0,
            }
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="failed formal quality gate"):
        pipeline._check_quality_gate("tokenizer", checkpoint)
