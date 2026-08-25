import copy

import pytest

from eyemae.eyevq.pretrain.train import get_lr


def _handoff_horizon(train_cfg: dict, checkpoint_step: int) -> dict:
    result = copy.deepcopy(train_cfg)
    schedule = int(result.get("lr_schedule_total_steps", result["total_steps"]))
    if checkpoint_step > schedule:
        raise ValueError("checkpoint exceeds schedule")
    result["total_steps"] = checkpoint_step
    result["lr_schedule_total_steps"] = schedule
    return result


def test_40k_checkpoint_preserves_original_50k_lr_trajectory():
    cfg = _handoff_horizon({"total_steps": 50_000}, 40_000)
    assert cfg == {"total_steps": 40_000, "lr_schedule_total_steps": 50_000}
    reproduced = get_lr(39_999, 2_000, cfg["lr_schedule_total_steps"], 5e-4, 5e-5)
    original = get_lr(39_999, 2_000, 50_000, 5e-4, 5e-5)
    incorrectly_retimed = get_lr(39_999, 2_000, 40_000, 5e-4, 5e-5)
    assert reproduced == pytest.approx(original)
    assert abs(reproduced - incorrectly_retimed) > 4e-5


def test_50k_final_checkpoint_uses_50k_schedule():
    cfg = _handoff_horizon({"total_steps": 50_000}, 50_000)
    assert cfg == {"total_steps": 50_000, "lr_schedule_total_steps": 50_000}
