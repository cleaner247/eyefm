from __future__ import annotations

import numpy as np
import pytest

from eyemae.config import load_config
from eyemae.preprocess import preprocess_trial, validate_area_normalization_contract


def _trial(subject_id: str, labels_left, labels_right, x=0.0, y=0.0, area=1000.0):
    t = len(labels_left)
    eye = np.zeros((t, 8), dtype=np.float32)
    eye[:, 0] = x
    eye[:, 1] = y
    eye[:, 2] = area
    eye[:, 3] = labels_left
    eye[:, 4] = x
    eye[:, 5] = y
    eye[:, 6] = area
    eye[:, 7] = labels_right
    return {
        "eye": eye,
        "task_id": np.array(0),
        "fix_on": np.zeros(t, dtype=np.float32),
        "stim": np.zeros((t, 3), dtype=np.float32),
        "subject_id": subject_id,
        "trial_id": "t",
    }


def test_missing_blink_and_subject_suffix_rules() -> None:
    cfg = load_config("tests/fixtures/preprocessing.yaml")
    stats = {"global": {"median": np.log1p(1000), "mad": 1.0}, "subjects": {}}
    p = preprocess_trial(_trial("s001D", [2, 1, 0], [0, 0, 0], x=0, y=0, area=1000), cfg, stats)
    assert p["quality"][0, 0, 0] == 1
    assert p["content"][0, 0, 3] == 0
    assert p["content"][0, 0, :3].sum() == 0
    assert p["quality"][1, 0, 0] == 0
    assert p["content"][1, 0, 3] == 1
    assert p["content"][1, 0, :3].sum() == 0
    assert p["quality"][2, 0, 0] == 0
    assert p["content"][2, 0, 0] == 0
    assert p["content"][2, 0, 1] == 0

    left_only = preprocess_trial(_trial("s001L", [0, 0], [0, 0]), cfg, stats)
    assert left_only["quality"][:, 1, 0].all()
    right_only = preprocess_trial(_trial("s001R", [0, 0], [0, 0]), cfg, stats)
    assert right_only["quality"][:, 0, 0].all()
    cfg_not_enforced = load_config("tests/fixtures/preprocessing.yaml")
    cfg_not_enforced["data"]["enforce_suffix_eye_availability"] = False
    left_only_not_enforced = preprocess_trial(_trial("s001L", [0, 0], [0, 0]), cfg_not_enforced, stats)
    assert not left_only_not_enforced["quality"].any()
    right_only_not_enforced = preprocess_trial(_trial("s001R", [0, 0], [0, 0]), cfg_not_enforced, stats)
    assert not right_only_not_enforced["quality"].any()
    both = preprocess_trial(_trial("s001D", [0, 0], [0, 0], area=0), cfg, stats)
    assert both["quality"].all()
    assert not both["content"].any()


def test_per_eye_area_normalization_uses_distinct_eye_stats() -> None:
    cfg = load_config("tests/fixtures/preprocessing.yaml")
    cfg["area"].update({
        "use_log1p": False,
        "per_eye": True,
        "mad_scale": 1.0,
        "mad_floor": 0.0,
        "min_subject_valid_frames": 2,
    })
    stats = {
        "global": {"median": 0.0, "mad": 1.0},
        "global_by_eye": {
            "left": {"median": 10.0, "mad": 2.0, "num_valid_frames": 100},
            "right": {"median": 20.0, "mad": 4.0, "num_valid_frames": 100},
        },
        "subjects": {
            "s001D": {
                "median": 150.0,
                "mad": 50.0,
                "num_valid_frames": 8,
                "eyes": {
                    "left": {"median": 100.0, "mad": 10.0, "num_valid_frames": 4},
                    "right": {"median": 200.0, "mad": 20.0, "num_valid_frames": 4},
                },
            }
        },
    }
    trial = _trial("s001D", [0, 0], [0, 0], area=0.0)
    trial["eye"][:, 2] = 110.0
    trial["eye"][:, 6] = 220.0

    processed = preprocess_trial(trial, cfg, stats)

    np.testing.assert_allclose(processed["content"][:, 0, 2], 1.0, atol=2e-7)
    np.testing.assert_allclose(processed["content"][:, 1, 2], 1.0, atol=2e-7)


def test_sparse_eye_falls_back_to_subject_pooled_stats() -> None:
    cfg = load_config("tests/fixtures/preprocessing.yaml")
    cfg["area"].update({
        "use_log1p": False,
        "per_eye": True,
        "mad_scale": 1.0,
        "mad_floor": 0.0,
        "min_subject_valid_frames": 10,
    })
    stats = {
        "global": {"median": 0.0, "mad": 1.0},
        "global_by_eye": {
            "left": {"median": 10.0, "mad": 2.0, "num_valid_frames": 100},
            "right": {"median": 20.0, "mad": 4.0, "num_valid_frames": 100},
        },
        "subjects": {
            "s001D": {
                "median": 100.0,
                "mad": 20.0,
                "num_valid_frames": 20,
                "eyes": {
                    "left": {"median": 90.0, "mad": 10.0, "num_valid_frames": 2},
                    "right": {"median": 110.0, "mad": 10.0, "num_valid_frames": 18},
                },
            }
        },
    }
    trial = _trial("s001D", [0, 0], [0, 0], area=120.0)

    processed = preprocess_trial(trial, cfg, stats)

    np.testing.assert_allclose(processed["content"][:, 0, 2], 1.0, atol=2e-7)
    np.testing.assert_allclose(processed["content"][:, 1, 2], 1.0, atol=2e-7)


def test_per_eye_contract_rejects_pooled_statistics() -> None:
    cfg = {
        "require_contract": True,
        "use_log1p": False,
        "per_eye": True,
        "mad_scale": 1.4826,
        "mad_floor": 32.0,
        "clip": 5.0,
    }
    stats = {
        "global": {"median": 100.0, "mad": 20.0},
        "global_by_eye": {
            "left": {"median": 90.0, "mad": 18.0},
            "right": {"median": 110.0, "mad": 22.0},
        },
        "subjects": {
            "s001D": {
                "eyes": {
                    "left": {"median": 90.0, "mad": 18.0},
                    "right": {"median": 110.0, "mad": 22.0},
                }
            }
        },
        "source": {
            "invalid_eye_policy": "trial_final_keep_then_frame_qc",
            "normalization": {
                "transform": "identity",
                "center": "per_subject_eye_median",
                "scale": "per_subject_eye_raw_mad",
                "fallback": "subject_pooled_then_global_eye_then_global_pooled",
                "mad_to_sigma": 1.4826,
                "mad_floor": 32.0,
                "clip": 5.0,
            },
        },
    }
    validate_area_normalization_contract(stats, cfg)

    stats["source"]["normalization"]["center"] = "per_subject_median"
    with pytest.raises(ValueError, match="center"):
        validate_area_normalization_contract(stats, cfg)
