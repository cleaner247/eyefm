from __future__ import annotations

import numpy as np

from eyemae.config import load_config
from eyemae.preprocess import preprocess_trial


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
    cfg = load_config("configs/debug.yaml")
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
    both = preprocess_trial(_trial("s001D", [0, 0], [0, 0], area=0), cfg, stats)
    assert not both["quality"].any()
