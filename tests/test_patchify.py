from __future__ import annotations

import numpy as np

from eyemae.config import load_config
from eyemae.patching import patchify_preprocessed_trial


def _processed(t: int):
    content = np.zeros((t, 2, 4), dtype=np.float32)
    quality = np.zeros((t, 2, 1), dtype=np.float32)
    stim = np.zeros((t, 4), dtype=np.float32)
    return {"content": content, "quality": quality, "stim": stim, "task_id": 0, "subject_id": "sD", "trial_id": "t"}


def test_patch_counts_and_shapes() -> None:
    cfg = load_config("configs/debug.yaml")
    assert patchify_preprocessed_trial(_processed(1000), cfg)["content"].shape == (50, 2, 20, 4)
    assert patchify_preprocessed_trial(_processed(1025), cfg)["content"].shape[0] == 51
    assert patchify_preprocessed_trial(_processed(19), cfg) is None


def test_nonmissing_fraction_and_validity() -> None:
    cfg = load_config("configs/debug.yaml")
    trial = _processed(40)
    trial["quality"][:20, 0, 0] = 1.0
    patched = patchify_preprocessed_trial(trial, cfg)
    assert patched["quality"].shape == (2, 2, 20, 1)
    assert patched["stim"].shape == (2, 20, 4)
    assert patched["eye_nonmissing_frac"][0, 0] == 0.0
    assert not patched["eye_token_valid"][0, 0]
    assert patched["eye_nonmissing_frac"][1, 0] == 1.0
