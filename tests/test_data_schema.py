from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eyemae.config import load_config
from eyemae.data import load_npz_trial, read_split_file


def _write_npz(path: Path, *, eye_shape=(40, 8), task_id=0, bad_label=False, bad_fix=False, bad_stim=False) -> None:
    eye = np.zeros(eye_shape, dtype=np.float32)
    if eye_shape[1] == 8:
        eye[:, 2] = 1000
        eye[:, 6] = 1000
        eye[:, 3] = 0
        eye[:, 7] = 3 if bad_label else 0
    t = eye_shape[0]
    fix_on = np.zeros(t + 1 if bad_fix else t, dtype=np.float32)
    stim = np.zeros((t, 4 if bad_stim else 3), dtype=np.float32)
    np.savez(path, eye=eye, task_id=np.array(task_id), fix_on=fix_on, stim=stim, subject_id=np.array("s001D"), trial_id=np.array("trial"))


def test_validate_schema_errors(tmp_path: Path) -> None:
    cfg = load_config("configs/debug.yaml")
    for kwargs, text in [
        ({"eye_shape": (20, 7)}, "eye"),
        ({"bad_fix": True}, "fix_on"),
        ({"bad_stim": True}, "stim"),
        ({"task_id": 9}, "task_id"),
        ({"bad_label": True}, "labels"),
    ]:
        path = tmp_path / "trial.npz"
        _write_npz(path, **kwargs)
        with pytest.raises(ValueError, match=text):
            load_npz_trial(path, tmp_path, cfg)


def test_split_txt_ignores_comments(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    path.write_text("\n# comment\na.npz\n\n b.npz \n", encoding="utf-8")
    assert read_split_file(path) == ["a.npz", "b.npz"]


def test_nan_policy_can_mark_missing(tmp_path: Path) -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["nan_policy"] = "mark_missing"
    path = tmp_path / "trial.npz"
    _write_npz(path)
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files}
    eye = payload["eye"].copy()
    eye[0, 0] = np.nan
    payload["eye"] = eye
    np.savez(path, **payload)
    trial = load_npz_trial(path, tmp_path, cfg)
    assert trial["eye"][0, 3] == cfg["label"]["missing_value"]
    assert trial["eye"][0, 0] == 0.0
