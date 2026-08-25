from __future__ import annotations

from pathlib import Path

import numpy as np

from eyemae.config import load_config
from eyemae.compute_area_stats import _valid_log_area, compute_area_stats


def _write(path: Path, area: float, subject: str) -> None:
    t = 40
    eye = np.zeros((t, 8), dtype=np.float32)
    eye[:, 2] = area
    eye[:, 6] = area
    np.savez(path, eye=eye, task_id=np.array(0), fix_on=np.zeros(t, dtype=np.float32), stim=np.zeros((t, 3), dtype=np.float32), subject_id=np.array(subject), trial_id=np.array(path.stem))


def test_area_stats_train_only_and_fallback(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "s001D").mkdir(parents=True)
    (data / "s002D").mkdir(parents=True)
    _write(data / "s001D" / "train.npz", 1000, "s001D")
    _write(data / "s002D" / "val.npz", 1_000_000, "s002D")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "pretrain_train.txt").write_text("s001D/train.npz\n", encoding="utf-8")
    (splits / "pretrain_val.txt").write_text("s002D/val.npz\n", encoding="utf-8")
    (splits / "pretrain_test.txt").write_text("", encoding="utf-8")
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(data)
    cfg["data"]["pretrain_train_split"] = str(splits / "pretrain_train.txt")
    cfg["data"]["pretrain_val_split"] = str(splits / "pretrain_val.txt")
    cfg["data"]["pretrain_test_split"] = str(splits / "pretrain_test.txt")
    cfg["area"]["stats_path"] = str(tmp_path / "stats.json")
    stats = compute_area_stats(cfg)
    assert "s001D" in stats["subjects"]
    assert "s002D" not in stats["subjects"]
    assert np.isfinite(stats["global"]["mad"])
    assert set(stats["global_by_eye"]) == {"left", "right"}
    assert set(stats["subjects"]["s001D"]["eyes"]) == {"left", "right"}


def test_area_stats_enforces_suffix_eye_availability_by_default(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "s001L").mkdir(parents=True)
    _write(data / "s001L" / "train.npz", 1000, "s001L")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "pretrain_train.txt").write_text("s001L/train.npz\n", encoding="utf-8")
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(data)
    cfg["data"]["pretrain_train_split"] = str(splits / "pretrain_train.txt")
    cfg["area"]["stats_path"] = str(tmp_path / "stats.json")

    stats = compute_area_stats(cfg)
    assert stats["subjects"]["s001L"]["num_valid_frames"] == 40

    cfg["data"]["enforce_suffix_eye_availability"] = False
    cfg["area"]["stats_path"] = str(tmp_path / "stats_not_enforced.json")
    stats_not_enforced = compute_area_stats(cfg)
    assert stats_not_enforced["subjects"]["s001L"]["num_valid_frames"] == 80


def test_area_stats_trial_final_keep_overrides_decodable_d_suffix() -> None:
    cfg = load_config("configs/debug.yaml")
    eye = np.zeros((4, 8), dtype=np.float32)
    eye[:, 2] = 100.0
    eye[:, 6] = 200.0
    trial = {
        "subject_id": "s001D",
        "left_eye_available": False,
        "right_eye_available": True,
        "eye": eye,
    }
    values = _valid_log_area(trial, cfg)
    assert values["left"].size == 0
    np.testing.assert_allclose(values["right"], np.log1p(200.0))


def test_area_stats_excludes_nonfinite_and_nonpositive_area() -> None:
    cfg = load_config("configs/debug.yaml")
    eye = np.zeros((5, 8), dtype=np.float32)
    eye[:, 2] = [100.0, 0.0, -1.0, np.nan, np.inf]
    eye[:, 6] = 200.0
    trial = {"subject_id": "s001D", "eye": eye}

    values = _valid_log_area(trial, cfg)

    np.testing.assert_allclose(values["left"], np.log1p([100.0]))
    np.testing.assert_allclose(values["right"], np.log1p(np.full(5, 200.0)))


def test_area_stats_keeps_distinct_left_right_statistics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "s001D").mkdir(parents=True)
    path = data / "s001D" / "train.npz"
    eye = np.zeros((40, 8), dtype=np.float32)
    eye[:, 2] = np.arange(100, 140, dtype=np.float32)
    eye[:, 6] = np.arange(300, 340, dtype=np.float32)
    np.savez(
        path,
        eye=eye,
        task_id=np.array(0),
        fix_on=np.zeros(40, dtype=np.float32),
        stim=np.zeros((40, 3), dtype=np.float32),
        subject_id=np.array("s001D"),
        trial_id=np.array("train"),
    )
    split = tmp_path / "train.txt"
    split.write_text("s001D/train.npz\n", encoding="utf-8")
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(data)
    cfg["data"]["pretrain_train_split"] = str(split)
    cfg["area"]["stats_path"] = str(tmp_path / "stats.json")

    stats = compute_area_stats(cfg)

    subject = stats["subjects"]["s001D"]
    assert subject["eyes"]["left"]["median"] == np.median(np.log1p(np.arange(100, 140)))
    assert subject["eyes"]["right"]["median"] == np.median(np.log1p(np.arange(300, 340)))
    assert subject["eyes"]["left"]["num_valid_frames"] == 40
    assert subject["eyes"]["right"]["num_valid_frames"] == 40
