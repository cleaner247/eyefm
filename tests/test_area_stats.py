from __future__ import annotations

from pathlib import Path

import numpy as np

from eyemae.config import load_config
from eyemae.compute_area_stats import compute_area_stats


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
