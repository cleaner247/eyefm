from __future__ import annotations

from pathlib import Path

from eyemae.config import load_config
from eyemae.data import get_split_subject_key
from eyemae.make_splits import make_splits


def test_trial_random_split_outputs(synthetic_dir: Path, tmp_path: Path) -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(synthetic_dir)
    cfg["split"]["out_dir"] = str(tmp_path / "splits")
    summary = make_splits(cfg)
    assert (tmp_path / "splits" / "pretrain_train.txt").exists()
    assert (tmp_path / "splits" / "pretrain_val.txt").exists()
    assert (tmp_path / "splits" / "pretrain_test.txt").exists()
    assert summary["task_counts"]["train"]
    assert summary["eye_availability_counts"]["train"]


def test_subject_heldout_base_subject_not_crossing(synthetic_dir: Path, tmp_path: Path) -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["data"]["data_dir"] = str(synthetic_dir)
    cfg["split"].update({"strategy": "subject_heldout", "group_by_base_subject_id": True, "out_dir": str(tmp_path / "splits")})
    make_splits(cfg)
    seen = {}
    for name in ("train", "val", "test"):
        rows = (tmp_path / "splits" / f"pretrain_{name}.txt").read_text(encoding="utf-8").splitlines()
        for row in rows:
            subject = Path(row).parts[0]
            base = get_split_subject_key(subject, True)
            assert base not in seen or seen[base] == name
            seen[base] = name
