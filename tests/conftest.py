from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from eyemae.config import load_config
from fixtures.make_synthetic_npz import make_synthetic_dataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def debug_cfg() -> dict:
    return load_config(str(ROOT / "configs" / "debug.yaml"))


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic_npz"
    make_synthetic_dataset(out, num_trials=64)
    return out


def write_manifest(root: Path, subject: str, rows: Iterable[dict[str, str]]) -> None:
    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "info_id",
        "info_id_with_suffix",
        "task",
        "source_suffix",
        "relative_npz",
        "trial_npz",
        "n_samples",
        "left_final_keep",
        "right_final_keep",
    ]
    with open(subject_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_npz(path: Path, n: int = 40, left_area: float = 10.0, right_area: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y = np.linspace(1.0, -1.0, n, dtype=np.float32)
    gaze = np.stack(
        [
            x,
            y,
            np.full(n, left_area, dtype=np.float32),
            x + 0.5,
            y - 0.5,
            np.full(n, right_area, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    stimulus = np.stack(
        [
            np.full(n, 3.0, dtype=np.float32),
            np.full(n, -2.0, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ],
        axis=1,
    )
    np.savez_compressed(path, gaze=gaze, stimulus=stimulus)
