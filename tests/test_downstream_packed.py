from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from eyemae.config import load_config
from eyemae.data import filter_packed_rows_with_usable_eye, packed_row_has_usable_eye
from eyemae.downstream_data import PackedDownstreamDataset, collate_downstream_trials


def test_packed_downstream_patch_count_respects_configured_patch_size() -> None:
    dataset = object.__new__(PackedDownstreamDataset)
    dataset.rows = [{"frame_length": "80", "num_patches_20ms": "4"}]
    dataset.cfg = {"patch": {"samples": 40}}
    assert dataset.get_num_patches(0) == 2
    dataset.cfg = {"patch": {"samples": 20}}
    assert dataset.get_num_patches(0) == 4


def _write_packed_fixture(root: Path, rows: list[dict[str, str]]) -> Path:
    shard = root / "shards" / "shard_000000"
    shard.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([40 for _ in rows], dtype=np.int64)
    offsets = np.asarray([i * 40 for i in range(len(rows))], dtype=np.int64)
    total = int(lengths.sum())
    x = np.zeros((total, 10), dtype=np.float32)
    x[:, 0] = 1.0
    x[:, 2] = 100.0
    x[:, 3] = 1.0
    x[:, 5] = 120.0
    x[:, 8] = 1.0
    y = np.zeros((total, 2), dtype=np.int8)
    np.save(shard / "X_data.npy", x)
    np.save(shard / "y_frame.npy", y)
    np.save(shard / "X_offsets.npy", offsets)
    np.save(shard / "X_lengths.npy", lengths)
    for i, row in enumerate(rows):
        row.update(
            {
                "shard_id": "shard_000000",
                "local_trial_index": str(i),
                "frame_offset": str(int(offsets[i])),
                "frame_length": str(int(lengths[i])),
                "num_patches_20ms": "2",
                "task_id": "0",
                "source_suffix": "D",
                "left_final_keep": row.get("left_final_keep", "True"),
                "right_final_keep": row.get("right_final_keep", "True"),
            }
        )
    index = root / "index.csv"
    fieldnames = [
        "global_trial_id",
        "shard_id",
        "local_trial_index",
        "frame_offset",
        "frame_length",
        "num_patches_20ms",
        "ml_subject_id",
        "subject",
        "trial_id",
        "task_id",
        "source_suffix",
        "left_final_keep",
        "right_final_keep",
        "health_label",
        "pd_disease_label",
        "class_label",
    ]
    with index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return index


def _cfg(tmp_path: Path, label_cfg: dict) -> dict:
    cfg = load_config("tests/fixtures/preprocessing.yaml")
    cfg["data"]["format"] = "packed_mmap"
    cfg["data"]["data_dir"] = str(tmp_path)
    cfg["data"]["max_open_shards_per_worker"] = 2
    cfg["data"]["validate_offsets"] = True
    cfg["area"]["stats_path"] = str(tmp_path / "missing_area_stats.json")
    cfg["label"].update(label_cfg)
    return cfg


def test_packed_downstream_binary_dataset(tmp_path: Path) -> None:
    index = _write_packed_fixture(
        tmp_path,
        [
            {"global_trial_id": "g0", "ml_subject_id": "s0", "subject": "s0", "trial_id": "t0", "health_label": "0", "pd_disease_label": "-1"},
            {"global_trial_id": "g1", "ml_subject_id": "s1", "subject": "s1", "trial_id": "t1", "health_label": "1", "pd_disease_label": "-1"},
        ],
    )
    cfg = _cfg(tmp_path, {"type": "binary", "task_name": "ad_binary"})
    dataset = PackedDownstreamDataset(tmp_path, index, cfg)
    item = dataset[1]
    assert item["label"] == 1
    assert item["ml_subject_id"] == "s1"
    assert item["global_trial_id"] == "g1"
    batch = collate_downstream_trials([dataset[0], item])
    assert batch["label"].shape == (2,)
    assert batch["ml_subject_id"] == ["s0", "s1"]


def test_packed_downstream_filters_trials_with_no_usable_eye(tmp_path: Path) -> None:
    index = _write_packed_fixture(
        tmp_path,
        [
            {"global_trial_id": "good", "ml_subject_id": "s0", "subject": "s0", "trial_id": "t0", "health_label": "0", "pd_disease_label": "-1"},
            {"global_trial_id": "bad", "ml_subject_id": "s1", "subject": "s1", "trial_id": "t1", "health_label": "1", "pd_disease_label": "-1", "left_final_keep": "False", "right_final_keep": "False"},
        ],
    )
    cfg = _cfg(tmp_path, {"type": "binary", "task_name": "mci_binary"})
    cfg["data"]["require_any_eye_keep"] = True
    dataset = PackedDownstreamDataset(tmp_path, index, cfg)
    assert len(dataset) == 1
    assert dataset.rows[0]["global_trial_id"] == "good"
    assert dataset.num_rows_before_eye_filter == 2
    assert [row["global_trial_id"] for row in dataset.excluded_no_usable_eye_rows] == ["bad"]


def test_shared_packed_eye_filter_uses_keep_flags_and_suffix_fallback() -> None:
    rows = [
        {"global_trial_id": "both", "source_suffix": "D", "left_final_keep": "True", "right_final_keep": "True"},
        {"global_trial_id": "left", "source_suffix": "L", "left_final_keep": "", "right_final_keep": ""},
        {"global_trial_id": "bad", "source_suffix": "D", "left_final_keep": "False", "right_final_keep": "False"},
    ]
    assert packed_row_has_usable_eye(rows[0])
    assert packed_row_has_usable_eye(rows[1])
    assert not packed_row_has_usable_eye(rows[2])
    kept, excluded = filter_packed_rows_with_usable_eye(rows)
    assert [row["global_trial_id"] for row in kept] == ["both", "left"]
    assert [row["global_trial_id"] for row in excluded] == ["bad"]


def test_packed_downstream_pd_multiclass_labels(tmp_path: Path) -> None:
    index = _write_packed_fixture(
        tmp_path,
        [
            {"global_trial_id": "g0", "ml_subject_id": "s0", "subject": "s0", "trial_id": "t0", "health_label": "0", "pd_disease_label": "-1"},
            {"global_trial_id": "g1", "ml_subject_id": "s1", "subject": "s1", "trial_id": "t1", "health_label": "1", "pd_disease_label": "2"},
        ],
    )
    cfg = _cfg(tmp_path, {"type": "multiclass", "task_name": "pd_related_5class", "num_classes": 5})
    dataset = PackedDownstreamDataset(tmp_path, index, cfg)
    assert dataset[0]["label"] == 0
    assert dataset[1]["label"] == 3
    batch = collate_downstream_trials([dataset[0], dataset[1]])
    assert str(batch["label"].dtype) == "torch.int64"


def test_packed_downstream_pd3_scheme_b_remap(tmp_path: Path) -> None:
    rows = [
        {
            "global_trial_id": f"g{source_class}",
            "ml_subject_id": f"s{source_class}",
            "subject": f"s{source_class}",
            "trial_id": f"t{source_class}",
            "health_label": "0" if source_class == 0 else "1",
            "pd_disease_label": "-1" if source_class == 0 else str(source_class - 1),
        }
        for source_class in range(5)
    ]
    index = _write_packed_fixture(tmp_path, rows)
    cfg = _cfg(
        tmp_path,
        {
            "type": "multiclass",
            "task_name": "pd_related_3class_scheme_b",
            "num_classes": 3,
            "class_names": ["control", "parkinson_spectrum", "tremor_spectrum"],
            "remap": {"0": 0, "1": 1, "2": 2, "3": 2, "4": 1},
        },
    )
    dataset = PackedDownstreamDataset(tmp_path, index, cfg)
    assert dataset.labels == [0, 1, 2, 2, 1]


def test_packed_downstream_direct_multiclass_label(tmp_path: Path) -> None:
    index = _write_packed_fixture(
        tmp_path,
        [
            {
                "global_trial_id": f"g{class_id}",
                "ml_subject_id": f"s{class_id}",
                "subject": f"s{class_id}",
                "trial_id": f"t{class_id}",
                "health_label": "",
                "pd_disease_label": "",
                "class_label": str(class_id),
            }
            for class_id in range(3)
        ],
    )
    cfg = _cfg(
        tmp_path,
        {
            "type": "multiclass",
            "task_name": "depression",
            "source_column": "class_label",
            "num_classes": 3,
            "class_names": ["healthy", "high_risk", "mdd"],
        },
    )
    dataset = PackedDownstreamDataset(tmp_path, index, cfg)
    assert dataset.labels == [0, 1, 2]
