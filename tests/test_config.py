from __future__ import annotations

import pytest

from eyemae.config import load_config, validate_config


def test_debug_and_main_configs_are_complete() -> None:
    for path in ("configs/debug.yaml", "configs/eyemae_cnn_512_12l.yaml"):
        cfg = load_config(path)
        validate_config(cfg, require_splits=False)
        assert cfg["input"]["content_dim"] == 4
        assert cfg["input"]["quality_dim"] == 1
        assert cfg["stim"]["stim_dim"] == cfg["input"]["stim_dim"] == 4
        assert cfg["model"]["sequence_format"] == "stim_eye_triplet_no_cls"
        assert cfg["model"]["pretrain_style"] == "bert_masked_reconstruction"


def test_validate_config_rejects_bad_values() -> None:
    cfg = load_config("configs/debug.yaml")
    cfg["label"]["blink_value"] = cfg["label"]["missing_value"]
    with pytest.raises(ValueError, match="distinct"):
        validate_config(cfg, require_splits=False)
