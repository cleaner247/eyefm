from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wait_v5_tokenizer_then_select_mask_and_retrain import (  # noqa: E402
    latest_tokenizer_checkpoint,
)
from run_eyevq_v4_optimal_search import bert_passes_gate  # noqa: E402


def test_latest_tokenizer_checkpoint_uses_numeric_step_order(tmp_path: Path):
    for name in ("ckpt_step002500.pt", "ckpt_step010000.pt", "ckpt_step007500.pt"):
        (tmp_path / name).touch()
    (tmp_path / "ckpt_best.pt").touch()
    assert latest_tokenizer_checkpoint(tmp_path).name == "ckpt_step010000.pt"


def test_latest_tokenizer_checkpoint_returns_none_when_empty(tmp_path: Path):
    assert latest_tokenizer_checkpoint(tmp_path) is None


def test_factorized_bert_gate_accepts_valid_ce_when_legacy_accuracy_is_nan():
    checkpoint = {
        "val_metrics": {
            "val/loss": 2.4,
            "val/acc": float("nan"),
            "val/per_dim_ce": [0.66, 0.53, 0.64, 0.57],
        }
    }
    assert bert_passes_gate(checkpoint)


def test_factorized_bert_gate_rejects_missing_or_inconsistent_ce():
    missing = {"val_metrics": {"val/loss": 2.4, "val/acc": float("nan")}}
    inconsistent = {
        "val_metrics": {
            "val/loss": 2.4,
            "val/acc": float("nan"),
            "val/per_dim_ce": [0.1, 0.1, 0.1, 0.1],
        }
    }
    assert not bert_passes_gate(missing)
    assert not bert_passes_gate(inconsistent)
