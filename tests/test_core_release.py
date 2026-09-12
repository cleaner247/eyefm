from pathlib import Path
import sys

import pytest
import torch
import yaml

from eyemae.eyevq.artifacts import checkpoint_has_identity, dataset_dependency_paths
from eyemae.eyevq.config import build_bert, build_tokenizer
from eyemae.eyevq.pipeline import Pipeline
from eyemae.eyevq.pretrain.masking import generate_eyemae_mask_paired_fixed_blocks


FINAL = Path(__file__).resolve().parents[1] / "configs/eyevq/final"


def config(name):
    return yaml.safe_load((FINAL / f"{name}.yaml").read_text())


@pytest.mark.parametrize("task", ["mci", "pd3", "ad", "detox", "epilepsy", "migraine"])
def test_all_published_tasks_share_full_finetune_reference(task):
    cfg = config(task)
    assert cfg["model"]["freeze_embedding"] is False
    assert cfg["model"]["freeze_bottom_layers"] == 0
    assert cfg["model"]["bert_checkpoint"] == "outputs/eyevq/final/bert/ckpt_final.pt"
    assert cfg["mil"]["trials_per_task"] == 16
    assert cfg["mil"]["subjects_per_gpu"] == 4
    assert cfg["train"]["periodic_test_every_epochs"] == 0
    assert cfg["train"]["epochs"] == (15 if task == "pd3" else 10)
    assert cfg["train"]["encoder_lr"] == pytest.approx(5e-5)
    if task == "pd3":
        assert cfg["label"]["type"] == "hierarchical_pd3"
        assert cfg["train"]["hierarchical_head_weighting"] == "equal"


def test_pipeline_records_separate_world_sizes(tmp_path):
    pipeline = Pipeline(tmp_path, Path(sys.prefix), 4)
    assert pipeline.tokenizer_cfg["reproducibility"]["world_size"] == 4
    assert pipeline.downstream_cfgs["mci"]["reproducibility"]["world_size"] == 1
    assert "--nproc_per_node=1" in pipeline._ddp("module", nproc=1)
    assert "--nproc_per_node=4" in pipeline._ddp("module")


def test_test_index_is_part_of_downstream_identity(tmp_path):
    cfg = {"train": {}, "data": {"data_dir": str(tmp_path), "test_index": "test.csv"}}
    assert dataset_dependency_paths(cfg)["test_index"] == tmp_path / "test.csv"


def test_final_checkpoint_must_reach_expected_step(tmp_path):
    path = tmp_path / "ckpt_final.pt"
    identity = {"stage": "bert"}
    torch.save({"run_identity": identity, "step": 5000}, path)
    assert checkpoint_has_identity(path, identity)
    assert not checkpoint_has_identity(path, identity, required_step=20000)
    torch.save({"run_identity": identity, "step": 20000}, path)
    assert checkpoint_has_identity(path, identity, required_step=20000)


def test_advisory_quality_gate_still_rejects_nan(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.recipe = config("recipe")
    path = tmp_path / "checkpoint.pt"
    metrics = {"val/L_eye": 1.0, "val/L_feat": 1.0, "val/active_codes": 1,
               "val/code_perplexity": 1.0, "val/top1_code_frequency": 1.0}
    torch.save({"val_metrics": metrics}, path)
    with pytest.warns(RuntimeWarning, match="quality gate"):
        pipeline._check_quality_gate("tokenizer", path)
    metrics["val/L_eye"] = float("nan")
    torch.save({"val_metrics": metrics}, path)
    with pytest.raises(RuntimeError, match="non-finite"):
        pipeline._check_quality_gate("tokenizer", path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full reference GPU smoke test")
def test_full_twelve_layer_tokenizer_and_dual_scale_bert_backward():
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch, patches, samples = 2, 128, 40
    stim = torch.randn(batch, patches, 4, samples, device=device)
    eyes = torch.randn(batch, patches, 2, 4, samples, device=device)
    eyes[..., 3, :] = 0
    quality = torch.zeros(batch, patches, 2, samples, 1, device=device)
    pad = torch.zeros(batch, patches, dtype=torch.bool, device=device)
    nonmissing = torch.ones(batch, patches, 2, device=device)
    tokenizer = build_tokenizer(config("tokenizer")).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = tokenizer(stim, eyes, quality, pad, nonmissing)
        loss = output["eye_recon"].float().square().mean() + output["manual_feat_pred"].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in tokenizer.parameters() if p.grad is not None)
    codes = output["code_ids"].detach()
    assert codes.shape == (batch, patches, 2)
    assert int(codes.min()) >= 0 and int(codes.max()) < 1575
    del output, tokenizer, loss

    bert = build_bert(config("bert")).to(device)
    valid = nonmissing >= .85
    short, _ = generate_eyemae_mask_paired_fixed_blocks(valid, pad, num_blocks=15, span_min=1, span_max=3, min_gap=1)
    long, _ = generate_eyemae_mask_paired_fixed_blocks(valid, pad, num_blocks=5, span_min=4, span_max=6, min_gap=1)
    duplicate = lambda value: torch.cat([value, value], dim=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, stats = bert(duplicate(stim), duplicate(eyes), duplicate(quality),
                           duplicate(pad), duplicate(nonmissing),
                           torch.zeros(batch * 2, dtype=torch.long, device=device),
                           duplicate(codes), torch.cat([short, long], dim=0))
    loss.backward()
    assert torch.isfinite(loss)
    assert int(stats["n_masked"]) > 0
    assert all(torch.isfinite(p.grad).all() for p in bert.parameters() if p.grad is not None)
