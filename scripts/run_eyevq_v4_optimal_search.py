#!/usr/bin/env python3
"""One-stop, resumable V5 EyeVQ model search and final downstream training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch
import yaml

from eyemae.eyevq.config import validate_bert_config, validate_tokenizer_config
from eyemae.eyevq.search import (
    cache_manifest_matches,
    ensemble_prediction_csvs,
    latest_step_checkpoint,
    select_joint_candidate,
    select_task_configuration,
    sha256_file,
    standardized_joint_scores,
)
from eyemae.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/eyevq/v5_fsq9755_nodecay_optimal_search"
TOKENIZER_TEMPLATE = PROJECT_ROOT / "configs/eyevq/tokenizer_joint_per_subject_v5_fsq9755_recommended40k.yaml"
BERT_TEMPLATE = PROJECT_ROOT / "configs/eyevq/pretrain_joint_per_subject_v5_fsq9755_tok40k_mask015.yaml"
DOWNSTREAM_TEMPLATES = {
    "mci": PROJECT_ROOT / "configs/eyevq/downstream_mci_v5_demoraw16_fsq9755_newloss.yaml",
    "pd5": PROJECT_ROOT / "configs/eyevq/downstream_pd5_v5_demoraw16_fsq9755_newloss.yaml",
}
TOKENIZER_STEPS = (25_000, 30_000, 35_000, 40_000)
BERT_STEPS = (30_000, 40_000, 50_000)
MASK_RATIOS = (0.25, 0.50)
# mode, filesystem label, minimum span, maximum span.  Span bounds remain in
# every generated config so each run has a complete, hashable identity.
# 先测 span（限制长度 [2,6]），再测随机 mask。
MASK_MODES = (
    ("paired_span", "span2to6", 2, 6),
    ("paired_random", "random", 2, 6),
)
UNFROZEN_LAYERS = (6, 8, 10)
ENCODER_LRS = (5e-6, 1e-5)
SEEDS = (42, 43, 44)


class Pipeline:
    def __init__(self, root: Path, *, dry_run: bool = False) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dry_run = bool(dry_run)
        self.python_env = Path(os.environ.get("PYTHON_ENV", "/home/jinfanhe/miniconda3/envs/jinf"))
        self.python = self.python_env / "bin/python"
        self.torchrun = self.python_env / "bin/torchrun"
        self.nproc = int(os.environ.get("NPROC_PER_NODE", "4"))
        self.log_path = self.root / "pipeline.log"
        self.status_path = self.root / "pipeline_status.json"
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (
            os.pathsep + self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else ""
        )
        self.env.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")
        self.status: dict[str, Any] = {
            "state": "initialized",
            "root": str(self.root),
            "test_results_used_for_selection": False,
            "test_evaluated_for_all_candidates": True,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {},
        }
        write_json(self.status_path, self.status)

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def mark(self, stage: str, state: str, **payload: Any) -> None:
        self.status["state"] = state if state == "failed" else "running"
        self.status["current_stage"] = stage
        self.status["stages"][stage] = {
            "state": state,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **payload,
        }
        write_json(self.status_path, self.status)

    def run(self, stage: str, command: list[str]) -> None:
        quoted = " ".join(shlex.quote(part) for part in command)
        self.log(f"[{stage}] {quoted}")
        self.mark(stage, "running", command=command)
        if self.dry_run:
            self.mark(stage, "dry_run_complete", command=command)
            return
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self.env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            self.mark(stage, "failed", command=command, returncode=completed.returncode)
            raise RuntimeError(f"Stage failed ({completed.returncode}): {stage}")
        self.mark(stage, "complete", command=command)

    def distributed(self, *args: str) -> list[str]:
        return [
            str(self.torchrun), "--standalone", f"--nproc_per_node={self.nproc}", *args
        ]

    def finish(self, summary: dict[str, Any]) -> None:
        self.status.update({
            "state": "complete",
            "current_stage": None,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_summary": str(self.root / "final_summary.json"),
        })
        write_json(self.status_path, self.status)
        write_json(self.root / "final_summary.json", summary)
        self.log("Pipeline complete")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    # fitted_spec 是训练时根据训练集拟合的人口学特征规格（均值/标准差/类别等），
    # 属于运行时数据而非配置身份，校验时应忽略，否则会与模板重新解析的配置不一致。
    demographics = result.get("demographics")
    if isinstance(demographics, dict):
        demographics.pop("fitted_spec", None)
    return result


def assert_config_matches(actual: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    if comparable_config(actual) != comparable_config(expected):
        raise RuntimeError(f"Checkpoint/result config does not match resolved config: {path}")


def audit_optimizer_groups(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing optimizer group audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("policy") != (
        "decay_matrix_or_kernel_weights_only; no_decay=bias,norm,vector_state"
    ):
        raise RuntimeError(f"Unexpected optimizer grouping policy: {path}")
    groups = audit.get("groups", [])
    if not groups or sum(int(group["num_parameters"]) for group in groups) != int(
        audit["num_trainable_parameters"]
    ):
        raise RuntimeError(f"Incomplete optimizer group audit: {path}")


def audit_split_subjects(config: dict[str, Any]) -> dict[str, int]:
    data = config["data"]
    data_dir = Path(data["data_dir"])
    split_subjects: dict[str, set[str]] = {}
    for split, key in (("train", "train_index"), ("val", "val_index"), ("test", "test_index")):
        with (data_dir / data[key]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        split_subjects[split] = {str(row["ml_subject_id"]) for row in rows}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_subjects[left] & split_subjects[right]
        if overlap:
            raise RuntimeError(f"Subject leakage between {left}/{right}: {sorted(overlap)[:10]}")
    return {split: len(subjects) for split, subjects in split_subjects.items()}


def preflight(root: Path) -> dict[str, Any]:
    tokenizer = load_yaml(TOKENIZER_TEMPLATE)
    bert = load_yaml(BERT_TEMPLATE)
    validate_tokenizer_config(tokenizer)
    validate_bert_config(bert)
    expected_loss = {
        "eye_xy_weight": 1.0,
        "eye_area_weight": 0.1,
        "eye_blink_weight": 0.1,
        "eye_blink_pos_weight": 1.0,
        "eye_velocity_weight": 0.0,
        "eye_recon_group_weight": 1.0,
        "eye_commit_group_weight": 1.0,
        "manual_feature_group_weight": 0.0015,
        "manual_feature_binary_weight": 0.25,
        "manual_feature_continuous_weight": 1.5,
        "manual_feature_balanced_bce": False,
    }
    for key, value in expected_loss.items():
        if tokenizer["loss"].get(key) != value:
            raise RuntimeError(f"Tokenizer loss mismatch: {key}")
    required_values = [
        (tokenizer["model"]["architecture"], "joint"),
        (tokenizer["vq"]["fsq_L"], [9, 7, 5, 5]),
        (tokenizer["vq"]["fsq_activation"], "tanh"),
        (tokenizer["train"]["stage_b_steps"], 40_000),
        (bert["bert"]["d_model"], 384),
        (bert["bert"]["n_layers"], 12),
        (bert["mask"]["mode"], "paired_random"),
        (bert["mask"]["span_min_patches"], 2),
        (bert["mask"]["span_max_patches"], 6),
    ]
    for actual, expected in required_values:
        if actual != expected:
            raise RuntimeError(f"Preflight value mismatch: {actual!r} != {expected!r}")
    required_paths = [
        Path(tokenizer["train"]["data_path"]),
        PROJECT_ROOT / tokenizer["train"]["area_stats_path"],
        PROJECT_ROOT / tokenizer["manual_features"]["cache_path"],
        PROJECT_ROOT / tokenizer["manual_features"]["stats_path"],
        PROJECT_ROOT / tokenizer["manual_features"]["task_metric_weight_path"],
    ]
    for task, path in DOWNSTREAM_TEMPLATES.items():
        cfg = load_yaml(path)
        required_paths.extend([
            Path(cfg["data"]["data_dir"]),
            Path(cfg["data"]["area_stats_path"]),
        ])
        if cfg["demographics"].get("projection_dim") != 0:
            raise RuntimeError(f"{task} demographics must be raw 16D")
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preflight artifacts: {missing}")
    split_counts = {
        task: audit_split_subjects(load_yaml(path))
        for task, path in DOWNSTREAM_TEMPLATES.items()
    }
    result = {
        "dataset": "v5",
        "fsq": [9, 7, 5, 5],
        "activation": "tanh",
        "loss": expected_loss,
        "bert_mask_ablation": [
            {
                "mode": mode,
                "label": label,
                "mask_ratio": ratio,
                "span_min_patches": span_min,
                "span_max_patches": span_max,
                "paired_lr": True,
            }
            for mode, label, span_min, span_max in MASK_MODES
            for ratio in MASK_RATIOS
        ],
        "split_subject_counts": split_counts,
        "output_root": str(root),
    }
    write_json(root / "preflight.json", result)
    return result


def load_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError(f"Checkpoint step mismatch: {path}")
    cfg = checkpoint.get("cfg", {})
    if cfg.get("vq", {}).get("fsq_L") != [9, 7, 5, 5]:
        raise RuntimeError(f"Checkpoint FSQ mismatch: {path}")
    if cfg.get("vq", {}).get("fsq_activation") != "tanh":
        raise RuntimeError(f"Checkpoint activation mismatch: {path}")
    if "eyemae_fast_dataset_v5" not in str(cfg.get("train", {}).get("data_path", "")):
        raise RuntimeError(f"Checkpoint is not V5: {path}")
    return checkpoint


def checkpoint_for_step(directory: Path, step: int) -> Path:
    candidate = directory / f"ckpt_step{step:06d}.pt"
    if candidate.is_file():
        return candidate
    final = directory / "ckpt_final.pt"
    if final.is_file() and int(torch.load(final, map_location="cpu", weights_only=False).get("step", -1)) == step:
        return final
    raise FileNotFoundError(f"No checkpoint at step {step}: {directory}")


def resume_arguments(directory: Path, target_step: int) -> list[str]:
    latest = latest_step_checkpoint(directory)
    if latest is None:
        return []
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    step = int(checkpoint.get("step", -1))
    return ["--resume", str(latest)] if 0 < step < target_step else []


def ensure_cache(
    pipeline: Pipeline, *, tokenizer: Path, bert_config: Path, directory: Path, stage: str
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "code_ids_train_val.npz"
    manifest = directory / "code_ids_train_val.manifest.json"
    if not cache_manifest_matches(cache, manifest, tokenizer, bert_config):
        pipeline.run(stage, pipeline.distributed(
            "-m", "eyemae.eyevq.precompute_codes",
            "--config", str(bert_config),
            "--tokenizer-checkpoint", str(tokenizer),
            "--out", str(cache), "--split", "all", "--n-batch", "512",
        ))
        if pipeline.dry_run:
            return cache
        write_json(manifest, {
            "tokenizer_checkpoint": str(tokenizer.resolve()),
            "tokenizer_sha256": sha256_file(tokenizer),
            "bert_config": str(bert_config.resolve()),
            "bert_config_sha256": sha256_file(bert_config),
            "cache_size_bytes": cache.stat().st_size,
            "split": "all",
        })
    if not pipeline.dry_run and not cache_manifest_matches(
        cache, manifest, tokenizer, bert_config
    ):
        raise RuntimeError(f"Invalid code-ID cache after generation: {cache}")
    return cache


def ensure_bert(
    pipeline: Pipeline,
    *,
    tokenizer: Path,
    cache: Path,
    output: Path,
    config: dict[str, Any],
    total_steps: int,
    mask_ratio: float,
    stage: str,
    mask_mode: str = "paired_random",
    mask_span_min: int = 2,
    mask_span_max: int = 6,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["mask"]["eye_masking_ratio"] = float(mask_ratio)
    cfg["mask"]["mode"] = str(mask_mode)
    cfg["mask"]["span_min_patches"] = int(mask_span_min)
    cfg["mask"]["span_max_patches"] = int(mask_span_max)
    cfg["train"].update({
        "total_steps": int(total_steps),
        "tokenizer_checkpoint": str(tokenizer),
        "code_ids_cache": str(cache),
        "val_every_steps": 5_000,
        "save_every_steps": 5_000,
    })
    config_path = output / "config.yaml"
    write_yaml(config_path, cfg)
    final = output / "ckpt_final.pt"
    if final.is_file():
        checkpoint = load_checkpoint(final, total_steps)
        assert_config_matches(checkpoint["cfg"], cfg, final)
        audit_optimizer_groups(output / "optimizer_groups.json")
        return final
    pipeline.run(stage, pipeline.distributed(
        "-m", "eyemae.eyevq.pretrain.train",
        "--config", str(config_path), "--output_dir", str(output),
        "--tokenizer-checkpoint", str(tokenizer),
        "--code-ids-cache", str(cache),
        "--max-steps", str(total_steps), "--mask-ratio", str(mask_ratio),
        "--mask-mode", str(mask_mode),
        "--mask-span-min", str(mask_span_min),
        "--mask-span-max", str(mask_span_max),
        *resume_arguments(output, total_steps),
    ))
    if not pipeline.dry_run:
        checkpoint = load_checkpoint(final, total_steps)
        assert_config_matches(checkpoint["cfg"], cfg, final)
        audit_optimizer_groups(output / "optimizer_groups.json")
    return final


def ensure_mil(
    pipeline: Pipeline,
    *,
    task: str,
    bert_checkpoint: Path,
    output: Path,
    freeze_bottom: int = 4,
    encoder_lr: float = 1e-5,
    seed: int = 42,
    stage: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(DOWNSTREAM_TEMPLATES[task])
    cfg.setdefault("label", {"type": "binary", "num_classes": 2})
    cfg["model"].update({
        "bert_checkpoint": str(bert_checkpoint),
        "freeze_embedding": True,
        "freeze_bottom_layers": int(freeze_bottom),
        "classifier_hidden": 128,
        "classifier_head": "mlp",
        "dropout": 0.3,
    })
    cfg["mil"].update({
        "subjects_per_gpu": 4,
        "trials_per_task": 4,
        "train_require_all_tasks": True,
        "task_pooling": "shared_head_mean",
        "trial_pooling": "logit_mean",
        "eval_use_all_trials": True,
    })
    cfg["train"].update({
        "seed": int(seed),
        "epochs": 100,
        "encoder_lr": float(encoder_lr),
        "encoder_min_lr": float(encoder_lr) / 10.0,
        "head_lr": 1e-5,
        "head_min_lr": 1e-6,
        "layer_decay": 1.0,
        "warmup_epochs": 4,
        "early_stopping_min_epochs": 27,
        "early_stopping_patience_epochs": 20,
        "num_trial_views": 1,
        "trial_view_consistency_weight": 0.0,
        "auxiliary_task_loss_weight": 0.0,
        "feature_mixup_alpha": 0.0,
    })
    config_path = output / "config.yaml"
    write_yaml(config_path, cfg)
    metrics_path = output / "metrics_test.json"
    val_path = output / "metrics_val_best.json"
    if not (metrics_path.is_file() and val_path.is_file()):
        pipeline.run(stage, pipeline.distributed(
            "-m", "eyemae.eyevq.downstream.train_mil",
            "--config", str(config_path), "--output_dir", str(output),
            "--bert-checkpoint", str(bert_checkpoint),
        ))
    if pipeline.dry_run:
        return {"best_val_auroc": 0.0, "cfg": cfg}
    if not metrics_path.is_file() or not val_path.is_file():
        raise RuntimeError(f"Incomplete downstream result: {output}")
    result = json.loads(val_path.read_text(encoding="utf-8"))
    assert_config_matches(result["cfg"], cfg, val_path)
    return result


def tokenizer_metrics(checkpoint: dict[str, Any]) -> dict[str, float]:
    metrics = checkpoint.get("val_metrics") or {}
    required = (
        "val/L_eye", "val/L_feat", "val/code_perplexity",
        "val/active_codes", "val/top1_code_frequency",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"Tokenizer checkpoint lacks validation metrics: {missing}")
    return {key: float(metrics[key]) for key in required}


def tokenizer_not_collapsed(metrics: dict[str, float]) -> bool:
    return (
        all(math.isfinite(value) for value in metrics.values())
        and metrics["val/active_codes"] >= 945
        and metrics["val/code_perplexity"] >= 100
        and metrics["val/top1_code_frequency"] <= 0.10
    )


def tokenizer_passes_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["val/L_eye"] <= 3.5e-4
        and metrics["val/L_feat"] <= 0.12
        and metrics["val/active_codes"] >= 1200
        and metrics["val/code_perplexity"] >= 500
        and metrics["val/top1_code_frequency"] <= 0.03
    )


def bert_passes_gate(checkpoint: dict[str, Any]) -> bool:
    metrics = checkpoint.get("val_metrics") or {}
    loss = float(metrics.get("val/loss", math.inf))
    accuracy = float(metrics.get("val/acc", -math.inf))
    # A common strict accuracy/loss cutoff biases this ablation against larger
    # ratios and contiguous spans, whose prediction task is intrinsically
    # harder.  Only catastrophic/non-learning candidates are rejected here;
    # downstream validation AUC remains the selection signal.
    if not math.isfinite(loss) or loss >= math.log(1575):
        return False
    if math.isfinite(accuracy):
        return accuracy > 0.05

    # Older factorized-head checkpoints can have NaN validation accuracy when
    # a zero-supervision validation batch poisoned the weighted accumulator.
    # Their CE values are still valid.  Accept only when all four coordinate
    # losses are finite, agree with the reported joint NLL, and are far below
    # the random-code baseline.  This preserves the catastrophic-failure gate
    # without rejecting an otherwise healthy completed run on a report-only bug.
    per_dim_ce = metrics.get("val/per_dim_ce")
    if not isinstance(per_dim_ce, (list, tuple)) or not per_dim_ce:
        return False
    dimension_losses = [float(value) for value in per_dim_ce]
    return (
        all(math.isfinite(value) and value >= 0.0 for value in dimension_losses)
        and math.isclose(sum(dimension_losses), loss, rel_tol=1e-3, abs_tol=1e-3)
        and loss < 0.8 * math.log(1575)
    )


def train_tokenizer(pipeline: Pipeline) -> Path:
    output = pipeline.root / "tokenizer"
    output.mkdir(parents=True, exist_ok=True)
    config = load_yaml(TOKENIZER_TEMPLATE)
    config_path = output / "config.yaml"
    write_yaml(config_path, config)
    final = output / "ckpt_final.pt"
    if final.is_file():
        checkpoint = load_checkpoint(final, 40_000)
        assert_config_matches(checkpoint["cfg"], config, final)
        audit_optimizer_groups(output / "optimizer_groups.json")
        return final
    pipeline.run("tokenizer_40k", pipeline.distributed(
        "-m", "eyemae.eyevq.tokenizer.train",
        "--config", str(config_path), "--output_dir", str(output),
        *resume_arguments(output, 40_000),
    ))
    if not pipeline.dry_run:
        checkpoint = load_checkpoint(final, 40_000)
        assert_config_matches(checkpoint["cfg"], config, final)
        audit_optimizer_groups(output / "optimizer_groups.json")
    return final


def select_tokenizer(pipeline: Pipeline, base_bert: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    if pipeline.dry_run:
        for step in TOKENIZER_STEPS:
            candidate_root = pipeline.root / "tokenizer_candidates" / f"step{step:06d}"
            tokenizer = pipeline.root / "tokenizer" / f"ckpt_step{step:06d}.pt"
            config_path = candidate_root / "bert_proxy_config.yaml"
            write_yaml(config_path, base_bert)
            cache = ensure_cache(
                pipeline, tokenizer=tokenizer, bert_config=config_path,
                directory=candidate_root / "cache", stage=f"cache_tokenizer_{step}",
            )
            proxy = ensure_bert(
                pipeline, tokenizer=tokenizer, cache=cache,
                output=candidate_root / "bert_proxy15k", config=base_bert,
                total_steps=15_000, mask_ratio=0.15,
                mask_mode="paired_random",
                stage=f"bert_proxy_tokenizer_{step}",
            )
            for task in ("mci", "pd5"):
                ensure_mil(
                    pipeline, task=task, bert_checkpoint=proxy,
                    output=candidate_root / task,
                    stage=f"proxy_{task}_tokenizer_{step}",
                )
        return (
            pipeline.root / "tokenizer/ckpt_step040000.pt",
            pipeline.root / "tokenizer_candidates/step040000/cache/code_ids_train_val.npz",
            [],
        )
    for step in TOKENIZER_STEPS:
        tokenizer = checkpoint_for_step(pipeline.root / "tokenizer", step)
        checkpoint = load_checkpoint(tokenizer, step)
        metrics = tokenizer_metrics(checkpoint)
        if not tokenizer_not_collapsed(metrics):
            pipeline.log(f"[tokenizer_step{step}] rejected by collapse gate: {metrics}")
            continue
        candidate_root = pipeline.root / "tokenizer_candidates" / f"step{step:06d}"
        config_path = candidate_root / "bert_proxy_config.yaml"
        proxy_cfg = copy.deepcopy(base_bert)
        write_yaml(config_path, proxy_cfg)
        cache = ensure_cache(
            pipeline, tokenizer=tokenizer, bert_config=config_path,
            directory=candidate_root / "cache", stage=f"cache_tokenizer_{step}",
        )
        proxy = ensure_bert(
            pipeline, tokenizer=tokenizer, cache=cache,
            output=candidate_root / "bert_proxy15k", config=base_bert,
            total_steps=15_000, mask_ratio=0.15, mask_mode="paired_random",
            stage=f"bert_proxy_tokenizer_{step}",
        )
        task_values: dict[str, float] = {}
        for task in ("mci", "pd5"):
            result = ensure_mil(
                pipeline, task=task, bert_checkpoint=proxy,
                output=candidate_root / task, stage=f"proxy_{task}_tokenizer_{step}",
            )
            task_values[task] = float(result["best_val_auroc"])
        candidates.append({
            "name": f"tokenizer_step{step:06d}", "step": step,
            "tokenizer_checkpoint": str(tokenizer), "cache": str(cache),
            "mci_val_auroc": task_values["mci"],
            "pd5_val_macro_auroc": task_values["pd5"],
            "val_L_eye": metrics["val/L_eye"], "val_L_feat": metrics["val/L_feat"],
            "tokenizer_metrics": metrics,
        })
    if not candidates:
        raise RuntimeError("No tokenizer checkpoint passed the collapse gate")
    scored = standardized_joint_scores(candidates)
    best_score = max(float(item["joint_val_score"]) for item in scored)
    tied = [item for item in scored if best_score - float(item["joint_val_score"]) <= 0.05]
    selected = min(tied, key=lambda item: (item["val_L_eye"], item["val_L_feat"], item["step"]))
    if not tokenizer_passes_gate(selected["tokenizer_metrics"]):
        raise RuntimeError(f"Selected tokenizer failed the formal quality gate: {selected}")
    summary = {"selected": selected, "candidates": scored, "test_used_for_selection": False}
    write_json(pipeline.root / "tokenizer_selection.json", summary)
    return Path(selected["tokenizer_checkpoint"]), Path(selected["cache"]), scored


def use_tokenizer_40k(pipeline: Pipeline, base_bert: dict[str, Any]) -> tuple[Path, Path]:
    """固定使用 40K 步 tokenizer，跳过 tokenizer 候选 proxy 搜索。

    只为 40K tokenizer 生成 code_ids cache，随后直接进入 BERT mask ablation。
    """
    step = 40_000
    tokenizer = checkpoint_for_step(pipeline.root / "tokenizer", step)
    checkpoint = load_checkpoint(tokenizer, step)
    metrics = tokenizer_metrics(checkpoint)
    if not tokenizer_not_collapsed(metrics):
        raise RuntimeError(f"40K tokenizer failed collapse gate: {metrics}")
    candidate_root = pipeline.root / "tokenizer_candidates" / f"step{step:06d}"
    config_path = candidate_root / "bert_proxy_config.yaml"
    write_yaml(config_path, copy.deepcopy(base_bert))
    cache = ensure_cache(
        pipeline, tokenizer=tokenizer, bert_config=config_path,
        directory=candidate_root / "cache", stage=f"cache_tokenizer_{step}",
    )
    pipeline.log(
        "[tokenizer_step%06d] fixed 40K tokenizer: "
        "val/L_eye=%.3e val/L_feat=%.3e active_codes=%.0f code_perplexity=%.0f"
        % (step, metrics["val/L_eye"], metrics["val/L_feat"],
           metrics["val/active_codes"], metrics["val/code_perplexity"])
    )
    return tokenizer, cache


def select_bert(
    pipeline: Pipeline, *, tokenizer: Path, cache: Path, base_bert: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for mask_mode, mode_label, span_min, span_max in MASK_MODES:
        for mask_ratio in MASK_RATIOS:
            mask_name = f"{mode_label}_mask{int(mask_ratio * 100):03d}"
            output = pipeline.root / "bert_candidates" / mask_name
            ensure_bert(
                pipeline, tokenizer=tokenizer, cache=cache, output=output,
                config=base_bert, total_steps=50_000, mask_ratio=mask_ratio,
                mask_mode=mask_mode, mask_span_min=span_min,
                mask_span_max=span_max, stage=f"bert_{mask_name}_50k",
            )
            if pipeline.dry_run:
                for step in BERT_STEPS:
                    checkpoint_path = output / f"ckpt_step{step:06d}.pt"
                    for task in ("mci", "pd5"):
                        ensure_mil(
                            pipeline, task=task, bert_checkpoint=checkpoint_path,
                            output=output / f"downstream_step{step:06d}" / task,
                            stage=f"bert_screen_{mask_name}_{step}_{task}",
                        )
                continue
            for step in BERT_STEPS:
                checkpoint_path = checkpoint_for_step(output, step)
                checkpoint = load_checkpoint(checkpoint_path, step)
                if not bert_passes_gate(checkpoint):
                    pipeline.log(f"[{mask_name}_step{step}] rejected by BERT quality gate")
                    continue
                task_values: dict[str, float] = {}
                candidate_root = output / f"downstream_step{step:06d}"
                for task in ("mci", "pd5"):
                    result = ensure_mil(
                        pipeline, task=task, bert_checkpoint=checkpoint_path,
                        output=candidate_root / task,
                        stage=f"bert_screen_{mask_name}_{step}_{task}",
                    )
                    task_values[task] = float(result["best_val_auroc"])
                candidates.append({
                    "name": f"{mask_name}_step{step:06d}",
                    "mask_mode": mask_mode,
                    "mask_ratio": mask_ratio,
                    "span_min_patches": span_min,
                    "span_max_patches": span_max,
                    "paired_lr": True,
                    "step": step,
                    "bert_checkpoint": str(checkpoint_path),
                    "mci_val_auroc": task_values["mci"],
                    "pd5_val_macro_auroc": task_values["pd5"],
                    "bert_val_metrics": checkpoint["val_metrics"],
                })
    if pipeline.dry_run:
        return pipeline.root / "bert_candidates/random_mask015/ckpt_final.pt", candidates
    if not candidates:
        raise RuntimeError("No BERT checkpoint passed the quality gate")
    selected, ranked = select_joint_candidate(candidates, tie_tolerance=0.05)
    write_json(pipeline.root / "bert_selection.json", {
        "selected": selected, "candidates": ranked, "test_used_for_selection": False,
    })
    return Path(selected["bert_checkpoint"]), ranked


def downstream_grid(
    pipeline: Pipeline, *, task: str, bert_checkpoint: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_root = pipeline.root / "downstream_grid" / task
    # 固定配置：解冻 8 层、编码器 LR 1e-5；三种子训练结果平均。
    unfrozen = 8
    freeze_bottom = 12 - unfrozen
    encoder_lr = 1e-5
    name = f"top{unfrozen}_lr1e5"
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        output = task_root / name / f"seed{seed}"
        result = ensure_mil(
            pipeline, task=task, bert_checkpoint=bert_checkpoint,
            output=output, freeze_bottom=freeze_bottom,
            encoder_lr=encoder_lr, seed=seed,
            stage=f"repeat_{task}_{name}_seed{seed}",
        )
        runs.append({
            "seed": seed, "val_auroc": float(result["best_val_auroc"]),
            "output": str(output),
        })
    if pipeline.dry_run:
        return {}, runs
    values = [run["val_auroc"] for run in runs]
    aggregates: list[dict[str, Any]] = [{
        "name": name,
        "unfrozen_layers": unfrozen,
        "freeze_bottom_layers": freeze_bottom,
        "encoder_lr": encoder_lr,
        "mean_val_auroc": fmean(values),
        "std_val_auroc": pstdev(values),
        "runs": runs,
    }]
    ensemble_root = pipeline.root / "final" / task / "ensemble"
    num_classes = 2 if task == "mci" else 5
    ensemble_metrics: dict[str, Any] = {}
    for split in ("val_best", "test"):
        filename = "predictions_val_best.csv" if split == "val_best" else "predictions_test.csv"
        metric_split = "val" if split == "val_best" else "test"
        ensemble_metrics[metric_split] = ensemble_prediction_csvs(
            [Path(run["output"]) / filename for run in runs],
            output_csv=ensemble_root / f"predictions_{metric_split}.csv",
            output_metrics=ensemble_root / f"metrics_{metric_split}.json",
            num_classes=num_classes,
            split=metric_split,
        )
    median_run = sorted(runs, key=lambda run: run["val_auroc"])[1]
    selected = {
        **aggregates[0],
        "median_validation_seed_checkpoint": str(Path(median_run["output"]) / "ckpt_best.pt"),
        "ensemble_metrics": ensemble_metrics,
        "selection_uses_test": False,
    }
    write_json(task_root / "selection.json", {
        "selected": selected,
        "seed42_screen": [],
        "three_seed_finalists": aggregates,
        "test_used_for_selection": False,
    })
    return selected, aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    pipeline = Pipeline(root, dry_run=args.dry_run)
    try:
        if not pipeline.python.is_file() or not pipeline.torchrun.is_file():
            raise FileNotFoundError(f"Missing training environment: {pipeline.python_env}")
        pipeline.log("Starting V5 FSQ9755 no-decay optimal search")
        preflight_result = preflight(root)
        pipeline.mark("preflight", "complete", result=preflight_result)
        train_tokenizer(pipeline)
        base_bert = load_yaml(BERT_TEMPLATE)
        tokenizer, cache = use_tokenizer_40k(pipeline, base_bert)
        bert, bert_candidates = select_bert(
            pipeline, tokenizer=tokenizer, cache=cache, base_bert=base_bert
        )
        final_tasks: dict[str, Any] = {}
        grid_results: dict[str, Any] = {}
        for task in ("mci", "pd5"):
            selected, aggregates = downstream_grid(
                pipeline, task=task, bert_checkpoint=bert
            )
            final_tasks[task] = selected
            grid_results[task] = aggregates
        summary = {
            "dataset": "v5",
            "tokenizer_checkpoint": str(tokenizer),
            "tokenizer_sha256": sha256_file(tokenizer) if tokenizer.is_file() else None,
            "code_ids_cache": str(cache),
            "bert_checkpoint": str(bert),
            "bert_sha256": sha256_file(bert) if bert.is_file() else None,
            "tokenizer_candidates": [],
            "tokenizer_selection": "fixed_40k",
            "bert_candidates": bert_candidates,
            "downstream_final": final_tasks,
            "downstream_three_seed_finalists": grid_results,
            "test_results_used_for_selection": False,
            "result_scope": "internal exploratory test; test split was historically observed",
        }
        pipeline.finish(summary)
    except BaseException as error:
        pipeline.status.update({
            "state": "failed",
            "error": repr(error),
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        write_json(pipeline.status_path, pipeline.status)
        pipeline.log(f"Pipeline failed: {error!r}")
        raise


if __name__ == "__main__":
    main()
