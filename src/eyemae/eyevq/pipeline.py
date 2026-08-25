"""Strict one-command tokenizer -> cache -> BERT -> downstream pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
import torch

from eyemae.eyevq.artifacts import (
    build_run_identity,
    cache_contract,
    checkpoint_has_identity,
    sha256_file,
    sha256_json,
    validate_cache_identity,
)
from eyemae.eyevq.config import validate_bert_config, validate_tokenizer_config
from eyemae.eyevq.search import checkpoint_step
from eyemae.downstream_metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    sigmoid,
    softmax,
)
from eyemae.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_CONFIG_DIR = PROJECT_ROOT / "configs/eyevq/final"


def _source_identity() -> dict[str, Any]:
    """Hash the complete executable/configuration surface of a formal run."""
    # Local research scratch modules are deliberately outside the formal
    # dependency graph and must not make an otherwise identical run identity
    # machine-specific.
    excluded_scratch_modules = {"dual_finetune.py", "feature_extracture.py"}
    files = sorted(
        path
        for path in (PROJECT_ROOT / "src/eyemae").rglob("*.py")
        if path.name not in excluded_scratch_modules
    )
    files.extend(sorted(FINAL_CONFIG_DIR.glob("*.yaml")))
    files.extend((PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "scripts/run_eyevq_final.sh"))
    identities = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    digest = hashlib.sha256()
    for item in identities:
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode())
    return {"sha256": digest.hexdigest(), "files": identities}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _quarantine(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    suffix = time.strftime("%Y%m%d_%H%M%S")
    destination = path.with_name(path.name + f".stale_{suffix}")
    path.replace(destination)
    write_json(
        destination.with_name(destination.name + ".reason.json"),
        {"original": str(path), "quarantined": str(destination), "reason": reason},
    )
    return destination


class Pipeline:
    def __init__(self, output_root: Path, python_env: Path, nproc: int) -> None:
        self.output_root = output_root.resolve()
        self.python = python_env / "bin/python"
        self.torchrun = python_env / "bin/torchrun"
        self.nproc = int(nproc)
        self.state_path = self.output_root / "pipeline_state.json"
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("output_root") != str(self.output_root):
                raise ValueError("Existing pipeline state belongs to another output root")
        else:
            self.state = {
                "schema_version": 1,
                "output_root": str(self.output_root),
                "status": "created",
                "stages": {},
            }

        self.tokenizer_dir = self.output_root / "tokenizer"
        self.cache_dir = self.output_root / "cache"
        self.bert_dir = self.output_root / "bert"
        self.tokenizer_checkpoint = self.tokenizer_dir / "ckpt_final.pt"
        self.cache = self.cache_dir / "code_ids_train_val.npz"
        self.bert_checkpoint = self.bert_dir / "ckpt_final.pt"
        self.resolved_dir = self.output_root / "resolved_configs"

        self.recipe = _load_yaml(FINAL_CONFIG_DIR / "recipe.yaml")
        self.source_identity = _source_identity()
        self.tokenizer_cfg = _load_yaml(FINAL_CONFIG_DIR / "tokenizer.yaml")
        self.bert_cfg = _load_yaml(FINAL_CONFIG_DIR / "bert.yaml")
        self.bert_cfg["train"]["tokenizer_checkpoint"] = str(self.tokenizer_checkpoint)
        self.bert_cfg["train"]["code_ids_cache"] = str(self.cache)
        self.downstream_cfgs = {
            task: _load_yaml(FINAL_CONFIG_DIR / f"{task}.yaml")
            for task in ("mci", "pd5")
        }
        for cfg in self.downstream_cfgs.values():
            cfg["model"]["bert_checkpoint"] = str(self.bert_checkpoint)
        for cfg in (self.tokenizer_cfg, self.bert_cfg, *self.downstream_cfgs.values()):
            cfg["reproducibility"] = {
                "recipe_version": int(self.recipe["recipe_version"]),
                "source_sha256": self.source_identity["sha256"],
            }

    def _assert_source_unchanged(self) -> None:
        current = _source_identity()
        if current["sha256"] != self.source_identity["sha256"]:
            raise RuntimeError(
                "Project source/configuration changed after pipeline creation; "
                "refusing to mix code versions in one run"
            )

    def _validate_formal_recipe(self) -> None:
        dataset = self.recipe["dataset"]
        dataset_root = Path(dataset["root"]).resolve()
        manifest = dataset_root / "pretrain/dataset_manifest.json"
        audit = Path(dataset["build_audit"]).resolve()
        if sha256_file(manifest) != str(dataset["manifest_sha256"]):
            raise ValueError("Formal dataset manifest SHA256 does not match recipe.yaml")
        if sha256_file(audit) != str(dataset["build_audit_sha256"]):
            raise ValueError("Formal dataset build audit SHA256 does not match recipe.yaml")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("dataset_version") != dataset["version"]:
            raise ValueError("Formal dataset version does not match recipe.yaml")

        validate_tokenizer_config(self.tokenizer_cfg)
        validate_bert_config(self.bert_cfg)
        if self.tokenizer_cfg["model"]["architecture"] != "joint":
            raise ValueError("Formal tokenizer must use joint self-attention")
        for name, cfg in (("tokenizer", self.tokenizer_cfg), ("bert", self.bert_cfg)):
            attention = cfg["attention"]
            if attention.get("stim_isolated") is not True or attention.get("stim_attend_cls") is not False:
                raise ValueError(f"Formal {name} must isolate stimulus from CLS and L/R")
        if self.bert_cfg["bert"].get("factorized_fsq") is not True:
            raise ValueError("Formal BERT must use factorized FSQ prediction")
        expected_mask = {
            "mode": "paired_span",
            "eye_masking_ratio": 0.60,
            "span_min_patches": 1,
            "span_max_patches": 5,
            "span_length_distribution": "uniform",
        }
        for key, value in expected_mask.items():
            if self.bert_cfg["mask"].get(key) != value:
                raise ValueError(f"Formal BERT mask mismatch for {key}")
        for task in ("mci", "pd5"):
            mil = self.downstream_cfgs[task]["mil"]
            if mil["trials_per_task"] != 16:
                raise ValueError(f"Formal {task.upper()} recipe must use K16")
            if mil["eligibility_min_trials_per_task"] != 16:
                raise ValueError(
                    f"Formal {task.upper()} training eligibility must require K16"
                )

        expected_pretrain = dataset_root / "pretrain"
        if Path(self.tokenizer_cfg["train"]["data_path"]).resolve() != expected_pretrain:
            raise ValueError("Tokenizer data path does not match formal dataset")
        if Path(self.bert_cfg["train"]["data_path"]).resolve() != expected_pretrain:
            raise ValueError("BERT data path does not match formal dataset")
        for task, directory in (("mci", "mci_binary"), ("pd5", "pd_related_5class")):
            expected = dataset_root / "finetune" / directory
            if Path(self.downstream_cfgs[task]["data"]["data_dir"]).resolve() != expected:
                raise ValueError(f"{task} data path does not match formal dataset")

    def _check_quality_gate(self, stage: str, checkpoint_path: Path) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        metrics = checkpoint.get("val_metrics") or {}
        gate = self.recipe["quality_gates"][stage]
        if stage == "tokenizer":
            checks = {
                "val/L_eye": float(metrics.get("val/L_eye", math.inf))
                <= float(gate["max_val_eye_loss"]),
                "val/L_feat": float(metrics.get("val/L_feat", math.inf))
                <= float(gate["max_val_feature_loss"]),
                "val/active_codes": float(metrics.get("val/active_codes", -math.inf))
                >= float(gate["min_active_codes"]),
                "val/code_perplexity": float(metrics.get("val/code_perplexity", -math.inf))
                >= float(gate["min_code_perplexity"]),
                "val/top1_code_frequency": float(metrics.get("val/top1_code_frequency", math.inf))
                <= float(gate["max_top1_code_frequency"]),
            }
        else:
            per_dimension = metrics.get("val/per_dim_acc") or []
            checks = {
                "val/loss": float(metrics.get("val/loss", math.inf))
                <= float(gate["max_val_loss"]),
                "val/acc": float(metrics.get("val/acc", -math.inf))
                >= float(gate["min_exact_code_accuracy"]),
                "val/per_dim_acc": len(per_dimension) == 4
                and min(map(float, per_dimension))
                >= float(gate["min_per_dimension_accuracy"]),
            }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"{stage} failed formal quality gate for {failed}: {metrics}"
            )

    def save_state(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        write_json(self.state_path, self.state)

    def set_stage(self, name: str, status: str, **details: Any) -> None:
        self.state["stages"][name] = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **details,
        }
        self.save_state()

    def preflight(self) -> None:
        if self.nproc <= 0:
            raise ValueError("nproc must be positive")
        if not self.python.is_file() or not self.torchrun.is_file():
            raise FileNotFoundError(f"Invalid Python environment: {self.python.parent.parent}")
        self._validate_formal_recipe()
        # Building identities proves that every declared dataset/index/stat/
        # manual-feature dependency exists and is readable.
        build_run_identity(
            "tokenizer",
            self.tokenizer_cfg,
            dependencies={
                f"manual_features.{key}": value
                for key, value in self.tokenizer_cfg.get("manual_features", {}).items()
                if key in {"cache_path", "stats_path", "task_metric_weight_path"} and value
            },
        )
        for task, cfg in self.downstream_cfgs.items():
            # BERT does not exist during a fresh preflight, so dataset files are
            # checked here and the external checkpoint is checked at its stage.
            base = deepcopy(cfg)
            base["model"].pop("bert_checkpoint", None)
            data_root = Path(base["data"]["data_dir"])
            for key in ("train_index", "val_index", "test_index"):
                if not (data_root / base["data"][key]).is_file():
                    raise FileNotFoundError(f"{task}: missing {key}")
            if not Path(base["data"]["area_stats_path"]).is_file():
                raise FileNotFoundError(f"{task}: missing area stats")
        for cfg in (self.tokenizer_cfg, self.bert_cfg):
            if int(cfg["patch"]["samples"]) != 40 or int(cfg["patch"]["stride"]) != 40:
                raise ValueError("Formal EyeVQ requires non-overlapping 40-sample patches")
        write_json(self.output_root / "source_identity.json", self.source_identity)
        self.set_stage(
            "preflight", "complete",
            recipe_version=int(self.recipe["recipe_version"]),
            source_sha256=self.source_identity["sha256"],
        )

    def _run(self, name: str, command: list[str], log_path: Path) -> None:
        self._assert_source_unchanged()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.set_stage(name, "running", command=command, log=str(log_path))
        env = dict(os.environ)
        source_path = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = source_path + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            self.set_stage(name, "failed", returncode=result.returncode, log=str(log_path))
            raise RuntimeError(f"Stage {name} failed; see {log_path}")

    def _ddp(self, module: str, *arguments: str) -> list[str]:
        return [
            str(self.torchrun), "--standalone", f"--nproc_per_node={self.nproc}",
            "-m", module, *map(str, arguments),
        ]

    def _resolved_config(self, name: str, cfg: dict[str, Any]) -> Path:
        path = self.resolved_dir / f"{name}.yaml"
        _write_yaml(path, cfg)
        return path

    def _valid_checkpoint(
        self,
        checkpoint: Path,
        stage: str,
        cfg: dict[str, Any],
        dependencies: dict[str, Path] | None = None,
    ) -> bool:
        if not checkpoint.is_file():
            return False
        expected = build_run_identity(stage, cfg, dependencies=dependencies)
        return checkpoint_has_identity(checkpoint, expected)

    def _latest_valid_step_checkpoint(
        self,
        directory: Path,
        stage: str,
        cfg: dict[str, Any],
        dependencies: dict[str, Path] | None = None,
    ) -> Path | None:
        candidates = sorted(
            directory.glob("ckpt_step*.pt"), key=checkpoint_step, reverse=True
        )
        for candidate in candidates:
            if self._valid_checkpoint(candidate, stage, cfg, dependencies):
                return candidate
            _quarantine(candidate, f"{stage} resume identity mismatch")
        return None

    def train_tokenizer(self) -> None:
        cfg_path = self._resolved_config("tokenizer", self.tokenizer_cfg)
        manual = self.tokenizer_cfg.get("manual_features", {})
        dependencies = {
            f"manual_features.{key}": Path(value)
            for key, value in manual.items()
            if key in {"cache_path", "stats_path", "task_metric_weight_path"} and value
        }
        if self._valid_checkpoint(
            self.tokenizer_checkpoint, "tokenizer", self.tokenizer_cfg, dependencies
        ):
            self._check_quality_gate("tokenizer", self.tokenizer_checkpoint)
            self.set_stage("tokenizer", "reused", checkpoint=str(self.tokenizer_checkpoint))
            return
        if self.tokenizer_checkpoint.exists():
            _quarantine(self.tokenizer_checkpoint, "tokenizer final identity mismatch")
        resume = self._latest_valid_step_checkpoint(
            self.tokenizer_dir, "tokenizer", self.tokenizer_cfg, dependencies
        )
        command = self._ddp(
            "eyemae.eyevq.tokenizer.train",
            "--config", str(cfg_path), "--output_dir", str(self.tokenizer_dir),
        )
        if resume:
            command += ["--resume", str(resume)]
        self._run("tokenizer", command, self.tokenizer_dir / "pipeline.log")
        if not self._valid_checkpoint(
            self.tokenizer_checkpoint, "tokenizer", self.tokenizer_cfg, dependencies
        ):
            raise RuntimeError("Tokenizer finished without a valid content-addressed final checkpoint")
        self._check_quality_gate("tokenizer", self.tokenizer_checkpoint)
        self.set_stage("tokenizer", "complete", checkpoint=str(self.tokenizer_checkpoint))

    def materialize_cache(self) -> None:
        cfg_path = self._resolved_config("bert", self.bert_cfg)
        contract_sha = sha256_json(cache_contract(self.bert_cfg, split="all"))
        try:
            validate_cache_identity(
                self.cache,
                tokenizer_checkpoint=self.tokenizer_checkpoint,
                contract_sha256=contract_sha,
            )
            self.set_stage("code_cache", "reused", cache=str(self.cache))
            return
        except (FileNotFoundError, KeyError, OSError, ValueError):
            if self.cache.exists():
                _quarantine(self.cache, "code cache identity mismatch")
            manifest = self.cache.with_name(self.cache.name + ".manifest.json")
            if manifest.exists():
                _quarantine(manifest, "code cache identity mismatch")
        command = self._ddp(
            "eyemae.eyevq.precompute_codes",
            "--config", str(cfg_path),
            "--tokenizer-checkpoint", str(self.tokenizer_checkpoint),
            "--out", str(self.cache), "--split", "all", "--n-batch", "512",
        )
        self._run("code_cache", command, self.cache_dir / "pipeline.log")
        validate_cache_identity(
            self.cache,
            tokenizer_checkpoint=self.tokenizer_checkpoint,
            contract_sha256=contract_sha,
        )
        self.set_stage("code_cache", "complete", cache=str(self.cache))

    def train_bert(self) -> None:
        cfg_path = self._resolved_config("bert", self.bert_cfg)
        dependencies = {
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "code_ids_cache": self.cache,
        }
        if self._valid_checkpoint(
            self.bert_checkpoint, "bert", self.bert_cfg, dependencies
        ):
            self._check_quality_gate("bert", self.bert_checkpoint)
            self.set_stage("bert", "reused", checkpoint=str(self.bert_checkpoint))
            return
        if self.bert_checkpoint.exists():
            _quarantine(self.bert_checkpoint, "BERT final identity mismatch")
        resume = self._latest_valid_step_checkpoint(
            self.bert_dir, "bert", self.bert_cfg, dependencies
        )
        command = self._ddp(
            "eyemae.eyevq.pretrain.train",
            "--config", str(cfg_path), "--output_dir", str(self.bert_dir),
        )
        if resume:
            command += ["--resume", str(resume)]
        self._run("bert", command, self.bert_dir / "pipeline.log")
        if not self._valid_checkpoint(
            self.bert_checkpoint, "bert", self.bert_cfg, dependencies
        ):
            raise RuntimeError("BERT finished without a valid content-addressed final checkpoint")
        self._check_quality_gate("bert", self.bert_checkpoint)
        self.set_stage("bert", "complete", checkpoint=str(self.bert_checkpoint))

    def train_downstream(self) -> None:
        for task, base_cfg in self.downstream_cfgs.items():
            for seed in (42, 43, 44):
                name = f"{task}_seed{seed}"
                cfg = deepcopy(base_cfg)
                cfg["train"]["seed"] = seed
                cfg_path = self._resolved_config(name, cfg)
                output = self.output_root / "downstream" / task / f"seed{seed}"
                metrics_path = output / "metrics_test.json"
                expected = build_run_identity(
                    "downstream", cfg,
                    dependencies={"bert_checkpoint": self.bert_checkpoint},
                )
                if metrics_path.is_file():
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics.get("run_identity") == expected and metrics.get("test_evaluated") is True:
                            self.set_stage(name, "reused", metrics=str(metrics_path))
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass
                    _quarantine(metrics_path, "downstream metrics identity mismatch")
                self._run(
                    name,
                    self._ddp(
                        "eyemae.eyevq.downstream.train_mil",
                        "--config", str(cfg_path), "--output_dir", str(output),
                    ),
                    output / "pipeline.log",
                )
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("run_identity") != expected or metrics.get("test_evaluated") is not True:
                    raise RuntimeError(f"{name} did not produce identity-matched test metrics")
                self.set_stage(name, "complete", metrics=str(metrics_path))

    def summarize(self) -> None:
        payload: dict[str, Any] = {
            "recipe": self.recipe,
            "source_sha256": self.source_identity["sha256"],
            "selection_policy": "configuration fixed from validation-only ablations; test never ranks candidates",
            "test_status": "internal exploratory; test was observed during development",
            "tokenizer_checkpoint": str(self.tokenizer_checkpoint),
            "bert_checkpoint": str(self.bert_checkpoint),
            "tasks": {},
        }
        for task in ("mci", "pd5"):
            payload["tasks"][task] = [
                json.loads(
                    (self.output_root / "downstream" / task / f"seed{seed}" / "metrics_test.json")
                    .read_text(encoding="utf-8")
                )
                for seed in (42, 43, 44)
            ]
            payload["tasks"][task + "_ensemble"] = self._ensemble_task(task)
        write_json(self.output_root / "search_summary.json", {
            "selection_uses_test": False,
            "selected": {
                "data": "V4 validation-selected internal reference",
                "tokenizer": "40K, joint stimulus-isolated tanh FSQ [9,7,5,5]",
                "bert": "50K, paired span 1-5 uniform, mask 0.60, factorized heads",
                "downstream": "top-8, LR 1e-5, MCI/PD5 K16, shared hidden-128 head",
            },
            "evidence": "docs/iclr_eyevq_paper.md#41-final-training-recipe-and-parameter-selection",
        })
        write_json(self.output_root / "final_summary.json", payload)
        self.set_stage("summary", "complete", path=str(self.output_root / "final_summary.json"))

    def _ensemble_task(self, task: str) -> dict[str, Any]:
        output_dir = self.output_root / "downstream" / task / "ensemble"
        output_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "method": "mean logits across seeds 42/43/44",
            "checkpoint_selection": "each seed independently selected by validation only",
        }
        for split, filename in (
            ("val", "predictions_val_best.csv"),
            ("test", "predictions_test.csv"),
        ):
            seed_rows: list[dict[str, dict[str, str]]] = []
            for seed in (42, 43, 44):
                path = self.output_root / "downstream" / task / f"seed{seed}" / filename
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = {
                        row["subject_key"]: row for row in csv.DictReader(handle)
                    }
                seed_rows.append(rows)
            subjects = sorted(seed_rows[0])
            if any(sorted(rows) != subjects for rows in seed_rows[1:]):
                raise RuntimeError(f"{task} {split} ensemble subject sets differ across seeds")
            labels = [int(seed_rows[0][subject]["label"]) for subject in subjects]
            output_rows: list[dict[str, Any]] = []
            if task == "mci":
                logits = [
                    sum(float(rows[subject]["logit"]) for rows in seed_rows) / 3.0
                    for subject in subjects
                ]
                metrics = compute_binary_metrics(
                    labels, logits, threshold=0.5, prefix=f"{split}/subject"
                )
                for subject, label, logit in zip(subjects, labels, logits):
                    probability = sigmoid(logit)
                    output_rows.append({
                        "subject_key": subject,
                        "label": label,
                        "logit": logit,
                        "prob": probability,
                        "pred": int(probability >= 0.5),
                    })
            else:
                logits = [
                    [
                        sum(float(rows[subject][f"logit_{class_id}"]) for rows in seed_rows) / 3.0
                        for class_id in range(5)
                    ]
                    for subject in subjects
                ]
                metrics = compute_multiclass_metrics(
                    labels, logits, num_classes=5, prefix=f"{split}/subject"
                )
                for subject, label, row_logits in zip(subjects, labels, logits):
                    probabilities = softmax(row_logits)
                    row: dict[str, Any] = {
                        "subject_key": subject,
                        "label": label,
                        "pred": max(range(5), key=lambda class_id: probabilities[class_id]),
                    }
                    row.update({f"logit_{i}": value for i, value in enumerate(row_logits)})
                    row.update({f"prob_{i}": value for i, value in enumerate(probabilities)})
                    output_rows.append(row)
            prediction_path = output_dir / f"predictions_{split}.csv"
            temporary = prediction_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)
            temporary.replace(prediction_path)
            result[split] = metrics
        write_json(output_dir / "metrics.json", result)
        return result

    def run(self, *, preflight_only: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state["status"] = "running"
        self.save_state()
        try:
            self.preflight()
            if preflight_only:
                self.state["status"] = "preflight_complete"
                self.save_state()
                return
            self.train_tokenizer()
            self.materialize_cache()
            self.train_bert()
            self.train_downstream()
            self.summarize()
            self.state["status"] = "complete"
            self.save_state()
        except Exception as error:
            self.state["status"] = "failed"
            self.state["error"] = f"{type(error).__name__}: {error}"
            self.save_state()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/eyevq/final")
    parser.add_argument("--python-env", default="/home/jinfanhe/miniconda3/envs/jinf")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    Pipeline(Path(args.output_root), Path(args.python_env), args.nproc).run(
        preflight_only=args.preflight_only
    )


if __name__ == "__main__":
    main()
