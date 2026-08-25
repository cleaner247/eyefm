#!/usr/bin/env python3
"""GPU-0 low-cost full-12-layer diagnostic at epochs 10/20/30/40."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/eyevq/v6_full12_periodic10_ckpt30k_20260826"
DEFAULT_BERT = DEFAULT_OUTPUT / "bert_snapshot/ckpt_best_step030000.pt"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def task_config(task: str, bert_checkpoint: Path) -> dict:
    template = ROOT / f"configs/eyevq/final/{task}.yaml"
    cfg = deepcopy(load_yaml(template))
    cfg["model"].update(
        {
            "bert_checkpoint": str(bert_checkpoint),
            "freeze_embedding": True,
            "freeze_bottom_layers": 0,
        }
    )
    cfg["mil"].update(
        {
            "subjects_per_gpu": 4,
            "trials_per_task": 16,
            "eligibility_min_trials_per_task": 16,
            "evaluation_min_trials_per_task": 4,
            "sample_without_replacement": True,
            "epoch_random_trial_sampling": True,
            "eval_use_all_trials": True,
        }
    )
    cfg["train"].update(
        {
            "seed": 42,
            "epochs": 40,
            "encoder_lr": 5.0e-6,
            "encoder_min_lr": 5.0e-7,
            "head_lr": 1.0e-5,
            "head_min_lr": 1.0e-6,
            "warmup_epochs": 2,
            "early_stopping_min_epochs": 40,
            "early_stopping_patience_epochs": 40,
            "periodic_test_every_epochs": 10,
            "num_trial_views": 1,
            "trial_view_consistency_weight": 0.0,
        }
    )
    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bert-checkpoint", type=Path, default=DEFAULT_BERT)
    args = parser.parse_args()
    output = args.output.resolve()
    bert_checkpoint = args.bert_checkpoint.resolve()
    if not bert_checkpoint.is_file():
        raise FileNotFoundError(bert_checkpoint)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    for task in ("mci", "pd5"):
        task_output = output / task
        config_path = task_output / "config.yaml"
        write_yaml(config_path, task_config(task, bert_checkpoint))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eyemae.eyevq.downstream.train_mil",
                "--config",
                str(config_path),
                "--output_dir",
                str(task_output),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
