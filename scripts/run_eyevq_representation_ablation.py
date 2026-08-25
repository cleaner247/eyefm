#!/usr/bin/env python3
"""Materialize and optionally execute the staged EyeVQ ablation matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from eyemae.eyevq.config import validate_bert_config, validate_tokenizer_config


def set_dotted(config: dict, key: str, value) -> None:
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(config: dict, overrides: dict | None) -> dict:
    result = deepcopy(config)
    for key, value in (overrides or {}).items():
        set_dotted(result, str(key), value)
    return result


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
    body = " ".join(shlex.quote(part) for part in command)
    return f"{prefix} {body}".strip()


def materialize(matrix_path: Path, phase: str, selected: set[str] | None) -> tuple[Path, list[dict]]:
    matrix = read_yaml(matrix_path)
    root = matrix_path.parents[3]
    base = matrix["base"]
    execution = matrix["execution"]
    output_root = root / base["output_root"] / phase
    tokenizer_base = read_yaml(root / base["tokenizer_config"])
    bert_base = read_yaml(root / base["bert_config"])
    downstream_base = {
        "mci": read_yaml(root / base["mci_config"]),
        "pd5": read_yaml(root / base["pd5_config"]),
    }
    steps = int(execution[f"{phase}_bert_steps"])
    tokenizer_steps = int(execution[f"{phase}_tokenizer_steps"])
    epochs = int(execution.get(f"{phase}_downstream_epochs", 0))
    seeds = [int(seed) for seed in execution[f"{phase}_seeds"]]
    env = {"CUDA_VISIBLE_DEVICES": str(execution["gpu_ids"])}
    nproc = str(execution["nproc_per_node"])
    plan: list[dict] = []
    known_names = {str(candidate["name"]) for candidate in matrix["candidates"]}
    if selected:
        unknown = selected - known_names
        if unknown:
            raise ValueError(f"Unknown candidate name(s): {sorted(unknown)}")

    for candidate in matrix["candidates"]:
        name = str(candidate["name"])
        if selected and name not in selected:
            continue
        run_dir = output_root / name
        config_dir = run_dir / "configs"
        tokenizer_cfg = apply_overrides(tokenizer_base, candidate.get("tokenizer"))
        bert_cfg = apply_overrides(bert_base, candidate.get("bert"))
        bert_cfg["train"]["total_steps"] = steps
        tokenizer_checkpoint = root / base["tokenizer_checkpoint"]
        code_cache = root / base["code_ids_cache"]
        commands: list[dict] = []

        if candidate.get("retrain_tokenizer", False):
            stage_a_steps = int(tokenizer_cfg["train"].get("stage_a_steps", 0))
            tokenizer_cfg["train"]["stage_b_steps"] = tokenizer_steps - stage_a_steps
            if tokenizer_cfg["train"]["stage_b_steps"] < 1:
                raise ValueError(f"{name}: tokenizer stage_a_steps exceeds the phase budget")
            validate_tokenizer_config(tokenizer_cfg)
            tokenizer_path = config_dir / "tokenizer.yaml"
            write_yaml(tokenizer_path, tokenizer_cfg)
            tokenizer_checkpoint = run_dir / "tokenizer" / "ckpt_final.pt"
            code_cache = run_dir / "cache" / "code_ids_all.npz"
            commands.append({
                "stage": "tokenizer",
                "command": ["torchrun", f"--nproc_per_node={nproc}", "-m", "eyemae.eyevq.tokenizer.train", "--config", str(tokenizer_path), "--output_dir", str(run_dir / "tokenizer")],
                "env": env,
            })

        no_tokenizer = bool(candidate.get("no_tokenizer", False))
        if no_tokenizer:
            bert_cfg["train"].pop("tokenizer_checkpoint", None)
            bert_cfg["train"].pop("code_ids_cache", None)
        else:
            if not candidate.get("retrain_tokenizer", False):
                for artifact in (tokenizer_checkpoint, code_cache):
                    if not artifact.is_file():
                        raise FileNotFoundError(
                            f"{name}: required reusable artifact does not exist: {artifact}"
                        )
            bert_cfg["train"]["tokenizer_checkpoint"] = str(tokenizer_checkpoint)
            bert_cfg["train"]["code_ids_cache"] = str(code_cache)
            if candidate.get("retrain_tokenizer", False):
                cache_cfg_path = config_dir / "bert.yaml"
                commands.append({
                    "stage": "cache",
                    "command": [sys.executable, "-m", "eyemae.eyevq.precompute_codes", "--config", str(cache_cfg_path), "--tokenizer-checkpoint", str(tokenizer_checkpoint), "--out", str(code_cache), "--split", "all", "--device", "cuda"],
                    "env": {"CUDA_VISIBLE_DEVICES": str(execution["gpu_ids"]).split(",")[0]},
                })
        validate_bert_config(bert_cfg)
        bert_path = config_dir / "bert.yaml"
        write_yaml(bert_path, bert_cfg)
        commands.append({
            "stage": "bert",
            "command": ["torchrun", f"--nproc_per_node={nproc}", "-m", "eyemae.eyevq.pretrain.train", "--config", str(bert_path), "--output_dir", str(run_dir / "bert")],
            "env": env,
        })

        bert_checkpoint = run_dir / "bert" / "ckpt_best.pt"
        for task, task_cfg_base in downstream_base.items():
            if phase == "probe":
                task_cfg = deepcopy(task_cfg_base)
                task_cfg["model"]["bert_checkpoint"] = str(bert_checkpoint)
                task_path = config_dir / f"{task}_probe.yaml"
                write_yaml(task_path, task_cfg)
                task_run = run_dir / task
                cache_path = task_run / "train_cls.pt"
                result_path = task_run / "probe_5fold.json"
                minimum = int(task_cfg["mil"].get("eligibility_min_trials_per_task", 4))
                probe_env = {
                    "CUDA_VISIBLE_DEVICES": str(execution["gpu_ids"]).split(",")[0]
                }
                commands.append({
                    "stage": f"{task}_cache_train_cls",
                    "command": [sys.executable, "-m", "eyemae.eyevq.downstream.cache_trial_cls", "--config", str(task_path), "--output", str(cache_path), "--min-trials-per-task", str(minimum), "--train-only"],
                    "env": probe_env,
                })
                commands.append({
                    "stage": f"{task}_train_only_probe",
                    "command": [sys.executable, "-m", "eyemae.eyevq.downstream.screen_bert_representation", "--cache", str(cache_path), "--output", str(result_path), "--folds", "5", "--seed", "42", "--c", "0.1"],
                    "env": probe_env,
                })
                continue
            for seed in seeds:
                task_cfg = deepcopy(task_cfg_base)
                task_cfg["model"]["bert_checkpoint"] = str(bert_checkpoint)
                task_cfg["train"]["seed"] = seed
                task_cfg["train"]["epochs"] = epochs
                task_path = config_dir / f"{task}_seed{seed}.yaml"
                write_yaml(task_path, task_cfg)
                command = [sys.executable, "-m", "eyemae.eyevq.downstream.train_mil", "--config", str(task_path), "--output_dir", str(run_dir / task / f"seed{seed}")]
                if phase != "confirm":
                    command.append("--skip-test")
                commands.append({"stage": f"{task}_seed{seed}", "command": command, "env": {"CUDA_VISIBLE_DEVICES": str(execution["gpu_ids"]).split(",")[0]}})

        plan.append({
            "name": name,
            "group": candidate["group"],
            "phase": phase,
            "commands": commands,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "execution_plan.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    return manifest, plan


def execute(plan: list[dict]) -> None:
    for candidate in plan:
        candidate_root = Path(candidate["commands"][-1]["command"][-1]).parents[1]
        terminal_outputs = list(candidate_root.glob("**/ckpt_final.pt"))
        if terminal_outputs:
            raise FileExistsError(
                f"Refusing to overwrite completed outputs for {candidate['name']}: "
                f"{terminal_outputs[0]}"
            )
        for item in candidate["commands"]:
            print(f"[{candidate['name']}] {item['stage']}: {command_text(item['command'], item['env'])}", flush=True)
            environment = dict(__import__("os").environ)
            environment.update(item["env"])
            subprocess.run(item["command"], env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/eyevq/ablations/v6_representation_ablation.yaml")
    parser.add_argument("--phase", choices=("probe", "screen", "confirm"), default="probe")
    parser.add_argument("--only", default=None, help="Comma-separated candidate names")
    parser.add_argument("--execute", action="store_true", help="Run sequentially; default only materializes and validates")
    args = parser.parse_args()
    selected = None if not args.only else {value.strip() for value in args.only.split(",") if value.strip()}
    manifest, plan = materialize(Path(args.matrix).resolve(), args.phase, selected)
    print(f"Validated {len(plan)} candidates; plan: {manifest}")
    for candidate in plan:
        for item in candidate["commands"]:
            print(command_text(item["command"], item["env"]))
    if args.execute:
        execute(plan)


if __name__ == "__main__":
    main()
