from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def read_config_list(path: Path) -> list[Path]:
    configs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(Path(line))
    return configs


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def task_mode(cfg: dict[str, Any]) -> tuple[str, str]:
    task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
    mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
    return task, mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--out-dir", default="configs/downstream_epoch1")
    parser.add_argument("--out-list", default="configs/downstream_epoch1/queue_epoch1.txt")
    parser.add_argument("--output-root", default="outputs/downstream_v3_fast_convergence/epoch1_train")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    output_root = Path(args.output_root)
    generated: list[Path] = []
    for cfg_path in read_config_list(Path(args.config_list_file)):
        cfg = read_yaml(cfg_path)
        task, mode = task_mode(cfg)
        cfg.setdefault("experiment", {})["output_dir"] = str(output_root / task / mode)
        cfg.setdefault("downstream_train", {})["max_epochs"] = 1
        cfg["downstream_train"]["early_stopping_patience"] = 10
        cfg["downstream_train"]["min_epochs_before_early_stopping"] = 0
        cfg.setdefault("debug", {}).pop("max_train_batches", None)
        cfg.setdefault("debug", {}).pop("max_eval_batches", None)
        target = out_dir / f"{cfg_path.stem}_epoch1.yaml"
        write_yaml(target, cfg)
        generated.append(target)

    out_list = Path(args.out_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(str(path) for path in generated) + "\n", encoding="utf-8")
    print({"configs": len(generated), "out_list": str(out_list), "output_root": str(output_root)})


if __name__ == "__main__":
    main()
