from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DATASET_ROOT = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2")
OLD_PRETRAIN_CHECKPOINT = Path(
    "outputs/pretrain_v3_fast_cache44_nooffset/eyemae_cnn_512_12l_patch20_stimtoken/checkpoint_best.pt"
)
OLD_AREA_STATS = Path("outputs/area_stats_fast_packed_seed42.json")
V2_AREA_STATS = Path("outputs/area_stats_fast_packed_v2_clean_full_subject_seed20260622.json")
MODES = ("scratch", "linear_probe", "partial", "full")
MODE_OUTPUT = {
    "scratch": "scratch_full",
    "linear_probe": "pretrained_linear_probe",
    "partial": "pretrained_partial",
    "full": "pretrained_full",
}
TASK_BASE = {
    "pd_related_5class": "pd_related_5class_random_seed20260620_fast",
    "pd_binary": "pd_binary_random_seed20260620_fast",
    "epilepsy_binary": "epilepsy_binary",
    "detox_binary": "detox_binary",
    "migraine_binary": "migraine_binary",
    "ad_binary": "ad_binary",
    "mci_binary": "mci_original_only_binary",
    "mci_matched_binary": "mci_matched_binary_random_seed20260622_label_fixed",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def update_pretrain_config(*, config_dir: Path) -> Path:
    cfg = read_yaml(Path("configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml"))
    cfg["experiment"]["name"] = "pretrain_v2_clean_eyemae_cnn_512_12l_patch20_stimtoken"
    cfg["experiment"]["output_dir"] = "outputs/pretrain_v2_clean/eyemae_cnn_512_12l_patch20_stimtoken"
    cfg["data"]["data_dir"] = str(DATASET_ROOT / "pretrain")
    cfg["data"]["train_index"] = "pretrain/pretrain_train.csv"
    cfg["data"]["val_index"] = "pretrain/pretrain_validation.csv"
    cfg["data"]["test_index"] = "pretrain/pretrain_test.csv"
    cfg["data"]["max_open_shards_per_worker"] = 44
    cfg["data"]["validate_offsets"] = False
    cfg["split"]["seed"] = 20260622
    cfg["split"]["split_summary"] = "pretrain/pretrain_split_summary.json"
    cfg["area"]["stats_path"] = str(V2_AREA_STATS)
    cfg["area"]["max_frames_per_subject"] = None
    cfg["area"]["max_global_frames"] = 2_000_000
    cfg["train"]["max_seq_tokens_per_gpu"] = 90000
    cfg["train"]["max_trials_per_gpu"] = 256
    out = config_dir / "eyemae_cnn_512_12l_v2_clean.yaml"
    write_yaml(out, cfg)
    return out


def update_downstream_config(
    *,
    task: str,
    mode: str,
    config_dir: Path,
    output_root: Path,
    checkpoint: Path,
    area_stats: Path,
    max_epochs: int,
) -> Path:
    base_name = TASK_BASE[task]
    cfg = read_yaml(Path("configs/downstream") / f"{base_name}_{mode}.yaml")
    cfg["experiment"]["name"] = f"downstream_v2_clean_{task}_{mode}"
    cfg["experiment"]["output_dir"] = str(output_root / task / MODE_OUTPUT[mode])
    cfg["data"]["data_dir"] = str(DATASET_ROOT / "finetune" / task)
    cfg["data"]["train_index"] = "train.csv"
    cfg["data"]["val_index"] = "validation.csv"
    cfg["data"]["test_index"] = "test.csv"
    cfg["data"]["max_open_shards_per_worker"] = 44
    cfg["data"]["validate_offsets"] = False
    cfg["split"]["strategy"] = "provided_subject_heldout"
    cfg["split"]["seed"] = 20260622
    cfg["split"]["train_ratio"] = 0.64
    cfg["split"]["val_ratio"] = 0.16
    cfg["split"]["test_ratio"] = 0.20
    cfg["split"]["split_summary"] = "split_summary.json"
    cfg["label"]["task_name"] = task
    cfg["label"]["view"] = task
    cfg["area"]["stats_path"] = str(area_stats)
    cfg["downstream"]["task_name"] = task
    cfg["downstream"]["disease"] = task
    cfg["downstream"]["mode"] = mode
    cfg["downstream"]["pretrained_checkpoint"] = None if mode == "scratch" else str(checkpoint)
    if "pretrained" in cfg:
        cfg["pretrained"]["checkpoint"] = None if mode == "scratch" else str(checkpoint)
        cfg["pretrained"]["pretrain_config"] = "configs/eyemae_cnn_512_12l_fast_cache44_nooffset.yaml"
    if "finetune" in cfg:
        cfg["finetune"]["mode"] = mode
    cfg["downstream_train"]["max_epochs"] = int(max_epochs)
    cfg["downstream_train"]["early_stopping_patience"] = 10
    cfg["downstream_train"]["min_epochs_before_early_stopping"] = 0
    cfg["downstream_train"]["prefetch_factor"] = 4
    cfg["downstream_checkpoint"]["monitor"] = (
        "validation/subject_macro_auroc_ovr" if cfg["label"].get("type") == "multiclass" else "validation/subject_auroc"
    )
    cfg["downstream_checkpoint"]["mode"] = "max"
    out = config_dir / f"{task}_{mode}.yaml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/v2_corrected_max30_oldpretrain"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/downstream_v2_corrected_max30_oldpretrain"))
    parser.add_argument("--queue", type=Path, default=Path("configs/v2_corrected_max30_oldpretrain/queue.txt"))
    parser.add_argument("--checkpoint", type=Path, default=OLD_PRETRAIN_CHECKPOINT)
    parser.add_argument("--area-stats", type=Path, default=OLD_AREA_STATS)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    pretrain_config = update_pretrain_config(config_dir=args.config_dir)
    downstream_configs = []
    for task in TASK_BASE:
        for mode in MODES:
            downstream_configs.append(
                update_downstream_config(
                    task=task,
                    mode=mode,
                    config_dir=args.config_dir,
                    output_root=args.output_root,
                    checkpoint=args.checkpoint,
                    area_stats=args.area_stats,
                    max_epochs=args.max_epochs,
                )
            )
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.queue.write_text(
        "# corrected v2 downstream configs using the existing v3 pretrained checkpoint and area stats.\n"
        "# This isolates the cleaned data/split effect from rerunning pretraining.\n"
        + "\n".join(str(path) for path in downstream_configs)
        + "\n",
        encoding="utf-8",
    )
    print("pretrain_config", pretrain_config)
    print("queue", args.queue)
    print("num_downstream_configs", len(downstream_configs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
