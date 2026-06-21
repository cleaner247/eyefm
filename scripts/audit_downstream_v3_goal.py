from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from eyemae.downstream_metrics import compute_binary_metrics, compute_multiclass_metrics


EXPECTED_TASKS = [
    "pd_related_5class",
    "pd_binary",
    "epilepsy_binary",
    "detox_binary",
    "migraine_binary",
    "ad_binary",
    "mci_original_only_binary",
    "mci_matched_binary_random_seed20260621",
]
EXPECTED_MODES = ["scratch", "linear_probe", "partial", "full"]
PLAN_SPLIT_NAMES = {"val": "validation"}
EXPECTED_JOB_COUNT = len(EXPECTED_TASKS) * len(EXPECTED_MODES)
PD_RANDOM_VIEW_BY_TASK = {
    "pd_related_5class": "PD相关_random_seed20260620",
    "pd_binary": "PD相关_binary_random_seed20260620",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def read_config_list(path: Path) -> list[Path]:
    configs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(Path(line))
    return configs


def status_item_config(item: Any) -> str | None:
    if isinstance(item, dict):
        config = item.get("config")
        if config is None:
            return None
        return str(config)
    if isinstance(item, str):
        return item
    return None


def plan_split(split: str) -> str:
    return PLAN_SPLIT_NAMES.get(split, split)


def index_path(cfg: dict[str, Any], split: str) -> Path:
    data_dir = Path(str(cfg["data"]["data_dir"]))
    key = "val_index" if split == "val" else f"{split}_index"
    return data_dir / str(cfg["data"][key])


def expected_split_summary_rel(cfg: dict[str, Any]) -> str:
    train_index = Path(str(cfg["data"]["train_index"]))
    return str(train_index.parent / "split_summary.json")


def read_subject_ids(path: Path, subject_key: str) -> set[str]:
    subjects: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if subject_key not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing subject column {subject_key}")
        for row in reader:
            subjects.add(str(row[subject_key]))
    return subjects


def split_index_summary(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    path = index_path(cfg, split)
    subject_key = str(cfg.get("data", {}).get("subject_key", "ml_subject_id"))
    subjects: set[str] = set()
    subject_eye: dict[str, str] = {}
    trial_eye_counts: dict[str, int] = {}
    trial_label_counts: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject = str(row[subject_key])
            subjects.add(subject)
            eye = str(row.get("source_suffix", ""))
            subject_eye[subject] = eye
            trial_eye_counts[eye] = trial_eye_counts.get(eye, 0) + 1
            label = str(row.get("health_label", ""))
            trial_label_counts[label] = trial_label_counts.get(label, 0) + 1
    subject_eye_counts: dict[str, int] = {}
    for eye in subject_eye.values():
        subject_eye_counts[eye] = subject_eye_counts.get(eye, 0) + 1
    return {
        "num_subjects": len(subjects),
        "label_counts": dict(sorted(trial_label_counts.items())),
        "trial_eye_availability_counts": dict(sorted(trial_eye_counts.items())),
        "subject_eye_availability_counts": dict(sorted(subject_eye_counts.items())),
    }


def metric_keys_for(cfg: dict[str, Any], split: str) -> list[str]:
    prefix = f"{split}/subject"
    if str(cfg.get("label", {}).get("type", "binary")) == "multiclass":
        return [
            f"{prefix}/macro_auroc_ovr",
            f"{prefix}/macro_auprc_ovr",
            f"{prefix}/balanced_accuracy",
            f"{prefix}/macro_f1",
            f"{prefix}/weighted_f1",
            f"{prefix}/cohen_kappa",
        ]
    return [
        f"{prefix}/auroc",
        f"{prefix}/auprc",
        f"{prefix}/balanced_accuracy",
        f"{prefix}/f1",
        f"{prefix}/weighted_f1",
        f"{prefix}/cohen_kappa",
    ]


def prediction_candidates(out_dir: Path, split: str, level: str) -> list[Path]:
    if level not in {"trial", "subject"}:
        raise ValueError(f"unknown prediction level: {level}")
    split_alias = plan_split(split)
    if level == "trial":
        names = [
            f"trial_predictions_{split}.csv",
            f"{split}_predictions.csv",
        ]
        if split_alias != split:
            names.extend([f"trial_predictions_{split_alias}.csv", f"{split_alias}_predictions.csv"])
    else:
        names = [
            f"subject_predictions_{split}.csv",
            f"{split}_subject_predictions.csv",
        ]
        if split_alias != split:
            names.extend([f"subject_predictions_{split_alias}.csv", f"{split_alias}_subject_predictions.csv"])
    return [out_dir / name for name in names]


def read_prediction_rows(paths: list[Path]) -> list[dict[str, str]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return []


def compute_prediction_metrics(
    rows: list[dict[str, str]],
    cfg: dict[str, Any],
    *,
    split: str,
    level: str,
) -> dict[str, float]:
    label_type = str(cfg.get("label", {}).get("type", "binary"))
    labels = [int(float(row["label"])) for row in rows]
    prefix = f"{split}/{level}"
    if label_type == "multiclass":
        num_classes = int(cfg.get("label", {}).get("num_classes", 5))
        logits = [[float(row[f"logit_{c}"]) for c in range(num_classes)] for row in rows]
        return compute_multiclass_metrics(labels, logits, num_classes=num_classes, prefix=prefix)
    threshold = float(cfg.get("downstream_eval", {}).get("threshold", 0.5))
    logits = [float(row["logit"]) for row in rows]
    return compute_binary_metrics(labels, logits, threshold=threshold, prefix=prefix)


def backfill_prediction_metrics(out_dir: Path, cfg: dict[str, Any]) -> bool:
    metrics_path = out_dir / "metrics_final.json"
    if not metrics_path.exists():
        return False
    metrics = read_json(metrics_path)
    changed = False
    for split in ["train", "val", "test"]:
        for level in ["trial", "subject"]:
            prefix = f"{split}/{level}"
            if f"{prefix}/weighted_f1" in metrics and f"{prefix}/cohen_kappa" in metrics:
                continue
            rows = read_prediction_rows(prediction_candidates(out_dir, split, level))
            if not rows:
                continue
            computed = compute_prediction_metrics(rows, cfg, split=split, level=level)
            for key, value in computed.items():
                if key not in metrics:
                    metrics[key] = value
                    changed = True
    if changed:
        write_json(metrics_path, metrics)
    return changed


def materialize_plan_artifacts(out_dir: Path, cfg_path: Path) -> None:
    cfg_json = out_dir / "config.json"
    resolved_yaml = out_dir / "resolved_config.yaml"
    if not resolved_yaml.exists():
        if cfg_json.exists():
            cfg = read_json(cfg_json)
        else:
            cfg = read_yaml(cfg_path)
        resolved_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    elif cfg_json.exists():
        cfg = read_json(cfg_json)
    else:
        cfg = read_yaml(cfg_path)

    metrics_final = out_dir / "metrics_final.json"
    if metrics_final.exists():
        backfill_prediction_metrics(out_dir, cfg)
        metrics = read_json(metrics_final)
        for split in ["train", "val", "test"]:
            split_metrics = {key: value for key, value in metrics.items() if key.startswith(f"{split}/")}
            if split_metrics:
                write_json(out_dir / f"{plan_split(split)}_metrics.json", split_metrics)

    aliases = {
        "val_predictions.csv": "validation_predictions.csv",
        "val_subject_predictions.csv": "validation_subject_predictions.csv",
        "trial_predictions_val.csv": "trial_predictions_validation.csv",
        "subject_predictions_val.csv": "subject_predictions_validation.csv",
        "confusion_matrix_val.json": "confusion_matrix_validation.json",
    }
    for source_name, target_name in aliases.items():
        source = out_dir / source_name
        target = out_dir / target_name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)

    run_summary_path = out_dir / "run_summary.json"
    if run_summary_path.exists():
        run_summary = read_json(run_summary_path)
    else:
        run_summary = {}
    if metrics_final.exists() or run_summary:
        cfg = read_json(cfg_json) if cfg_json.exists() else read_yaml(cfg_path)
        task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name", ""))
        mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode", ""))
        split_summaries = dict(run_summary.get("splits", {}))
        for split in ["train", "val", "test"]:
            current = dict(split_summaries.get(split, {}))
            try:
                index_summary = split_index_summary(cfg, split)
            except Exception:
                index_summary = {}
            for key, value in index_summary.items():
                current.setdefault(key, value)
            split_summaries[split] = current
        run_summary.setdefault("task_name", task)
        run_summary.setdefault("mode", mode)
        run_summary.setdefault("finetune_mode", mode)
        run_summary.setdefault("pretrained_checkpoint", cfg.get("downstream", {}).get("pretrained_checkpoint") if mode != "scratch" else None)
        run_summary.setdefault(
            "pretraining_exposure",
            {
                "mode": cfg.get("pretraining_exposure", {}).get("mode", "all_unlabeled_or_unknown"),
                "pretrain_subject_manifest": cfg.get("pretraining_exposure", {}).get("pretrain_subject_manifest"),
            },
        )
        run_summary.setdefault("num_train_subjects", split_summaries.get("train", {}).get("num_subjects", 0))
        run_summary.setdefault("num_validation_subjects", split_summaries.get("val", {}).get("num_subjects", 0))
        run_summary.setdefault("num_test_subjects", split_summaries.get("test", {}).get("num_subjects", 0))
        run_summary.setdefault("label_counts", {name: summary.get("label_counts", {}) for name, summary in split_summaries.items()})
        run_summary.setdefault(
            "subject_eye_availability_counts",
            {name: summary.get("subject_eye_availability_counts", {}) for name, summary in split_summaries.items()},
        )
        run_summary["splits"] = split_summaries
        write_json(run_summary_path, run_summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--status-json", default="outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/status.json")
    parser.add_argument("--output-root", default="outputs/downstream_v3_fast")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--materialize-plan-artifacts", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "config_list_file": args.config_list_file,
        "status_json": args.status_json,
        "output_root": args.output_root,
        "jobs": [],
    }

    config_paths = read_config_list(Path(args.config_list_file))
    if len(config_paths) != EXPECTED_JOB_COUNT:
        errors.append(f"Expected {EXPECTED_JOB_COUNT} configs, found {len(config_paths)}")
    if len({str(path) for path in config_paths}) != len(config_paths):
        errors.append("Config list contains duplicate paths")

    seen_pairs: set[tuple[str, str]] = set()
    output_dirs: list[Path] = []
    config_complete: dict[str, bool] = {}
    split_cache: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}

    for cfg_path in config_paths:
        if not cfg_path.exists():
            errors.append(f"Missing config: {cfg_path}")
            continue
        cfg = read_yaml(cfg_path)
        task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
        mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
        pair = (task, mode)
        seen_pairs.add(pair)
        out_dir = Path(str(cfg.get("experiment", {}).get("output_dir", "")))
        output_dirs.append(out_dir)

        job_report: dict[str, Any] = {
            "config": str(cfg_path),
            "task": task,
            "mode": mode,
            "output_dir": str(out_dir),
            "complete": False,
        }
        report["jobs"].append(job_report)

        if out_dir == Path(".") or not str(out_dir).startswith(str(args.output_root)):
            errors.append(f"{cfg_path}: output_dir is not under {args.output_root}: {out_dir}")
        if int(cfg.get("downstream_train", {}).get("max_epochs", -1)) != 100:
            errors.append(f"{cfg_path}: max_epochs is not 100")
        if int(cfg.get("downstream_train", {}).get("early_stopping_patience", -1)) != 10:
            errors.append(f"{cfg_path}: early_stopping_patience is not 10")
        if int(cfg.get("downstream_train", {}).get("min_epochs_before_early_stopping", -1)) != 0:
            errors.append(f"{cfg_path}: min_epochs_before_early_stopping is not 0")

        if task in PD_RANDOM_VIEW_BY_TASK:
            expected_view = PD_RANDOM_VIEW_BY_TASK[task]
            for split in ["train", "val", "test"]:
                path = str(cfg.get("data", {}).get("val_index" if split == "val" else f"{split}_index", ""))
                if expected_view not in path:
                    errors.append(f"{cfg_path}: PD split {split} does not use {expected_view}: {path}")
            if int(cfg.get("split", {}).get("seed", -1)) != 20260620:
                errors.append(f"{cfg_path}: PD random split seed is not 20260620")
        else:
            for split in ["train", "val", "test"]:
                path = str(cfg.get("data", {}).get("val_index" if split == "val" else f"{split}_index", ""))
                if "PD相关_random_seed20260620" in path or "PD相关_binary_random_seed20260620" in path:
                    errors.append(f"{cfg_path}: non-PD task unexpectedly uses PD random split for {split}")

        try:
            subject_key = str(cfg.get("data", {}).get("subject_key", "ml_subject_id"))
            split_paths = {split: index_path(cfg, split) for split in ["train", "val", "test"]}
            for split, path in split_paths.items():
                if not path.exists():
                    errors.append(f"{cfg_path}: missing {split} index {path}")
            expected_summary = expected_split_summary_rel(cfg)
            configured_summary = str(cfg.get("split", {}).get("split_summary", ""))
            if configured_summary != expected_summary:
                errors.append(
                    f"{cfg_path}: split.split_summary should be {expected_summary}, got {configured_summary}"
                )
            summary_path = Path(str(cfg["data"]["data_dir"])) / expected_summary
            if not summary_path.exists():
                errors.append(f"{cfg_path}: missing split summary {summary_path}")
            cache_key = tuple(str(split_paths[split]) for split in ["train", "val", "test"]) + (subject_key,)
            if all(path.exists() for path in split_paths.values()) and cache_key not in split_cache:
                split_cache[cache_key] = {split: read_subject_ids(path, subject_key) for split, path in split_paths.items()}
            subjects = split_cache.get(cache_key)
            if subjects:
                overlaps = {
                    "train_val": subjects["train"] & subjects["val"],
                    "train_test": subjects["train"] & subjects["test"],
                    "val_test": subjects["val"] & subjects["test"],
                }
                for name, overlap in overlaps.items():
                    if overlap:
                        errors.append(f"{cfg_path}: subject overlap {name} has {len(overlap)} subjects")
                job_report["subjects"] = {split: len(values) for split, values in subjects.items()}
        except Exception as exc:  # noqa: BLE001 - audit should continue and report all gaps.
            errors.append(f"{cfg_path}: split audit failed: {exc}")

        if args.materialize_plan_artifacts and out_dir.exists():
            materialize_plan_artifacts(out_dir, cfg_path)

        metrics_path = out_dir / "metrics_final.json"
        config_complete[str(cfg_path)] = metrics_path.exists()
        if not metrics_path.exists():
            message = f"{cfg_path}: incomplete, missing {metrics_path}"
            if args.allow_incomplete:
                warnings.append(message)
            else:
                errors.append(message)
            continue

        job_report["complete"] = True
        metrics = read_json(metrics_path)
        for split in ["train", "val", "test"]:
            for key in metric_keys_for(cfg, split):
                if key not in metrics:
                    errors.append(f"{cfg_path}: metrics_final.json missing {key}")
            split_metrics_path = out_dir / f"{plan_split(split)}_metrics.json"
            if not split_metrics_path.exists():
                errors.append(f"{cfg_path}: missing plan split metrics file {split_metrics_path}")

        required_files = [
            "resolved_config.yaml",
            "run_summary.json",
            "checkpoint_last.pt",
            "checkpoint_best.pt",
            "trial_predictions_validation.csv",
            "subject_predictions_validation.csv",
            "trial_predictions_test.csv",
            "subject_predictions_test.csv",
            "confusion_matrix_validation.json",
            "confusion_matrix_test.json",
        ]
        for name in required_files:
            if not (out_dir / name).exists():
                errors.append(f"{cfg_path}: missing required output artifact {out_dir / name}")

        run_summary_path = out_dir / "run_summary.json"
        if run_summary_path.exists():
            run_summary = read_json(run_summary_path)
            for key in [
                "task_name",
                "mode",
                "pretrained_checkpoint",
                "num_train_subjects",
                "num_validation_subjects",
                "num_test_subjects",
                "label_counts",
                "subject_eye_availability_counts",
            ]:
                if key not in run_summary:
                    errors.append(f"{cfg_path}: run_summary.json missing {key}")
            exposure = run_summary.get("pretraining_exposure")
            if not isinstance(exposure, dict) or "mode" not in exposure:
                errors.append(f"{cfg_path}: run_summary.json missing pretraining_exposure.mode")
            if run_summary.get("task_name") not in {task, None}:
                errors.append(f"{cfg_path}: run_summary task_name mismatch: {run_summary.get('task_name')} != {task}")
            if run_summary.get("mode") not in {mode, None}:
                errors.append(f"{cfg_path}: run_summary mode mismatch: {run_summary.get('mode')} != {mode}")

    expected_pairs = {(task, mode) for task in EXPECTED_TASKS for mode in EXPECTED_MODES}
    missing_pairs = sorted(expected_pairs - seen_pairs)
    extra_pairs = sorted(seen_pairs - expected_pairs)
    if missing_pairs:
        errors.append(f"Missing task/mode pairs: {missing_pairs}")
    if extra_pairs:
        errors.append(f"Unexpected task/mode pairs: {extra_pairs}")
    if len(set(output_dirs)) != len(output_dirs):
        errors.append("Output directories are not unique")

    status_path = Path(args.status_json)
    if status_path.exists():
        status = read_json(status_path)
        unresolved_status: dict[str, list[Any]] = {}
        for key in ["running", "pending", "failures"]:
            unresolved: list[Any] = []
            for item in status.get(key, []):
                config = status_item_config(item)
                if config is not None and config_complete.get(config):
                    continue
                unresolved.append(item)
            unresolved_status[key] = unresolved
        report["status"] = {
            "completed": len(status.get("completed", [])),
            "running": len(status.get("running", [])),
            "pending": len(status.get("pending", [])),
            "failures": len(status.get("failures", [])),
            "unresolved_running": len(unresolved_status["running"]),
            "unresolved_pending": len(unresolved_status["pending"]),
            "unresolved_failures": len(unresolved_status["failures"]),
        }
        if not args.allow_incomplete:
            if unresolved_status["running"]:
                errors.append(f"Queue still has unresolved running jobs: {len(unresolved_status['running'])}")
            if unresolved_status["pending"]:
                errors.append(f"Queue still has unresolved pending jobs: {len(unresolved_status['pending'])}")
            if unresolved_status["failures"]:
                errors.append(f"Queue has unresolved failures: {unresolved_status['failures']}")
    else:
        errors.append(f"Missing status json: {status_path}")

    metrics_files = sorted(Path(args.output_root).glob("**/metrics_final.json"))
    official_metrics_final_count = sum(1 for complete in config_complete.values() if complete)
    report["official_metrics_final_count"] = official_metrics_final_count
    report["metrics_final_count"] = len(metrics_files)
    report["extra_metrics_final_count"] = len(metrics_files) - official_metrics_final_count
    if not args.allow_incomplete and official_metrics_final_count != EXPECTED_JOB_COUNT:
        errors.append(
            f"Expected {EXPECTED_JOB_COUNT} official metrics_final.json files, found {official_metrics_final_count}"
        )

    report["errors"] = errors
    report["warnings"] = warnings
    if args.report_json:
        write_json(Path(args.report_json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
