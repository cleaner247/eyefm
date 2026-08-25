"""Pure helpers for the auditable V4 staged model search."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np

from eyemae.downstream_metrics import compute_binary_metrics, compute_multiclass_metrics
from eyemae.utils import write_json


_STEP_PATTERN = re.compile(r"ckpt_step0*(\d+)\.pt$")


def checkpoint_step(path: str | Path) -> int:
    match = _STEP_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def latest_step_checkpoint(directory: str | Path) -> Path | None:
    candidates = list(Path(directory).glob("ckpt_step*.pt"))
    return max(candidates, key=checkpoint_step) if candidates else None


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest_matches(
    cache_path: str | Path,
    manifest_path: str | Path,
    tokenizer_checkpoint: str | Path,
    bert_config: str | Path,
) -> bool:
    """Validate cache size and both producer identities without loading code arrays."""
    cache = Path(cache_path)
    manifest_file = Path(manifest_path)
    tokenizer = Path(tokenizer_checkpoint)
    config = Path(bert_config)
    if not all(path.is_file() for path in (cache, manifest_file, tokenizer, config)):
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not (
            manifest["tokenizer_sha256"] == sha256_file(tokenizer)
            and manifest["bert_config_sha256"] == sha256_file(config)
            and int(manifest["cache_size_bytes"]) == cache.stat().st_size
            and Path(manifest["tokenizer_checkpoint"]).resolve() == tokenizer.resolve()
        ):
            return False
        with np.load(cache, allow_pickle=True) as payload:
            return (
                int(payload["format_version"]) in {3, 4}
                and Path(str(payload["tokenizer_checkpoint"].item())).resolve()
                == tokenizer.resolve()
                and len(payload["gids"]) > 0
                and len(payload["gids"]) == len(payload["code_ids"])
                and len(payload["gids"]) == len(payload["num_patches"])
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def standardized_joint_scores(
    candidates: list[dict[str, Any]],
    *,
    mci_key: str = "mci_val_auroc",
    pd5_key: str = "pd5_val_macro_auroc",
) -> list[dict[str, Any]]:
    """Add task z-scores and their equal-weight mean without reading test fields."""
    if not candidates:
        raise ValueError("No candidates to score")
    output = [dict(candidate) for candidate in candidates]
    for source, target in ((mci_key, "mci_z"), (pd5_key, "pd5_z")):
        values = [float(candidate[source]) for candidate in output]
        center = fmean(values)
        scale = pstdev(values)
        for candidate, value in zip(output, values):
            candidate[target] = 0.0 if scale == 0.0 else (value - center) / scale
    for candidate in output:
        candidate["joint_val_score"] = 0.5 * (
            float(candidate["mci_z"]) + float(candidate["pd5_z"])
        )
        candidate["worst_task_z"] = min(
            float(candidate["mci_z"]), float(candidate["pd5_z"])
        )
    return output


def select_joint_candidate(
    candidates: list[dict[str, Any]],
    *,
    tie_tolerance: float = 0.05,
    step_key: str = "step",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = standardized_joint_scores(candidates)
    best_score = max(float(candidate["joint_val_score"]) for candidate in scored)
    tied = [
        candidate
        for candidate in scored
        if best_score - float(candidate["joint_val_score"]) <= tie_tolerance
    ]
    selected = max(
        tied,
        key=lambda candidate: (
            float(candidate["worst_task_z"]),
            -int(candidate.get(step_key, 10**12)),
        ),
    )
    ranked = sorted(
        scored,
        key=lambda candidate: (
            float(candidate["joint_val_score"]),
            float(candidate["worst_task_z"]),
            -int(candidate.get(step_key, 10**12)),
        ),
        reverse=True,
    )
    return selected, ranked


def select_task_configuration(
    aggregates: list[dict[str, Any]], *, tie_tolerance: float = 0.002
) -> dict[str, Any]:
    """Select by mean validation only; break near-ties by stability and simplicity."""
    if not aggregates:
        raise ValueError("No downstream configurations to select")
    best_mean = max(float(item["mean_val_auroc"]) for item in aggregates)
    tied = [
        item for item in aggregates
        if best_mean - float(item["mean_val_auroc"]) <= tie_tolerance
    ]
    return min(
        tied,
        key=lambda item: (
            float(item["std_val_auroc"]),
            int(item["unfrozen_layers"]),
            float(item["encoder_lr"]),
        ),
    )


def _read_prediction_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["subject_key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate subject_key in {path}")
    return result


def ensemble_prediction_csvs(
    prediction_paths: Iterable[str | Path],
    *,
    output_csv: str | Path,
    output_metrics: str | Path,
    num_classes: int,
    split: str,
) -> dict[str, float]:
    paths = [Path(path) for path in prediction_paths]
    if len(paths) < 2:
        raise ValueError("An ensemble requires at least two prediction files")
    runs = [_read_prediction_rows(path) for path in paths]
    keys = sorted(runs[0])
    if any(sorted(run) != keys for run in runs[1:]):
        raise ValueError("Ensemble prediction files contain different subjects")

    output_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    if num_classes == 2:
        logits: list[float] = []
        for key in keys:
            label_values = {int(run[key]["label"]) for run in runs}
            if len(label_values) != 1:
                raise ValueError(f"Inconsistent ensemble label for {key}")
            label = label_values.pop()
            logit = fmean(float(run[key]["logit"]) for run in runs)
            labels.append(label)
            logits.append(logit)
            probability = 1.0 / (1.0 + math.exp(-logit))
            output_rows.append({
                "subject_key": key,
                "split": split,
                "label": label,
                "logit": logit,
                "prob": probability,
                "pred": int(probability >= 0.5),
            })
        metrics = compute_binary_metrics(labels, logits, threshold=0.5, prefix=f"{split}/subject")
    else:
        logits_by_subject: list[list[float]] = []
        for key in keys:
            label_values = {int(run[key]["label"]) for run in runs}
            if len(label_values) != 1:
                raise ValueError(f"Inconsistent ensemble label for {key}")
            label = label_values.pop()
            logits = [
                fmean(float(run[key][f"logit_{class_id}"]) for run in runs)
                for class_id in range(num_classes)
            ]
            labels.append(label)
            logits_by_subject.append(logits)
            row: dict[str, Any] = {
                "subject_key": key,
                "split": split,
                "label": label,
                "pred": max(range(num_classes), key=logits.__getitem__),
            }
            row.update({f"logit_{class_id}": value for class_id, value in enumerate(logits)})
            output_rows.append(row)
        metrics = compute_multiclass_metrics(
            labels,
            logits_by_subject,
            num_classes=num_classes,
            prefix=f"{split}/subject",
        )

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(destination)
    write_json(output_metrics, metrics)
    return metrics
