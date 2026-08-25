#!/usr/bin/env python3
"""Train-only cached-CLS screening for the high-impact MCI ablations.

This module intentionally never reads validation or test tensors.  It uses
subject-level stratified folds made from the cached training split and is a
cheap gate before any encoder fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eyemae.utils import write_json


TASK_COUNT = 4
QC_COLUMNS = ("one_eye", "blink_fraction", "missing_fraction", "patch_count")


@dataclass(frozen=True)
class Variant:
    name: str
    trials_per_bag: int = 4
    num_bags: int = 1
    robust_bags: bool = False
    qc_stratified_trials: bool = False
    qc_balance_loss: bool = False
    demographics: str = "none"  # none | concat | late_additive
    constrained_tasks: bool = False
    task_dropout: float = 0.0


VARIANTS = {
    # The production baseline appends the fold-safe raw 16-D demographics to
    # each normalized trial CLS before the shared head.
    "d0_baseline": Variant("d0_baseline", demographics="concat"),
    "d1_qc_balanced": Variant(
        "d1_qc_balanced", qc_stratified_trials=True, qc_balance_loss=True,
        demographics="concat"
    ),
    "d2_4xk4_mean": Variant(
        "d2_4xk4_mean", num_bags=4, demographics="concat"
    ),
    "d2_4xk4_robust": Variant(
        "d2_4xk4_robust", num_bags=4, robust_bags=True, demographics="concat"
    ),
    "d3_eye_only": Variant("d3_eye_only"),
    "d3_demo_concat": Variant("d3_demo_concat", demographics="concat"),
    "d3_demo_late": Variant("d3_demo_late", demographics="late_additive"),
    "d4_task_residual": Variant(
        "d4_task_residual", constrained_tasks=True, demographics="concat"
    ),
    "d4_task_residual_dropout": Variant(
        "d4_task_residual_dropout", constrained_tasks=True, task_dropout=0.10,
        demographics="concat"
    ),
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subject_seed(subject: str, seed: int, epoch: int, task: int) -> int:
    value = f"{subject}|{seed}|{epoch}|{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


class TrainCache:
    """Validated, train-only view of a raw trial CLS cache."""

    def __init__(self, path: Path, *, eligibility_min_trials: int = 4) -> None:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("selection_source") != "train_only_subject_folds":
            raise ValueError("Cache is not marked for Train-only subject folds")
        # Deliberately take only this key.  Val/Test remain unreachable here.
        split = raw["splits"]["train"]
        self.cache_path = str(path.resolve())
        self.bert_sha256 = str(raw["bert_sha256"])
        self.features = split["features"].float()
        self.labels = split["labels"].long()
        self.tasks = split["task_ids"].long()
        self.quality = split["quality"].float()
        self.quality_names = tuple(split["quality_names"])
        self.subject_keys = tuple(map(str, split["subject_keys"]))
        demo_keys = tuple(map(str, split["demographic_subject_keys"]))
        self.demographics = {
            key: value.float()
            for key, value in zip(demo_keys, split["demographics"], strict=True)
        }
        n = self.features.shape[0]
        if not (
            self.labels.shape[0] == self.tasks.shape[0] == self.quality.shape[0]
            == len(self.subject_keys) == n
        ):
            raise ValueError("Cache trial axes are inconsistent")
        self.subject_indices: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        by_subject_task: dict[tuple[str, int], list[int]] = defaultdict(list)
        subject_label: dict[str, int] = {}
        for index, (subject, task, label) in enumerate(
            zip(self.subject_keys, self.tasks.tolist(), self.labels.tolist(), strict=True)
        ):
            if not 0 <= task < TASK_COUNT:
                raise ValueError(f"Unexpected task id {task}")
            if subject in subject_label and subject_label[subject] != label:
                raise ValueError(f"Inconsistent labels for subject {subject}")
            subject_label[subject] = label
            by_subject_task[(subject, task)].append(index)
        if eligibility_min_trials < 4:
            raise ValueError("eligibility_min_trials must be at least four")
        self.subjects = [
            subject for subject in sorted(subject_label)
            if all(
                len(by_subject_task.get((subject, task), ()))
                >= eligibility_min_trials
                for task in range(TASK_COUNT)
            )
        ]
        if not self.subjects:
            raise ValueError("No subjects satisfy the Stage A eligibility threshold")
        self.subject_labels = torch.tensor(
            [subject_label[subject] for subject in self.subjects], dtype=torch.float32
        )
        for subject in self.subjects:
            for task in range(TASK_COUNT):
                indices = by_subject_task.get((subject, task), [])
                if len(indices) < 4:
                    raise ValueError(f"Strict subject {subject} task {task} has <4 trials")
                self.subject_indices[subject][task] = torch.tensor(indices, dtype=torch.long)
        self.subject_qc = torch.stack([self._subject_qc(s) for s in self.subjects])

    def _subject_qc(self, subject: str) -> torch.Tensor:
        indices = torch.cat(list(self.subject_indices[subject].values()))
        positions = [self.quality_names.index(name) for name in QC_COLUMNS]
        return self.quality[indices][:, positions].mean(dim=0)

    def bags(
        self, subject_positions: Iterable[int], variant: Variant, *, seed: int, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [subjects,tasks,bags,dim] and demographics.

        Sampling is without replacement whenever enough trials exist.  QC
        stratification sorts by a label-independent burden score and samples
        across its quantiles; labels never affect trial selection.
        """
        output: list[torch.Tensor] = []
        demos: list[torch.Tensor] = []
        required = variant.trials_per_bag * variant.num_bags
        qc_positions = [self.quality_names.index(name) for name in QC_COLUMNS[:3]]
        for subject_position in subject_positions:
            subject = self.subjects[int(subject_position)]
            task_bags: list[torch.Tensor] = []
            for task in range(TASK_COUNT):
                indices = self.subject_indices[subject][task]
                generator = torch.Generator().manual_seed(
                    _subject_seed(subject, seed, epoch, task)
                )
                if variant.qc_stratified_trials:
                    q = self.quality[indices][:, qc_positions]
                    burden = q[:, 0] + q[:, 1] + q[:, 2]
                    ordered = indices[torch.argsort(burden)]
                    # Draw once from each equal-mass QC quantile.  This is
                    # label-independent and avoids replacing the natural
                    # subject distribution with a class-conditioned sampler.
                    if len(ordered) >= required:
                        bounds = torch.linspace(
                            0, len(ordered), required + 1
                        ).floor().long()
                        picks = []
                        for left, right in zip(bounds[:-1], bounds[1:], strict=True):
                            width = max(int(right - left), 1)
                            position = int(left) + int(
                                torch.randint(width, (1,), generator=generator)
                            )
                            picks.append(ordered[position])
                        choose = torch.stack(picks)
                        choose = choose[
                            torch.randperm(len(choose), generator=generator)
                        ]
                    else:
                        choose = ordered
                else:
                    choose = indices[torch.randperm(len(indices), generator=generator)[:required]]
                if len(choose) < required:
                    # Strict data always has K>=4. For 4xK4, use every unique
                    # trial first and fill the final slots by a new permutation.
                    extra = indices[torch.randperm(len(indices), generator=generator)]
                    repeats = math.ceil((required - len(choose)) / len(extra))
                    choose = torch.cat([choose, extra.repeat(repeats)])[:required]
                bag = self.features[choose].reshape(
                    variant.num_bags, variant.trials_per_bag, -1
                ).mean(dim=1)
                task_bags.append(bag)
            output.append(torch.stack(task_bags))
            demos.append(self.demographics[subject])
        return torch.stack(output), torch.stack(demos)


def qc_balance_weights(qc: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Stabilized inverse propensity weights fitted on a training fold only."""
    mean = qc.mean(axis=0, keepdims=True)
    std = qc.std(axis=0, keepdims=True).clip(min=1e-6)
    model = LogisticRegression(C=0.1, max_iter=2000, class_weight=None)
    model.fit((qc - mean) / std, labels)
    probability = model.predict_proba((qc - mean) / std)
    observed = probability[np.arange(len(labels)), labels.astype(int)].clip(0.05, 0.95)
    prevalence = np.bincount(labels.astype(int), minlength=2) / len(labels)
    weights = prevalence[labels.astype(int)] / observed
    weights = np.clip(weights, 0.5, 2.0)
    return weights / weights.mean()


class CachedMILHead(nn.Module):
    def __init__(self, variant: Variant, feature_dim: int, demographic_dim: int) -> None:
        super().__init__()
        self.variant = variant
        input_dim = feature_dim + (demographic_dim if variant.demographics == "concat" else 0)
        self.norm = nn.LayerNorm(feature_dim)
        self.shared_head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        if variant.demographics == "late_additive":
            self.demo_head = nn.Linear(demographic_dim, 1)
            self.demo_alpha_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.demo_head = None
            self.register_parameter("demo_alpha_logit", None)
        if variant.constrained_tasks:
            self.task_residual = nn.Parameter(torch.zeros(TASK_COUNT))
        else:
            self.register_parameter("task_residual", None)

    def task_weights(self) -> torch.Tensor:
        if self.task_residual is None:
            return torch.full((TASK_COUNT,), 1.0 / TASK_COUNT, device=next(self.parameters()).device)
        # This parameterization stays in approximately [0.18, 0.33].
        return torch.softmax(0.2 * torch.tanh(self.task_residual), dim=0)

    def forward(self, bags: torch.Tensor, demographics: torch.Tensor) -> torch.Tensor:
        # bags: subject, task, bag, feature
        x = self.norm(bags)
        if self.variant.demographics == "concat":
            demo = demographics[:, None, None, :].expand(*x.shape[:-1], -1)
            x = torch.cat([x, demo], dim=-1)
        logits = self.shared_head(x).squeeze(-1)  # subject, task, bag
        if self.variant.robust_bags and logits.shape[-1] >= 4:
            ordered = logits.sort(dim=-1).values
            task_logits = ordered[..., 1:-1].mean(dim=-1)
        else:
            task_logits = logits.mean(dim=-1)
        if self.training and self.variant.task_dropout > 0:
            keep = torch.rand_like(task_logits) >= self.variant.task_dropout
            # Always retain at least one task.
            none = ~keep.any(dim=-1)
            keep[none, torch.randint(TASK_COUNT, (int(none.sum()),), device=keep.device)] = True
            weights = self.task_weights()[None, :] * keep
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            subject_logit = (task_logits * weights).sum(dim=-1)
        else:
            subject_logit = (task_logits * self.task_weights()).sum(dim=-1)
        if self.demo_head is not None:
            alpha = 0.3 * torch.sigmoid(self.demo_alpha_logit)
            subject_logit = subject_logit + alpha * self.demo_head(demographics).squeeze(-1)
        return subject_logit


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    pred = (probs >= 0.5).astype(int)
    loss = -np.mean(labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8))
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "brier": float(np.mean((probs - labels) ** 2)),
        "loss": float(loss),
    }


def run_fold(
    cache: TrainCache,
    variant: Variant,
    train_positions: np.ndarray,
    held_positions: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(seed)
    train_labels = cache.subject_labels[train_positions].to(device)
    held_labels = cache.subject_labels[held_positions].cpu().numpy()
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum().clamp_min(1)
    sample_weight = np.ones(len(train_positions), dtype=np.float32)
    if variant.qc_balance_loss:
        sample_weight = qc_balance_weights(
            cache.subject_qc[train_positions].numpy(), train_labels.cpu().numpy()
        ).astype(np.float32)
    sample_weight_tensor = torch.from_numpy(sample_weight).to(device)
    model = CachedMILHead(
        variant, cache.features.shape[1], next(iter(cache.demographics.values())).numel()
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    fixed_held_bags, fixed_held_demo = cache.bags(
        held_positions, variant, seed=seed, epoch=10_000
    )
    fixed_held_bags, fixed_held_demo = fixed_held_bags.to(device), fixed_held_demo.to(device)
    for epoch in range(max_epochs):
        model.train()
        bags, demo = cache.bags(train_positions, variant, seed=seed, epoch=epoch)
        logits = model(bags.to(device), demo.to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits, train_labels, reduction="none", pos_weight=pos_weight
        )
        loss = (loss * sample_weight_tensor).sum() / sample_weight_tensor.sum()
        if model.task_residual is not None:
            loss = loss + 0.1 * model.task_residual.square().mean()
        if model.demo_head is not None:
            loss = loss + 0.01 * model.demo_head.weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            held_logits = model(fixed_held_bags, fixed_held_demo).cpu().numpy()
        auc = roc_auc_score(held_labels, held_logits)
        if auc > best_auc + 1e-6:
            best_auc, best_epoch = float(auc), epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 29 and stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        held_logits = model(fixed_held_bags, fixed_held_demo).cpu().numpy()
    details: dict[str, Any] = {
        "seed": seed,
        "best_epoch": best_epoch,
        "metrics": _metrics(held_labels, held_logits),
        "held_positions": held_positions.tolist(),
        "held_subjects": [cache.subjects[int(i)] for i in held_positions],
        "held_labels": held_labels.astype(int).tolist(),
        "held_logits": held_logits.tolist(),
        "task_weights": model.task_weights().detach().cpu().tolist(),
    }
    if model.demo_alpha_logit is not None:
        details["demographic_alpha"] = float(
            (0.3 * torch.sigmoid(model.demo_alpha_logit)).detach().cpu()
        )
    return details


def screen_variant(
    cache: TrainCache,
    variant: Variant,
    *,
    folds: int,
    fold_seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = cache.subject_labels.numpy().astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    fold_results = []
    oof_logits = np.full(len(labels), np.nan, dtype=np.float64)
    started = time.time()
    for fold, (train_positions, held_positions) in enumerate(splitter.split(labels, labels)):
        result = run_fold(
            cache, variant, train_positions, held_positions,
            seed=fold_seed + fold, max_epochs=max_epochs, patience=patience, device=device,
        )
        result["fold"] = fold
        fold_results.append(result)
        oof_logits[held_positions] = np.asarray(result["held_logits"])
    if not np.isfinite(oof_logits).all():
        raise RuntimeError("OOF predictions are incomplete")
    aucs = [item["metrics"]["auroc"] for item in fold_results]
    return {
        "variant": asdict(variant),
        "selection_split": "train_subject_stratified_folds_only",
        "test_used_for_selection": False,
        "folds": folds,
        "fold_seed": fold_seed,
        "fold_auc_mean": float(np.mean(aucs)),
        "fold_auc_std": float(np.std(aucs, ddof=1)),
        "oof_metrics": _metrics(labels, oof_logits),
        "runtime_seconds": time.time() - started,
        "fold_results": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--eligibility-min-trials", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    cache = TrainCache(
        Path(args.cache),
        eligibility_min_trials=int(args.eligibility_min_trials),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.variants:
        result = screen_variant(
            cache, VARIANTS[name], folds=args.folds, fold_seed=args.fold_seed,
            max_epochs=args.max_epochs, patience=args.patience, device=torch.device(args.device),
        )
        results.append(result)
        write_json(output_dir / f"{name}.json", result)
        print(json.dumps({
            "variant": name,
            "fold_auc_mean": result["fold_auc_mean"],
            "fold_auc_std": result["fold_auc_std"],
            "oof_auc": result["oof_metrics"]["auroc"],
        }), flush=True)
    ranked = sorted(results, key=lambda item: item["fold_auc_mean"], reverse=True)
    summary = {
        "cache_path": cache.cache_path,
        "bert_sha256": cache.bert_sha256,
        "selection_split": "train_subject_stratified_folds_only",
        "eligible_subjects": len(cache.subjects),
        "eligibility_min_trials_per_task": int(args.eligibility_min_trials),
        "test_used_for_selection": False,
        "ranking_metric": "fold_auc_mean",
        "results": [
            {
                "variant": item["variant"]["name"],
                "fold_auc_mean": item["fold_auc_mean"],
                "fold_auc_std": item["fold_auc_std"],
                "oof_metrics": item["oof_metrics"],
            }
            for item in ranked
        ],
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
