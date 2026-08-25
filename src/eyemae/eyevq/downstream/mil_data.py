"""Subject-level multi-instance data pipeline for EyeVQ downstream tasks.

Each training item is one subject bag with a fixed number of slots per task.
Tasks with enough valid trials are sampled independently and reproducibly for
``(seed, subject, task, sample step)``.  Optional partial-task training retains
all available trials when a task has fewer than K. Missing/padded task slots
contain gradient-masked placeholders so DDP shapes remain fixed while the model
uses only genuinely present trials.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from eyemae.downstream_data import PackedDownstreamDataset, collate_downstream_trials


TASK_NAMES = ("ProSaccade", "AntiSaccade", "MemorySaccade", "DoubleSaccade")
TASK_IDS = tuple(range(len(TASK_NAMES)))


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class SubjectRecord:
    subject_key: str
    label: int
    task_trial_indices: tuple[tuple[int, ...], ...]

    @property
    def task_present_mask(self) -> tuple[bool, ...]:
        return tuple(bool(indices) for indices in self.task_trial_indices)


class SubjectBagDataset(Dataset):
    """Group a packed trial dataset into deterministic subject/task bags."""

    def __init__(
        self,
        trial_dataset: PackedDownstreamDataset,
        *,
        trials_per_task: int = 4,
        task_ids: Sequence[int] = TASK_IDS,
        require_all_tasks: bool = True,
        sample_all_available_below_k: bool = False,
        seed: int = 42,
        subject_features: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if trials_per_task <= 0:
            raise ValueError("trials_per_task must be positive")
        if len(set(int(task_id) for task_id in task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique")

        self.trial_dataset = trial_dataset
        self.trials_per_task = int(trials_per_task)
        self.task_ids = tuple(int(task_id) for task_id in task_ids)
        self.require_all_tasks = bool(require_all_tasks)
        self.sample_all_available_below_k = bool(
            sample_all_available_below_k
        )
        if self.require_all_tasks and self.sample_all_available_below_k:
            raise ValueError(
                "sample_all_available_below_k requires partial-task training"
            )
        self.seed = int(seed)
        self.subject_features = subject_features

        grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        labels: dict[str, int] = {}
        for trial_index, (row, label) in enumerate(zip(trial_dataset.rows, trial_dataset.labels)):
            subject_key = str(row["ml_subject_id"])
            task_id = int(row["task_id"])
            if task_id not in self.task_ids:
                continue
            label = int(label)
            previous = labels.setdefault(subject_key, label)
            if previous != label:
                raise ValueError(f"Subject has inconsistent labels: {subject_key}")
            grouped[subject_key][task_id].append(trial_index)

        records: list[SubjectRecord] = []
        excluded: list[str] = []
        for subject_key in sorted(grouped):
            raw_per_task = tuple(
                tuple(sorted(grouped[subject_key].get(task_id, ())))
                for task_id in self.task_ids
            )
            if self.sample_all_available_below_k:
                per_task = raw_per_task
            else:
                per_task = tuple(
                    indices if len(indices) >= self.trials_per_task else ()
                    for indices in raw_per_task
                )
            task_present = tuple(bool(indices) for indices in per_task)
            if self.require_all_tasks and not all(task_present):
                excluded.append(subject_key)
                continue
            if not any(task_present):
                excluded.append(subject_key)
                continue
            records.append(SubjectRecord(subject_key, labels[subject_key], per_task))

        if not records:
            raise ValueError("No subjects satisfy the subject-bag requirements")
        self.records = records
        self.excluded_subjects = tuple(excluded)
        self.labels = [record.label for record in records]
        self.subject_keys = [record.subject_key for record in records]
        self.subjects_with_missing_tasks = tuple(
            record.subject_key
            for record in records
            if not all(record.task_present_mask)
        )
        self.task_count_by_subject = {
            record.subject_key: int(sum(record.task_present_mask))
            for record in records
        }

    def __len__(self) -> int:
        return len(self.records)

    def _sample_task_indices(
        self, record: SubjectRecord, task_position: int, sample_step: int
    ) -> list[int]:
        indices = record.task_trial_indices[task_position]
        if len(indices) < self.trials_per_task:
            if self.sample_all_available_below_k:
                return list(indices)
            if not self.require_all_tasks and not indices:
                return []
            raise ValueError(
                f"Subject {record.subject_key} has only {len(indices)} trials for "
                f"task_id={self.task_ids[task_position]}; need {self.trials_per_task}"
            )
        # Include the step in the seed so every subject appearance draws a new
        # uniform subset. A fixed per-task permutation followed by cyclic
        # windows severely restricts the trial combinations (for example,
        # K=2 from six trials can collapse to only three fixed pairs).
        generator = torch.Generator().manual_seed(
            _stable_seed(
                self.seed,
                record.subject_key,
                self.task_ids[task_position],
                int(sample_step),
            )
        )
        positions = torch.randperm(len(indices), generator=generator)[
            : self.trials_per_task
        ].tolist()
        selected = [indices[position] for position in positions]
        if len(set(selected)) != self.trials_per_task:
            raise RuntimeError("Random task sampling produced a duplicate trial")
        return selected

    def sampled_trial_indices(
        self, subject_index: int, sample_step: int
    ) -> tuple[tuple[int, ...], ...]:
        record = self.records[int(subject_index)]
        return tuple(
            tuple(self._sample_task_indices(record, task_position, int(sample_step)))
            for task_position in range(len(self.task_ids))
        )

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(key, tuple):
            subject_index, sample_step = key
        else:
            subject_index, sample_step = key, 0
        record = self.records[int(subject_index)]
        sampled = self.sampled_trial_indices(int(subject_index), int(sample_step))
        fallback_index = next(
            trial_index
            for task_indices in sampled
            for trial_index in task_indices
        )
        fallback_trial: dict[str, Any] | None = None
        trials: list[dict[str, Any]] = []
        trial_slot_mask: list[list[bool]] = []
        for task_position, task_indices in enumerate(sampled):
            if task_indices:
                trials.extend(
                    self.trial_dataset[trial_index]
                    for trial_index in task_indices
                )
            task_slot_mask = [True] * len(task_indices)

            # Keep the dense [subject, task, K] layout without allowing a
            # missing/padded task slot to contribute to the forward loss. The
            # trial-slot mask gives these placeholders exactly zero gradient.
            if fallback_trial is None:
                fallback_trial = self.trial_dataset[fallback_index]
            for slot in range(len(task_indices), self.trials_per_task):
                placeholder = dict(fallback_trial)
                placeholder["task_id"] = self.task_ids[task_position]
                placeholder["global_trial_id"] = (
                    f"__masked_trial_slot__:{record.subject_key}:"
                    f"{self.task_ids[task_position]}:{slot}"
                )
                trials.append(placeholder)
                task_slot_mask.append(False)
            if len(task_slot_mask) != self.trials_per_task:
                raise RuntimeError("A task bag did not produce exactly K slots")
            trial_slot_mask.append(task_slot_mask)
        item = {
            "trials": trials,
            "subject_key": record.subject_key,
            "label": record.label,
            "task_present_mask": record.task_present_mask,
            "trial_slot_mask": trial_slot_mask,
            "sampled_trial_indices": sampled,
        }
        if self.subject_features is not None:
            if record.subject_key not in self.subject_features:
                raise KeyError(f"Missing subject features for {record.subject_key}")
            item["subject_features"] = self.subject_features[record.subject_key]
        return item


def collate_subject_bags(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dense subject bags while retaining the true task-present mask."""
    if not items:
        raise ValueError("Cannot collate an empty subject batch")
    trials_per_subject = len(items[0]["trials"])
    if trials_per_subject <= 0:
        raise ValueError("Subject bags must contain trials")
    if any(len(item["trials"]) != trials_per_subject for item in items):
        raise ValueError("All subject bags in a training batch must have equal size")

    flat_trials = [trial for item in items for trial in item["trials"]]
    batch = collate_downstream_trials(flat_trials)
    batch["subject_label"] = torch.tensor(
        [int(item["label"]) for item in items], dtype=torch.float32
    )
    batch["subject_ids"] = [str(item["subject_key"]) for item in items]
    batch["task_present_mask"] = torch.tensor(
        [item["task_present_mask"] for item in items], dtype=torch.bool
    )
    batch["trial_slot_mask"] = torch.tensor(
        [item["trial_slot_mask"] for item in items], dtype=torch.bool
    )
    batch["num_subjects"] = len(items)
    batch["trials_per_subject"] = trials_per_subject
    batch["sampled_trial_indices"] = [item["sampled_trial_indices"] for item in items]
    has_subject_features = ["subject_features" in item for item in items]
    if any(has_subject_features):
        if not all(has_subject_features):
            raise ValueError("Subject features must be present for every item or none")
        batch["subject_features"] = torch.stack(
            [torch.as_tensor(item["subject_features"], dtype=torch.float32) for item in items]
        )
    return batch
