"""Distributed subject samplers for subject-level MIL."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class DistributedSubjectEpochSampler(Sampler[list[tuple[int, int]]]):
    """Shuffle subjects by epoch without class quotas or within-epoch repeats.

    A full DDP optimizer step must contain the same local batch size on every
    rank. Subjects that do not fill the final global batch are omitted with a
    deterministic rotating schedule, so long-run exposure differs by at most
    one while no subject is duplicated to pad an epoch.
    """

    def __init__(
        self,
        labels: Sequence[int],
        *,
        num_replicas: int,
        rank: int,
        subjects_per_rank: int,
        epochs: int,
        seed: int = 42,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if subjects_per_rank <= 0:
            raise ValueError("subjects_per_rank must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not labels:
            raise ValueError("labels cannot be empty")
        self.labels = tuple(int(label) for label in labels)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.subjects_per_rank = int(subjects_per_rank)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.epoch = 0
        self.global_batch_size = self.num_replicas * self.subjects_per_rank
        self.steps_per_epoch = len(self.labels) // self.global_batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError(
                f"Need at least {self.global_batch_size} subjects, got {len(self.labels)}"
            )
        self.subjects_per_epoch = self.steps_per_epoch * self.global_batch_size
        self.dropped_per_epoch = len(self.labels) - self.subjects_per_epoch
        self.omission_order = list(range(len(self.labels)))
        random.Random(self.seed + 7_919).shuffle(self.omission_order)

    def set_epoch(self, epoch: int) -> None:
        if not 0 <= int(epoch) < self.epochs:
            raise ValueError(f"epoch must be in [0, {self.epochs}), got {epoch}")
        self.epoch = int(epoch)

    def dropped_indices(self, epoch: int) -> list[int]:
        if self.dropped_per_epoch == 0:
            return []
        start = (int(epoch) * self.dropped_per_epoch) % len(self.labels)
        return [
            self.omission_order[(start + offset) % len(self.labels)]
            for offset in range(self.dropped_per_epoch)
        ]

    def epoch_indices(self, epoch: int) -> list[int]:
        dropped = set(self.dropped_indices(epoch))
        selected = [index for index in range(len(self.labels)) if index not in dropped]
        random.Random(self.seed + 1_000_003 * int(epoch)).shuffle(selected)
        if len(selected) != self.subjects_per_epoch or len(selected) != len(set(selected)):
            raise RuntimeError("Epoch subject selection is not unique")
        return selected

    def all_rank_indices(self, epoch: int, batch_index: int) -> list[list[int]]:
        if not 0 <= int(batch_index) < self.steps_per_epoch:
            raise ValueError(
                f"batch_index must be in [0, {self.steps_per_epoch}), got {batch_index}"
            )
        selected = self.epoch_indices(epoch)
        start = int(batch_index) * self.global_batch_size
        global_batch = selected[start : start + self.global_batch_size]
        rank_batches = [
            global_batch[
                rank * self.subjects_per_rank : (rank + 1) * self.subjects_per_rank
            ]
            for rank in range(self.num_replicas)
        ]
        flat = [index for batch in rank_batches for index in batch]
        if len(flat) != self.global_batch_size or len(flat) != len(set(flat)):
            raise RuntimeError("A global epoch batch contains a repeated subject")
        return rank_batches

    def audit_epoch(self, epoch: int) -> dict[str, object]:
        selected = self.epoch_indices(epoch)
        dropped = self.dropped_indices(epoch)
        selected_counts = Counter(self.labels[index] for index in selected)
        dropped_counts = Counter(self.labels[index] for index in dropped)
        class_ids = sorted(set(self.labels))
        return {
            "epoch": int(epoch) + 1,
            "steps": self.steps_per_epoch,
            "num_selected": len(selected),
            "num_dropped": len(dropped),
            "selected_label_counts": {
                str(label): int(selected_counts[label]) for label in class_ids
            },
            "dropped_label_counts": {
                str(label): int(dropped_counts[label]) for label in class_ids
            },
            "unique_within_epoch": len(selected) == len(set(selected)),
        }

    def exposure_audit(self) -> dict[str, object]:
        counts = Counter(
            index
            for epoch in range(self.epochs)
            for index in self.epoch_indices(epoch)
        )
        exposures = [counts[index] for index in range(len(self.labels))]
        return {
            "epochs": self.epochs,
            "subjects_per_epoch": self.subjects_per_epoch,
            "dropped_per_epoch": self.dropped_per_epoch,
            "min_subject_exposure": min(exposures),
            "max_subject_exposure": max(exposures),
        }

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for batch_index in range(self.steps_per_epoch):
            sample_step = self.epoch * self.steps_per_epoch + batch_index
            yield [
                (subject_index, sample_step)
                for subject_index in self.all_rank_indices(
                    self.epoch, batch_index
                )[self.rank]
            ]

    def __len__(self) -> int:
        return self.steps_per_epoch
