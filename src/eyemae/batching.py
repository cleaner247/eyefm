from __future__ import annotations

import math
import random
from typing import Iterator

from torch.utils.data import Sampler


class TokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        *,
        max_seq_tokens: int,
        max_trials: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        bucket_by_length: bool = False,
        bucket_size: int | None = None,
        infinite: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.max_seq_tokens = int(max_seq_tokens)
        self.max_trials = int(max_trials)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.bucket_by_length = bool(bucket_by_length)
        self.bucket_size = int(bucket_size) if bucket_size is not None else None
        self.infinite = bool(infinite)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Resume an infinite deterministic stream without replaying samples.

        ``start_batch`` counts already-consumed batches across all internal
        epochs.  Skipping happens on lists of indices inside the sampler, so a
        resumed job does not waste time loading and collating old batches.
        """
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        self.start_batch = int(start_batch)

    def _seq_tokens(self, idx: int) -> int:
        return 3 * max(1, int(self.dataset.get_num_patches(idx)))

    def _padded_seq_tokens(self, batch_len: int, max_patches: int) -> int:
        return 3 * max(1, int(max_patches)) * max(1, int(batch_len))

    def _ordered_indices(self, epoch: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + int(epoch))
        if self.shuffle:
            rng.shuffle(indices)
        if self.world_size > 1:
            indices = indices[self.rank :: self.world_size]
        if self.bucket_by_length:
            bucket_size = self.bucket_size or max(1024, self.max_trials * 10)
            ordered: list[int] = []
            for start in range(0, len(indices), bucket_size):
                window = indices[start : start + bucket_size]
                window.sort(key=self.dataset.get_num_patches)
                ordered.extend(window)
            indices = ordered
        return indices

    def _make_batches(self, epoch: int) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        max_patches = 0
        for idx in self._ordered_indices(epoch):
            patches = max(1, int(self.dataset.get_num_patches(idx)))
            next_max_patches = max(max_patches, patches)
            next_tokens = self._padded_seq_tokens(len(batch) + 1, next_max_patches)
            would_overflow = batch and (
                len(batch) >= self.max_trials or next_tokens > self.max_seq_tokens
            )
            if would_overflow:
                batches.append(batch)
                batch = []
                max_patches = 0
                next_max_patches = patches
            batch.append(idx)
            max_patches = next_max_patches
        if batch and not self.drop_last:
            batches.append(batch)
        # Keep deterministic order for validation. Training still interleaves
        # length buckets, using the same global seed on every rank.
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        batches_to_skip = self.start_batch
        while True:
            batches = self._make_batches(epoch)
            if batches_to_skip >= len(batches):
                batches_to_skip -= len(batches)
            else:
                yield from batches[batches_to_skip:]
                batches_to_skip = 0
            if not self.infinite:
                break
            epoch += 1

    def __len__(self) -> int:
        if self.infinite:
            return math.ceil(len(self.dataset) / max(1, self.world_size))
        return len(self._make_batches(self.epoch))
