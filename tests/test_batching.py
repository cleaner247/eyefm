from __future__ import annotations

from itertools import islice

from eyemae.batching import TokenBatchSampler


class FakeLengthDataset:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def get_num_patches(self, index: int) -> int:
        return self.lengths[index]


def _tokens(dataset: FakeLengthDataset, batch: list[int]) -> int:
    return 3 * max(dataset.get_num_patches(i) for i in batch) * len(batch)


def test_token_batch_sampler_respects_token_and_trial_limits() -> None:
    dataset = FakeLengthDataset([2, 4, 6, 8, 10, 12])
    sampler = TokenBatchSampler(
        dataset,
        max_seq_tokens=36,
        max_trials=2,
        shuffle=False,
        bucket_by_length=False,
    )
    batches = list(sampler)
    assert batches == [[0, 1], [2], [3], [4], [5]]
    assert all(len(batch) <= 2 for batch in batches)
    assert all(_tokens(dataset, batch) <= 36 for batch in batches)


def test_token_batch_sampler_buckets_by_length_inside_random_windows() -> None:
    dataset = FakeLengthDataset([8, 1, 7, 2, 6, 3, 5, 4])
    sampler = TokenBatchSampler(
        dataset,
        max_seq_tokens=1000,
        max_trials=8,
        shuffle=False,
        bucket_by_length=True,
        bucket_size=4,
    )
    assert list(sampler) == [[1, 3, 2, 0, 5, 7, 6, 4]]


def test_token_batch_sampler_infinite_ddp_ranks_do_not_exhaust() -> None:
    dataset = FakeLengthDataset([1, 10, 2, 9, 3, 8, 4, 7, 5])
    rank_batches = [
        list(
            islice(
            TokenBatchSampler(
                dataset,
                max_seq_tokens=30,
                max_trials=2,
                shuffle=False,
                bucket_by_length=True,
                bucket_size=4,
                infinite=True,
                rank=rank,
                world_size=3,
            ),
            6,
            )
        )
        for rank in range(3)
    ]
    assert [len(batches) for batches in rank_batches] == [6, 6, 6]
    for batches in rank_batches:
        for batch in batches:
            assert len(batch) <= 2
            assert _tokens(dataset, batch) <= 30


def test_token_batch_sampler_budget_counts_padding() -> None:
    dataset = FakeLengthDataset([1, 10, 1])
    sampler = TokenBatchSampler(
        dataset,
        max_seq_tokens=36,
        max_trials=10,
        shuffle=False,
        bucket_by_length=False,
    )
    assert list(sampler) == [[0], [1], [2]]
