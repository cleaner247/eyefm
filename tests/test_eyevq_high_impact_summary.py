import numpy as np

from eyemae.eyevq.downstream.summarize_high_impact import _paired_bootstrap


def test_paired_bootstrap_detects_better_candidate() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([0.1, 0.4, 0.7, 0.3, 0.6, 0.9])
    candidate = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = _paired_bootstrap(
        labels, baseline, candidate, samples=500, seed=3
    )
    assert result["mean_delta"] > 0
    assert result["probability_delta_gt_zero"] > 0.5


def test_overlapping_bootstrap_interval_does_not_support_gain() -> None:
    labels = np.array([0, 0, 1, 1])
    logits = np.array([0.1, 0.2, 0.8, 0.9])
    result = _paired_bootstrap(labels, logits, logits, samples=200, seed=4)
    assert result["ci_2p5"] == 0.0
    assert not (result["ci_2p5"] > 0.0)
