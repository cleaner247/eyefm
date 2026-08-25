import numpy as np
import torch

from eyemae.eyevq.downstream.stage_a_screen import (
    CachedMILHead,
    VARIANTS,
    qc_balance_weights,
)


def test_qc_balance_weights_are_finite_and_normalized():
    qc = np.array([[0, 0], [0, 0.1], [1, 0.8], [1, 0.9]] * 5, dtype=float)
    labels = np.array([0, 0, 1, 1] * 5)
    weights = qc_balance_weights(qc, labels)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.mean(), 1.0)
    assert weights.min() >= 0.5 / 2.0  # normalization can shift clipped bounds


def test_constrained_task_weights_are_normalized_and_bounded():
    model = CachedMILHead(VARIANTS["d4_task_residual"], 8, 3)
    with torch.no_grad():
        model.task_residual.copy_(torch.tensor([-100.0, -1.0, 1.0, 100.0]))
    weights = model.task_weights()
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert float(weights.detach().min()) >= 0.15
    assert float(weights.detach().max()) <= 0.35


def test_late_demographic_alpha_is_bounded_and_near_zero_at_init():
    model = CachedMILHead(VARIANTS["d3_demo_late"], 8, 3)
    alpha = 0.3 * torch.sigmoid(model.demo_alpha_logit)
    alpha_value = float(alpha.detach())
    assert 0 <= alpha_value <= 0.3
    assert alpha_value < 0.01


def test_robust_four_bag_forward_shape():
    model = CachedMILHead(VARIANTS["d2_4xk4_robust"], 8, 3)
    logits = model(torch.randn(7, 4, 4, 8), torch.randn(7, 3))
    assert logits.shape == (7,)
