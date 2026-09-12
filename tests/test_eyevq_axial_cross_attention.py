from __future__ import annotations

import torch

from eyemae.eyevq.tokenizer.model import AxialCrossTransformerBlock


def test_axial_cross_attention_shapes_calls_and_backward() -> None:
    torch.manual_seed(0)
    batch, n_time, dim = 2, 4, 16
    block = AxialCrossTransformerBlock(dim, heads=4, ffn_hidden=32, dropout=0.0)
    x = torch.randn(batch, 1 + 3 * n_time, dim, requires_grad=True)
    padding = torch.zeros(batch, 1 + 3 * n_time, dtype=torch.bool)
    padding[1, -3:] = True
    calls: list[tuple[str, tuple[int, ...]]] = []

    temporal_hook = block.temporal_qkv.register_forward_hook(
        lambda _module, inputs, _output: calls.append(
            ("temporal", tuple(inputs[0].shape))
        )
    )
    local_hook = block.local_qkv.register_forward_hook(
        lambda _module, inputs, _output: calls.append(
            ("local", tuple(inputs[0].shape))
        )
    )
    output = block(x, padding, n_time=n_time)
    temporal_hook.remove()
    local_hook.remove()

    assert output.shape == x.shape
    assert calls == [
        ("temporal", (batch, n_time, dim)),
        ("temporal", (batch, 1 + 2 * n_time, dim)),
        ("local", (batch * n_time, 3, dim)),
    ]
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.local_qkv.weight.grad is not None
    assert block.temporal_qkv.weight.grad is not None


def test_axial_cross_attention_ignores_padded_keys() -> None:
    torch.manual_seed(1)
    batch, n_time, dim = 1, 3, 16
    block = AxialCrossTransformerBlock(dim, heads=4, ffn_hidden=32, dropout=0.0)
    block.eval()
    x = torch.randn(batch, 1 + 3 * n_time, dim)
    padding = torch.zeros(batch, 1 + 3 * n_time, dtype=torch.bool)
    padding[:, -3:] = True
    changed = x.clone()
    changed[:, -3:] = torch.randn_like(changed[:, -3:]) * 1000

    first = block(x, padding, n_time=n_time)
    second = block(changed, padding, n_time=n_time)
    valid_queries = ~padding
    assert torch.allclose(first[valid_queries], second[valid_queries], atol=1e-5, rtol=1e-5)
