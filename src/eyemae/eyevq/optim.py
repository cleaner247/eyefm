"""Shared, auditable AdamW parameter grouping for EyeVQ."""

from __future__ import annotations

import math
from typing import Any

import torch


def uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    """Decay matrix weights, but never biases, norms, or vector/scalar state."""
    normalized = name.replace("module.", "").replace("_orig_mod.", "").lower()
    if parameter.ndim <= 1 or normalized.endswith(".bias"):
        return False
    # CLS/mask tokens are commonly stored as [1, 1, D].  They are semantic
    # vectors rather than matrix/kernel weights and must follow vector policy.
    if parameter.ndim > 1 and math.prod(parameter.shape[:-1]) == 1:
        return False
    if "norm" in normalized:
        return False
    return True


def build_adamw_param_groups(
    module: torch.nn.Module,
    *,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return complete, duplicate-free AdamW groups and a JSON audit payload."""
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    buckets: dict[float, list[torch.nn.Parameter]] = {0.0: [], float(weight_decay): []}
    names: dict[float, list[str]] = {0.0: [], float(weight_decay): []}
    seen: set[int] = set()
    expected: set[int] = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        expected.add(parameter_id)
        if parameter_id in seen:
            raise RuntimeError(f"Optimizer parameter appears more than once: {name}")
        seen.add(parameter_id)
        decay = float(weight_decay) if uses_weight_decay(name, parameter) else 0.0
        buckets[decay].append(parameter)
        names[decay].append(name)
    if seen != expected or not seen:
        raise RuntimeError("Optimizer grouping did not cover every trainable parameter")

    groups: list[dict[str, Any]] = []
    audit_groups: list[dict[str, Any]] = []
    for decay in (float(weight_decay), 0.0):
        parameters = buckets[decay]
        if not parameters:
            continue
        groups.append({"params": parameters, "weight_decay": decay})
        audit_groups.append({
            "weight_decay": decay,
            "num_tensors": len(parameters),
            "num_parameters": sum(parameter.numel() for parameter in parameters),
            "parameter_names": names[decay],
        })
    audit = {
        "policy": "decay_matrix_or_kernel_weights_only; no_decay=bias,norm,vector_state",
        "weight_decay": float(weight_decay),
        "num_trainable_tensors": len(seen),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "groups": audit_groups,
    }
    return groups, audit
