"""Paired left/right masking for formal EyeVQ-BERT pretraining."""

from __future__ import annotations

import math

import torch


PAIRED_RANDOM = "paired_random"
PAIRED_SPAN = "paired_span"
PAIRED_MULTIBLOCK = "paired_multiblock"
SUPPORTED_MASK_MODES = frozenset({PAIRED_RANDOM, PAIRED_SPAN, PAIRED_MULTIBLOCK})
UNIFORM_SPAN_LENGTHS = "uniform"
SYMMETRIC_POWER_SPAN_LENGTHS = "symmetric_power"
EXPLICIT_SPAN_LENGTHS = "explicit"
TOKEN_BALANCED_SPAN_LENGTHS = "token_balanced"
SUPPORTED_SPAN_LENGTH_DISTRIBUTIONS = frozenset(
    {
        UNIFORM_SPAN_LENGTHS,
        SYMMETRIC_POWER_SPAN_LENGTHS,
        EXPLICIT_SPAN_LENGTHS,
        TOKEN_BALANCED_SPAN_LENGTHS,
    }
)


def span_length_probabilities(
    span_min: int,
    span_max: int,
    *,
    distribution: str = UNIFORM_SPAN_LENGTHS,
    power: float = 1.0,
    explicit_probabilities: list[float] | tuple[float, ...] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the categorical prior over span lengths.

    ``token_balanced`` samples a length with probability proportional to its
    reciprocal.  Before feasibility conditioning, every length therefore
    contributes the same expected number of masked patches because
    ``p(length) * length`` is constant.

    ``symmetric_power`` is a discrete, symmetric beta/power shape.  It gives
    both endpoints less mass than the middle lengths, unlike a monotone
    Pareto law whose shortest span must be most frequent.
    """
    span_min = int(span_min)
    span_max = int(span_max)
    distribution = str(distribution)
    power = float(power)
    if span_min < 1 or span_max < span_min:
        raise ValueError(
            f"span bounds must satisfy 1 <= span_min <= span_max, got [{span_min}, {span_max}]"
        )
    if distribution not in SUPPORTED_SPAN_LENGTH_DISTRIBUTIONS:
        raise ValueError(
            "span length distribution must be one of "
            f"{sorted(SUPPORTED_SPAN_LENGTH_DISTRIBUTIONS)}, got {distribution!r}"
        )
    if not math.isfinite(power) or power <= 0:
        raise ValueError(f"span length power must be finite and positive, got {power}")
    lengths = torch.arange(span_min, span_max + 1, device=device, dtype=torch.float32)
    if distribution == EXPLICIT_SPAN_LENGTHS:
        if explicit_probabilities is None:
            raise ValueError("explicit span distribution requires probabilities")
        weights = torch.as_tensor(
            explicit_probabilities, dtype=torch.float32, device=device
        )
        if weights.ndim != 1 or weights.numel() != lengths.numel():
            raise ValueError(
                "explicit span probabilities must have one value per length; "
                f"expected {lengths.numel()}, got shape {tuple(weights.shape)}"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("explicit span probabilities must be finite and non-negative")
        if not float(weights.sum()) > 0:
            raise ValueError("explicit span probabilities must have a positive sum")
    elif distribution == TOKEN_BALANCED_SPAN_LENGTHS:
        weights = lengths.reciprocal()
    elif distribution == UNIFORM_SPAN_LENGTHS or lengths.numel() == 1:
        weights = torch.ones_like(lengths)
    else:
        left = lengths - float(span_min) + 0.5
        right = float(span_max) - lengths + 0.5
        weights = (left * right).pow(power)
    return weights / weights.sum()


def _validate_inputs(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float,
) -> None:
    if eye_valid.ndim != 3 or eye_valid.shape[-1] != 2:
        raise ValueError(f"eye_valid must have shape [B, N, 2], got {tuple(eye_valid.shape)}")
    if pad_mask.shape != eye_valid.shape[:2]:
        raise ValueError(
            f"pad_mask must have shape {tuple(eye_valid.shape[:2])}, got {tuple(pad_mask.shape)}"
        )
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")


def paired_time_eligibility(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Return time positions where both eyes are valid and the patch is not padding."""
    if eye_valid.ndim != 3 or eye_valid.shape[-1] != 2:
        raise ValueError(f"eye_valid must have shape [B, N, 2], got {tuple(eye_valid.shape)}")
    if pad_mask.shape != eye_valid.shape[:2]:
        raise ValueError(
            f"pad_mask must have shape {tuple(eye_valid.shape[:2])}, got {tuple(pad_mask.shape)}"
        )
    return eye_valid.bool().all(dim=-1) & ~pad_mask.bool()


def _pack_paired_time_mask(
    selected_time: torch.Tensor,
    *,
    with_cls: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand one shared time mask into the interleaved ``S,L,R`` token layout."""
    batch_size, n_time = selected_time.shape
    body_mask = torch.zeros(
        batch_size,
        n_time * 3,
        dtype=torch.bool,
        device=selected_time.device,
    )
    body_mask[:, 1::3] = selected_time
    body_mask[:, 2::3] = selected_time
    if not with_cls:
        return body_mask, body_mask
    cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=selected_time.device)
    input_mask = torch.cat([cls_mask, body_mask], dim=1)
    return input_mask, input_mask


def generate_eyemae_mask_paired_random(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float = 0.25,
    generator: torch.Generator | None = None,
    with_cls: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly mask shared L/R time positions.

    The quota is ``floor(mask_ratio * n_joint_valid)`` with a minimum of one
    when any jointly valid time position exists.  A selected time position
    always masks both its L and R tokens.  Stimulus, CLS, padding, and
    single-eye-only positions are never selected.
    """
    _validate_inputs(eye_valid, pad_mask, mask_ratio)
    joint_valid = paired_time_eligibility(eye_valid, pad_mask)
    batch_size, n_time = joint_valid.shape
    n_valid = joint_valid.sum(dim=-1)
    quota = torch.floor(n_valid.float() * mask_ratio).long()
    quota = torch.where(n_valid > 0, quota.clamp_min(1), quota)

    scores = torch.rand(
        batch_size,
        n_time,
        generator=generator,
        device=eye_valid.device,
    ).masked_fill(~joint_valid, float("-inf"))
    order = scores.argsort(dim=-1, descending=True)
    keep = torch.arange(n_time, device=eye_valid.device).view(1, n_time) < quota.unsqueeze(-1)
    selected = torch.zeros_like(joint_valid)
    selected.scatter_(dim=-1, index=order, src=keep.expand_as(order))
    selected &= joint_valid
    return _pack_paired_time_mask(selected, with_cls=with_cls)


def _window_is_available(available: torch.Tensor, length: int) -> torch.Tensor:
    """Return ``[B, N-length+1]`` flags for fully available windows."""
    if available.shape[1] < length:
        return available.new_zeros((available.shape[0], 0))
    return available.unfold(1, length, 1).all(dim=-1)


def generate_eyemae_mask_paired_span(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float = 0.25,
    generator: torch.Generator | None = None,
    with_cls: bool = True,
    span_min: int = 2,
    span_max: int = 6,
    span_length_distribution: str = UNIFORM_SPAN_LENGTHS,
    span_length_power: float = 1.0,
    span_length_probabilities_values: list[float] | tuple[float, ...] | None = None,
    return_span_lengths: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask shared L/R positions as disjoint contiguous temporal spans.

    Every connected masked component is between ``span_min`` and
    ``span_max`` patches.  Different spans are separated by at least one
    unmasked patch so adjacent spans cannot silently merge into an interval
    longer than ``span_max``.  The target number of masked time positions is
    ``floor(mask_ratio * n_joint_valid)``; when a valid span exists, the
    target is raised to ``span_min``.  Fragmented trials may realize a lower
    ratio rather than violating validity or the requested span bounds.
    """
    _validate_inputs(eye_valid, pad_mask, mask_ratio)
    span_min = int(span_min)
    span_max = int(span_max)
    if span_min < 1 or span_max < span_min:
        raise ValueError(
            f"span bounds must satisfy 1 <= span_min <= span_max, got [{span_min}, {span_max}]"
        )

    joint_valid = paired_time_eligibility(eye_valid, pad_mask)
    batch_size, n_time = joint_valid.shape
    n_valid = joint_valid.sum(dim=-1)
    target = torch.floor(n_valid.float() * mask_ratio).long()
    target = torch.where(
        n_valid >= span_min,
        target.clamp_min(span_min),
        torch.zeros_like(target),
    )
    target = torch.minimum(target, n_valid)
    selected = torch.zeros_like(joint_valid)
    # Per-time-position metadata for an optional predictor-side conditioning
    # embedding.  Zero means visible/not selected; every selected position is
    # tagged with the length of the exact span event that selected it.
    selected_span_lengths = torch.zeros_like(joint_valid, dtype=torch.long)
    lengths = torch.arange(span_min, span_max + 1, device=eye_valid.device)
    length_probabilities = span_length_probabilities(
        span_min,
        span_max,
        distribution=span_length_distribution,
        power=span_length_power,
        explicit_probabilities=span_length_probabilities_values,
        device=eye_valid.device,
    )

    # At least span_min positions are added per iteration.  The extra two
    # iterations make the loop robust when some samples have fragmented
    # validity and need a different sequence of span lengths.
    max_iterations = math.ceil(n_time / max(span_min, 1)) + 2
    positions = torch.arange(n_time, device=eye_valid.device).view(1, n_time)
    for _ in range(max_iterations):
        selected_count = selected.sum(dim=-1)
        remaining = target - selected_count
        active = remaining >= span_min

        # Reserve a one-patch gap around every existing interval, ensuring
        # that connected components stay within [span_min, span_max].
        blocked = selected.clone()
        blocked[:, 1:] |= selected[:, :-1]
        blocked[:, :-1] |= selected[:, 1:]
        available = joint_valid & ~blocked

        length_has_window = []
        windows_by_length: dict[int, torch.Tensor] = {}
        for length in range(span_min, span_max + 1):
            windows = _window_is_available(available, length)
            windows_by_length[length] = windows
            length_has_window.append(windows.any(dim=-1))
        has_window = torch.stack(length_has_window, dim=1)

        allowed = lengths.view(1, -1) <= remaining.unsqueeze(1)
        # Do not leave a positive remainder smaller than the minimum span.
        # For span_min=1 every remainder is representable, including one.
        next_remaining = remaining.unsqueeze(1) - lengths.view(1, -1)
        allowed &= (next_remaining == 0) | (next_remaining >= span_min)
        allowed &= has_window & active.unsqueeze(1)
        any_allowed = allowed.any(dim=1)
        # Check termination every two rounds.  Once all rows become inactive,
        # the one possible intervening round is a no-op; this halves the
        # dominant CUDA->CPU control-flow synchronizations without a long
        # fixed unrolled tail.
        if _ % 2 == 1 and not bool(any_allowed.any()):
            break

        uniform_noise = torch.rand(
            batch_size,
            len(lengths),
            generator=generator,
            device=eye_valid.device,
        ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - 1e-7)
        # Gumbel-max samples from the requested categorical distribution;
        # masking disallowed lengths automatically renormalizes its support.
        length_scores = (
            length_probabilities.log().unsqueeze(0)
            - torch.log(-torch.log(uniform_noise))
        ).masked_fill(~allowed, float("-inf"))
        length_indices = length_scores.argmax(dim=1)
        chosen_lengths = lengths[length_indices]

        additions = torch.zeros_like(selected)
        # A row chooses exactly one length, so one shared i.i.d. score matrix
        # is sufficient for all candidate lengths.  Generating a separate
        # [B,N] matrix five times produced the same conditional uniform start
        # distribution but wasted most of the mask-generator work.
        start_noise = torch.rand(
            batch_size,
            n_time,
            generator=generator,
            device=eye_valid.device,
        )
        for length in range(span_min, span_max + 1):
            rows = any_allowed & (chosen_lengths == length)
            windows = windows_by_length[length]
            start_scores = start_noise[:, : windows.shape[1]].masked_fill(
                ~windows, float("-inf")
            )
            starts = start_scores.argmax(dim=1)
            additions |= rows.unsqueeze(1) & (
                (positions >= starts.unsqueeze(1))
                & (positions < (starts + length).unsqueeze(1))
            )
        selected_span_lengths = torch.where(
            additions,
            chosen_lengths.unsqueeze(1).expand_as(selected_span_lengths),
            selected_span_lengths,
        )
        selected |= additions

    selected &= joint_valid
    packed = _pack_paired_time_mask(selected, with_cls=with_cls)
    if return_span_lengths:
        selected_span_lengths.masked_fill_(~selected, 0)
        return packed[0], packed[1], selected_span_lengths
    return packed


def _contiguous_run_lengths(available: torch.Tensor) -> torch.Tensor:
    """Return the valid run length starting at every time position."""
    batch_size, n_time = available.shape
    positions = torch.arange(n_time, device=available.device).view(1, n_time)
    sentinel = torch.full_like(positions, n_time).expand(batch_size, -1)
    invalid_positions = torch.where(~available, positions.expand(batch_size, -1), sentinel)
    next_invalid = torch.flip(
        torch.cummin(torch.flip(invalid_positions, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    return torch.where(available, next_invalid - positions, 0)


def generate_eyemae_mask_paired_multiblock(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float = 0.60,
    generator: torch.Generator | None = None,
    with_cls: bool = True,
    *,
    num_blocks: int = 4,
    target_scale_min: float = 0.12,
    target_scale_max: float = 0.18,
    min_gap: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """JEPA-inspired paired temporal multi-block masking.

    Each trial samples one target scale and uses that primary length for all
    target blocks.  Block locations are sampled independently, but unlike
    image I-JEPA the one-dimensional blocks are strictly non-overlapping and
    separated by ``min_gap`` visible patches.  If fragmented validity makes
    the primary length infeasible late in placement, only that block is
    shortened to the longest remaining valid run.  Stimulus, padding, and
    single-eye-only positions are never selected.

    ``mask_ratio`` remains part of the common masking API and is validated,
    while the realized ratio for this policy is determined by target scale,
    block count, overlap prohibition, and eye validity.
    """
    _validate_inputs(eye_valid, pad_mask, mask_ratio)
    num_blocks = int(num_blocks)
    min_gap = int(min_gap)
    target_scale_min = float(target_scale_min)
    target_scale_max = float(target_scale_max)
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    if not 0.0 < target_scale_min <= target_scale_max < 1.0:
        raise ValueError(
            "target scales must satisfy 0 < min <= max < 1, got "
            f"[{target_scale_min}, {target_scale_max}]"
        )

    joint_valid = paired_time_eligibility(eye_valid, pad_mask)
    batch_size, n_time = joint_valid.shape
    n_valid = joint_valid.sum(dim=1)
    scales = torch.empty(
        batch_size, device=eye_valid.device, dtype=torch.float32
    ).uniform_(target_scale_min, target_scale_max, generator=generator)
    primary_lengths = torch.floor(n_valid.float() * scales).long()
    primary_lengths = torch.where(
        n_valid > 0,
        primary_lengths.clamp_min(1),
        torch.zeros_like(primary_lengths),
    )

    selected = torch.zeros_like(joint_valid)
    positions = torch.arange(n_time, device=eye_valid.device).view(1, n_time)
    for _ in range(num_blocks):
        blocked = selected.clone()
        for distance in range(1, min_gap + 1):
            blocked[:, distance:] |= selected[:, :-distance]
            blocked[:, :-distance] |= selected[:, distance:]
        available = joint_valid & ~blocked
        run_lengths = _contiguous_run_lengths(available)
        longest_remaining = run_lengths.max(dim=1).values
        block_lengths = torch.minimum(primary_lengths, longest_remaining)
        active = block_lengths > 0

        feasible_starts = run_lengths >= block_lengths.unsqueeze(1)
        feasible_starts &= active.unsqueeze(1)
        start_scores = torch.rand(
            batch_size,
            n_time,
            generator=generator,
            device=eye_valid.device,
        ).masked_fill(~feasible_starts, float("-inf"))
        starts = start_scores.argmax(dim=1)
        additions = active.unsqueeze(1) & (
            (positions >= starts.unsqueeze(1))
            & (positions < (starts + block_lengths).unsqueeze(1))
        )
        selected |= additions

    selected &= joint_valid
    return _pack_paired_time_mask(selected, with_cls=with_cls)


def generate_eyemae_mask(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float = 0.25,
    *,
    mode: str = PAIRED_RANDOM,
    generator: torch.Generator | None = None,
    with_cls: bool = True,
    span_min: int = 2,
    span_max: int = 6,
    span_length_distribution: str = UNIFORM_SPAN_LENGTHS,
    span_length_power: float = 1.0,
    span_length_probabilities_values: list[float] | tuple[float, ...] | None = None,
    multiblock_num_blocks: int = 4,
    multiblock_target_scale_min: float = 0.12,
    multiblock_target_scale_max: float = 0.18,
    multiblock_min_gap: int = 1,
    return_span_lengths: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to one of the formal paired-mask policies."""
    normalized_mode = PAIRED_RANDOM if mode == "uniform" else str(mode)
    if return_span_lengths and normalized_mode != PAIRED_SPAN:
        raise ValueError("span-length metadata is only defined for paired_span masking")
    if normalized_mode == PAIRED_RANDOM:
        return generate_eyemae_mask_paired_random(
            eye_valid,
            pad_mask,
            mask_ratio,
            generator=generator,
            with_cls=with_cls,
        )
    if normalized_mode == PAIRED_SPAN:
        return generate_eyemae_mask_paired_span(
            eye_valid,
            pad_mask,
            mask_ratio,
            generator=generator,
            with_cls=with_cls,
            span_min=span_min,
            span_max=span_max,
            span_length_distribution=span_length_distribution,
            span_length_power=span_length_power,
            span_length_probabilities_values=span_length_probabilities_values,
            return_span_lengths=return_span_lengths,
        )
    if normalized_mode == PAIRED_MULTIBLOCK:
        return generate_eyemae_mask_paired_multiblock(
            eye_valid,
            pad_mask,
            mask_ratio,
            generator=generator,
            with_cls=with_cls,
            num_blocks=multiblock_num_blocks,
            target_scale_min=multiblock_target_scale_min,
            target_scale_max=multiblock_target_scale_max,
            min_gap=multiblock_min_gap,
        )
    raise ValueError(
        f"mask mode must be one of {sorted(SUPPORTED_MASK_MODES)}, got {mode!r}"
    )


def generate_eyemae_mask_uniform(
    eye_valid: torch.Tensor,
    pad_mask: torch.Tensor,
    mask_ratio: float = 0.25,
    generator: torch.Generator | None = None,
    with_cls: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible name for paired random masking.

    The old independent-per-eye behavior is intentionally removed: all new
    and legacy callers now receive synchronized L/R masks.
    """
    return generate_eyemae_mask_paired_random(
        eye_valid,
        pad_mask,
        mask_ratio,
        generator=generator,
        with_cls=with_cls,
    )
