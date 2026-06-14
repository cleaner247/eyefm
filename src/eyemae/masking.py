from __future__ import annotations

from typing import Any

import torch


def _rand_float(generator: torch.Generator | None, device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device).item())


def _rand_int(low: int, high: int, generator: torch.Generator | None, device: torch.device) -> int:
    if high <= low:
        return low
    return int(torch.randint(low, high, (), generator=generator, device=device).item())


def mae_eligible_from_batch(batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    quality = batch["quality"]
    eye_nonmissing_frac = 1.0 - quality[..., 0].float().mean(dim=-1)
    return (
        (~batch["pad_mask"][:, :, None])
        & batch["eye_token_valid"].bool()
        & (eye_nonmissing_frac >= float(cfg["mask"]["min_nonmissing_frac_for_mae"]))
    )


def generate_mae_mask(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_mask = batch["pad_mask"].bool()
    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    eligible = mae_eligible_from_batch(batch, cfg)
    mae_mask = torch.zeros(bsz, nmax, 2, dtype=torch.bool, device=device)
    mask_type = torch.zeros(bsz, nmax, 2, dtype=torch.long, device=device)
    ratio_min = float(cfg["mask"]["ratio_min"])
    ratio_max = float(cfg["mask"]["ratio_max"])

    for b in range(bsz):
        elig = eligible[b]
        positions = torch.nonzero(elig, as_tuple=False)
        num_eligible = int(positions.shape[0])
        if num_eligible < 2:
            continue

        ratio = ratio_min + _rand_float(generator, device) * (ratio_max - ratio_min)
        target = int(ratio * num_eligible)
        target = max(1, min(target, num_eligible - 1))

        selected_eyes = torch.tensor([0, 1], dtype=torch.long, device=device)
        if _rand_float(generator, device) < float(cfg["mask"]["single_eye_mask_prob"]):
            selected_eyes = torch.tensor([_rand_int(0, 2, generator, device)], dtype=torch.long, device=device)
            if int(elig[:, selected_eyes].sum().item()) < 2:
                selected_eyes = torch.tensor([0, 1], dtype=torch.long, device=device)

        def apply_span(prob_key: str, min_key: str, max_key: str, typ: int) -> None:
            if _rand_float(generator, device) >= float(cfg["mask"][prob_key]):
                return
            span_len = _rand_int(int(cfg["mask"][min_key]), int(cfg["mask"][max_key]) + 1, generator, device)
            start = _rand_int(0, max(1, nmax), generator, device)
            end = min(nmax, start + span_len)
            for eye in selected_eyes.tolist():
                span = elig[start:end, eye]
                if span.numel() == 0:
                    continue
                mae_mask[b, start:end, eye] |= span
                update = span & (mask_type[b, start:end, eye] < typ)
                view = mask_type[b, start:end, eye]
                view[update] = typ

        apply_span("short_span_prob", "short_span_len_min", "short_span_len_max", 2)
        apply_span("long_span_prob", "long_span_len_min", "long_span_len_max", 3)

        selected_elig = torch.zeros_like(elig)
        selected_elig[:, selected_eyes] = elig[:, selected_eyes]
        if int(selected_elig.sum().item()) < target:
            selected_elig = elig
        remaining = torch.nonzero(selected_elig & (~mae_mask[b]), as_tuple=False)
        needed = target - int(mae_mask[b].sum().item())
        if needed > 0 and remaining.numel() > 0:
            perm = torch.randperm(remaining.shape[0], generator=generator, device=device)
            chosen = remaining[perm[: min(needed, remaining.shape[0])]]
            mae_mask[b, chosen[:, 0], chosen[:, 1]] = True
            current = mask_type[b, chosen[:, 0], chosen[:, 1]]
            current = torch.where(current == 0, torch.ones_like(current), current)
            mask_type[b, chosen[:, 0], chosen[:, 1]] = current

        mae_mask[b] &= elig
        mask_type[b][~mae_mask[b]] = 0
        visible_eligible = elig & (~mae_mask[b])
        if not bool(visible_eligible.any()):
            masked_positions = torch.nonzero(mae_mask[b], as_tuple=False)
            if masked_positions.numel() > 0:
                pick = masked_positions[_rand_int(0, masked_positions.shape[0], generator, device)]
                mae_mask[b, pick[0], pick[1]] = False
                mask_type[b, pick[0], pick[1]] = 0

    mae_mask[pad_mask] = False
    mask_type[~mae_mask] = 0
    return mae_mask, mask_type
