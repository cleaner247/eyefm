from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch


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
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate MAE mask: random numbers on GPU, logic on CPU via numpy.

    Strategy: generate all randomness in one GPU kernel (fast + deterministic),
    download tiny random buffer to CPU, run per-sample logic in numpy (no syncs),
    upload final boolean mask to GPU.

    When *stream* is given, GPU work (eligible + rand) runs on that stream so
    that ``.cpu()`` only synchronises the dedicated stream, NOT the default
    (compute) stream.  This lets mask generation overlap with the previous
    micro-batch's backward pass.
    """
    device = batch["pad_mask"].device
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        eligible_gpu = mae_eligible_from_batch(batch, cfg)  # (bsz, nmax, 2)
        bsz, nmax, _ = eligible_gpu.shape

        # ── Generate ALL random numbers on GPU in one shot ──
        # 10 randoms per sample: ratio, single_eye, eye_idx, short_trig, short_len,
        # short_start, long_trig, long_len, long_start, safety_pick
        rand_gpu = torch.rand(bsz, 10, generator=generator, device=device)

    # Synchronise ONLY the mask stream (if any), not the default compute stream.
    if stream is not None:
        stream.synchronize()
    eligible_np = eligible_gpu.cpu().numpy()
    rand_np = rand_gpu.cpu().numpy()

    ratio_min = float(cfg["mask"]["ratio_min"])
    ratio_max = float(cfg["mask"]["ratio_max"])
    single_eye_prob = float(cfg["mask"]["single_eye_mask_prob"])
    short_prob = float(cfg["mask"]["short_span_prob"])
    short_min = int(cfg["mask"]["short_span_len_min"])
    short_max = int(cfg["mask"]["short_span_len_max"])
    long_prob = float(cfg["mask"]["long_span_prob"])
    long_min = int(cfg["mask"]["long_span_len_min"])
    long_max = int(cfg["mask"]["long_span_len_max"])

    mae_mask_np = np.zeros((bsz, nmax, 2), dtype=bool)
    mask_type_np = np.zeros((bsz, nmax, 2), dtype=np.int64)

    for b in range(bsz):
        elig = eligible_np[b]
        positions = np.argwhere(elig)
        num_eligible = positions.shape[0]
        if num_eligible < 2:
            continue

        r = rand_np[b]
        ratio = ratio_min + r[0] * (ratio_max - ratio_min)
        target = max(1, min(int(ratio * num_eligible), num_eligible - 1))

        # Eye selection
        if r[1] < single_eye_prob:
            se = 0 if r[2] < 0.5 else 1
            sel = [se]
            if elig[:, sel].sum() < 2:
                sel = [0, 1]
        else:
            sel = [0, 1]

        # Short span
        if r[3] < short_prob:
            slen = short_min + int(r[4] * (short_max - short_min + 1))
            sstart = int(r[4] * max(1, nmax))
            send = min(nmax, sstart + slen)
            for eye in sel:
                span = elig[sstart:send, eye]
                if span.any():
                    mae_mask_np[b, sstart:send, eye] |= span
                    upd = span & (mask_type_np[b, sstart:send, eye] < 2)
                    mask_type_np[b, sstart:send, eye][upd] = 2

        # Long span
        if r[5] < long_prob:
            llen = long_min + int(r[6] * (long_max - long_min + 1))
            lstart = int(r[7] * max(1, nmax))
            lend = min(nmax, lstart + llen)
            for eye in sel:
                span = elig[lstart:lend, eye]
                if span.any():
                    mae_mask_np[b, lstart:lend, eye] |= span
                    upd = span & (mask_type_np[b, lstart:lend, eye] < 3)
                    mask_type_np[b, lstart:lend, eye][upd] = 3

        # Random fill to reach target ratio
        sel_mask = np.zeros(2, dtype=bool)
        for e in sel:
            sel_mask[e] = True
        selected_elig = elig & sel_mask[np.newaxis, :]
        if selected_elig.sum() < target:
            selected_elig = elig
        remaining = np.argwhere(selected_elig & (~mae_mask_np[b]))
        needed = target - int(mae_mask_np[b].sum())
        if needed > 0 and remaining.shape[0] > 0:
            n_pick = min(needed, remaining.shape[0])
            # Derive deterministic shuffle from pre-generated randoms
            local_rng = np.random.RandomState(int(r[8] * 2**31))
            idx = np.arange(remaining.shape[0])
            local_rng.shuffle(idx)
            chosen = remaining[idx[:n_pick]]
            for pos in chosen:
                mae_mask_np[b, pos[0], pos[1]] = True
                if mask_type_np[b, pos[0], pos[1]] == 0:
                    mask_type_np[b, pos[0], pos[1]] = 1

        mae_mask_np[b] &= elig
        mask_type_np[b][~mae_mask_np[b]] = 0
        if not (elig & (~mae_mask_np[b])).any():
            masked_positions = np.argwhere(mae_mask_np[b])
            if masked_positions.shape[0] > 0:
                pick_idx = int(r[9] * masked_positions.shape[0])
                p = masked_positions[pick_idx]
                mae_mask_np[b, p[0], p[1]] = False
                mask_type_np[b, p[0], p[1]] = 0

    # Upload to GPU
    mae_mask = torch.from_numpy(mae_mask_np).to(device)
    mask_type = torch.from_numpy(mask_type_np).to(device)

    pad_mask_bool = batch["pad_mask"].bool()
    mae_mask[pad_mask_bool] = False
    mask_type[~mae_mask] = 0
    return mae_mask, mask_type
