"""Finite Scalar Quantization and its distribution-matched iFSQ variant.

The codebook cardinality is always ``prod(L)``.  For odd ``L_i``, iFSQ maps
the encoder activation to ``[-1, 1]`` with ``2*sigmoid(1.6*z)-1``, scales it
by ``(L_i-1)/2``, rounds on that integer grid, and normalizes it back to
``[-1, 1]`` for the decoder.  This is the definition from Lin et al. (2026),
not a multiplication by ``L_i-1`` (which would silently create ``2*L_i-1``
levels).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FSQ(nn.Module):
    """Finite Scalar Quantizer with per-dimension levels.

    Args:
        d: number of quantization dimensions
        L: levels per dim (int→uniform, or list→per-dim)
        activation: ``"ifsq"`` for distribution-matched sigmoid or
            ``"tanh"`` for legacy FSQ.
        ifsq_alpha: sigmoid slope used by iFSQ (the paper selects 1.6).
    """

    def __init__(
        self,
        d: int = 4,
        L=5,
        *,
        activation: str = "tanh",
        ifsq_alpha: float = 1.6,
    ) -> None:
        super().__init__()
        if isinstance(L, int):
            L = [L] * d
        else:
            assert len(L) == d, f"L list len {len(L)} != d {d}"
        for li in L:
            assert li % 2 == 1, f"L must be odd, got {L}"
        if activation not in {"tanh", "ifsq"}:
            raise ValueError(
                f"FSQ activation must be 'tanh' or 'ifsq', got {activation!r}"
            )
        if ifsq_alpha <= 0:
            raise ValueError(f"ifsq_alpha must be positive, got {ifsq_alpha}")

        self.d = d
        self.L_per_dim = list(L)
        self.activation = activation
        self.ifsq_alpha = float(ifsq_alpha)

        codebook_size = 1
        for li in L:
            codebook_size *= li
        self.codebook_size = codebook_size

        # Per-dim level values: L_i=5 → [-1, -0.5, 0, 0.5, 1]
        self.levels_list = []
        for li in L:
            half = (li - 1) / 2
            lev = (torch.arange(li) - half) / half
            self.levels_list.append(lev)

        # Encoding strides: stride_i = prod_{j<i} L_j
        stride = 1
        base_list = []
        for li in L:
            base_list.append(stride)
            stride *= li
        self.register_buffer("base", torch.tensor(base_list).long(), persistent=False)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map unbounded encoder outputs to the decoder's ``[-1, 1]`` domain."""
        if self.activation == "ifsq":
            return 2.0 * torch.sigmoid(self.ifsq_alpha * z) - 1.0
        return z.tanh()

    def usage_stats_from_counts(self, cnt: torch.Tensor) -> dict[str, float]:
        """Return joint and per-scalar utilization from mixed-radix counts."""
        if cnt.ndim != 1 or cnt.numel() != self.codebook_size:
            raise ValueError(
                f"Expected {self.codebook_size} code counts, got {tuple(cnt.shape)}"
            )
        cnt = cnt.float()
        total = cnt.sum().clamp(min=1)
        nonzero = cnt > 0
        probs = cnt[nonzero] / total

        all_ids = torch.arange(self.codebook_size, device=cnt.device)
        dim_perplexities: list[float] = []
        dim_active_fractions: list[float] = []
        dim_top1: list[float] = []
        for level, base in zip(self.L_per_dim, self.base):
            digits = (all_ids // base) % level
            marginal = torch.zeros(level, device=cnt.device, dtype=cnt.dtype)
            marginal.scatter_add_(0, digits, cnt)
            marginal_probs = marginal / total
            positive = marginal_probs > 0
            dim_perplexities.append(
                float(torch.exp(-(marginal_probs[positive] * marginal_probs[positive].log()).sum()).item())
            )
            dim_active_fractions.append(float((marginal > 0).float().mean().item()))
            dim_top1.append(float(marginal_probs.max().item()))

        return {
            "active_code_fraction": float(nonzero.float().mean().item()),
            "code_perplexity": float(torch.exp(-(probs * probs.log()).sum()).item()),
            "dead_code_count": float((~nonzero).sum().item()),
            "top1_code_frequency": float((cnt.max() / total).item()),
            "code_dim_perplexity_mean": sum(dim_perplexities) / len(dim_perplexities),
            "code_dim_perplexity_min": min(dim_perplexities),
            "code_dim_active_fraction_min": min(dim_active_fractions),
            "code_dim_top1_frequency_max": max(dim_top1),
        }

    def forward(self, z: torch.Tensor):
        lead_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.d)
        z_bounded = self.bound(z_flat)

        idxs, zq = [], []
        for i, (lev, li) in enumerate(zip(self.levels_list, self.L_per_dim)):
            lev = lev.to(z.device)
            half_width = (li - 1) / 2.0
            quantized_integer = torch.round(z_bounded[:, i] * half_width)
            quantized_integer = quantized_integer.clamp(-half_width, half_width)
            idx = (quantized_integer + half_width).long()
            idxs.append(idx)
            # Equivalent to lev[idx], kept explicit to mirror the iFSQ paper.
            zq.append(quantized_integer / half_width)

        indices = torch.stack(idxs, dim=-1)
        z_q_flat = torch.stack(zq, dim=-1)
        z_q_st_flat = z_bounded + (z_q_flat - z_bounded).detach()
        code_ids_flat = (indices * self.base.to(z.device)).sum(dim=-1)

        z_q_st = z_q_st_flat.reshape(*lead_shape, self.d)
        z_q = z_q_flat.reshape(*lead_shape, self.d)
        code_ids = code_ids_flat.reshape(*lead_shape)

        with torch.no_grad():
            cnt = torch.bincount(code_ids_flat, minlength=self.codebook_size)
            stats = self.usage_stats_from_counts(cnt)
            stats.update({"n_reinitialized": 0, "commitment_loss": 0.0})
        return z_q_st, code_ids, z_q, torch.tensor(0.0, device=z.device), stats

    def init_from_kmeans(self, centroids: torch.Tensor) -> None:
        pass
