"""A small, standard straight-through VQ-VAE quantizer.

The module deliberately follows the same return contract as :class:`FSQ`, so
the tokenizer, cache writer and code-usage audit can compare quantizers without
forking the training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEQuantizer(nn.Module):
    """Learned nearest-neighbour codebook with the original VQ-VAE loss."""

    def __init__(self, d: int, codebook_size: int, commitment_beta: float = 0.25) -> None:
        super().__init__()
        if d < 1 or codebook_size < 2:
            raise ValueError("VQ-VAE requires d >= 1 and codebook_size >= 2")
        if commitment_beta < 0:
            raise ValueError("commitment_beta must be non-negative")
        self.d = int(d)
        self.codebook_size = int(codebook_size)
        self.commitment_beta = float(commitment_beta)
        self.embedding = nn.Embedding(self.codebook_size, self.d)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size,
        )

    def usage_stats_from_counts(self, counts: torch.Tensor) -> dict[str, float]:
        if counts.ndim != 1 or counts.numel() != self.codebook_size:
            raise ValueError(
                f"Expected {self.codebook_size} code counts, got {tuple(counts.shape)}"
            )
        counts = counts.float()
        total = counts.sum().clamp_min(1)
        nonzero = counts > 0
        probabilities = counts[nonzero] / total
        entropy = -(probabilities * probabilities.log()).sum()
        return {
            "active_code_fraction": float(nonzero.float().mean().item()),
            "code_perplexity": float(entropy.exp().item()),
            "dead_code_count": float((~nonzero).sum().item()),
            "top1_code_frequency": float((counts.max() / total).item()),
        }

    def forward(self, z: torch.Tensor):
        if z.shape[-1] != self.d:
            raise ValueError(f"Expected latent dimension {self.d}, got {z.shape[-1]}")
        lead_shape = z.shape[:-1]
        flat = z.reshape(-1, self.d)
        codebook = self.embedding.weight
        distances = (
            flat.float().square().sum(dim=1, keepdim=True)
            + codebook.float().square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat.float() @ codebook.float().t()
        )
        code_ids = distances.argmin(dim=1)
        quantized = self.embedding(code_ids).to(flat.dtype)
        codebook_loss = F.mse_loss(quantized, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized.detach())
        quantizer_loss = codebook_loss + self.commitment_beta * commitment_loss
        straight_through = flat + (quantized - flat).detach()

        with torch.no_grad():
            counts = torch.bincount(code_ids, minlength=self.codebook_size)
            stats = self.usage_stats_from_counts(counts)
            stats.update(
                {
                    "n_reinitialized": 0,
                    "codebook_loss": float(codebook_loss.detach().item()),
                    "commitment_loss": float(commitment_loss.detach().item()),
                }
            )
        return (
            straight_through.reshape(*lead_shape, self.d),
            code_ids.reshape(*lead_shape),
            quantized.reshape(*lead_shape, self.d),
            quantizer_loss,
            stats,
        )

    @torch.no_grad()
    def init_from_kmeans(self, centroids: torch.Tensor) -> None:
        if centroids.shape != self.embedding.weight.shape:
            raise ValueError(
                "K-means centroids must match the VQ-VAE codebook: "
                f"expected {tuple(self.embedding.weight.shape)}, got {tuple(centroids.shape)}"
            )
        self.embedding.weight.copy_(centroids.to(self.embedding.weight))
