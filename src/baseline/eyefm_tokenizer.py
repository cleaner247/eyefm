"""Eyefm 3-stream tokenizer (foundation-model style).

Splits the v8 (B, K, T, 12) input into 3 streams (eye-L / eye-R / stim),
projects each to d_model, and adds per-stream token type embedding.
Task (saccade paradigm) is per-trial, mean-pooled across K trials, then
concatenated to the head.

Reference architecture: foundation model (EyeMAE) config
  `eyefm/configs/eyemae_cnn_512_12l.yaml`
which uses `sequence_format: stim_eye_triplet_no_cls` with 3 token types.
The DL baseline simplifies by NOT patching (per-frame tokens instead of
per-patch triplets) and shares the per-stream heads as parallel
siblings rather than 1 Transformer over interleaved tokens.

Column layout for the (T, 12) input (set in data_loader.py):
  L: cols 0, 1, 2, 10   (L_x, L_y, L_area, L_quality)
  R: cols 3, 4, 5, 11   (R_x, R_y, R_area, R_quality)
  S: cols 6, 7, 8,  9   (S_x, S_y, S_on, S_fix)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Column indices in the v8 (T, 12) input layout.
COL_L = (0, 1, 2, 10)
COL_R = (3, 4, 5, 11)
COL_S = (6, 7, 8, 9)


class ThreeStreamTokenizer(nn.Module):
    """Project 3 streams to d_model, add token type embedding, layernorm.

    Input:  (B, K, T, 12)
    Output: 3 tensors, each (B*K, T, d_model) — one per stream.
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        # Per-stream linear projection (4-dim raw → d_model).
        self.proj_l = nn.Linear(4, d_model)
        self.proj_r = nn.Linear(4, d_model)
        self.proj_s = nn.Linear(4, d_model)
        # Token-type embedding: 3 types (L, R, S).
        self.token_type = nn.Embedding(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.token_type.weight, std=0.02)

    def split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, K, T, 12) → 3 streams of (B, K, T, 4)."""
        l = x[..., COL_L]
        r = x[..., COL_R]
        s = x[..., COL_S]
        return l, r, s

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, K, T, 12) → (h_l, h_r, h_s) each (B*K, T, d_model).

        v8.1 fix: removed LayerNorm — was destroying stream magnitude info
        that the head needs to discriminate classes. Keep small dropout only.
        """
        B, K, T, _ = x.shape
        l, r, s = self.split_input(x)
        l = l.reshape(B * K, T, 4)
        r = r.reshape(B * K, T, 4)
        s = s.reshape(B * K, T, 4)
        h_l = self.proj_l(l) + self.token_type.weight[0]
        h_r = self.proj_r(r) + self.token_type.weight[1]
        h_s = self.proj_s(s) + self.token_type.weight[2]
        if self.dropout.p > 0:
            h_l = self.dropout(h_l)
            h_r = self.dropout(h_r)
            h_s = self.dropout(h_s)
        return h_l, h_r, h_s


class TaskEmbedding(nn.Module):
    """Per-trial saccade task id embedding, mean-pooled across K trials.

    Output: (B, d_task) per subject. Mean-pool across K because the
    saccade paradigm is a subject-level attribute (every trial of a
    given run uses the same paradigm).
    """

    def __init__(self, n_tasks: int = 4, d_task: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_tasks, d_task)
        self.d_task = d_task
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, task_idx: torch.Tensor | None) -> torch.Tensor:
        """task_idx: (B, K) → (B, d_task). If None, return zeros."""
        if task_idx is None:
            return None  # caller handles None as zeros
        B, K = task_idx.shape
        emb = self.emb(task_idx.reshape(-1))        # (B*K, d_task)
        emb = emb.reshape(B, K, -1).mean(dim=1)     # (B, d_task)
        return emb


class ThreeStreamHead(nn.Module):
    """Fuse 3 stream pools + task embedding → n_classes logits.

    Inputs: (h_l_pool, h_r_pool, h_s_pool) each (B, d_model),
            task_emb (B, d_task) or None.
    Output: (B, n_classes).
    """

    def __init__(self, d_model: int = 64, d_task: int = 16, n_classes: int = 2,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.d_task = d_task
        in_dim = 3 * d_model + d_task
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, h_l: torch.Tensor, h_r: torch.Tensor, h_s: torch.Tensor,
                task_emb: torch.Tensor | None) -> torch.Tensor:
        """Concatenate 3 stream pools + task_emb, then MLP → (B, n_classes)."""
        B = h_l.shape[0]
        if task_emb is None:
            task_emb = torch.zeros(B, self.d_task, device=h_l.device, dtype=h_l.dtype)
        feat = torch.cat([h_l, h_r, h_s, task_emb], dim=-1)
        return self.head(feat)
