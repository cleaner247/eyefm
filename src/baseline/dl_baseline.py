"""EyeFM DL baseline: 4 E2Mo-style architectures with 50-epoch val-pick.

Architectures (paper-replicable from E2Mo 表格):
  - TCN:              Temporal Convolutional Network (Bai et al. 2018)
  - TimesNet:         FFT-period 2D Inception (Wu et al. IJCAI 2023)
  - NST:              Non-stationary Transformer (Liu NeurIPS 2022)
  - CNNTransformer:   1D conv front-end + Transformer encoder (E2Mo simplified)

Training protocol (paper-standard hp, no grid search):
  - max 50 epochs
  - LR=3e-4, weight_decay=0.01 (TCN) / 0.05 (others)
  - dropout=0.3, batch=16, T_LEN=1024
  - class_weight balanced (inverse-freq from train labels, mean-normalized)
  - val AUROC per epoch: if higher than previous best → save best ckpt
  - after 50 epochs → reload best ckpt, run final test

Reference:  docs/eyemae_baseline.md
"""
from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data_loader import (
    ShardCache,
    SplitData,
    TrialDataset,
    build_split_data,
    class_weights_from_labels,
    collate_subjects,
    make_subject_batches,
)
from .eyefm_tokenizer import ThreeStreamHead, ThreeStreamTokenizer, TaskEmbedding

LOGGER = logging.getLogger(__name__)

# ==== v8 defaults: 3-stream + task-conditioning ====
DEFAULT_N_TASKS = 4
DEFAULT_D_MODEL = 64
DEFAULT_D_TASK = 16


# ==== Hyper-params (paper-standard) ====
DEFAULT_T_LEN = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 50
DEFAULT_LR = 3e-4
DEFAULT_WD = 0.05
DEFAULT_WD_TCN = 0.01
DEFAULT_DROPOUT = 0.3
SEED = 42


# ==== Architectures ====
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        h = self.conv1(x)
        h = self.chomp1(h)
        h = self.relu1(h)
        h = self.drop1(h)
        h = self.conv2(h)
        h = self.chomp2(h)
        h = self.relu2(h)
        h = self.drop2(h)
        return self.relu(h + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network (Bai et al. 2018) — 3-stream + task conditioning.

    Foundation-model style: split (B, K, T, 12) input into L/R/S streams via
    ThreeStreamTokenizer, run 3 independent TCN stacks (one per stream),
    pool each stream, fuse with task embedding in ThreeStreamHead.
    """

    def __init__(self, in_dim: int = 12, hidden: int = 64, n_blocks: int = 2,
                 n_classes: int = 2, dropout: float = DEFAULT_DROPOUT,
                 n_tasks: int = DEFAULT_N_TASKS, d_model: int = DEFAULT_D_MODEL,
                 d_task: int = DEFAULT_D_TASK) -> None:
        super().__init__()
        self.tokenizer = ThreeStreamTokenizer(d_model=d_model, dropout=dropout)
        self.task_emb = TaskEmbedding(n_tasks=n_tasks, d_task=d_task)
        # 3 parallel TCN stacks (input dim = d_model from tokenizer).
        def make_tcn() -> nn.Sequential:
            layers: list[nn.Module] = []
            in_ch = d_model
            for i in range(n_blocks):
                layers.append(TCNBlock(in_ch, hidden, kernel_size=3, dilation=2 ** i, dropout=dropout))
                in_ch = hidden
            return nn.Sequential(*layers)
        self.tcn_l = make_tcn()
        self.tcn_r = make_tcn()
        self.tcn_s = make_tcn()
        self.head = ThreeStreamHead(d_model=hidden, d_task=d_task, n_classes=n_classes, dropout=dropout)

    def _stream_pool(self, tcn: nn.Sequential, h: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """Run TCN over (B*K, T, d_model) → pool → reshape to (B, hidden)."""
        # Conv1d wants (N, C, T); we have (N, T, C) from tokenizer.
        h_in = h.transpose(1, 2)
        h_out = tcn(h_in)  # (N, hidden, T)
        h_pool = h_out.mean(dim=-1)  # (N, hidden) — avg over time
        h_pool = h_pool.reshape(B, K, -1).mean(dim=1)  # (B, hidden)
        return h_pool

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, K, T, 12); task_idx: (B, K) or None → (B, n_classes)."""
        B, K, T, _ = x.shape
        h_l, h_r, h_s = self.tokenizer(x)  # each (B*K, T, d_model)
        p_l = self._stream_pool(self.tcn_l, h_l, B, K)
        p_r = self._stream_pool(self.tcn_r, h_r, B, K)
        p_s = self._stream_pool(self.tcn_s, h_s, B, K)
        t = self.task_emb(task_idx)  # (B, d_task) or None
        return self.head(p_l, p_r, p_s, t)


class Inception2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch // 4, kernel_size=1)
        self.branch3 = nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_ch, out_ch // 4, kernel_size=5, padding=2)
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(in_ch, out_ch // 4, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(torch.cat([self.branch1(x), self.branch3(x),
                                  self.branch5(x), self.branch_pool(x)], dim=1))


class TimesBlock(nn.Module):
    """FFT-period 2D Inception block (TimesNet, Wu IJCAI 2023, simplified)."""

    def __init__(self, hidden: int, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.inception = Inception2D(hidden, hidden)
        self.ff = nn.Conv2d(hidden, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, T, D)
        B, T, C = x.shape
        fft = torch.fft.rfft(x, dim=1)
        amp = torch.abs(fft).mean(dim=-1)
        amp[:, 0] = 0  # exclude DC
        top_k = min(self.top_k, amp.shape[1])
        _, idx = torch.topk(amp, top_k, dim=1)
        out_stack: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for i in range(top_k):
            f = idx[:, i].clamp(min=1)
            f_val = max(int(f[0].item()), 1)
            p = max(int(T // f_val), 1)
            new_T = p * f_val
            if new_T > T:
                x_padded = F.pad(x, (0, 0, 0, new_T - T))
            elif new_T < T:
                x_padded = x[:, :new_T, :]
            else:
                x_padded = x
            x_2d = x_padded.reshape(B, p, f_val, C).permute(0, 3, 1, 2)
            x_2d = self.inception(x_2d)
            x_2d = self.ff(x_2d)
            x_back = x_2d.permute(0, 2, 3, 1).reshape(B, new_T, C)
            if new_T < T:
                x_back = F.pad(x_back, (0, 0, 0, T - new_T))
            elif new_T > T:
                x_back = x_back[:, :T, :]
            out_stack.append(x_back)
            w_i = amp[torch.arange(B, device=x.device), idx[:, i]]
            weights.append(w_i)
        weights = torch.stack(weights, dim=1)
        weights = F.softmax(weights, dim=1).unsqueeze(-1)
        out_stack_t = torch.stack(out_stack, dim=1)
        return (out_stack_t * weights.unsqueeze(-1)).sum(dim=1)


class TimesNetModel(nn.Module):
    """TimesNet (Wu IJCAI 2023, simplified) — 3-stream + task conditioning."""

    def __init__(self, in_dim: int = 12, hidden: int = 64, n_blocks: int = 2,
                 top_k: int = 2, n_classes: int = 2, dropout: float = DEFAULT_DROPOUT,
                 n_tasks: int = DEFAULT_N_TASKS, d_model: int = DEFAULT_D_MODEL,
                 d_task: int = DEFAULT_D_TASK) -> None:
        super().__init__()
        self.tokenizer = ThreeStreamTokenizer(d_model=d_model, dropout=dropout)
        self.task_emb = TaskEmbedding(n_tasks=n_tasks, d_task=d_task)
        # 3 parallel TimesBlock stacks.
        self.blocks_l = nn.ModuleList([TimesBlock(hidden, top_k) for _ in range(n_blocks)])
        self.blocks_r = nn.ModuleList([TimesBlock(hidden, top_k) for _ in range(n_blocks)])
        self.blocks_s = nn.ModuleList([TimesBlock(hidden, top_k) for _ in range(n_blocks)])
        self.in_proj_l = nn.Linear(d_model, hidden)
        self.in_proj_r = nn.Linear(d_model, hidden)
        self.in_proj_s = nn.Linear(d_model, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = ThreeStreamHead(d_model=hidden, d_task=d_task, n_classes=n_classes, dropout=dropout)

    def _stream_forward(self, h: torch.Tensor, in_proj: nn.Linear,
                        blocks: nn.ModuleList) -> torch.Tensor:
        """Run in_proj → n_blocks TimesBlock with residual → norm → mean over T."""
        h = in_proj(h)
        for blk in blocks:
            h = blk(h) + h
        h = self.norm(h).mean(dim=1)
        return h

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, K, T, 12); task_idx: (B, K) or None → (B, n_classes)."""
        B, K, T, _ = x.shape
        h_l, h_r, h_s = self.tokenizer(x)  # each (B*K, T, d_model)
        p_l = self._stream_forward(h_l, self.in_proj_l, self.blocks_l).reshape(B, K, -1).mean(dim=1)
        p_r = self._stream_forward(h_r, self.in_proj_r, self.blocks_r).reshape(B, K, -1).mean(dim=1)
        p_s = self._stream_forward(h_s, self.in_proj_s, self.blocks_s).reshape(B, K, -1).mean(dim=1)
        t = self.task_emb(task_idx)
        return self.head(p_l, p_r, p_s, t)


class DSAttention(nn.Module):
    """Dual-statistic attention (NST, Liu NeurIPS 2022, OOM-safe chunked)."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        chunk_size = 1  # OOM-safe
        out = torch.zeros_like(x)
        for s in range(0, B, chunk_size):
            e = min(s + chunk_size, B)
            xc, tc, dc = x[s:e], tau[s:e], delta[s:e]
            Bc = e - s
            qkv = self.qkv(xc).reshape(Bc, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn * tc.unsqueeze(1) + dc.unsqueeze(1)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            outc = (attn @ v).transpose(1, 2).reshape(Bc, T, C)
            out[s:e] = outc
        return self.proj(out)


class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.backbone(stats)


class NSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = DSAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, tau, delta)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class NSTModel(nn.Module):
    """Non-stationary Transformer (Liu NeurIPS 2022) — 3-stream + task conditioning.

    Simplification: tau/delta are kept as constant ones/zeros (per-trial
    subject-level statistics, not learned from each stream's input). The
    3-stream tokenizer handles the per-stream differentiation.
    """

    def __init__(self, in_dim: int = 12, d_model: int = DEFAULT_D_MODEL, n_heads: int = 2, n_layers: int = 1,
                 d_ff: int = 128, n_classes: int = 2, dropout: float = 0.1, max_T: int = DEFAULT_T_LEN,
                 n_tasks: int = DEFAULT_N_TASKS, d_task: int = DEFAULT_D_TASK) -> None:
        super().__init__()
        if max_T is None:
            max_T = DEFAULT_T_LEN
        self.max_T = max_T
        self.tokenizer = ThreeStreamTokenizer(d_model=d_model, dropout=dropout)
        self.task_emb = TaskEmbedding(n_tasks=n_tasks, d_task=d_task)
        # Per-stream position embedding (shared across streams, separate tokens).
        self.pos_emb = nn.Embedding(max_T, d_model)
        # 3 parallel NST encoder stacks.
        self.layers_l = nn.ModuleList([NSTEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.layers_r = nn.ModuleList([NSTEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.layers_s = nn.ModuleList([NSTEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = ThreeStreamHead(d_model=d_model, d_task=d_task, n_classes=n_classes, dropout=dropout)

    def _stream_attend(self, h: torch.Tensor, layers: nn.ModuleList, T: int) -> torch.Tensor:
        """Add pos_emb → n_layers NST encoder → norm → mean over T.
        tau=ones, delta=zeros (no per-stream statistics, simplified v8).
        """
        pos = torch.arange(T, device=h.device).unsqueeze(0).expand(h.shape[0], T)
        h = h + self.pos_emb(pos)
        B, T_, _ = h.shape
        tau = torch.ones(B, 1, 1, device=h.device, dtype=h.dtype)
        delta = torch.zeros(B, 1, T_, device=h.device, dtype=h.dtype)
        for layer in layers:
            h = layer(h, tau, delta)
        h = self.norm(h).mean(dim=1)
        return h

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, K, T, 12); task_idx: (B, K) or None → (B, n_classes)."""
        B, K, T, _ = x.shape
        h_l, h_r, h_s = self.tokenizer(x)  # each (B*K, T, d_model)
        p_l = self._stream_attend(h_l, self.layers_l, T).reshape(B, K, -1).mean(dim=1)
        p_r = self._stream_attend(h_r, self.layers_r, T).reshape(B, K, -1).mean(dim=1)
        p_s = self._stream_attend(h_s, self.layers_s, T).reshape(B, K, -1).mean(dim=1)
        t = self.task_emb(task_idx)
        return self.head(p_l, p_r, p_s, t)


class CNNTransformerBlock(nn.Module):
    """1D conv front-end + Transformer encoder (E2Mo 简化) — 3-stream + task conditioning."""

    def __init__(self, d_model: int = DEFAULT_D_MODEL, n_heads: int = 2, n_layers: int = 1, in_dim: int = 12,
                 n_classes: int = 2, dropout: float = DEFAULT_DROPOUT, max_T: int = DEFAULT_T_LEN,
                 n_tasks: int = DEFAULT_N_TASKS, d_task: int = DEFAULT_D_TASK) -> None:
        super().__init__()
        if max_T is None:
            max_T = DEFAULT_T_LEN
        self.tokenizer = ThreeStreamTokenizer(d_model=d_model, dropout=dropout)
        self.task_emb = TaskEmbedding(n_tasks=n_tasks, d_task=d_task)
        # 3 separate conv1d front-ends (replaces the in_proj from raw concat).
        self.conv_l = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.conv_r = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.conv_s = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.pos_emb = nn.Embedding(max_T, d_model)
        # 3 separate transformer encoder stacks.
        def make_encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            return nn.TransformerEncoder(layer, num_layers=n_layers)
        self.enc_l = make_encoder()
        self.enc_r = make_encoder()
        self.enc_s = make_encoder()
        self.norm = nn.LayerNorm(d_model)
        self.head = ThreeStreamHead(d_model=d_model, d_task=d_task, n_classes=n_classes, dropout=dropout)

    def _stream_forward(self, h: torch.Tensor, conv: nn.Conv1d,
                        enc: nn.TransformerEncoder, T: int) -> torch.Tensor:
        """conv1d front-end → add pos_emb → transformer encoder → norm → mean over T."""
        # h: (N, T, d_model) → conv1d wants (N, d_model, T)
        h = conv(h.transpose(1, 2)).transpose(1, 2)  # (N, T, d_model)
        pos = torch.arange(T, device=h.device).unsqueeze(0).expand(h.shape[0], T)
        h = h + self.pos_emb(pos)
        h = enc(h)
        h = self.norm(h).mean(dim=1)
        return h

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, K, T, 12); task_idx: (B, K) or None → (B, n_classes)."""
        B, K, T, _ = x.shape
        h_l, h_r, h_s = self.tokenizer(x)
        p_l = self._stream_forward(h_l, self.conv_l, self.enc_l, T).reshape(B, K, -1).mean(dim=1)
        p_r = self._stream_forward(h_r, self.conv_r, self.enc_r, T).reshape(B, K, -1).mean(dim=1)
        p_s = self._stream_forward(h_s, self.conv_s, self.enc_s, T).reshape(B, K, -1).mean(dim=1)
        t = self.task_emb(task_idx)
        return self.head(p_l, p_r, p_s, t)


def make_model(arch: str, n_classes: int, t_len: int = DEFAULT_T_LEN, dropout: float = DEFAULT_DROPOUT,
               n_tasks: int = DEFAULT_N_TASKS, d_model: int = DEFAULT_D_MODEL,
               d_task: int = DEFAULT_D_TASK) -> nn.Module:
    """Dispatch a model by name (v8: 3-stream + task conditioning)."""
    if arch == "TCN":
        return TCNModel(n_classes=n_classes, dropout=dropout,
                        n_tasks=n_tasks, d_model=d_model, d_task=d_task)
    if arch == "TimesNet":
        return TimesNetModel(n_classes=n_classes, dropout=dropout,
                             n_tasks=n_tasks, d_model=d_model, d_task=d_task)
    if arch == "NST":
        return NSTModel(n_classes=n_classes, dropout=dropout, max_T=t_len,
                        n_tasks=n_tasks, d_model=d_model, d_task=d_task)
    if arch == "CNNTransformer":
        return CNNTransformerBlock(n_classes=n_classes, dropout=dropout, max_T=t_len,
                                   n_tasks=n_tasks, d_model=d_model, d_task=d_task)
    raise ValueError(f"Unknown arch: {arch}")


# ==== P1: SWA + logit adjustment + Focal Loss ====

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) for multi-class classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    For multi-class with logits:
      p = softmax(logits)
      p_t = p[range(N), y]  # true-class probability
      focal_weight = (1 - p_t)^γ
      loss = -α_t * focal_weight * log(p_t)

    α is optional per-class weight (typically inverse-freq class_weight).
    γ focuses learning on hard examples; γ=2 is the paper default.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha  # (n_classes,) per-class weight, on the loss device
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        idx = torch.arange(len(y), device=y.device)
        log_p_t = log_p[idx, y]
        p_t = p[idx, y]
        focal_weight = (1.0 - p_t).clamp(min=0.0).pow(self.gamma)
        loss = -focal_weight * log_p_t
        if self.alpha is not None:
            alpha_t = self.alpha[y]
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SWAAverager:
    """Simple stochastic weight averaging buffer (no BN update needed).

    Collects state_dict snapshots, averages on demand. Used in v8.3 to
    stabilize late-epoch weight oscillation and improve generalization.
    """

    def __init__(self, model: nn.Module) -> None:
        self.swa_state: dict[str, torch.Tensor] = {
            k: v.detach().clone().float().cpu()
            for k, v in model.state_dict().items()
        }
        self.n = 1

    def update(self, model: nn.Module) -> None:
        """Add current model weights to the running average."""
        cur = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
        for k in self.swa_state:
            self.swa_state[k] = self.swa_state[k] + cur[k]
        self.n += 1

    def apply_to(self, model: nn.Module) -> None:
        """Load SWA-averaged weights into model (cast back to model dtype/device)."""
        ref_state = model.state_dict()
        avg = {k: (self.swa_state[k] / self.n).to(ref_state[k].dtype).to(ref_state[k].device)
               for k in self.swa_state if k in ref_state}
        model.load_state_dict(avg)


def compute_logit_adjust_bias(train_labels: np.ndarray, n_classes: int = 2,
                              device: torch.device | None = None) -> torch.Tensor:
    """Compute log(π_pos / (1 - π_pos)) bias for binary class prior adjustment (Menon 2021).

    Returns a (n_classes,) tensor: [0, log(π_pos/(1-π_pos))] for binary, all zeros
    for multiclass (we don't apply the bias when n_classes > 2 to keep multiclass
    calibration simple).
    """
    if n_classes != 2:
        return torch.zeros(n_classes, device=device) if device is not None else torch.zeros(n_classes)
    # Fix: np.mean(list == 1) returns 0.0 because list==int is False. We must
    # convert to numpy array FIRST so the comparison broadcasts elementwise.
    arr = np.asarray(train_labels, dtype=np.int64) if not isinstance(train_labels, np.ndarray) else train_labels
    pi_pos = float(np.mean(arr == 1))
    pi_pos = min(max(pi_pos, 0.05), 0.95)  # clamp to avoid ±inf
    tau = math.log(pi_pos / (1.0 - pi_pos))
    bias = torch.zeros(n_classes)
    bias[1] = tau
    if device is not None:
        bias = bias.to(device)
    return bias


# ==== Training protocol ====
@dataclass
class DLConfig:
    arch: str
    n_classes: int
    t_len: int = DEFAULT_T_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float | None = None  # auto: 0.01 TCN / 0.05 others
    dropout: float = DEFAULT_DROPOUT
    seed: int = SEED
    n_tasks: int = DEFAULT_N_TASKS   # v8: saccade paradigm count
    d_model: int = DEFAULT_D_MODEL  # v8: stream token dim
    d_task: int = DEFAULT_D_TASK    # v8: task embedding dim
    # v8.3 P1: SWA + Focal + logit adjustment
    use_swa: bool = False
    swa_last_n: int = 5   # collect last N epoch weights into SWA
    use_focal: bool = False
    focal_gamma: float = 2.0
    use_logit_adjust: bool = False
    # v8.4 P2 detox ablation
    d_model: int = 64     # tokenizer/head dim
    dropout: float = 0.3  # model dropout
    d_task: int = 16      # task embedding dim (0 = no task conditioning)


def _wd_for(arch: str, override: float | None) -> float:
    if override is not None:
        return override
    return DEFAULT_WD_TCN if arch == "TCN" else DEFAULT_WD


def _safe_logits(model_out: torch.Tensor) -> torch.Tensor:
    """Replace ±inf in logits to avoid cross_entropy NaN."""
    return torch.where(torch.isfinite(model_out), model_out, torch.zeros_like(model_out))


def _eval_batches(model: nn.Module, batches: list[list[int]], dataset: TrialDataset,
                  device: torch.device, _batch_size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model in eval mode over subject-batches; return (y_true, probs, subj_idx).

    v8: model accepts (X, task_idx) where task_idx = (1, K) per subject.
    Backward compatible: if model doesn't accept task_idx, fall back to
    model(X) (so 3-stream v7b wrappers without task conditioning still work).
    """
    model.eval()
    ys_all, probs_all, subj_all = [], [], []
    with torch.no_grad():
        for batch in batches:
            xs, ys, tids = [], [], []
            for i in batch:
                x, y, t = dataset[i]
                xs.append(x)
                ys.append(int(y))
                tids.append(int(t))
            if not xs:
                continue
            # X: (K, T, D) → (1, K, T, D) for subject-level model forward.
            X = torch.stack(xs).unsqueeze(0).to(device, non_blocking=True)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            task_idx = torch.tensor(tids, dtype=torch.long, device=device).unsqueeze(0)  # (1, K)
            try:
                logits = _safe_logits(model(X, task_idx)).squeeze(0)  # (n_classes,)
            except TypeError:
                # Backward compat: model without task_idx support
                logits = _safe_logits(model(X)).squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().numpy()  # (n_classes,)
            # broadcast same prob to all K trials (model does subject-level prediction)
            probs_per_trial = np.tile(probs[None, :], (len(ys), 1))
            ys_all.append(y.cpu().numpy())
            probs_all.append(probs_per_trial)
            subj_all.append(np.array(batch))
    return np.concatenate(ys_all), np.concatenate(probs_all), np.concatenate(subj_all)


def _select_metric(metrics: dict[str, float], n_classes: int) -> float:
    """v8.2: Use val_auroc for val_pick (more sensitive to small improvements
    than val_bal_acc; doesn't get stuck at 0.5 when argmax fails).

    Multiclass still uses auroc_macro for parity with v4 behavior.
    """
    if n_classes == 2:
        return metrics.get("auroc", float("-inf"))
    return metrics.get("auroc_macro", float("-inf"))


def _best_threshold_for_bal_acc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Sweep threshold on val OOF to maximize balanced_accuracy.
    Post-hoc decision threshold optimization (threshold moving, Elkan 2001).
    Default 0.5 fallback if no threshold improves on default.
    """
    if probs.shape[1] != 2 or len(np.unique(y_true)) < 2:
        return 0.5
    p1 = probs[:, 1]
    default_ba = balanced_accuracy_score(y_true, (p1 >= 0.5).astype(int))
    best_thr, best_ba = 0.5, default_ba
    for thr in np.arange(0.05, 0.96, 0.05):
        y_pred = (p1 >= thr).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        ba = balanced_accuracy_score(y_true, y_pred)
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return float(best_thr)


def train_one_dl(
    model: nn.Module,
    train_split: SplitData,
    val_split: SplitData,
    test_split: SplitData,
    cfg: DLConfig,
    device: torch.device,
    shard_cache: ShardCache,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train one arch, val-pick best ckpt, return (val_metrics, test_metrics).

    v8.3 P1 extensions:
      - use_focal: Focal Loss (γ=focal_gamma, α=class_w) instead of CE
      - use_logit_adjust: add log(π_pos/(1-π_pos)) to class-1 logits during training
      - use_swa: collect last N epoch weights, average them, compare SWA vs best ckpt
        on val_auroc and keep the better one for final test.
    """
    n_classes = cfg.n_classes
    train_ds = TrialDataset(train_split, shard_cache, t_len=cfg.t_len)
    val_ds = TrialDataset(val_split, shard_cache, t_len=cfg.t_len)
    test_ds = TrialDataset(test_split, shard_cache, t_len=cfg.t_len)
    train_batches = make_subject_batches(train_split, seed=cfg.seed)
    val_batches = make_subject_batches(val_split, seed=cfg.seed + 100)
    test_batches = make_subject_batches(test_split, seed=cfg.seed + 200)

    class_weight = class_weights_from_labels(train_split.labels, n_classes).to(device)
    if cfg.use_focal:
        loss_fn = FocalLoss(alpha=class_weight, gamma=cfg.focal_gamma)
        LOGGER.info("  v8.3 P1.3: using FocalLoss(γ=%.1f, α=class_w)", cfg.focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optim_ = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=_wd_for(cfg.arch, cfg.weight_decay))

    # v8.3 P1.2: logit adjustment (train-time bias on class 1)
    if cfg.use_logit_adjust:
        logit_bias = compute_logit_adjust_bias(train_split.labels, n_classes, device=device)
        # Fix: log the clamped pi_pos (matches what was actually used in bias)
        arr = np.asarray(train_split.labels, dtype=np.int64) if not isinstance(train_split.labels, np.ndarray) else train_split.labels
        pi_pos_raw = float(np.mean(arr == 1))
        pi_pos_clamped = min(max(pi_pos_raw, 0.05), 0.95)
        LOGGER.info("  v8.3 P1.2: using logit_adjust bias=%s (train prior π_pos=%.3f, clamped=%.3f)",
                    logit_bias.tolist(), pi_pos_raw, pi_pos_clamped)
    else:
        logit_bias = None

    # v8.3 P1.1: SWA buffer
    swa = SWAAverager(model) if cfg.use_swa else None
    swa_start_epoch = max(1, cfg.max_epochs - cfg.swa_last_n + 1)
    if cfg.use_swa:
        LOGGER.info("  v8.3 P1.1: SWA enabled, collecting last %d epoch weights (from epoch %d)",
                    cfg.swa_last_n, swa_start_epoch)

    best_val = float("-inf")
    best_state: dict | None = None
    best_epoch = 0
    no_improve = 0
    # v8.2: 3-stream model needs longer warmup; 20 vs prior 10 prevents premature
    # early-stopping when val_auroc plateaus then resumes climbing.
    patience = 20
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        np.random.shuffle(train_batches)
        for batch in train_batches:
            xs, ys, tids = [], [], []
            for i in batch:
                x, y, t = train_ds[i]
                xs.append(x)
                ys.append(int(y))
                tids.append(int(t))
            if not xs:
                continue
            # X: (K, T, D) → add batch dim → (1, K, T, D) for subject-level model forward.
            X = torch.stack(xs).unsqueeze(0).to(device, non_blocking=True)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            task_idx = torch.tensor(tids, dtype=torch.long, device=device).unsqueeze(0)  # (1, K)
            optim_.zero_grad()
            try:
                logits = _safe_logits(model(X, task_idx))  # (1, n_classes)
            except TypeError:
                logits = _safe_logits(model(X))  # backward compat
            # v8.3 P1.2: add logit-adjust bias to the per-trial-expanded logits
            if logit_bias is not None:
                logits = logits + logit_bias.unsqueeze(0)  # (1, n_classes) + (n_classes,)
            # Repeat logits across K trials so the cross-entropy loss is well-defined
            # (the model has already done subject-level averaging internally).
            logits = logits.expand(y.shape[0], -1)  # (K, n_classes)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
        # val
        y_va, p_va, _ = _eval_batches(model, val_batches, val_ds, device, cfg.batch_size)
        va_metrics = _compute_metrics_arr(y_va, p_va, n_classes)
        score = _select_metric(va_metrics, n_classes)
        LOGGER.info("  epoch %d/%d  val_%s=%.4f  val_bal_acc=%.4f",
                    epoch, cfg.max_epochs,
                    "auroc" if n_classes == 2 else "auroc_macro",
                    score, va_metrics.get("balanced_accuracy", float("nan")))
        if score > best_val and score > 0.55:  # require above-trivial to count as improvement
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info("  early stop at epoch %d (no improvement in %d)", epoch, patience)
                break
        # v8.3 P1.1: append current weights to SWA buffer (only after warmup epochs)
        if swa is not None and epoch >= swa_start_epoch:
            swa.update(model)
    train_time = time.time() - t0
    LOGGER.info("  best_epoch=%d  best_val=%.4f  train_time=%.1fs", best_epoch, best_val, train_time)
    if best_state is not None:
        model.load_state_dict(best_state)

    # v8.3 P1.1: compare SWA-averaged model against best ckpt on val_auroc, keep the better
    swa_state_backup = None
    swa_val = float("-inf")
    use_swa_final = False
    if swa is not None and swa.n > 1:
        # Backup the best-ckpt weights so we can restore if SWA is worse
        swa_state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa.apply_to(model)
        y_va_swa, p_va_swa, _ = _eval_batches(model, val_batches, val_ds, device, cfg.batch_size)
        swa_metrics = _compute_metrics_arr(y_va_swa, p_va_swa, n_classes)
        swa_val = _select_metric(swa_metrics, n_classes)
        LOGGER.info("  SWA: n=%d  val_%s=%.4f  vs best=%.4f  → %s",
                    swa.n, "auroc" if n_classes == 2 else "auroc_macro",
                    swa_val, best_val,
                    "SWA" if swa_val > best_val else "best-ckpt")
        if swa_val > best_val:
            use_swa_final = True
        else:
            # Restore best-ckpt weights
            model.load_state_dict(swa_state_backup)
    y_va, p_va, _ = _eval_batches(model, val_batches, val_ds, device, cfg.batch_size)
    y_te, p_te, s_te = _eval_batches(model, test_batches, test_ds, device, cfg.batch_size)
    va = _compute_metrics_arr(y_va, p_va, n_classes)
    te = _compute_metrics_arr(y_te, p_te, n_classes)
    # Post-hoc threshold calibration (Phase 1): find best thr on val OOF, apply to test
    val_threshold = _best_threshold_for_bal_acc(y_va, p_va) if n_classes == 2 else 0.5
    te_tuned = _compute_metrics_arr_at_threshold(y_te, p_te, n_classes, val_threshold) if n_classes == 2 else te
    LOGGER.info("  Phase1 threshold: val_thr=%.2f  test_ba=%.4f (tuned=%.4f)  used_swa=%s",
                val_threshold, te["balanced_accuracy"],
                te_tuned.get("balanced_accuracy", te["balanced_accuracy"]), use_swa_final)
    return va, te, best_epoch, train_time, (y_te, p_te, s_te, y_va, p_va, val_threshold, te_tuned, use_swa_final)


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap CI for AUROC. Same algo as ml_baseline.bootstrap_auc_ci."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan")
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt, ys)))
        except Exception:
            continue
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def _compute_metrics_arr_at_threshold(y_true: np.ndarray, probs: np.ndarray, n_classes: int, threshold: float) -> dict[str, float]:
    """Same as _compute_metrics_arr but uses custom threshold instead of argmax.
    Used for post-hoc decision threshold optimization (Phase 1)."""
    out = _compute_metrics_arr(y_true, probs, n_classes)
    if n_classes != 2:
        return out
    from sklearn.metrics import recall_score
    y_pred = (probs[:, 1] >= threshold).astype(int)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["sensitivity"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["specificity"] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    out["auc_mr"] = float((out["sensitivity"] + out["specificity"]) / 2)
    out["threshold"] = float(threshold)
    return out


def _compute_metrics_arr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    """Same as ml_baseline.compute_metrics but takes numpy arrays directly. Also adds AUROC CI for binary."""
    y_pred = probs.argmax(axis=1)
    out: dict[str, float] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if n_classes == 2:
        try:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["auroc_ci_low"], out["auroc_ci_high"] = _bootstrap_auc_ci(y_true, probs[:, 1])
            out["auprc"] = float(average_precision_score(y_true, probs[:, 1]))
        except Exception:
            out["auroc"] = float("nan")
            out["auroc_ci_low"] = float("nan")
            out["auroc_ci_high"] = float("nan")
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0
        out["sensitivity"] = float(tp / max(tp + fn, 1))
        out["specificity"] = float(tn / max(tn + fp, 1))
        try:
            out["auc_mr"] = float((out["sensitivity"] + out["specificity"]) / 2)
        except Exception:
            out["auc_mr"] = float("nan")
    else:
        try:
            out["auroc_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            out["auroc_macro"] = float("nan")
        out["auroc"] = out["auroc_macro"]
    return out



CSV_FIELDS = (
    "arch", "n_classes", "best_epoch", "best_val_score", "train_time_sec",
    "val_threshold",  # NEW: Phase 1 post-hoc threshold
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa",
    "auroc", "auroc_macro", "auroc_ci_low", "auroc_ci_high",
    "auprc",
    "sensitivity", "specificity", "auc_mr",
    "balanced_accuracy_tuned",  # NEW: Phase 1 threshold-tuned
    "sensitivity_tuned",
    "specificity_tuned",
)


def run_dl_task(
    task: str,
    data_root: Path,
    out_dir: Path,
    archs: Sequence[str] = ("TCN", "TimesNet", "NST", "CNNTransformer"),
    t_len: int = DEFAULT_T_LEN,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
    use_swa: bool = False,
    swa_last_n: int = 5,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    use_logit_adjust: bool = False,
    seeds: list[int] | None = None,  # v8.3 P1.6: multi-seed ensemble
    d_task: int = 16,                 # v8.4 P2 detox ablation
    d_model: int = 64,
    dropout: float = 0.3,
) -> list[dict[str, Any]]:
    """Run all 4 DL baselines on a single task; write per-arch test csv + summary.

    v8.3 P1 extensions (all default False, backward compatible with v8.2):
      - use_swa: SWA over last N epoch weights
      - use_focal: Focal Loss with γ=focal_gamma
      - use_logit_adjust: train-time log-prior bias on class-1 logits
      - seeds: list of int seeds to run per arch (default [SEED]); when >1,
        probs are averaged across seeds (subject-level alignment by subj_idx).
    """
    from .data_loader import get_n_classes
    n_classes = get_n_classes(task)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seeds is None:
        seeds = [SEED]
    splits = build_split_data(task, data_root)
    shard_cache = ShardCache(data_root, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_csv = out_dir / f"test_final_dl_{task}.csv"
    rows: list[dict[str, Any]] = []
    for arch in archs:
        LOGGER.info("=== arch: %s  task: %s  seeds=%s ===", arch, task, seeds)
        # Per-arch multi-seed ensemble: collect probs from each seed, then average
        all_seed_preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (y, probs, subj)
        all_seed_va_preds: list[tuple[np.ndarray, np.ndarray]] = []  # (y_val, probs_val)
        seed_metrics: list[dict[str, float]] = []
        seed_best_epochs: list[int] = []
        seed_train_times: list[float] = []
        seed_va_thresholds: list[float] = []
        last_seed_va_threshold = 0.5
        last_seed_use_swa = False
        last_seed_best_epoch = 0
        last_seed_train_time = 0.0
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = DLConfig(arch=arch, n_classes=n_classes, t_len=t_len,
                           max_epochs=max_epochs, batch_size=batch_size,
                           seed=seed,
                           use_swa=use_swa, swa_last_n=swa_last_n,
                           use_focal=use_focal, focal_gamma=focal_gamma,
                           use_logit_adjust=use_logit_adjust,
                           d_task=d_task, d_model=d_model, dropout=dropout)
            model = make_model(arch, n_classes, t_len=t_len, dropout=cfg.dropout,
                               n_tasks=cfg.n_tasks, d_model=cfg.d_model, d_task=cfg.d_task).to(device)
            LOGGER.info("  v8.4 P2 ablation: arch=%s d_model=%d d_task=%d dropout=%.2f focal_γ=%.2f swa=%s",
                        arch, cfg.d_model, cfg.d_task, cfg.dropout, cfg.focal_gamma, use_swa)
            try:
                val_metrics, test_metrics, best_epoch, train_time, test_preds = train_one_dl(
                    model, splits["train"], splits["validation"], splits["test"], cfg, device, shard_cache
                )
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("arch %s seed %d FAIL: %s", arch, seed, e)
                continue
            if len(test_preds) == 8:
                y_te, p_te, s_te, y_va, p_va, val_threshold, te_tuned, use_swa_final = test_preds
            else:
                y_te, p_te, s_te, y_va, p_va, val_threshold, te_tuned = test_preds
                use_swa_final = False
            all_seed_preds.append((y_te, p_te, s_te))
            all_seed_va_preds.append((y_va, p_va))
            seed_metrics.append(test_metrics)
            seed_best_epochs.append(best_epoch)
            seed_train_times.append(train_time)
            seed_va_thresholds.append(val_threshold)
            last_seed_va_threshold = val_threshold
            last_seed_use_swa = use_swa_final
            last_seed_best_epoch = best_epoch
            last_seed_train_time = train_time
            LOGGER.info("  seed %d done: best_val=%.4f test_auroc=%.4f thr=%.2f swa=%s",
                        seed, _select_metric(val_metrics, n_classes),
                        test_metrics.get("auroc", float("nan")), val_threshold, use_swa_final)

            # Save per-seed test/val predictions (1 npz per seed) so downstream
            # tools (see scripts/aggregate_per_seed.py) can recompute per-seed
            # metrics (auroc, auprc, bal-acc, f1, kappa) with mean ± std across
            # seeds, instead of relying on the ensemble-only baseline_summary.csv.
            per_seed_npz = out_dir / f"per_seed_preds_dl_{arch}_{task}_seed{seed}.npz"
            np.savez(
                per_seed_npz,
                y_true=y_te,
                probs=p_te,
                subj_idx=s_te,
                y_val=y_va,
                probs_val=p_va,
                val_threshold=np.asarray(val_threshold, dtype=np.float64),
                use_swa=np.asarray(use_swa_final),
                seed=np.asarray(seed, dtype=np.int64),
                n_classes=np.asarray(n_classes, dtype=np.int64),
            )
            LOGGER.info("  Saved per-seed preds: %s", per_seed_npz)

        if not all_seed_preds:
            LOGGER.warning("arch %s: all seeds failed; skip", arch)
            continue

        # Average probs across seeds (subject-level alignment by subj_idx)
        y_ref, _, s_ref = all_seed_preds[0]
        # Stack probs aligned by subj_idx (use y_ref order, which is the same for all seeds
        # because seed only changes model init + batch shuffle, not split subj order)
        probs_aligned = np.stack([p for (_, p, _) in all_seed_preds], axis=0)  # (n_seeds, n_subj, n_classes)
        p_avg = probs_aligned.mean(axis=0)  # (n_subj, n_classes)
        # Same for val
        y_va_ref, p_va_ref = all_seed_va_preds[0]
        probs_va_aligned = np.stack([p for (_, p) in all_seed_va_preds], axis=0)
        p_va_avg = probs_va_aligned.mean(axis=0)

        # Recompute final metrics from averaged probs
        va = _compute_metrics_arr(y_va_ref, p_va_avg, n_classes)
        te = _compute_metrics_arr(y_ref, p_avg, n_classes)
        val_threshold = _best_threshold_for_bal_acc(y_va_ref, p_va_avg) if n_classes == 2 else 0.5
        te_tuned = _compute_metrics_arr_at_threshold(y_ref, p_avg, n_classes, val_threshold) if n_classes == 2 else te
        LOGGER.info("  arch %s ENSEMBLE(%d seeds): val_auroc=%.4f test_auroc=%.4f test_ba=%.4f (tuned=%.4f thr=%.2f)",
                    arch, len(all_seed_preds), _select_metric(va, n_classes),
                    te.get("auroc", float("nan")), te["balanced_accuracy"],
                    te_tuned.get("balanced_accuracy", te["balanced_accuracy"]), val_threshold)

        # Save per-subj averaged probs
        preds_npz = out_dir / f"test_preds_dl_{arch}_{task}.npz"
        np.savez(preds_npz, y_true=y_ref, probs=p_avg, subj_idx=s_ref,
                 y_val=y_va_ref, probs_val=p_va_avg, val_threshold=val_threshold,
                 use_swa=last_seed_use_swa, n_seeds=len(all_seed_preds))
        LOGGER.info("Saved per-subj preds: %s (val_thr=%.2f, n_seeds=%d)", preds_npz, val_threshold, len(all_seed_preds))
        # Save best ckpt from last seed (for re-eval if needed)
        ckpt_path = out_dir / f"best_ckpt_dl_{arch}_{task}.pt"
        torch.save({"arch": arch, "state_dict": model.state_dict(), "n_classes": n_classes,
                    "best_epoch": last_seed_best_epoch, "t_len": cfg.t_len,
                    "use_swa": last_seed_use_swa, "n_seeds": len(all_seed_preds)}, ckpt_path)
        LOGGER.info("Saved best ckpt: %s", ckpt_path)
        row = {
            "arch": arch,
            "n_classes": n_classes,
            "best_epoch": last_seed_best_epoch,
            "best_val_score": _select_metric(va, n_classes),
            "train_time_sec": last_seed_train_time,
            "val_threshold": float(te_tuned.get("threshold", 0.5)) if n_classes == 2 else 0.5,
            "balanced_accuracy_tuned": float(te_tuned.get("balanced_accuracy", float("nan"))) if n_classes == 2 else float("nan"),
            "sensitivity_tuned": float(te_tuned.get("sensitivity", float("nan"))) if n_classes == 2 else float("nan"),
            "specificity_tuned": float(te_tuned.get("specificity", float("nan"))) if n_classes == 2 else float("nan"),
            "used_swa": bool(last_seed_use_swa),
            "n_seeds": len(all_seed_preds),
            **{k: te.get(k, float("nan")) for k in CSV_FIELDS
               if k not in {"arch", "n_classes", "best_epoch", "best_val_score", "train_time_sec",
                            "val_threshold", "balanced_accuracy_tuned", "sensitivity_tuned", "specificity_tuned"}},
        }
        rows.append(row)
        LOGGER.info("  %s: best_val=%.4f  test_auroc=%.4f  swa=%s  n_seeds=%d", arch, row["best_val_score"],
                    te.get("auroc", te.get("auroc_macro", float("nan"))),
                    last_seed_use_swa, len(all_seed_preds))
    with test_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            LOGGER.info("DEBUG3 row keys=%s val_threshold=%.4f ba_tuned=%.4f", list(r.keys())[:8], r.get("val_threshold", -999), r.get("balanced_accuracy_tuned", -999))
            w.writerow({k: r.get(k, float("nan")) for k in CSV_FIELDS})
    LOGGER.info("Wrote %s", test_csv)
    return rows
