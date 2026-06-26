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

LOGGER = logging.getLogger(__name__)


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
    """Temporal Convolutional Network (Bai et al. 2018)."""

    def __init__(self, in_dim: int = 10, hidden: int = 64, n_blocks: int = 2,
                 n_classes: int = 2, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = in_dim
        for i in range(n_blocks):
            layers.append(TCNBlock(in_ch, hidden, kernel_size=3, dilation=2 ** i, dropout=dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, T, D) → (B*K, T, D) → conv1d needs (B*K, D, T)
        B, K, T, D = x.shape
        h = x.view(B * K, T, D).transpose(1, 2)
        h = self.tcn(h)
        h = h.mean(dim=-1)  # avg over time
        h = h.view(B, K, -1).mean(dim=1)
        return self.head(h)


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
    def __init__(self, in_dim: int = 10, hidden: int = 96, n_blocks: int = 2,
                 top_k: int = 2, n_classes: int = 2, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([TimesBlock(hidden, top_k) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, T, D = x.shape
        h = self.in_proj(x.view(B * K, T, D))
        for blk in self.blocks:
            h = blk(h) + h
        h = self.norm(h).mean(dim=1)
        h = h.view(B, K, -1).mean(dim=1)
        return self.head(h)


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
    """Non-stationary Transformer (Liu NeurIPS 2022)."""

    def __init__(self, in_dim: int = 10, d_model: int = 64, n_heads: int = 2, n_layers: int = 1,
                 d_ff: int = 128, n_classes: int = 2, dropout: float = 0.1, max_T: int = DEFAULT_T_LEN) -> None:
        super().__init__()
        if max_T is None:
            max_T = DEFAULT_T_LEN
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.tau_learner = Projector(in_dim * 2, [d_model, d_model], 1)
        self.delta_learner = Projector(in_dim * 2, [d_model, d_model], max_T)
        self.layers = nn.ModuleList([NSTEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, T, D = x.shape
        x_raw = x.view(B * K, T, D)
        mu = x_raw.mean(dim=1, keepdim=True)
        sigma = x_raw.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x_raw - mu) / sigma
        stats = torch.cat([mu.squeeze(1), sigma.squeeze(1)], dim=-1)
        tau = F.softplus(self.tau_learner(stats).unsqueeze(1))
        delta = self.delta_learner(stats)[:, :T].unsqueeze(1)
        h = self.in_proj(x_norm)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B * K, T)
        h = h + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h, tau, delta)
        h = self.norm(h).mean(dim=1)
        h = h.view(B, K, -1).mean(dim=1)
        return self.head(h)


class CNNTransformerBlock(nn.Module):
    """1D conv front-end + Transformer encoder (E2Mo 简化)."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, n_layers: int = 1, in_dim: int = 10,
                 n_classes: int = 2, dropout: float = DEFAULT_DROPOUT, max_T: int = DEFAULT_T_LEN) -> None:
        super().__init__()
        if max_T is None:
            max_T = DEFAULT_T_LEN
        self.in_proj = nn.Conv1d(in_dim, d_model, kernel_size=7, padding=3)
        self.pos_emb = nn.Embedding(max_T, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, T, D = x.shape
        h = x.view(B * K, T, D).transpose(1, 2)  # (B*K, D, T) for Conv1d
        h = self.in_proj(h).transpose(1, 2)     # (B*K, T, d_model)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B * K, T)
        h = h + self.pos_emb(pos)
        h = self.encoder(h)
        h = self.norm(h).mean(dim=1)
        h = h.view(B, K, -1).mean(dim=1)
        return self.head(h)


def make_model(arch: str, n_classes: int, t_len: int = DEFAULT_T_LEN, dropout: float = DEFAULT_DROPOUT) -> nn.Module:
    """Dispatch a model by name. `t_len` is used by NST/CNNTransformer pos_emb max."""
    if arch == "TCN":
        return TCNModel(n_classes=n_classes, dropout=dropout)
    if arch == "TimesNet":
        return TimesNetModel(n_classes=n_classes, dropout=dropout)
    if arch == "NST":
        return NSTModel(n_classes=n_classes, dropout=dropout, max_T=t_len)
    if arch == "CNNTransformer":
        return CNNTransformerBlock(n_classes=n_classes, dropout=dropout, max_T=t_len)
    raise ValueError(f"Unknown arch: {arch}")


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

    Each entry in `batches` is one subject's full list of trial indices. The
    model is called once per subject with shape (1, K, T, D) and returns logits
    of shape (1, n_classes) which we squeeze to (n_classes,) and turn into
    per-trial probability vectors by broadcasting. Subject-level prediction
    is therefore the per-trial argmax (the model averages over K trials
    internally before the final head).
    """
    model.eval()
    ys_all, probs_all, subj_all = [], [], []
    with torch.no_grad():
        for batch in batches:
            xs, ys = [], []
            for i in batch:
                x, y = dataset[i]
                xs.append(x)
                ys.append(int(y))
            if not xs:
                continue
            # X: (K, T, D) → (1, K, T, D) for subject-level model forward.
            X = torch.stack(xs).unsqueeze(0).to(device, non_blocking=True)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = _safe_logits(model(X)).squeeze(0)  # (n_classes,)
            probs = F.softmax(logits, dim=-1).cpu().numpy()  # (n_classes,)
            # broadcast same prob to all K trials (model does subject-level prediction)
            probs_per_trial = np.tile(probs[None, :], (len(ys), 1))
            ys_all.append(y.cpu().numpy())
            probs_all.append(probs_per_trial)
            subj_all.append(np.array(batch))
    return np.concatenate(ys_all), np.concatenate(probs_all), np.concatenate(subj_all)


def _select_metric(metrics: dict[str, float], n_classes: int) -> float:
    """Primary metric for val pick: AUROC (binary) / AUROC-macro (multiclass)."""
    if n_classes == 2:
        return metrics.get("auroc", float("-inf"))
    return metrics.get("auroc_macro", float("-inf"))


def train_one_dl(
    model: nn.Module,
    train_split: SplitData,
    val_split: SplitData,
    test_split: SplitData,
    cfg: DLConfig,
    device: torch.device,
    shard_cache: ShardCache,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train one arch, val-pick best ckpt, return (val_metrics, test_metrics)."""
    n_classes = cfg.n_classes
    train_ds = TrialDataset(train_split, shard_cache, t_len=cfg.t_len)
    val_ds = TrialDataset(val_split, shard_cache, t_len=cfg.t_len)
    test_ds = TrialDataset(test_split, shard_cache, t_len=cfg.t_len)
    train_batches = make_subject_batches(train_split, seed=cfg.seed)
    val_batches = make_subject_batches(val_split, seed=cfg.seed + 100)
    test_batches = make_subject_batches(test_split, seed=cfg.seed + 200)

    class_weight = class_weights_from_labels(train_split.labels, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optim_ = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=_wd_for(cfg.arch, cfg.weight_decay))

    best_val = float("-inf")
    best_state: dict | None = None
    best_epoch = 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        np.random.shuffle(train_batches)
        for batch in train_batches:
            xs, ys = [], []
            for i in batch:
                x, y = train_ds[i]
                xs.append(x)
                ys.append(int(y))
            if not xs:
                continue
            # X: (K, T, D) → add batch dim → (1, K, T, D) for subject-level model forward.
            X = torch.stack(xs).unsqueeze(0).to(device, non_blocking=True)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            optim_.zero_grad()
            logits = _safe_logits(model(X))  # (1, n_classes)
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
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
    train_time = time.time() - t0
    LOGGER.info("  best_epoch=%d  best_val=%.4f  train_time=%.1fs", best_epoch, best_val, train_time)
    if best_state is not None:
        model.load_state_dict(best_state)
    y_va, p_va, _ = _eval_batches(model, val_batches, val_ds, device, cfg.batch_size)
    y_te, p_te, _ = _eval_batches(model, test_batches, test_ds, device, cfg.batch_size)
    return _compute_metrics_arr(y_va, p_va, n_classes), _compute_metrics_arr(y_te, p_te, n_classes)


def _compute_metrics_arr(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    """Same as ml_baseline.compute_metrics but takes numpy arrays directly."""
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
        except Exception:
            out["auroc"] = float("nan")
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
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa",
    "auroc", "auroc_macro", "sensitivity", "specificity", "auc_mr",
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
) -> list[dict[str, Any]]:
    """Run all 4 DL baselines on a single task; write per-arch test csv + summary."""
    from .data_loader import get_n_classes
    n_classes = get_n_classes(task)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = build_split_data(task, data_root)
    shard_cache = ShardCache(data_root, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_csv = out_dir / f"test_final_dl_{task}.csv"
    rows: list[dict[str, Any]] = []
    for arch in archs:
        LOGGER.info("=== arch: %s  task: %s ===", arch, task)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        cfg = DLConfig(arch=arch, n_classes=n_classes, t_len=t_len,
                       max_epochs=max_epochs, batch_size=batch_size)
        model = make_model(arch, n_classes, t_len=t_len).to(device)
        try:
            val_metrics, test_metrics = train_one_dl(
                model, splits["train"], splits["validation"], splits["test"], cfg, device, shard_cache
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("arch %s FAIL: %s", arch, e)
            continue
        row = {
            "arch": arch,
            "n_classes": n_classes,
            "best_epoch": 0,
            "best_val_score": _select_metric(val_metrics, n_classes),
            "train_time_sec": 0.0,
            **{k: test_metrics.get(k, float("nan")) for k in CSV_FIELDS
               if k not in {"arch", "n_classes", "best_epoch", "best_val_score", "train_time_sec"}},
        }
        rows.append(row)
        LOGGER.info("  %s: best_val=%.4f  test_auroc=%.4f", arch, row["best_val_score"],
                    test_metrics.get("auroc", test_metrics.get("auroc_macro", float("nan"))))
    with test_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in CSV_FIELDS})
    LOGGER.info("Wrote %s", test_csv)
    return rows
