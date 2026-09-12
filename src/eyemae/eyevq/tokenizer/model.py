"""
EyeVQ Tokenizer Model: ViT Encoder + VQ + ViT Decoder.

Architecture:
  PatchEmbedding:  Conv1d stack → [B, 256]  (S/L/R 同结构,不同实例)
  ViTEncoder:      S/L/R + enc_cls → 12层 Transformer (d=512, FFN=1536)
  VQProjection:    Linear(512→128)  → VQ Quantizer (L/R only)
  ViTDecoder:      dec_cls + z_q_LR → 3层 Transformer → 手工特征 + eye重建
  PatchReconHead:  ConvTranspose → [4,20]

Transformer blocks aligned with EyeMAE: RMSNorm + SwiGLU + MultiheadAttention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from eyemae.eyevq.tokenizer.fsq import FSQ
from eyemae.eyevq.tokenizer.vqvae import VQVAEQuantizer


# ──────────────────────────────────────────────
# EyeMAE-aligned Transformer Components
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2)
        self.out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, gate = self.w12(x).chunk(2, dim=-1)
        return self.out(self.dropout(a * F.silu(gate)))


class TransformerBlock(nn.Module):
    """EyeMAE-style pre-norm block: RMSNorm + FlashAttn + RMSNorm + SwiGLU."""
    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-norm block with optional structural attention mask.

        Args:
            x: [B, N, D]
            key_padding_mask: [B, N] bool — True = padding (ignored by attention).
            attn_mask: [N, N] bool — True = allowed to attend.
                       If provided, it is combined with key_padding_mask.
        """
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        # Combine structural mask + key padding mask into a single float mask.
        # PyTorch SDPA expects: add_mask (True → keep, False → -inf) as float/bool.
        mask = None
        if attn_mask is not None:
            # [N, N] bool → broadcast to [B, 1, N, N]
            mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        if key_padding_mask is not None:
            kpm = key_padding_mask  # [B, N] True=padding
            # Expand to [B, 1, 1, N] (key dim), invert: False=padding→blocked
            kpm_b = (~kpm).unsqueeze(1).unsqueeze(2)  # [B,1,1,N] True=valid key
            if mask is None:
                mask = kpm_b
            else:
                mask = mask & kpm_b  # [B,1,N,N] & [B,1,1,N] → broadcast

        # Flash attention via PyTorch SDPA (uses flash-attn on A100 automatically)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0 if not self.training else self.drop1.p,
            is_causal=False,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.drop1(self.out_proj(attn_out))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class AxialCrossTransformerBlock(nn.Module):
    """Two-axis S/L/R block with two attentions and one feed-forward network.

    The input layout is ``[CLS, S0, L0, R0, S1, L1, R1, ...]``.  Attention is
    applied in two stages:

    1. temporal self-attention on S alone and on ``[CLS, L, R]`` alone;
    2. local cross-channel attention within each aligned ``(S_t, L_t, R_t)``.

    The two temporal groups share the second attention's parameters.  This
    keeps the definition at exactly two attention modules per layer instead of
    accidentally doubling both attention and FFN capacity.
    """

    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.local_norm = RMSNorm(dim)
        self.local_qkv = nn.Linear(dim, dim * 3)
        self.local_out = nn.Linear(dim, dim)
        self.temporal_norm = RMSNorm(dim)
        self.temporal_qkv = nn.Linear(dim, dim * 3)
        self.temporal_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def _attention(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = qkv_proj(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        valid_keys = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid_keys,
            dropout_p=0.0 if not self.training else self.attn_drop.p,
            is_causal=False,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, dim)
        return out_proj(attended)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        n_time: int,
    ) -> torch.Tensor:
        batch, total_len, dim = x.shape
        expected_len = 1 + 3 * n_time
        if total_len != expected_len or key_padding_mask.shape != (batch, expected_len):
            raise ValueError(
                "axial-cross attention requires [CLS,S,L,R] interleaving: "
                f"x={tuple(x.shape)}, mask={tuple(key_padding_mask.shape)}, "
                f"n_time={n_time}"
            )

        # Axis 1: one shared temporal attention, evaluated independently for
        # the S stream and the [CLS,L,R] stream. No edge crosses the streams.
        cls = x[:, :1]
        body = x[:, 1:].reshape(batch, n_time, 3, dim)
        body_mask = key_padding_mask[:, 1:].reshape(batch, n_time, 3)
        temporal_body = self.temporal_norm(body)
        s_stream = temporal_body[:, :, 0]
        s_mask = body_mask[:, :, 0]
        s_update = self._attention(
            s_stream, s_mask, self.temporal_qkv, self.temporal_out
        )

        lr_stream = temporal_body[:, :, 1:].reshape(batch, n_time * 2, dim)
        lr_mask = body_mask[:, :, 1:].reshape(batch, n_time * 2)
        cls_lr = torch.cat([self.temporal_norm(cls), lr_stream], dim=1)
        cls_lr_mask = torch.cat([key_padding_mask[:, :1], lr_mask], dim=1)
        cls_lr_update = self._attention(
            cls_lr, cls_lr_mask, self.temporal_qkv, self.temporal_out
        )

        cls = cls + self.attn_drop(cls_lr_update[:, :1])
        s_body = body[:, :, 0] + self.attn_drop(s_update)
        lr_body = body[:, :, 1:] + self.attn_drop(
            cls_lr_update[:, 1:].reshape(batch, n_time, 2, dim)
        )
        body = torch.cat([s_body.unsqueeze(2), lr_body], dim=2)

        # Axis 2: aligned local S/L/R attention after each stream has first
        # built its temporal context. CLS bypasses this cross-channel stage.
        local = self.local_norm(body).reshape(batch * n_time, 3, dim)
        local_mask = body_mask.reshape(batch * n_time, 3)
        local = self._attention(
            local, local_mask, self.local_qkv, self.local_out
        ).reshape(batch, n_time, 3, dim)
        body = body + self.attn_drop(local)

        x = torch.cat([cls, body.reshape(batch, n_time * 3, dim)], dim=1)
        return x + self.ffn_drop(self.ffn(self.ffn_norm(x)))


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

TYPE_CLS = 0
TYPE_S   = 1
TYPE_L   = 2
TYPE_R   = 3

Q_BINS = 20  # quality embedding bins


def build_stim_isolated_attn_mask(
    total_len: int,
    n_time: int,
    device,
    with_cls: bool = True,
    stim_attend_cls: bool = True,
) -> torch.Tensor:
    """Structural mask for stimulus-isolated joint self-attention.

    Stimulus queries may read stimulus keys, never L/R keys.  ``stim_attend_cls``
    controls the remaining directed edge independently: disabling it prevents
    information accumulated by CLS from flowing back into stimulus tokens,
    while CLS and eye queries can still read the complete valid sequence.

    序列布局: [CLS(0), S₀(1), L₀(2), R₀(3), S₁(4), ...]  (with_cls=True)
            或 [S₀(0), L₀(1), R₀(2), S₁(3), ...]          (with_cls=False)

    Returns: [total_len, total_len] bool — True = allowed to attend.
    """
    mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)
    for t in range(n_time):
        if with_cls:
            s_pos = 1 + 3 * t
        else:
            s_pos = 3 * t
        # Stim row: stimulus keys, plus optional CLS key, but never L/R keys.
        row = torch.zeros(total_len, dtype=torch.bool, device=device)
        if with_cls and stim_attend_cls:
            row[0] = True  # CLS
        for t2 in range(n_time):
            s2 = 1 + 3 * t2 if with_cls else 3 * t2
            row[s2] = True
        mask[s_pos] = row
    return mask


# ──────────────────────────────────────────────
# Patch Embedding (S/L/R 同结构,不同实例)
# ──────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Raw patch [4, patch_samples] → d_model.

    3层 Conv1d, stride=2 下采样 patch_samples → patch_samples//2 → //4,
    flatten → Linear → LayerNorm.  S 和 Eye 各用独立实例, L/R 之间共享 EyePatchEmbed.
    (patch_samples=20 → 20→10→5; 100 → 100→50→25)
    """

    def __init__(self, d_model: int = 256, patch_samples: int = 20):
        super().__init__()
        self.patch_samples = patch_samples
        self.conv1 = nn.Conv1d(4, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)   # L→L/2
        self.conv3 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)  # L/2→L/4
        self.flatten_dim = 128 * (patch_samples // 4)
        self.fc = nn.Linear(self.flatten_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, patch_samples]  raw patch
        Returns:
            [B, d_model]
        """
        x = F.gelu(self.conv1(x))                     # [B, 64, L]
        x = F.gelu(self.conv2(x))                     # [B, 128, L/2]
        x = F.gelu(self.conv3(x))                     # [B, 128, L/4]
        x = x.reshape(x.shape[0], -1)                 # [B, 128*(L/4)]
        x = self.fc(x)                                # [B, d_model]
        x = self.norm(x)
        return x


# ──────────────────────────────────────────────
# ViT Encoder
# ──────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """ViT Encoder: S/L/R patches + enc_cls → 12层 Transformer.

    enc_cls 仅 encoder 内部使用,不传入 decoder。
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        dim_ff: int = 768,
        dropout: float = 0.0,
        max_patches: int = 512,
        patch_samples: int = 20,
        stim_isolated_attn: bool = False,
        stim_attend_cls: bool = True,
        attention_layout: str = "joint",
        include_stim: bool = True,   # False → L/R-only encoder (分离架构)
        with_cls: bool = True,       # False → 无 enc_cls (分离架构 L/R encoder)
        min_nonmissing_frac: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_samples = patch_samples
        self.stim_isolated_attn = stim_isolated_attn
        self.stim_attend_cls = bool(stim_attend_cls)
        if attention_layout not in {"joint", "axial_cross"}:
            raise ValueError(
                "attention_layout must be 'joint' or 'axial_cross', "
                f"got {attention_layout!r}"
            )
        if attention_layout == "axial_cross" and (not include_stim or not with_cls):
            raise ValueError("axial_cross attention requires S and CLS tokens")
        self.attention_layout = attention_layout
        self.include_stim = include_stim
        self.with_cls = with_cls
        self.min_nonmissing_frac = float(min_nonmissing_frac)

        # Patch embedding — 同结构,不同实例 (include_stim=False 时无 S)
        self.stim_patch_embed = PatchEmbedding(d_model, patch_samples) if include_stim else None
        self.eye_patch_embed = PatchEmbedding(d_model, patch_samples)   # L/R 共享

        # enc_cls token (仅 encoder 内部使用; with_cls=False 时不创建)
        if with_cls:
            self.enc_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.enc_cls_token, std=0.02)
        else:
            self.enc_cls_token = None

        # Type embedding: CLS=0, S=1, L=2, R=3
        self.type_embed = nn.Embedding(4, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)

        # Position embedding (按时间步共享: S/L/R 同 time step 用相同 pos)
        self.pos_embed = nn.Embedding(max_patches, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # Quality embedding (仅 L/R)
        self.quality_embed = nn.Embedding(Q_BINS, d_model)
        nn.init.normal_(self.quality_embed.weight, std=0.02)

        # Pre-norm Transformer blocks (EyeMAE-aligned)
        block_cls = (
            AxialCrossTransformerBlock
            if attention_layout == "axial_cross"
            else TransformerBlock
        )
        self.blocks = nn.ModuleList([
            block_cls(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

        # 缓存 stim-isolated 结构 mask (用 max_patches 预构建, forward 只切片到实际 N)
        self._stim_mask_base: torch.Tensor | None = None
        if stim_isolated_attn:
            total = 1 + 3 * max_patches
            self._stim_mask_base = build_stim_isolated_attn_mask(
                total,
                max_patches,
                "cpu",
                with_cls=True,
                stim_attend_cls=self.stim_attend_cls,
            )

    def _get_stim_mask(self, n_time: int, device) -> torch.Tensor | None:
        if self._stim_mask_base is None:
            return None
        total = 1 + 3 * n_time
        return self._stim_mask_base[:total, :total].to(device)

    def forward(
        self,
        stim: torch.Tensor | None,       # [B, N, 4, 20] (include_stim=False 时传 None)
        eye_l: torch.Tensor,             # [B, N, 4, 20]
        eye_r: torch.Tensor,             # [B, N, 4, 20]
        eye_nonmissing_frac: torch.Tensor,  # [B, N, 2]
        pad_mask: torch.Tensor,          # [B, N]
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            enc_cls:  [B, 1, d_model]
            s_hidden: [B, N, d_model]
            l_hidden: [B, N, d_model]
            r_hidden: [B, N, d_model]
        """
        B, N = eye_l.shape[:2]           # eye_l 始终存在 (stim 可能为 None)
        device = eye_l.device

        # ── 1. Patch embedding ──
        ps = self.patch_samples
        # S (可选 — include_stim=False 时跳过, 只编 L/R)
        if self.include_stim and stim is not None:
            s_flat = stim.reshape(B * N, 4, ps)
            s_emb = self.stim_patch_embed(s_flat).reshape(B, N, self.d_model)  # [B, N, d]
        else:
            s_emb = None
        # L
        l_flat = eye_l.reshape(B * N, 4, ps)
        l_emb = self.eye_patch_embed(l_flat).reshape(B, N, self.d_model)   # [B, N, d]
        # R
        r_flat = eye_r.reshape(B * N, 4, ps)
        r_emb = self.eye_patch_embed(r_flat).reshape(B, N, self.d_model)   # [B, N, d]

        # ── 2. Add type + position embeddings ──
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)     # [B, N]
        pos_emb = self.pos_embed(pos)                                        # [B, N, d]

        if s_emb is not None:
            s_emb = s_emb + self.type_embed.weight[TYPE_S] + pos_emb        # [B, N, d]
        l_emb = l_emb + self.type_embed.weight[TYPE_L] + pos_emb
        r_emb = r_emb + self.type_embed.weight[TYPE_R] + pos_emb

        # ── 3. Add quality embedding (L/R only) ──
        q_l = (eye_nonmissing_frac[:, :, 0] * Q_BINS).long().clamp(0, Q_BINS - 1)
        q_r = (eye_nonmissing_frac[:, :, 1] * Q_BINS).long().clamp(0, Q_BINS - 1)
        l_emb = l_emb + self.quality_embed(q_l)
        r_emb = r_emb + self.quality_embed(q_r)

        # ── 4. Build sequence: [CLS?, S₀, L₀, R₀, ...] 或 [CLS?, L₀, R₀, ...] ──
        if s_emb is not None:
            # Interleave S, L, R: stack → [B, N, 3, d] → reshape → [B, N*3, d]
            seq_body = torch.stack([s_emb, l_emb, r_emb], dim=2).reshape(B, N * 3, self.d_model)
            n_mod = 3
        else:
            # L/R only: [B, N, 2, d] → [B, N*2, d]
            seq_body = torch.stack([l_emb, r_emb], dim=2).reshape(B, N * 2, self.d_model)
            n_mod = 2

        if self.with_cls:
            enc_cls = self.enc_cls_token.expand(B, 1, -1)  # [B, 1, d]
            seq = torch.cat([enc_cls, seq_body], dim=1)     # [B, 1+N*n_mod, d]
        else:
            seq = seq_body                                  # [B, N*n_mod, d]

        # ── 5. Build key_padding_mask ──
        eye_invalid = eye_nonmissing_frac < self.min_nonmissing_frac
        if n_mod == 3:
            body_mask = torch.stack(
                [pad_mask, pad_mask | eye_invalid[:, :, 0], pad_mask | eye_invalid[:, :, 1]],
                dim=2,
            ).reshape(B, N * 3)
        else:
            body_mask = torch.stack(
                [pad_mask | eye_invalid[:, :, 0], pad_mask | eye_invalid[:, :, 1]],
                dim=2,
            ).reshape(B, N * 2)
        if self.with_cls:
            cls_mask = pad_mask.new_zeros(B, 1)              # [B, 1]
            key_padding_mask = torch.cat([cls_mask, body_mask], dim=1)  # [B, 1+N*n_mod]
        else:
            key_padding_mask = body_mask                     # [B, N*n_mod]

        # ── 6. Transformer (manual block loop) ──
        attn_mask = self._get_stim_mask(N, device) if (self.stim_isolated_attn and s_emb is not None) else None

        h = seq
        for block in self.blocks:
            if self.attention_layout == "axial_cross":
                h = block(h, key_padding_mask=key_padding_mask, n_time=N)
            else:
                h = block(h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        hidden = self.out_norm(h)

        # ── 7. Split back ──
        if self.with_cls:
            enc_cls_out = hidden[:, :1, :]                # [B, 1, d]
            body = hidden[:, 1:, :]
        else:
            enc_cls_out = None
            body = hidden
        body = body.reshape(B, N, n_mod, self.d_model)     # [B, N, n_mod, d]
        if n_mod == 3:
            s_out = body[:, :, 0, :]                      # [B, N, d]
            l_out = body[:, :, 1, :]
            r_out = body[:, :, 2, :]
        else:
            s_out = None
            l_out = body[:, :, 0, :]
            r_out = body[:, :, 1, :]

        return {
            "enc_cls": enc_cls_out,
            "s_hidden": s_out,
            "l_hidden": l_out,
            "r_hidden": r_out,
        }


# ──────────────────────────────────────────────
# S Encoder (分离架构: 只编刺激, 输出连续表示, 无 cls)
# ──────────────────────────────────────────────

class SEncoder(nn.Module):
    """轻量刺激编码器: S patches → 连续 s_hidden [B, N, d] (不量化, 无 cls).

    仅服务于特征预测 decoder; 重建路径完全不使用刺激。
    结构: PatchEmbedding + pos_embed + 若干 TransformerBlock (默认 3 层).
    输出每个 S patch 的连续表示, 供 FeatureDecoder 使用。
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        dim_ff: int = 768,
        dropout: float = 0.0,
        max_patches: int = 512,
        patch_samples: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_samples = patch_samples
        self.stim_patch_embed = PatchEmbedding(d_model, patch_samples)
        self.pos_embed = nn.Embedding(max_patches, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

    def forward(self, stim: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """stim: [B, N, 4, patch_samples] → s_hidden: [B, N, d_model]"""
        B, N = stim.shape[:2]
        device = stim.device
        ps = self.patch_samples
        s_flat = stim.reshape(B * N, 4, ps)
        s_emb = self.stim_patch_embed(s_flat).reshape(B, N, self.d_model)  # [B, N, d]
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        s_emb = s_emb + self.pos_embed(pos)
        h = s_emb
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)
        return self.out_norm(h)


# ──────────────────────────────────────────────
# VQ Projection (仅 L/R, 单层 Linear)
# ──────────────────────────────────────────────

class VQProjection(nn.Module):
    """Encoder hidden → FSQ dim 投影.

    Linear(d_model → fsq_d), 无激活, 无中间层.
    """

    def __init__(self, d_model: int = 512, fsq_d: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_model, fsq_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*N*2, d_model] → [B*N*2, fsq_d]"""
        return self.proj(x)


# ──────────────────────────────────────────────
# Patch Reconstruction Head (decoder 侧)
# ──────────────────────────────────────────────

class PatchReconHead(nn.Module):
    """d_model hidden → raw patch [4, patch_samples] via ConvTranspose upsampling.

    对称于 PatchEmbedding: Linear → 128×(L/4) → deconv(L/4→L/2→L) → Conv→[4,L]
    """

    def __init__(self, d_model: int = 256, out_channels: int = 4, patch_samples: int = 20):
        super().__init__()
        self.patch_samples = patch_samples
        self.flat = patch_samples // 4
        self.fc = nn.Linear(d_model, 128 * self.flat)                                     # d → 128*(L/4)
        self.deconv1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)   # L/4→L/2
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)    # L/2→L
        self.conv_out = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N*2, d_model]
        Returns:
            [B*N*2, 4, patch_samples]
        """
        x = self.fc(x)                                # [B*N*2, 128*(L/4)]
        x = x.reshape(x.shape[0], 128, self.flat)      # [B*N*2, 128, L/4]
        x = F.gelu(self.deconv1(x))                    # [B*N*2, 128, L/2]
        x = F.gelu(self.deconv2(x))                    # [B*N*2, 64, L]
        return self.conv_out(x)                        # [B*N*2, 4, L]


# ──────────────────────────────────────────────
# ViT Decoder
# ──────────────────────────────────────────────

class ViTDecoder(nn.Module):
    """ViT Decoder: 独立 dec_cls + 量化 L/R tokens → 手工特征 + eye重建.

    dec_cls 是 decoder 自己的可学习参数,非 encoder 传入。
    序列: [dec_cls, L₀, R₀, L₁, R₁, ..., Lₙ₋₁, Rₙ₋₁]  — 无 S,无 enc_cls.
    """

    def __init__(
        self,
        d_model: int = 256,
        code_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        dim_ff: int = 768,
        dropout: float = 0.0,
        num_manual_features: int = 6,
        max_patches: int = 512,
        patch_samples: int = 20,
        with_cls: bool = True,           # False → 重建专用 (无 dec_cls)
        build_feature_head: bool = True, # False → 重建专用 (无 cls_head)
        min_nonmissing_frac: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_samples = patch_samples
        self.with_cls = with_cls
        self.build_feature_head = build_feature_head
        self.min_nonmissing_frac = float(min_nonmissing_frac)

        # VQ code → d_model
        self.code_to_dmodel = nn.Linear(code_dim, d_model)

        # dec_cls token (decoder 独立; with_cls=False 时不创建)
        if with_cls:
            self.dec_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.dec_cls_token, std=0.02)
        else:
            self.dec_cls_token = None

        # Type embedding (decoder: CLS=0, L=1, R=2)
        self.type_embed = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self._t_cls, self._t_l, self._t_r = 0, 1, 2  # decoder-local type indices

        # Position embedding
        self.pos_embed = nn.Embedding(max_patches, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # Quality embedding (L/R)
        self.quality_embed = nn.Embedding(Q_BINS, d_model)
        nn.init.normal_(self.quality_embed.weight, std=0.02)

        # Transformer blocks (EyeMAE-aligned)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

        # Output heads
        # dec_cls → 手工特征 (无 pooling) — build_feature_head=False 时无
        if build_feature_head:
            self.cls_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_manual_features),
            )
        else:
            self.cls_head = None

        # L/R → raw patch reconstruction
        self.recon_head = PatchReconHead(d_model=d_model, patch_samples=patch_samples)

    def forward(
        self,
        z_q_lr: torch.Tensor,                 # [B, N, 2, code_dim]
        eye_nonmissing_frac: torch.Tensor,    # [B, N, 2]
        pad_mask: torch.Tensor,               # [B, N]
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            manual_feat_pred: [B, num_manual_features]
            eye_recon:        [B, N, 2, 4, 20]
        """
        B, N = z_q_lr.shape[:2]
        device = z_q_lr.device

        # ── 1. Project VQ codes → d_model ──
        z_q_flat = z_q_lr.reshape(B * N * 2, -1)          # [B*N*2, code_dim]
        lr_emb = self.code_to_dmodel(z_q_flat)             # [B*N*2, d_model]
        lr_emb = lr_emb.reshape(B, N, 2, self.d_model)     # [B, N, 2, d]
        l_emb, r_emb = lr_emb[:, :, 0, :], lr_emb[:, :, 1, :]  # [B, N, d]

        # ── 2. Add type + position + quality embeddings ──
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)

        l_emb = l_emb + self.type_embed.weight[self._t_l] + pos_emb
        r_emb = r_emb + self.type_embed.weight[self._t_r] + pos_emb

        q_l = (eye_nonmissing_frac[:, :, 0] * Q_BINS).long().clamp(0, Q_BINS - 1)
        q_r = (eye_nonmissing_frac[:, :, 1] * Q_BINS).long().clamp(0, Q_BINS - 1)
        l_emb = l_emb + self.quality_embed(q_l)
        r_emb = r_emb + self.quality_embed(q_r)

        # ── 3. Build sequence: [dec_cls?, L₀, R₀, ..., Lₙ₋₁, Rₙ₋₁] ──
        seq_body = torch.stack([l_emb, r_emb], dim=2).reshape(B, N * 2, self.d_model)
        if self.with_cls:
            dec_cls = self.dec_cls_token.expand(B, 1, -1)
            seq = torch.cat([dec_cls, seq_body], dim=1)  # [B, 1+N*2, d]
        else:
            seq = seq_body                                # [B, N*2, d]

        # ── 4. key_padding_mask ──
        eye_invalid = eye_nonmissing_frac < self.min_nonmissing_frac
        body_mask = torch.stack(
            [pad_mask | eye_invalid[:, :, 0], pad_mask | eye_invalid[:, :, 1]], dim=2
        ).reshape(B, N * 2)
        if self.with_cls:
            cls_mask = pad_mask.new_zeros(B, 1)
            key_padding_mask = torch.cat([cls_mask, body_mask], dim=1)
        else:
            key_padding_mask = body_mask

        # ── 5. Transformer (manual block loop) ──
        h = seq
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        hidden = self.out_norm(h)

        # ── 6. Split outputs ──
        # dec_cls → 手工特征 (可选; 重建专用时无)
        if self.build_feature_head:
            dec_cls_out = hidden[:, 0, :]              # [B, d_model]
            manual_feat_pred = self.cls_head(dec_cls_out)  # [B, num_manual_features]
            body = hidden[:, 1:, :]
        else:
            manual_feat_pred = None
            body = hidden

        # L/R → reconstruction
        body = body.reshape(B, N, 2, self.d_model)      # [B, N, 2, d]
        lr_hidden = body.reshape(B * N * 2, self.d_model)        # [B*N*2, d]
        eye_recon = self.recon_head(lr_hidden)                    # [B*N*2, 4, patch_samples]
        eye_recon = eye_recon.reshape(B, N, 2, 4, self.patch_samples)

        return {
            "manual_feat_pred": manual_feat_pred,
            "eye_recon": eye_recon,
        }


# ──────────────────────────────────────────────
# True cross-attention feature decoder
# ──────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: eye queries read stimulus key/value tokens."""

    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_norm = RMSNorm(dim)
        self.kv_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_query, dim = query.shape
        n_context = context.shape[1]
        q = self.q_proj(self.q_norm(query)).reshape(
            batch, n_query, self.heads, self.head_dim
        ).transpose(1, 2)
        context_norm = self.kv_norm(context)
        k = self.k_proj(context_norm).reshape(
            batch, n_context, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(context_norm).reshape(
            batch, n_context, self.heads, self.head_dim
        ).transpose(1, 2)
        allowed = (~context_padding_mask).unsqueeze(1).unsqueeze(2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, n_query, dim)
        query = query + self.attn_drop(self.out_proj(attended))
        query = query + self.ffn_drop(self.ffn(self.ffn_norm(query)))
        return query


class CrossAttentionFeatureDecoder(nn.Module):
    """Quantized eye queries cross-attend an independently encoded stimulus stream.

    Stimulus tokens first use stimulus-only self-attention in :class:`SEncoder`.
    The eye/feature stream then performs eye self-attention followed by genuine
    cross-attention with ``Q=eye`` and ``K,V=stimulus``.  Stimulus therefore
    never reads L/R tokens, while L/R and the feature token can read stimulus.
    """

    def __init__(
        self,
        d_model: int = 256,
        code_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        dim_ff: int = 768,
        dropout: float = 0.0,
        num_manual_features: int = 6,
        max_patches: int = 512,
        patch_samples: int = 20,
        min_nonmissing_frac: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_samples = patch_samples
        self.min_nonmissing_frac = float(min_nonmissing_frac)

        # VQ code → d_model (L/R)
        self.code_to_dmodel = nn.Linear(code_dim, d_model)
        self.s_proj = nn.Linear(d_model, d_model)

        # feat_cls token (decoder 独立)
        self.feat_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.feat_cls_token, std=0.02)

        self.type_embed = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self._t_cls, self._t_l, self._t_r = 0, 1, 2

        # Position embedding (按时间步共享: L/R/S 同 time step 用相同 pos)
        self.pos_embed = nn.Embedding(max_patches, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # Quality embedding (仅 L/R)
        self.quality_embed = nn.Embedding(Q_BINS, d_model)
        nn.init.normal_(self.quality_embed.weight, std=0.02)

        self.self_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

        # feat_cls → 手工特征 (无 pooling)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_manual_features),
        )

    def forward(
        self,
        z_q_lr: torch.Tensor,              # [B, N, 2, code_dim]
        s_hidden: torch.Tensor,            # [B, N, d_model] (连续, 不量化)
        eye_nonmissing_frac: torch.Tensor, # [B, N, 2]
        pad_mask: torch.Tensor,            # [B, N]
    ) -> dict[str, torch.Tensor]:
        """Returns: manual_feat_pred: [B, num_manual_features]"""
        B, N = z_q_lr.shape[:2]
        device = z_q_lr.device

        # ── 1. L/R: VQ codes → d_model ──
        z_q_flat = z_q_lr.reshape(B * N * 2, -1)
        lr_emb = self.code_to_dmodel(z_q_flat).reshape(B, N, 2, self.d_model)
        l_emb, r_emb = lr_emb[:, :, 0, :], lr_emb[:, :, 1, :]

        # ── 2. Add eye type + position + quality ──
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        l_emb = l_emb + self.type_embed.weight[self._t_l] + pos_emb
        r_emb = r_emb + self.type_embed.weight[self._t_r] + pos_emb

        q_l = (eye_nonmissing_frac[:, :, 0] * Q_BINS).long().clamp(0, Q_BINS - 1)
        q_r = (eye_nonmissing_frac[:, :, 1] * Q_BINS).long().clamp(0, Q_BINS - 1)
        l_emb = l_emb + self.quality_embed(q_l)
        r_emb = r_emb + self.quality_embed(q_r)

        # ── 3. Eye query sequence and independent stimulus context ──
        feat_cls = self.feat_cls_token.expand(B, 1, -1) + self.type_embed.weight[self._t_cls]
        lr_seq = torch.stack([l_emb, r_emb], dim=2).reshape(B, N * 2, self.d_model)
        query = torch.cat([feat_cls, lr_seq], dim=1)  # [B, 1+2N, d]
        # SEncoder already added its own time embedding before stimulus-only
        # self-attention. Do not add the eye decoder's position embedding again.
        context = self.s_proj(s_hidden)                # [B, N, d]

        # ── 4. Masks ──
        cls_mask = pad_mask.new_zeros(B, 1)
        eye_invalid = eye_nonmissing_frac < self.min_nonmissing_frac
        body_mask = torch.stack(
            [pad_mask | eye_invalid[:, :, 0], pad_mask | eye_invalid[:, :, 1]], dim=2
        ).reshape(B, N * 2)
        query_padding_mask = torch.cat([cls_mask, body_mask], dim=1)

        # ── 5. Eye self-attention, then Q=eye / K,V=stim cross-attention ──
        h = query
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            h = self_block(h, key_padding_mask=query_padding_mask)
            h = cross_block(h, context=context, context_padding_mask=pad_mask)
        hidden = self.out_norm(h)

        # ── 7. feat_cls → 特征 ──
        feat_cls_out = hidden[:, 0, :]
        manual_feat_pred = self.cls_head(feat_cls_out)
        return {"manual_feat_pred": manual_feat_pred}


# ──────────────────────────────────────────────
# Full EyeVQ Tokenizer
# ──────────────────────────────────────────────

class EyeVQTokenizer(nn.Module):
    """EyeVQ Tokenizer: ViT Encoder → VQ(L/R only) → ViT Decoder.

    Pipeline:
      1. ViTEncoder:  S/L/R + enc_cls → Transformer → hidden
      2. VQProjection: L/R hidden → Linear(256→64) → VQ Quantizer
      3. ViTDecoder:   dec_cls(独立) + z_q_LR → Transformer → 手工特征 + eye重建
    """

    def __init__(
        self,
        # Encoder
        d_model: int = 256,
        enc_n_layers: int = 12,
        enc_n_heads: int = 8,
        enc_dim_ff: int = 768,
        # VQ
        vq_type: str = "fsq",
        eye_code_dim: int = 8,
        eye_codebook_size: int = 512,
        fsq_L: int | list = 5,
        fsq_activation: str = "tanh",      # "ifsq" or legacy "tanh"
        ifsq_alpha: float = 1.6,
        commitment_beta: float = 0.25,
        # Decoder
        dec_n_layers: int = 3,
        dec_n_heads: int = 8,
        dec_dim_ff: int = 768,
        # Features
        num_manual_features: int = 6,
        max_patches: int = 512,
        patch_samples: int = 20,
        dropout: float = 0.0,
        stim_isolated_attn: bool = True,
        stim_attend_cls: bool = True,
        attention_layout: str = "joint",
        architecture: str = "joint",       # "joint" or "cross_attention"
        s_enc_n_layers: int = 3,
        feat_dec_n_layers: int = 3,
        min_nonmissing_frac: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.eye_code_dim = eye_code_dim
        self.eye_codebook_size = eye_codebook_size
        self.vq_type = vq_type
        if architecture not in {"joint", "cross_attention"}:
            raise ValueError(
                f"architecture must be 'joint' or 'cross_attention', got {architecture!r}"
            )
        self.architecture = architecture
        self.patch_samples = patch_samples

        is_cross = architecture == "cross_attention"
        self.encoder = ViTEncoder(
            d_model=d_model, n_layers=enc_n_layers, n_heads=enc_n_heads,
            dim_ff=enc_dim_ff, dropout=dropout, max_patches=max_patches,
            patch_samples=patch_samples, stim_isolated_attn=stim_isolated_attn,
            stim_attend_cls=stim_attend_cls,
            attention_layout=attention_layout,
            include_stim=not is_cross, with_cls=not is_cross,
            min_nonmissing_frac=min_nonmissing_frac,
        )

        self.vq_proj = VQProjection(d_model=d_model, fsq_d=eye_code_dim)
        if vq_type == "fsq":
            self.eye_codebook = FSQ(
                d=eye_code_dim,
                L=fsq_L,
                activation=fsq_activation,
                ifsq_alpha=ifsq_alpha,
            )
        elif vq_type == "vqvae":
            self.eye_codebook = VQVAEQuantizer(
                d=eye_code_dim,
                codebook_size=eye_codebook_size,
                commitment_beta=commitment_beta,
            )
        elif vq_type == "ae":
            # Continuous autoencoder baseline: no rounding, lookup table, or
            # straight-through estimator.  The latent is normalized immediately
            # before every decoder so its scale is fixed and can also be used as
            # a stable offline regression target for BERT.
            self.eye_codebook = nn.Identity()
        else:
            raise ValueError(f"Unsupported vq_type={vq_type!r}")
        self.decoder_latent_norm = (
            # Non-affine normalization is deliberate: a learnable gamma/beta
            # could undo the fixed-scale contract used by the BERT MSE cache.
            nn.LayerNorm(eye_code_dim, elementwise_affine=False)
            if vq_type == "ae" else nn.Identity()
        )

        if is_cross:
            self.recon_decoder = ViTDecoder(
                d_model=d_model, code_dim=eye_code_dim,
                n_layers=dec_n_layers, n_heads=dec_n_heads, dim_ff=dec_dim_ff,
                dropout=dropout, num_manual_features=num_manual_features,
                max_patches=max_patches, patch_samples=patch_samples,
                with_cls=False, build_feature_head=False,
                min_nonmissing_frac=min_nonmissing_frac,
            )
            self.s_encoder = SEncoder(
                d_model=d_model, n_layers=s_enc_n_layers, n_heads=enc_n_heads,
                dim_ff=enc_dim_ff, dropout=dropout, max_patches=max_patches,
                patch_samples=patch_samples,
            )
            self.feat_decoder = CrossAttentionFeatureDecoder(
                d_model=d_model, code_dim=eye_code_dim,
                n_layers=feat_dec_n_layers, n_heads=dec_n_heads, dim_ff=dec_dim_ff,
                dropout=dropout, num_manual_features=num_manual_features,
                max_patches=max_patches, patch_samples=patch_samples,
                min_nonmissing_frac=min_nonmissing_frac,
            )
        else:
            self.decoder = ViTDecoder(
                d_model=d_model, code_dim=eye_code_dim,
                n_layers=dec_n_layers, n_heads=dec_n_heads, dim_ff=dec_dim_ff,
                dropout=dropout, num_manual_features=num_manual_features,
                max_patches=max_patches, patch_samples=patch_samples,
                min_nonmissing_frac=min_nonmissing_frac,
            )

    def _encode_and_quantize(
        self,
        stim: torch.Tensor,
        content: torch.Tensor,
        quality: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor | None,
        quantize: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        """Shared encoder path with the invariant ``[L0,R0,L1,R1,...]``."""
        B, N = stim.shape[:2]
        eye_l = content[:, :, 0, :, :]
        eye_r = content[:, :, 1, :, :]
        if eye_nonmissing_frac is None:
            missing = quality > 0.5
            eye_nonmissing_frac = 1.0 - missing.float().mean(dim=(3, 4))

        if self.architecture == "cross_attention":
            enc = self.encoder(None, eye_l, eye_r, eye_nonmissing_frac, pad_mask)
            s_hidden = self.s_encoder(stim, pad_mask)
        else:
            enc = self.encoder(stim, eye_l, eye_r, eye_nonmissing_frac, pad_mask)
            s_hidden = enc["s_hidden"]

        lr_hidden = torch.stack(
            [enc["l_hidden"], enc["r_hidden"]], dim=2
        ).reshape(B, N * 2, self.d_model)
        z_e = self.vq_proj(lr_hidden)
        if self.vq_type == "ae":
            z_q_st = z_e
            code_ids = torch.zeros(B, N * 2, dtype=torch.long, device=z_e.device)
            commit = z_e.new_zeros(())
            vq_stats = {}
        elif quantize:
            z_q_st, code_ids, _z_q, commit, vq_stats = self.eye_codebook(z_e)
        else:
            # Legacy continuous warmup, when explicitly requested, removes
            # rounding but preserves the configured FSQ/iFSQ bounded domain.
            z_q_st = (
                self.eye_codebook.bound(z_e)
                if isinstance(self.eye_codebook, FSQ)
                else z_e
            )
            code_ids = torch.zeros(B, N * 2, dtype=torch.long, device=z_e.device)
            commit = z_e.new_zeros(())
            vq_stats = {}
        z_q_lr = z_q_st.reshape(B, N, 2, self.eye_code_dim)
        z_q_lr = self.decoder_latent_norm(z_q_lr)
        return enc, s_hidden, z_q_lr, code_ids.reshape(B, N, 2), vq_stats, commit

    def forward(
        self,
        stim: torch.Tensor,                    # [B, N, 4, 20]
        content: torch.Tensor,                 # [B, N, 2, 4, 20]  (L/R eye)
        quality: torch.Tensor,                 # [B, N, 2, 20, 1]
        pad_mask: torch.Tensor,                # [B, N]
        eye_nonmissing_frac: torch.Tensor | None = None,  # [B, N, 2]
        quantize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Full forward: encode → VQ(L/R) → decode."""
        B, N = stim.shape[:2]
        enc, s_hidden, z_q_lr, code_ids, vq_stats, commit = self._encode_and_quantize(
            stim, content, quality, pad_mask, eye_nonmissing_frac, quantize
        )
        if eye_nonmissing_frac is None:
            eye_nonmissing_frac = 1.0 - (quality > 0.5).float().mean(dim=(3, 4))

        # ── 3. Decode. Both formal paths predict features after quantization. ──
        if self.architecture == "cross_attention":
            recon = self.recon_decoder(z_q_lr, eye_nonmissing_frac, pad_mask)          # 只用 L/R
            feat = self.feat_decoder(z_q_lr, s_hidden, eye_nonmissing_frac, pad_mask)  # S + L/R
            dec = {**recon, **feat}
        else:
            dec = self.decoder(z_q_lr, eye_nonmissing_frac, pad_mask)

        return {
            **enc,
            **dec,
            "s_hidden": s_hidden,
            "code_ids": code_ids,
            "commit_eye": commit,
            "vq_stats": vq_stats,
        }

    @torch.no_grad()
    def encode_codes(
        self,
        stim: torch.Tensor,
        content: torch.Tensor,
        quality: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return correctly aligned ``[B, N, 2]`` eye code IDs."""
        _enc, _s, _zq, code_ids, _stats, _commit = self._encode_and_quantize(
            stim, content, quality, pad_mask, eye_nonmissing_frac, True
        )
        return code_ids

    @torch.no_grad()
    def encode_latents(
        self,
        stim: torch.Tensor,
        content: torch.Tensor,
        quality: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return decoder-normalized continuous eye latents ``[B,N,2,D]``."""
        if self.vq_type != "ae":
            raise ValueError("encode_latents is only defined for vq.type=ae")
        _enc, _s, latents, _ids, _stats, _commit = self._encode_and_quantize(
            stim, content, quality, pad_mask, eye_nonmissing_frac, False
        )
        return latents
