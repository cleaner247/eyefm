from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .patching import build_sequence_attention_mask


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
        return self.out(self.dropout(a * torch.nn.functional.silu(gate)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class ConvTokenizer(nn.Module):
    def __init__(self, in_dim: int, channels: tuple[int, ...], patch_samples: int, d_model: int, kernels: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for ch, kernel in zip(channels, kernels):
            layers.extend([nn.Conv1d(last, ch, kernel_size=kernel, padding=kernel // 2), nn.GELU()])
            last = ch
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(last * patch_samples, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x.shape[:-2]
        patch = x.shape[-2]
        dim = x.shape[-1]
        y = x.reshape(-1, patch, dim).transpose(1, 2)
        y = self.conv(y).flatten(1)
        y = self.proj(y)
        return y.reshape(*original, -1)


class EyeMAEModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        patch = int(cfg["patch"]["samples"])
        d_model = int(model_cfg["d_model"])
        self.cfg = cfg
        self.d_model = d_model
        self.max_patches = int(model_cfg["max_patches"])
        self.content_tokenizer = ConvTokenizer(4, (64, 128), patch, d_model, (5, 3))
        self.quality_tokenizer = ConvTokenizer(1, (32,), patch, d_model, (3,))
        self.stim_tokenizer = ConvTokenizer(4, (64,), patch, d_model, (3,))
        self.task_embedding = nn.Embedding(4, d_model)
        self.token_type_embedding = nn.Embedding(3, d_model)
        self.time_embedding = nn.Embedding(self.max_patches, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    int(model_cfg["n_heads"]),
                    int(model_cfg["ffn_hidden"]),
                    float(model_cfg["dropout"]),
                )
                for _ in range(int(model_cfg["n_layers"]))
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.pred_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch * 4))
        self.patch = patch
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)

    def assemble_tokens(
        self,
        content: torch.Tensor,
        quality: torch.Tensor,
        stim: torch.Tensor,
        task_id: torch.Tensor,
        mae_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n, _, _, _ = content.shape
        if n > self.max_patches:
            raise ValueError(f"num patches {n} exceeds model.max_patches {self.max_patches}")
        content_token = self.content_tokenizer(content)
        quality_token = self.quality_tokenizer(quality)
        stim_token = self.stim_tokenizer(stim)
        content_token = torch.where(mae_mask[..., None], self.mask_token.view(1, 1, 1, -1), content_token)

        task = self.task_embedding(task_id).view(bsz, 1, 1, self.d_model)
        times = self.time_embedding(torch.arange(n, device=content.device)).view(1, n, self.d_model)
        stim_type = self.token_type_embedding(torch.zeros((), dtype=torch.long, device=content.device))
        left_type = self.token_type_embedding(torch.ones((), dtype=torch.long, device=content.device))
        right_type = self.token_type_embedding(torch.full((), 2, dtype=torch.long, device=content.device))

        s_tokens = stim_token + task.squeeze(2) + times + stim_type.view(1, 1, -1)
        l_tokens = content_token[:, :, 0] + quality_token[:, :, 0] + task.squeeze(2) + times + left_type.view(1, 1, -1)
        r_tokens = content_token[:, :, 1] + quality_token[:, :, 1] + task.squeeze(2) + times + right_type.view(1, 1, -1)
        return self.fusion_norm(s_tokens), self.fusion_norm(l_tokens), self.fusion_norm(r_tokens)

    @staticmethod
    def interleave_sequence(s_tokens: torch.Tensor, l_tokens: torch.Tensor, r_tokens: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = s_tokens.shape
        seq = torch.empty(bsz, 3 * n, dim, dtype=s_tokens.dtype, device=s_tokens.device)
        seq[:, 0::3] = s_tokens
        seq[:, 1::3] = l_tokens
        seq[:, 2::3] = r_tokens
        return seq

    @staticmethod
    def extract_eye_hidden(hidden_seq: torch.Tensor) -> torch.Tensor:
        hidden_l = hidden_seq[:, 1::3]
        hidden_r = hidden_seq[:, 2::3]
        return torch.stack([hidden_l, hidden_r], dim=2)

    def forward(
        self,
        content: torch.Tensor,
        quality: torch.Tensor,
        stim: torch.Tensor,
        task_id: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_token_valid: torch.Tensor,
        mae_mask: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        s_tokens, l_tokens, r_tokens = self.assemble_tokens(content, quality, stim, task_id, mae_mask)
        seq = self.interleave_sequence(s_tokens, l_tokens, r_tokens)
        seq_attn_pad_mask = build_sequence_attention_mask(pad_mask, eye_token_valid)
        hidden = seq
        for block in self.blocks:
            hidden = block(hidden, seq_attn_pad_mask)
        hidden = self.final_norm(hidden)
        hidden = hidden.masked_fill(seq_attn_pad_mask[..., None], 0.0)
        hidden_eye = self.extract_eye_hidden(hidden)
        pred = self.pred_head(hidden_eye).view(content.shape[0], content.shape[1], 2, self.patch, 4)
        out = {"pred": pred, "seq_attn_pad_mask": seq_attn_pad_mask}
        if return_hidden:
            out.update({"hidden_seq": hidden, "hidden_eye": hidden_eye, "seq": seq})
        return out


def build_model(cfg: dict[str, Any]) -> EyeMAEModel:
    return EyeMAEModel(cfg)
