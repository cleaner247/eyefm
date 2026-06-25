from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _checkpoint_fn

# ── Increase torch.compile cache from default 8 to 256 so that
#     _fused_norm_masked (and other compiled helpers) can cache every
#     (B, nmax) shape we encounter without eviction.
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

from .patching import build_sequence_attention_mask


class RMSNorm(nn.Module):
    """RMSNorm using PyTorch's fused CUDA kernel (2.1+) with manual fallback."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.nn.functional, "rms_norm"):
            return torch.nn.functional.rms_norm(
                x, (x.shape[-1],), self.weight, self.eps
            )
        # Fallback: unfused path
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN with gate–up projection fused in a single linear layer.

    The activation part (split / SiLU / multiply) is compiled with
    torch.compile so inductor can fuse SiLU and the elementwise multiply
    into a single GPU kernel, avoiding an intermediate HBM round-trip.
    """

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2)
        self.out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.w12(x)
        gate_up = _swiglu_activation(projected)
        return self.out(self.dropout(gate_up))


@torch.compile(dynamic=True)
def _swiglu_activation(projected: torch.Tensor) -> torch.Tensor:
    """Fused SiLU + multiply: inductor should merge into one kernel."""
    gate, up = projected.chunk(2, dim=-1)
    return up * torch.nn.functional.silu(gate)


@torch.compile(dynamic=True)
def _fused_norm_masked(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse RMSNorm + masked_fill to avoid an intermediate HBM write."""
    h = torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), weight, eps)
    return h.masked_fill(mask[..., None], 0.0)


class TransformerBlock(nn.Module):
    """Transformer block with Megatron-style selective checkpointing.

    Only the FFN (SwiGLU) is checkpointed — it has the largest activation
    footprint (2× hidden dim).  Attention is left un‑checkpointed because
    flash attention already manages its scratch memory in SRAM.
    """

    def __init__(self, dim: int, heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        *,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        # ── Attention: NOT checkpointed ──
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        # ── FFN: selectively checkpointed ──
        if use_checkpoint:
            x = _checkpoint_fn(self._ffn_forward, x, use_reentrant=False)
        else:
            x = self._ffn_forward(x)
        return x

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN sub‑forward, separable so it can be wrapped by checkpoint.
        Uses out-of-place + to avoid version mismatch inside checkpoint."""
        return x + self.drop2(self.ffn(self.norm2(x)))


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
        y = x.reshape(-1, patch, dim).transpose(1, 2).contiguous()
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
        self.use_token_type_embedding = bool(model_cfg.get("use_token_type_embedding", True))
        self.token_type_embedding = nn.Embedding(3, d_model) if self.use_token_type_embedding else None
        self.time_embedding = nn.Embedding(self.max_patches, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.fusion_norm = nn.LayerNorm(d_model)
        compile_blocks = bool(model_cfg.get("compile_blocks", False))
        self.blocks = nn.ModuleList()
        if compile_blocks:
            # ── torch.compile pre‑flight for PyTorch 2.5.1 ──
            # 1) Bump Triton max X block (inductor may fuse to X=4096).
            import torch._inductor.runtime.triton_heuristics as _th
            _th.TRITON_MAX_BLOCK = {"X": 4096, "Y": 1024, "Z": 1024, "R": 65536}
            # 2) Force fork-based compile workers so the monkey‑patch above is
            #    inherited (SubprocPool uses subprocess.Popen which does not).
            import torch._inductor.config as _inductor_cfg
            _inductor_cfg.worker_start_method = "fork"
            # 3) Lower worker count: a single fork avoids CUDA‑context bloat
            #    while still offloading Triton compilation off the main thread.
            _inductor_cfg.compile_threads = 4

        for _ in range(int(model_cfg["n_layers"])):
            block = TransformerBlock(
                d_model,
                int(model_cfg["n_heads"]),
                int(model_cfg["ffn_hidden"]),
                float(model_cfg["dropout"]),
            )
            if compile_blocks:
                # Only compile FFN — the attention sub-module (MultiheadAttention)
                # generates too many intermediates for inductor to fuse safely on
                # shapes like (78, 1152, 512).  FFN (SwiGLU) is the compute-heavy
                # part and compiles cleanly.
                block.ffn = torch.compile(block.ffn, dynamic=False)
            self.blocks.append(block)
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
        task_sq = task.squeeze(2)  # (bsz, 1, d_model) — compute once
        times = self.time_embedding(torch.arange(n, device=content.device)).view(1, n, self.d_model)
        if self.token_type_embedding is None:
            stim_type = left_type = right_type = torch.zeros(self.d_model, dtype=content_token.dtype, device=content.device)
        else:
            stim_type = self.token_type_embedding(torch.zeros((), dtype=torch.long, device=content.device))
            left_type = self.token_type_embedding(torch.ones((), dtype=torch.long, device=content.device))
            right_type = self.token_type_embedding(torch.full((), 2, dtype=torch.long, device=content.device))

        # Accumulate with two adds instead of three chained +'s to reduce intermediates
        base = task_sq + times  # (bsz, n, d_model)
        s_tokens = self.fusion_norm(stim_token + base + stim_type.view(1, 1, -1))
        l_tokens = self.fusion_norm(content_token[:, :, 0] + quality_token[:, :, 0] + base + left_type.view(1, 1, -1))
        r_tokens = self.fusion_norm(content_token[:, :, 1] + quality_token[:, :, 1] + base + right_type.view(1, 1, -1))
        return s_tokens, l_tokens, r_tokens

    @staticmethod
    def interleave_sequence(s_tokens: torch.Tensor, l_tokens: torch.Tensor, r_tokens: torch.Tensor) -> torch.Tensor:
        # stack + reshape: contiguous copy (stack) + zero-copy view (reshape),
        # replacing 3 strided-scatter writes.  ~2x faster on GPU.
        stacked = torch.stack([s_tokens, l_tokens, r_tokens], dim=2)  # (B, N, 3, D)
        bsz, n = s_tokens.shape[:2]
        return stacked.reshape(bsz, 3 * n, -1)

    @staticmethod
    def extract_eye_hidden(hidden_seq: torch.Tensor) -> torch.Tensor:
        # reshape + slice: zero-copy view, replacing 2 strided reads + stack.
        # Returns (B, N, 2, D) with non-contiguous batch dims — pred_head handles this fine.
        h = hidden_seq.reshape(hidden_seq.shape[0], -1, 3, hidden_seq.shape[-1])
        return h[:, :, 1:]  # skip index 0 (stim tokens)

    def forward_features(
        self,
        content: torch.Tensor,
        quality: torch.Tensor,
        stim: torch.Tensor,
        task_id: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_token_valid: torch.Tensor,
        mae_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mae_mask is None:
            mae_mask = torch.zeros(
                content.shape[0],
                content.shape[1],
                2,
                dtype=torch.bool,
                device=content.device,
            )
        s_tokens, l_tokens, r_tokens = self.assemble_tokens(content, quality, stim, task_id, mae_mask)
        seq = self.interleave_sequence(s_tokens, l_tokens, r_tokens)
        seq_attn_pad_mask = build_sequence_attention_mask(pad_mask, eye_token_valid)

        # ── Pad nmax to nearest multiple of pad_to_tokens ──
        # e.g. pad_to_tokens=64 → nmax ∈ {64, 128, 192, 256, 320, 384}.
        # The sampler groups trials by rounded nmax, so every micro-batch
        # within a group has the same padded shape → torch.compile caches
        # one graph per (B, nmax) pair (≤ 6 distinct shapes).
        _pad_multiple = int(self.cfg.get("train", {}).get("pad_to_tokens", 0) or 0)
        _orig_seq_len = seq.shape[1]
        if _pad_multiple > 1:
            _n = _orig_seq_len // 3  # real nmax
            _target_n = ((_n + _pad_multiple - 1) // _pad_multiple) * _pad_multiple
            _target_seq = _target_n * 3
            if _target_seq > _orig_seq_len:
                pad_len = _target_seq - _orig_seq_len
                seq = torch.nn.functional.pad(seq, (0, 0, 0, pad_len))
                seq_attn_pad_mask = torch.nn.functional.pad(
                    seq_attn_pad_mask, (0, pad_len), value=True
                )

        hidden = seq
        use_checkpoint = bool(self.cfg.get("train", {}).get("gradient_checkpointing", False)) and self.training
        for block in self.blocks:
            hidden = block(hidden, seq_attn_pad_mask, use_checkpoint=use_checkpoint)

        # ── Keep padded length through final_norm / extract so that
        #     _fused_norm_masked only ever sees 6 distinct shapes
        #     (one per nmax bucket) instead of 309 (every real_nmax).
        #     The small amount of wasted compute on padding positions
        #     is far cheaper than re‑compilation on every micro‑batch.
        hidden = _fused_norm_masked(
            hidden,
            self.final_norm.weight,
            self.final_norm.eps,
            seq_attn_pad_mask,
        )
        hidden_eye = self.extract_eye_hidden(hidden)
        out: dict[str, torch.Tensor] = {
            "hidden_eye": hidden_eye,
            "seq_attn_pad_mask": seq_attn_pad_mask,
        }
        # Only keep the large tensors when explicitly requested (e.g. visualization);
        # during training they would waste ~200 MB per micro-batch.
        if not self.training:
            out["hidden_seq"] = hidden
            out["seq"] = seq
            out["eye_token_valid"] = eye_token_valid
        return out

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
        features = self.forward_features(content, quality, stim, task_id, pad_mask, eye_token_valid, mae_mask)
        hidden_eye = features["hidden_eye"]  # [B, _rounded_nmax, 2, D] — 6 shapes only
        # pred_head sees the (slightly larger) padded nmax; trim the output
        # back to the real trial length after the reshape.
        pred = self.pred_head(hidden_eye).view(content.shape[0], -1, 2, self.patch, 4)
        if pred.shape[1] > content.shape[1]:
            pred = pred[:, :content.shape[1], :, :, :]
        out = {"pred": pred, "seq_attn_pad_mask": features["seq_attn_pad_mask"]}
        if return_hidden:
            out.update({"hidden_seq": features["hidden_seq"], "hidden_eye": hidden_eye, "seq": features["seq"]})
        return out


def build_model(cfg: dict[str, Any]) -> EyeMAEModel:
    return EyeMAEModel(cfg)
