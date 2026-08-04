#!/usr/bin/env python3
"""Zh200M C3 model.

Architecture is kept compatible with the user's 196,153,216-parameter model,
including its intentionally preserved custom RoPE pairing:
  * cos/sin use repeat_interleave(2)
  * rotate_half splits the front/back halves

C3 stability changes do not alter parameter shapes:
  * trainable parameters remain FP32; BF16/FP16 is applied by autocast
  * RMSNorm statistics and RoPE tables are computed in FP32
  * optional non-reentrant activation checkpointing
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    # Preserve the original project's nonstandard RoPE layout exactly.
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Preserve the original front/back-half rotation exactly.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos_t = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin_t = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos_t + rotate_half(x) * sin_t


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype=input_dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, head_dim: int):
        super().__init__()
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.kv_dim = n_kv_head * head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:seq_len], sin[:seq_len])
        k = apply_rope(k, cos[:seq_len], sin[:seq_len])
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        ffn_dim: int,
        eps: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps)
        self.attn = Attention(dim, n_head, n_kv_head, head_dim)
        self.ffn_norm = RMSNorm(dim, eps)
        self.mlp = MLP(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class Zh200MC3(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 896,
        n_layer: int = 20,
        n_head: int = 14,
        n_kv_head: int = 2,
        ffn_dim: int = 2432,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError("dim must be divisible by n_head")
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.head_dim = dim // n_head
        self.rope_theta = rope_theta
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_head, n_kv_head, self.head_dim, ffn_dim, eps)
                for _ in range(n_layer)
            ]
        )
        self.norm = RMSNorm(dim, eps)
        self.gradient_checkpointing = False
        self._rope_cache: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index)
        cached = self._rope_cache.get(key)
        if cached is None or cached[0].shape[0] < seq_len:
            cached = precompute_rope(self.head_dim, seq_len, self.rope_theta, device=device)
            self._rope_cache[key] = cached
        return cached

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds {self.max_seq_len}")
        hidden = self.embed_tokens(input_ids)
        # Embedding is not always autocast-eligible. Cast the residual stream to
        # the active autocast dtype so activations stay BF16/FP16 while the
        # trainable parameters and Adam states remain FP32.
        if torch.is_autocast_enabled():
            try:
                autocast_dtype = torch.get_autocast_dtype(input_ids.device.type)
            except (AttributeError, TypeError):
                autocast_dtype = torch.get_autocast_gpu_dtype()
            hidden = hidden.to(dtype=autocast_dtype)
        cos, sin = self._rope(seq_len, input_ids.device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, cos, sin, use_reentrant=False)
            else:
                hidden = layer(hidden, cos, sin)
        hidden = self.norm(hidden)
        return F.linear(hidden, self.embed_tokens.weight)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(cfg: dict) -> Zh200MC3:
    model = Zh200MC3(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg["n_kv_head"],
        ffn_dim=cfg["ffn_dim"],
        max_seq_len=cfg["max_seq_len"],
        rope_theta=cfg["rope_theta"],
        eps=cfg["rms_norm_eps"],
    )
    count = model.count_parameters()
    expected = int(cfg["parameter_count"])
    if count != expected:
        raise RuntimeError(f"parameter_count={count} != expected={expected}")
    return model
