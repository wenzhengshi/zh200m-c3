#!/usr/bin/env python3
"""LLaMA-compatible 196M decoder-only model (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings, no bias)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: [..., n_head, seq, head_dim]; cos/sin: [seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, head_dim, rope_theta):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.kv_dim = n_kv_head * head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope_theta = rope_theta
        self._rope_cache = None

    def _rope(self, seq_len, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].device != device or self._rope_cache[0].shape[0] < seq_len:
            cos, sin = precompute_rope(self.head_dim, seq_len, self.rope_theta, device=device, dtype=dtype)
            self._rope_cache = (cos, sin)
        return self._rope_cache

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:s], sin[:s])
        k = apply_rope(k, cos[:s], sin[:s])
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.n_head * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, head_dim, ffn_dim, rope_theta, eps):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps)
        self.attn = Attention(dim, n_head, n_kv_head, head_dim, rope_theta)
        self.ffn_norm = RMSNorm(dim, eps)
        self.mlp = MLP(dim, ffn_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class Zh200M(nn.Module):
    def __init__(self, vocab_size=32000, dim=896, n_layer=20, n_head=14, n_kv_head=2,
                 ffn_dim=2432, max_seq_len=2048, rope_theta=10000.0, eps=1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.head_dim = dim // n_head
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_head, n_kv_head, self.head_dim, ffn_dim, rope_theta, eps)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(dim, eps)
        self.rope_theta = rope_theta
        self._cos = None
        self._sin = None
        self._rope_len = 0
        # tied embeddings: lm_head reuses embed_tokens.weight

    def _rope(self, seq_len, device, dtype):
        if self._cos is None or self._cos.device != device or self._rope_len < seq_len or self._cos.dtype != dtype:
            self._cos, self._sin = precompute_rope(self.head_dim, seq_len, self.rope_theta, device=device, dtype=dtype)
            self._rope_len = seq_len
        return self._cos, self._sin

    def forward(self, input_ids):
        b, s = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self._rope(s, input_ids.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        logits = F.linear(h, self.embed_tokens.weight)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(cfg):
    model = Zh200M(
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
    n = model.count_parameters()
    assert n == 196153216, f"parameter_count={n} != 196153216"
    return model
