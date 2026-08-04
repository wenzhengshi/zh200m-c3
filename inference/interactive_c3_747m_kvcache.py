#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zh200M C3 747M 原生 RoPE + FP32 + 预分配 KV Cache 实时续写

特点：
1. 保留训练时非标准 RoPE：
   - 频率 repeat_interleave(2)
   - rotate_half 使用前后半 chunk
2. 只缓存原始 2 个 KV 头，不缓存展开后的 14 个头。
3. KV Cache 按最大上下文一次预分配，逐 Token 写入，避免每步 torch.cat。
4. 首次输入做一次 Prefill；后续生成每步只输入 1 个 Token。
5. 每生成一个 Token 立即刷新到终端。
6. 统计 Prefill、首 Token 延迟、生成速度和总速度。
6. Windows/Linux 跨平台。
7. 启动时自动验证：
   - 原始前向与 KV Cache Prefill 的下一 Token 是否一致；
   - KV Cache 连续解码与完整重算的前三个贪心 Token 是否一致。
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import math
import os
import platform
import random
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR.parent
MODEL_PY = BASE_DIR / "model_c3.py"
TOKENIZER_FILE = MODEL_ROOT / "tokenizer_frozen" / "tokenizer.model"

DEFAULT_MODEL_CANDIDATES = (
    MODEL_ROOT / "checkpoint" / "checkpoint_tokens_747110400.pt",
    BASE_DIR / "checkpoint_tokens_747110400.pt",
)

HARD_MAX_NEW_TOKENS = 512
DEFAULT_TEST_PROMPT = "中国的首都是"


def peak_rss_mb() -> float:
    """返回当前进程峰值物理内存，Windows/Linux 均支持。"""
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()

            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return counters.PeakWorkingSetSize / 1024.0 / 1024.0
        except Exception:
            return float("nan")

        return float("nan")

    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return value / 1024.0 / 1024.0
        return value / 1024.0
    except Exception:
        return float("nan")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_model_path(explicit_path: str | None) -> Path:
    """解析未量化权重位置。"""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"指定的模型文件不存在：{path}")
        return path

    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path.resolve()

    checked = "\n".join(f"  - {path}" for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "没有找到未量化权重。已检查：\n"
        f"{checked}\n"
        "可用 --model-file 指定 checkpoint.pt 或 BF16 推理包。"
    )


def extract_unquantized_package(package: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    """
    支持两类文件：
    1. 原始训练 checkpoint.pt：package['model']
    2. 精简未量化推理包：package['state_dict']
    """
    if not isinstance(package, dict):
        raise TypeError(f"权重顶层必须为 dict，实际为 {type(package)}")

    config = package.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("权重中缺少 config 字典")

    if isinstance(package.get("model"), dict):
        return dict(config), package["model"], "原始训练 checkpoint.pt"

    if isinstance(package.get("state_dict"), dict):
        source = str(package.get("format", "未量化推理包"))
        return dict(config), package["state_dict"], source

    raise RuntimeError(
        "权重中既没有 model state_dict，也没有 state_dict"
    )


def state_dict_dtype_summary(state_dict: dict[str, torch.Tensor]) -> str:
    counts: dict[str, int] = {}
    for value in state_dict.values():
        if torch.is_tensor(value):
            key = str(value.dtype)
            counts[key] = counts.get(key, 0) + value.numel()

    if not counts:
        return "未知"

    return ", ".join(
        f"{dtype}={count:,}"
        for dtype, count in sorted(counts.items())
    )


def load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "zh200m_kvcache_native_runtime",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入模型代码：{path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["zh200m_kvcache_native_runtime"] = module
    spec.loader.exec_module(module)
    return module



def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"未知模型输出类型：{type(output)}")


def native_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    保留本模型训练时的非标准 rotate_half：
    把最后一维切成前后两半，再拼接 (-后半, 前半)。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_native_rope(
    *,
    start_pos: int,
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    保留训练时的频率布局：
    freqs 的每个频率 repeat_interleave(2)，
    但 rotate_half 仍使用前后半 chunk。
    """
    position = torch.arange(
        start_pos,
        start_pos + seq_len,
        device=device,
        dtype=torch.float32,
    )
    inv_freq = 1.0 / (
        float(theta)
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / float(head_dim)
        )
    )

    freqs = torch.outer(position, inv_freq)
    angles = freqs.repeat_interleave(2, dim=-1)

    # [1, T, 1, D]，与 [B, T, H, D] 广播。
    cos = angles.cos().to(dtype=dtype)[None, :, None, :]
    sin = angles.sin().to(dtype=dtype)[None, :, None, :]
    return cos, sin


def apply_native_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return x * cos + native_rotate_half(x) * sin


def kv_attention_forward(
    self: nn.Module,
    x: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    用预分配 KV Cache 替换每层原 Attention.forward。

    - Cache 为空/长度为 0：允许一次 Prefill T>=1；
    - Cache 已有历史：只允许单 Token Decode；
    - 仅保存原始 GQA 的 2 个 KV 头；
    - 保留训练时的非标准 RoPE。
    """
    del args, kwargs

    if x.ndim != 3:
        raise RuntimeError(f"Attention 输入应为 [B,T,C]，实际为 {tuple(x.shape)}")

    batch_size, seq_len, _ = x.shape
    n_head = int(self._kv_n_head)
    n_kv_head = int(self._kv_n_kv_head)
    head_dim = int(self._kv_head_dim)
    max_seq_len = int(self._kv_max_seq_len)
    rope_theta = float(self._kv_rope_theta)
    past_len = int(self._kv_cache_len)

    if past_len > 0 and seq_len != 1:
        raise RuntimeError(
            "KV Cache 已存在时只允许输入 1 个 Token；"
            f"当前 past_len={past_len}, seq_len={seq_len}"
        )

    new_len = past_len + seq_len
    if new_len > max_seq_len:
        raise RuntimeError(
            f"KV Cache 超过上下文上限：{past_len}+{seq_len}>{max_seq_len}"
        )

    q = self.q_proj(x).view(batch_size, seq_len, n_head, head_dim)
    k_new = self.k_proj(x).view(batch_size, seq_len, n_kv_head, head_dim)
    v_new = self.v_proj(x).view(batch_size, seq_len, n_kv_head, head_dim)

    cos, sin = build_native_rope(
        start_pos=past_len,
        seq_len=seq_len,
        head_dim=head_dim,
        theta=rope_theta,
        device=q.device,
        dtype=q.dtype,
    )
    q = apply_native_rope(q, cos, sin)
    k_new = apply_native_rope(k_new, cos, sin)

    cache_k = self._kv_cache_k
    cache_v = self._kv_cache_v
    need_allocate = (
        cache_k is None
        or cache_v is None
        or int(cache_k.shape[0]) != batch_size
        or cache_k.device != k_new.device
        or cache_k.dtype != k_new.dtype
    )
    if need_allocate:
        cache_shape = (
            batch_size,
            max_seq_len,
            n_kv_head,
            head_dim,
        )
        cache_k = torch.empty(
            cache_shape,
            device=k_new.device,
            dtype=k_new.dtype,
        )
        cache_v = torch.empty(
            cache_shape,
            device=v_new.device,
            dtype=v_new.dtype,
        )
        self._kv_cache_k = cache_k
        self._kv_cache_v = cache_v
        self._kv_cache_batch = batch_size
        if past_len != 0:
            raise RuntimeError("KV Cache 在非零历史位置意外重新分配")

    cache_k[:, past_len:new_len].copy_(k_new)
    cache_v[:, past_len:new_len].copy_(v_new)
    self._kv_cache_len = new_len

    # 只缓存 2 个 KV 头；计算注意力时临时映射到 14 个 Q 头。
    k_all = cache_k[:, :new_len]
    v_all = cache_v[:, :new_len]

    repeat_factor = n_head // n_kv_head
    if n_head % n_kv_head != 0:
        raise RuntimeError(
            f"n_head={n_head} 不能整除 n_kv_head={n_kv_head}"
        )

    k_attn = k_all.repeat_interleave(repeat_factor, dim=2)
    v_attn = v_all.repeat_interleave(repeat_factor, dim=2)

    q = q.transpose(1, 2)
    k_attn = k_attn.transpose(1, 2)
    v_attn = v_attn.transpose(1, 2)

    attention = F.scaled_dot_product_attention(
        q,
        k_attn,
        v_attn,
        dropout_p=0.0,
        is_causal=(past_len == 0 and seq_len > 1),
    )

    attention = (
        attention.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, n_head * head_dim)
    )
    return self.o_proj(attention)


class KVCacheController:
    def __init__(self, model: nn.Module, config: dict[str, Any]):
        self.model = model
        self.n_head = int(config["n_head"])
        self.n_kv_head = int(config["n_kv_head"])
        self.dim = int(config["dim"])
        self.head_dim = self.dim // self.n_head
        self.max_seq_len = int(config["max_seq_len"])
        self.rope_theta = float(config.get("rope_theta", 10000.0))

        layers = getattr(model, "layers", None)
        if layers is None:
            raise RuntimeError("模型中找不到 layers")

        self.attentions: list[nn.Module] = []
        self.original_forwards: list[Any] = []

        for layer_index, layer in enumerate(layers):
            attn = getattr(layer, "attn", None)
            if attn is None:
                raise RuntimeError(f"第 {layer_index} 层找不到 attn")
            for required in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if not hasattr(attn, required):
                    raise RuntimeError(
                        f"第 {layer_index} 层 Attention 缺少 {required}"
                    )
            self.attentions.append(attn)
            self.original_forwards.append(attn.forward)

    def install(self) -> None:
        for layer_index, attn in enumerate(self.attentions):
            attn._kv_layer_index = layer_index
            attn._kv_n_head = self.n_head
            attn._kv_n_kv_head = self.n_kv_head
            attn._kv_head_dim = self.head_dim
            attn._kv_max_seq_len = self.max_seq_len
            attn._kv_rope_theta = self.rope_theta
            attn._kv_cache_k = None
            attn._kv_cache_v = None
            attn._kv_cache_len = 0
            attn._kv_cache_batch = 0
            attn.forward = types.MethodType(kv_attention_forward, attn)

    def uninstall(self) -> None:
        for attn, original in zip(self.attentions, self.original_forwards):
            attn.forward = original
            for name in (
                "_kv_cache_k",
                "_kv_cache_v",
                "_kv_cache_len",
                "_kv_cache_batch",
            ):
                if hasattr(attn, name):
                    delattr(attn, name)

    def reset(self, *, release_memory: bool = False) -> None:
        for attn in self.attentions:
            attn._kv_cache_len = 0
            if release_memory:
                attn._kv_cache_k = None
                attn._kv_cache_v = None
                attn._kv_cache_batch = 0

    @property
    def cache_length(self) -> int:
        if not self.attentions:
            return 0
        return int(self.attentions[0]._kv_cache_len)

    @property
    def allocated_cache_mb(self) -> float:
        total_bytes = 0
        for attn in self.attentions:
            for tensor in (attn._kv_cache_k, attn._kv_cache_v):
                if torch.is_tensor(tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / 1024.0 / 1024.0

    def assert_consistent(self) -> None:
        lengths = [int(attn._kv_cache_len) for attn in self.attentions]
        if len(set(lengths)) != 1:
            raise RuntimeError(f"各层 KV Cache 长度不一致：{lengths}")

def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
    window: int = 128,
) -> torch.Tensor:
    if penalty <= 1.0:
        return logits

    result = logits.clone()
    for token_id in set(token_ids[-window:]):
        value = result[token_id]
        if value < 0:
            result[token_id] = value * penalty
        else:
            result[token_id] = value / penalty
    return result


def sample_next_token(
    logits: torch.Tensor,
    all_token_ids: list[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    logits = apply_repetition_penalty(
        logits.float(),
        all_token_ids,
        repetition_penalty,
    )

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.numel())
        threshold = torch.topk(logits, top_k).values[-1]
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits[remove] = float("-inf")

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits).item())

    return int(torch.multinomial(probabilities, num_samples=1).item())


class CachedGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        config: dict[str, Any],
        cache: KVCacheController,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.cache = cache
        self.max_seq_len = int(config["max_seq_len"])

        self.token_ids: list[int] = []
        self.next_logits: torch.Tensor | None = None

    def clear(self) -> None:
        self.token_ids = []
        self.next_logits = None
        self.cache.reset()

    def _forward(self, ids: list[int]) -> torch.Tensor:
        if not ids:
            raise ValueError("不能向模型输入空 Token")

        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            output = self.model(input_ids)
        return extract_logits(output)

    def rebuild(self, token_ids: list[int]) -> None:
        self.cache.reset()
        self.token_ids = list(token_ids[-self.max_seq_len :])

        if not self.token_ids:
            self.next_logits = None
            return

        logits = self._forward(self.token_ids)
        self.next_logits = logits[0, -1].float()
        self.cache.assert_consistent()

        if self.cache.cache_length != len(self.token_ids):
            raise RuntimeError(
                f"Prefill 后缓存长度 {self.cache.cache_length} "
                f"不等于 Token 长度 {len(self.token_ids)}"
            )

    def _make_room_for_one(self) -> None:
        if len(self.token_ids) < self.max_seq_len:
            return

        # 与旧脚本“只保留最后 max_seq_len 个 Token”的行为一致。
        # 为即将加入的一个新 Token 预留位置。
        keep = self.token_ids[-(self.max_seq_len - 1) :]
        self.rebuild(keep)

    def append_one(self, token_id: int) -> None:
        self._make_room_for_one()

        logits = self._forward([int(token_id)])
        self.token_ids.append(int(token_id))
        self.next_logits = logits[0, -1].float()
        self.cache.assert_consistent()

        if self.cache.cache_length != len(self.token_ids):
            raise RuntimeError(
                f"Decode 后缓存长度 {self.cache.cache_length} "
                f"不等于 Token 长度 {len(self.token_ids)}"
            )

    def append_tokens(self, new_ids: list[int]) -> None:
        if not new_ids:
            return

        if not self.token_ids:
            self.rebuild(new_ids)
            return

        # 缓存存在时逐 Token 追加，避免带 past 的多 Token mask 歧义。
        for token_id in new_ids:
            self.append_one(int(token_id))

    def append_text(self, text: str) -> int:
        ids = self._encode(text)
        if not ids:
            raise RuntimeError("输入文字编码后为空")
        self.append_tokens(ids)
        return len(ids)

    def _encode(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(text, out_type=int)
        except (RuntimeError, TypeError):
            # sentencepiece pybind 偶发兼容问题：退回底层稳定 API
            return list(self.tokenizer.EncodeAsIds(text))

    def generate_stream(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        request_started: float | None = None,
    ) -> tuple[str, int, float, float, float]:
        if self.next_logits is None or not self.token_ids:
            raise RuntimeError("当前上下文为空，请先输入文字")

        max_new_tokens = max(
            1,
            min(int(max_new_tokens), HARD_MAX_NEW_TOKENS),
        )

        generated: list[int] = []
        streamed_text = ""
        eos_id = int(self.tokenizer.eos_id())

        generation_started = time.perf_counter()
        if request_started is None:
            request_started = generation_started
        first_visible_at: float | None = None

        for _ in range(max_new_tokens):
            assert self.next_logits is not None

            next_id = sample_next_token(
                self.next_logits,
                self.token_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if eos_id >= 0 and next_id == eos_id:
                break

            generated.append(next_id)
            current_text = self.tokenizer.decode(generated)
            if current_text.startswith(streamed_text):
                delta = current_text[len(streamed_text):]
            else:
                delta = self.tokenizer.decode([next_id])

            if delta:
                print(delta, end="", flush=True)
                if first_visible_at is None:
                    first_visible_at = time.perf_counter()
            streamed_text = current_text

            # 每一步只喂回刚生成的 1 个 Token。
            self.append_one(next_id)

        finished = time.perf_counter()
        print(flush=True)

        generation_elapsed = finished - generation_started
        total_elapsed = finished - request_started
        if first_visible_at is None:
            ttft = total_elapsed
        else:
            ttft = first_visible_at - request_started

        return (
            self.tokenizer.decode(generated),
            len(generated),
            generation_elapsed,
            ttft,
            total_elapsed,
        )


def validate_kv_cache(
    model: nn.Module,
    tokenizer: spm.SentencePieceProcessor,
    controller: KVCacheController,
) -> None:
    test_ids = tokenizer.encode(DEFAULT_TEST_PROMPT, out_type=int)
    if not test_ids:
        raise RuntimeError("KV Cache 自检提示词编码为空")

    input_ids = torch.tensor([test_ids], dtype=torch.long)

    # 1. 先用原始 Attention 前向。
    with torch.inference_mode():
        original_logits = extract_logits(model(input_ids))[0, -1].float()

    original_next = int(torch.argmax(original_logits).item())

    # 2. 安装 KV Cache 后做同一 Prefill。
    controller.install()
    controller.reset()

    with torch.inference_mode():
        cached_logits = extract_logits(model(input_ids))[0, -1].float()

    cached_next = int(torch.argmax(cached_logits).item())
    max_diff = float((original_logits - cached_logits).abs().max().item())
    mean_diff = float((original_logits - cached_logits).abs().mean().item())

    print("===== KV Cache 前向等价性检查 =====")
    print("测试提示词：", DEFAULT_TEST_PROMPT)
    print(
        "原始下一Token：",
        original_next,
        repr(tokenizer.id_to_piece(original_next)),
    )
    print(
        "缓存下一Token：",
        cached_next,
        repr(tokenizer.id_to_piece(cached_next)),
    )
    print("最大Logit差：", f"{max_diff:.8f}")
    print("平均Logit差：", f"{mean_diff:.8f}")

    if original_next != cached_next:
        controller.uninstall()
        raise RuntimeError(
            "KV Cache Prefill 与原模型下一 Token 不一致，已拒绝启动"
        )

    if max_diff > 1e-3:
        print(
            "警告：Logit 差异大于 1e-3，但下一 Token 一致；"
            "继续执行三步缓存一致性检查。"
        )

    # 3. 比较三步贪心：缓存连续 Decode vs 每步完整重算。
    base_ids = list(test_ids)

    # 缓存路线。
    controller.reset()
    with torch.inference_mode():
        cache_logits = extract_logits(
            model(torch.tensor([base_ids], dtype=torch.long))
        )[0, -1].float()

    cached_generated: list[int] = []
    for _ in range(3):
        next_id = int(torch.argmax(cache_logits).item())
        cached_generated.append(next_id)

        with torch.inference_mode():
            cache_logits = extract_logits(
                model(torch.tensor([[next_id]], dtype=torch.long))
            )[0, -1].float()

    # 完整重算路线。
    full_ids = list(base_ids)
    full_generated: list[int] = []
    for _ in range(3):
        controller.reset()
        with torch.inference_mode():
            full_logits = extract_logits(
                model(torch.tensor([full_ids], dtype=torch.long))
            )[0, -1].float()

        next_id = int(torch.argmax(full_logits).item())
        full_generated.append(next_id)
        full_ids.append(next_id)

    print("缓存连续3 Token：", cached_generated)
    print("完整重算3 Token：", full_generated)

    if cached_generated != full_generated:
        controller.uninstall()
        raise RuntimeError(
            "KV Cache 三步生成与完整重算不一致，已拒绝启动"
        )

    controller.reset()
    print("KV Cache 自检：通过")
    print()


def print_help() -> None:
    print(
        """
命令：

  /continue
      不添加新文字，基于当前 KV Cache 再生成，最多 512 Token。

  /clear
      清空全文和全部 20 层 KV Cache。

  /undo
      撤销上一轮，并从恢复后的 Token 重建 KV Cache。

  /show
      显示当前完整文本。

  /status
      显示 Token 数、缓存长度、生成参数与内存。

  /max 1到512
      设置每轮生成 Token 数，硬上限固定为 512。

  /temp 数字
      设置温度；0 为贪心。

  /topk 数字
      设置 Top-K；0 为关闭。

  /topp 数字
      设置 Top-P，范围 (0,1]。

  /repeat 数字
      设置重复惩罚，必须 >= 1.0。

  /save
      自动命名保存当前全文。

  /save 文件名
      保存为指定 txt 文件。

  /exit
      退出。
"""
    )



def prompt_generation_settings(
    default_max_new_tokens: int,
    default_temperature: float,
) -> tuple[int, float]:
    """在加载模型之前，让用户填写本次启动的生成参数。"""
    print("=" * 72)
    print("启动前生成参数设置")
    print("直接按回车使用括号中的默认值；此时模型尚未加载。")
    print("温度 0 = 贪心；常用采样温度为 0.6～0.9。")
    print("=" * 72)

    while True:
        raw = input(
            f"最大输出 Token [默认 {default_max_new_tokens}，范围 1～{HARD_MAX_NEW_TOKENS}]："
        ).strip()
        if not raw:
            max_new_tokens = int(default_max_new_tokens)
            break
        try:
            max_new_tokens = int(raw)
            if not 1 <= max_new_tokens <= HARD_MAX_NEW_TOKENS:
                raise ValueError
            break
        except ValueError:
            print(f"请输入 1～{HARD_MAX_NEW_TOKENS} 之间的整数。")

    while True:
        raw = input(
            f"温度 [默认 {default_temperature:g}，范围 0～5]："
        ).strip()
        if not raw:
            temperature = float(default_temperature)
            break
        try:
            temperature = float(raw)
            if not 0.0 <= temperature <= 5.0:
                raise ValueError
            break
        except ValueError:
            print("请输入 0～5 之间的数字；0 表示贪心生成。")

    print()
    print("本次启动参数：")
    print("  最大输出 Token：", max_new_tokens)
    print("  温度：", temperature)
    print("参数已确认，开始加载模型……")
    print()
    return max_new_tokens, temperature

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zh200M FP32 未量化 + KV Cache 连续实时续写"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help=(
            "未量化权重路径。可以是原始 checkpoint.pt，"
            "也可以是 zh200m-native-rope-bf16.pt"
        ),
    )
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="每轮生成 Token 数，硬上限为 512",
    )
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-kv-self-test",
        action="store_true",
        help="跳过启动时 KV Cache 等价性测试，不建议使用",
    )
    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("--threads 必须大于 0")

    try:
        args.max_new_tokens, args.temperature = prompt_generation_settings(
            default_max_new_tokens=max(
                1,
                min(int(args.max_new_tokens), HARD_MAX_NEW_TOKENS),
            ),
            default_temperature=max(0.0, float(args.temperature)),
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消，模型没有加载。")
        return

    for path in (MODEL_PY, TOKENIZER_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    model_file = resolve_model_path(args.model_file)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("正在加载未量化模型……")
    print("操作系统：", platform.platform())
    print("Python：", sys.version.split()[0])
    print("PyTorch：", torch.__version__)
    print("权重位置：", model_file)
    print("运行精度：FP32（未量化）")
    print(flush=True)

    load_started = time.perf_counter()
    package = safe_torch_load(model_file)
    config, state_dict, package_source = extract_unquantized_package(package)

    print("权重类型：", package_source)
    print("存储权重 dtype：", state_dict_dtype_summary(state_dict))
    print("正在构造原生 FP32 模型……", flush=True)

    native_module = load_python_module(MODEL_PY)
    model = native_module.build_model(config)
    model = model.float().cpu().eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("模型参数量：", f"{parameter_count:,}")
    print("正在严格加载未量化权重并转换为 FP32……", flush=True)

    # 模型参数当前是 FP32。若 state_dict 为 BF16，load_state_dict 会复制并转成 FP32。
    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )
    model = model.float().cpu().eval()

    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(TOKENIZER_FILE)):
        raise RuntimeError("Tokenizer 加载失败")

    # 原始 checkpoint 还含有 optimizer 等内容，加载完成后立即释放。
    del state_dict
    del package
    gc.collect()

    controller = KVCacheController(model, config)

    if args.skip_kv_self_test:
        controller.install()
        controller.reset()
        print("警告：已跳过 KV Cache 等价性自检")
    else:
        validate_kv_cache(model, tokenizer, controller)

    generator = CachedGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        cache=controller,
    )

    max_new_tokens = max(
        1,
        min(int(args.max_new_tokens), HARD_MAX_NEW_TOKENS),
    )
    temperature = max(0.0, float(args.temperature))
    top_k = max(0, int(args.top_k))
    top_p = float(args.top_p)
    repetition_penalty = max(1.0, float(args.repetition_penalty))

    history: list[list[int]] = []
    load_seconds = time.perf_counter() - load_started

    print("=" * 72)
    print("Zh200M C3 747M FP32 + 预分配 KV Cache 实时续写")
    print("=" * 72)
    print("严格加载结果：", load_result)
    print("运行精度：FP32，未使用 INT8 量化")
    print("模型加载与自检：", f"{load_seconds:.3f}s")
    print("层数：", len(controller.attentions))
    print(
        "GQA：",
        f"{controller.n_head} 个 Q 头 / "
        f"{controller.n_kv_head} 个 KV 头",
    )
    print("每头维度：", controller.head_dim)
    print("每轮输出硬上限：", HARD_MAX_NEW_TOKENS, "Token")
    print("输入 /help 查看命令")
    print()

    while True:
        try:
            user_input = input("你输入> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        command = user_input.strip()
        if not command:
            print("请输入文字，或输入 /continue。")
            continue

        if command in ("/exit", "/quit", "/q"):
            print("已退出。")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/clear":
            history.append(list(generator.token_ids))
            generator.clear()
            print("全文和 KV Cache 已清空。")
            continue

        if command == "/undo":
            if not history:
                print("没有可以撤销的上一轮。")
            else:
                restore = history.pop()
                print("正在重建撤销后的 KV Cache……", flush=True)
                generator.rebuild(restore)
                print("已撤销上一轮。")
            continue

        if command == "/show":
            print()
            print("===== 当前完整文本 =====")
            if generator.token_ids:
                print(tokenizer.decode(generator.token_ids))
            else:
                print("（当前为空）")
            print("========================")
            print()
            continue

        if command == "/status":
            rss = peak_rss_mb()
            print()
            print("===== 当前状态 =====")
            print("运行精度：FP32 未量化")
            print("全文 Token：", len(generator.token_ids))
            print("KV Cache 长度：", controller.cache_length)
            print("KV Cache 已分配：", f"{controller.allocated_cache_mb:.1f} MiB")
            print("上下文上限：", generator.max_seq_len)
            print("每轮生成：", max_new_tokens)
            print("硬性上限：", HARD_MAX_NEW_TOKENS)
            print("温度：", temperature)
            print("Top-K：", top_k)
            print("Top-P：", top_p)
            print("重复惩罚：", repetition_penalty)
            if rss == rss:
                print("进程峰值内存：", f"{rss:.1f} MiB")
            print("====================")
            print()
            continue

        if command.startswith("/max "):
            try:
                value = int(command.split(maxsplit=1)[1])
                if not 1 <= value <= HARD_MAX_NEW_TOKENS:
                    raise ValueError
                max_new_tokens = value
                print("每轮生成 Token：", max_new_tokens)
            except ValueError:
                print("格式错误：请输入 1～512，例如 /max 64")
            continue

        if command.startswith("/temp "):
            try:
                value = float(command.split(maxsplit=1)[1])
                if value < 0:
                    raise ValueError
                temperature = value
                print("温度：", temperature)
            except ValueError:
                print("格式错误，例如 /temp 0.75")
            continue

        if command.startswith("/topk "):
            try:
                value = int(command.split(maxsplit=1)[1])
                if value < 0:
                    raise ValueError
                top_k = value
                print("Top-K：", top_k)
            except ValueError:
                print("格式错误，例如 /topk 40")
            continue

        if command.startswith("/topp "):
            try:
                value = float(command.split(maxsplit=1)[1])
                if not 0.0 < value <= 1.0:
                    raise ValueError
                top_p = value
                print("Top-P：", top_p)
            except ValueError:
                print("格式错误，例如 /topp 0.9")
            continue

        if command.startswith("/repeat "):
            try:
                value = float(command.split(maxsplit=1)[1])
                if value < 1.0:
                    raise ValueError
                repetition_penalty = value
                print("重复惩罚：", repetition_penalty)
            except ValueError:
                print("格式错误，例如 /repeat 1.08")
            continue

        if command == "/save" or command.startswith("/save "):
            if not generator.token_ids:
                print("当前没有文本可保存。")
                continue

            if command == "/save":
                filename = (
                    "FP32_KV续写_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".txt"
                )
            else:
                filename = command.split(maxsplit=1)[1].strip()
                if not filename.lower().endswith(".txt"):
                    filename += ".txt"

            save_path = BASE_DIR / filename
            save_path.write_text(
                tokenizer.decode(generator.token_ids),
                encoding="utf-8",
            )
            print("已保存：", save_path)
            continue

        is_continue = command == "/continue"
        if is_continue and not generator.token_ids:
            print("当前上下文为空，请先输入文字。")
            continue

        history.append(list(generator.token_ids))

        try:
            request_started = time.perf_counter()
            input_count = 0
            prefill_elapsed = 0.0

            if not is_continue:
                prefill_started = time.perf_counter()
                input_count = generator.append_text(user_input)
                prefill_elapsed = time.perf_counter() - prefill_started
                prefill_speed = (
                    input_count / prefill_elapsed
                    if prefill_elapsed > 0
                    else 0.0
                )
                print(
                    f"[输入 {input_count} Token | "
                    f"Prefill {prefill_elapsed:.3f}s | "
                    f"{prefill_speed:.2f} Token/s | "
                    f"KV={controller.cache_length}]"
                )

            print("模型续写> ", end="", flush=True)
            (
                generated_text,
                count,
                generation_elapsed,
                ttft,
                total_elapsed,
            ) = generator.generate_stream(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                request_started=request_started,
            )

            decode_speed = (
                count / generation_elapsed
                if generation_elapsed > 0
                else 0.0
            )
            total_speed = (
                count / total_elapsed
                if total_elapsed > 0
                else 0.0
            )
            print(
                f"[输出 {count} Token | "
                f"首Token {ttft:.3f}s | "
                f"生成 {generation_elapsed:.3f}s | "
                f"Decode {decode_speed:.2f} Token/s | "
                f"总速度 {total_speed:.2f} Token/s | "
                f"KV={controller.cache_length} | "
                f"KV内存={controller.allocated_cache_mb:.1f} MiB]"
            )

            if not generated_text:
                print("模型本轮没有生成可显示文本。")

            rss = peak_rss_mb()
            if rss == rss:
                print(f"[进程峰值内存 {rss:.1f} MiB]")
            print()

        except KeyboardInterrupt:
            restore = history.pop()
            print("\n本轮已取消，正在恢复 KV Cache……")
            generator.rebuild(restore)
            print("已恢复。")
        except Exception:
            restore = history.pop()
            generator.rebuild(restore)
            raise


if __name__ == "__main__":
    main()
