#!/usr/bin/env python3
"""Auditable single-A800 pretraining for Zh200M C3.

Core invariants:
- random initialization unless resuming a checkpoint produced by this script
- true micro-batch tensor shape is enforced
- global token counters come from input_ids.numel(), never from config alone
- data cursor movement, optimizer steps, and scheduler steps are asserted
- sequence-boundary labels are ignored instead of crossing shuffled 2048 blocks
- FP32 parameters/Adam states with BF16 autocast on A800
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model_c3 import build_model


EXPECTED_PARAMETER_COUNT = 196_153_216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--resume", default=None, help="exact C3 checkpoint file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument(
        "--max-val-seqs", type=int, default=None,
        help="cap baseline and interval validation sequences",
    )
    parser.add_argument(
        "--final-max-val-seqs", type=int, default=None,
        help="cap final validation; default is the full validation set",
    )
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--no-final-checkpoint", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--stop-new-step-after-seconds", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class PackedSequenceDataset:
    def __init__(
        self,
        path: Path,
        seq_len: int,
        vocab_size: int,
        expected_tokens: int | None = None,
        expected_sequences: int | None = None,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        size_bytes = path.stat().st_size
        if size_bytes % 4 != 0:
            raise RuntimeError(f"{path}: byte size {size_bytes} is not divisible by uint32 size")
        self.path = path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.token_count = int(self.tokens.size)
        self.sequence_count = self.token_count // seq_len
        self.remainder_tokens = self.token_count % seq_len
        self.rows = self.tokens[: self.sequence_count * seq_len].reshape(self.sequence_count, seq_len)
        if expected_tokens is not None and self.token_count != expected_tokens:
            raise RuntimeError(f"{path}: tokens={self.token_count} != expected={expected_tokens}")
        if expected_sequences is not None and self.sequence_count != expected_sequences:
            raise RuntimeError(
                f"{path}: full_sequences={self.sequence_count} != expected={expected_sequences}"
            )
        self._sample_validate()

    def _sample_validate(self) -> None:
        if self.sequence_count == 0:
            raise RuntimeError(f"{self.path}: no full sequences")
        probe = sorted({0, self.sequence_count - 1, self.sequence_count // 2})
        sample = np.asarray(self.rows[probe], dtype=np.uint32)
        max_token = int(sample.max())
        if max_token >= self.vocab_size:
            raise RuntimeError(
                f"{self.path}: sampled token id {max_token} >= vocab_size {self.vocab_size}"
            )

    def get_batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.sequence_count):
            raise IndexError("sequence index out of range")
        if indices.size > 0 and (indices.size == 1 or np.all(np.diff(indices) == 1)):
            start = int(indices[0])
            array = np.asarray(self.rows[start : start + indices.size], dtype=np.int64)
        else:
            array = np.asarray(self.rows[indices], dtype=np.int64)
        input_ids = torch.from_numpy(array)
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        # labels[:, -1] stays -100: blocks were globally shuffled, so crossing
        # from one 2048 block to the next would train a false transition.
        return input_ids, labels


class DeterministicEpochSampler:
    """Deterministic, resumable sampler with storage-friendly locality.

    The released train.bin is already globally deterministically shuffled. Epoch 0
    therefore reads it sequentially for maximum throughput. Later epochs shuffle
    large blocks while preserving contiguous reads inside each block.
    """

    def __init__(
        self,
        sequence_count: int,
        seed: int,
        epoch: int = 0,
        position: int = 0,
        block_size: int = 1024,
    ):
        if sequence_count <= 0:
            raise ValueError("sequence_count must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.sequence_count = sequence_count
        self.seed = seed
        self.block_size = block_size
        self.epoch = epoch
        self.position = position
        self.total_drawn = 0
        self._permutation = self._make_permutation(epoch)
        if not 0 <= position <= sequence_count:
            raise ValueError("invalid sampler position")

    def _make_permutation(self, epoch: int) -> np.ndarray:
        if epoch == 0:
            return np.arange(self.sequence_count, dtype=np.int64)
        starts = np.arange(0, self.sequence_count, self.block_size, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(self.seed + epoch))
        rng.shuffle(starts)
        pieces = [
            np.arange(start, min(start + self.block_size, self.sequence_count), dtype=np.int64)
            for start in starts
        ]
        return np.concatenate(pieces)

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._permutation = self._make_permutation(self.epoch)

    def next_indices(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            if self.position == self.sequence_count:
                self._advance_epoch()
            available = self.sequence_count - self.position
            take = min(remaining, available)
            pieces.append(self._permutation[self.position : self.position + take])
            self.position += take
            self.total_drawn += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def state_dict(self) -> dict[str, int]:
        return {
            "sequence_count": self.sequence_count,
            "seed": self.seed,
            "block_size": self.block_size,
            "epoch": self.epoch,
            "position": self.position,
            "total_drawn": self.total_drawn,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["sequence_count"]) != self.sequence_count:
            raise RuntimeError("sampler sequence_count mismatch")
        if int(state["seed"]) != self.seed:
            raise RuntimeError("sampler seed mismatch")
        if int(state.get("block_size", self.block_size)) != self.block_size:
            raise RuntimeError("sampler block_size mismatch")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.total_drawn = int(state["total_drawn"])
        self._permutation = self._make_permutation(self.epoch)


@dataclass
class TokenCosineScheduler:
    optimizer: torch.optim.Optimizer
    peak_lr: float
    min_lr_ratio: float
    warmup_steps: int
    total_steps: int
    completed_steps: int = 0

    def lr_for_update(self, update_number: int) -> float:
        if update_number <= 0:
            return 0.0
        if update_number <= self.warmup_steps:
            return self.peak_lr * update_number / max(1, self.warmup_steps)
        progress = (update_number - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def prepare_update(self, update_number: int) -> float:
        if update_number != self.completed_steps + 1:
            raise RuntimeError(
                f"scheduler expected update {self.completed_steps + 1}, got {update_number}"
            )
        lr = self.lr_for_update(update_number)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def mark_completed(self, update_number: int) -> None:
        if update_number != self.completed_steps + 1:
            raise RuntimeError("scheduler completion order mismatch")
        self.completed_steps = update_number

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("peak_lr", "min_lr_ratio", "warmup_steps", "total_steps"):
            current = getattr(self, key)
            incoming = state[key]
            if float(current) != float(incoming):
                raise RuntimeError(f"scheduler {key} mismatch: {incoming} != {current}")
        self.completed_steps = int(state["completed_steps"])


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_optimizer(model: torch.nn.Module, cfg: dict, device: torch.device):
    kwargs = dict(
        params=model.parameters(),
        lr=float(cfg["peak_lr"]),
        betas=tuple(cfg.get("adam_betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        weight_decay=float(cfg["weight_decay"]),
    )
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(**kwargs, fused=True), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**kwargs), False


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: TokenCosineScheduler,
    sampler: DeterministicEpochSampler,
    counters: dict[str, int],
    cfg: dict,
    run_manifest: dict,
    saved_milestones: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format": "zh200m_c3_v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "counters": counters,
        "config": cfg,
        "run_manifest": run_manifest,
        "saved_milestones": saved_milestones,
        "rng": capture_rng_state(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: TokenCosineScheduler,
    sampler: DeterministicEpochSampler,
    expected_cfg: dict,
) -> tuple[dict[str, int], list[int]]:
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_data.get("format") != "zh200m_c3_v1":
        raise RuntimeError("refusing non-C3 checkpoint; old C2 checkpoint cannot initialize this run")
    for key in (
        "parameter_count",
        "vocab_size",
        "max_seq_len",
        "dim",
        "n_layer",
        "n_head",
        "n_kv_head",
        "ffn_dim",
        "rope_theta",
    ):
        if checkpoint_data["config"][key] != expected_cfg[key]:
            raise RuntimeError(f"checkpoint config mismatch for {key}")
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    parameter_device = next(model.parameters()).device
    for optimizer_state in optimizer.state.values():
        for key, value in list(optimizer_state.items()):
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(parameter_device)
    scaler.load_state_dict(checkpoint_data["scaler"])
    scheduler.load_state_dict(checkpoint_data["scheduler"])
    sampler.load_state_dict(checkpoint_data["sampler"])
    restore_rng_state(checkpoint_data["rng"])
    return checkpoint_data["counters"], list(checkpoint_data.get("saved_milestones", []))


def evaluate(
    model: torch.nn.Module,
    dataset: PackedSequenceDataset,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    max_sequences: int | None,
) -> tuple[float, int]:
    model.eval()
    limit = dataset.sequence_count if max_sequences is None else min(max_sequences, dataset.sequence_count)
    total_loss = 0.0
    total_targets = 0
    with torch.no_grad():
        for start in range(0, limit, batch_size):
            indices = np.arange(start, min(start + batch_size, limit), dtype=np.int64)
            input_ids, labels = dataset.get_batch(indices)
            input_ids = input_ids.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            with autocast_context(device, dtype_name):
                logits = model(input_ids)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            targets = int((labels != -100).sum().item())
            total_loss += float(loss_sum.item())
            total_targets += targets
    model.train()
    return total_loss / max(1, total_targets), total_targets


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if int(cfg["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("unexpected parameter count")
    if not cfg.get("random_init", False) or cfg.get("init_checkpoint") is not None:
        raise RuntimeError("C3 config must require random_init=true and init_checkpoint=null")

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported")

    micro_batch = int(args.micro_batch or cfg["micro_batch_size"])
    grad_accum = int(args.grad_accum or cfg["gradient_accumulation_steps"])
    seq_len = int(cfg["max_seq_len"])
    expected_step_sequences = micro_batch * grad_accum
    expected_step_tokens = expected_step_sequences * seq_len
    configured_tokens = int(cfg["effective_batch_tokens"])
    if expected_step_tokens != configured_tokens:
        raise RuntimeError(
            f"true batch tokens {expected_step_tokens} != configured {configured_tokens}; "
            "change micro/accum only while preserving the configured total"
        )

    release = Path(args.data_release).resolve()
    training_dir = release / "training_data"
    tokenizer_dir = release / "tokenizer_frozen"
    tokenizer_path = tokenizer_dir / "tokenizer.model"
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != cfg["tokenizer_sha256"]:
        raise RuntimeError(f"tokenizer SHA256 mismatch: {tokenizer_hash}")

    train_ds = PackedSequenceDataset(
        training_dir / "train.bin",
        seq_len,
        int(cfg["vocab_size"]),
        int(cfg["expected_train_tokens"]),
        int(cfg["expected_train_sequences"]),
    )
    val_ds = PackedSequenceDataset(
        training_dir / "validation.bin",
        seq_len,
        int(cfg["vocab_size"]),
        int(cfg["expected_validation_tokens"]),
        int(cfg["expected_validation_sequences"]),
    )

    model = build_model(cfg)  # Keep parameters FP32.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    optimizer, fused_optimizer = make_optimizer(model, cfg, device)
    schedule_total_steps = int(cfg["lr_schedule_total_steps"])
    scheduler = TokenCosineScheduler(
        optimizer=optimizer,
        peak_lr=float(cfg["peak_lr"]),
        min_lr_ratio=float(cfg["min_lr_ratio"]),
        warmup_steps=int(cfg["warmup_steps"]),
        total_steps=schedule_total_steps,
    )
    use_fp16 = args.dtype == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    sampler = DeterministicEpochSampler(
        train_ds.sequence_count, seed, block_size=int(cfg.get("shuffle_block_sequences", 1024))
    )

    max_steps = int(args.max_steps if args.max_steps is not None else cfg["max_optimizer_steps"])
    max_tokens = int(args.max_tokens if args.max_tokens is not None else cfg["max_train_tokens"])
    if args.audit_only and max_steps > 10:
        raise RuntimeError("audit-only mode is limited to at most 10 optimizer steps")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "step_audit.jsonl"
    text_log_path = out_dir / "train.log"

    counters = {
        "global_step": 0,
        "global_input_tokens": 0,
        "global_target_tokens": 0,
        "global_sequences": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
    }
    saved_milestones: list[int] = []

    run_manifest = {
        "format": "zh200m_c3_run_v1",
        "created_at_unix": time.time(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "data_release": str(release),
        "tokenizer_sha256": tokenizer_hash,
        "train_file_bytes": (training_dir / "train.bin").stat().st_size,
        "train_tokens": train_ds.token_count,
        "train_sequences": train_ds.sequence_count,
        "validation_tokens": val_ds.token_count,
        "validation_sequences": val_ds.sequence_count,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "true_input_tokens_per_step": expected_step_tokens,
        "true_sequences_per_step": expected_step_sequences,
        "dtype": args.dtype,
        "fp32_parameters": True,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "fused_optimizer": fused_optimizer,
        "sampler_strategy": "epoch0_sequential_then_deterministic_block_shuffle",
        "shuffle_block_sequences": int(cfg.get("shuffle_block_sequences", 1024)),
        "random_init": args.resume is None,
        "resume": str(Path(args.resume).resolve()) if args.resume else None,
        "model_parameter_count": model.count_parameters(),
        "custom_rope_preserved": True,
    }
    atomic_json_dump(run_manifest, out_dir / "run_manifest.json")

    if args.resume:
        counters, saved_milestones = load_checkpoint(
            Path(args.resume).resolve(), model, optimizer, scaler, scheduler, sampler, cfg
        )
        if scheduler.completed_steps != counters["global_step"]:
            raise RuntimeError("resume scheduler/global_step mismatch")
        if sampler.total_drawn != counters["global_sequences"]:
            raise RuntimeError("resume sampler/global_sequences mismatch")
    else:
        # Explicitly refuse any hidden initialization path.
        if counters["global_step"] != 0 or counters["global_input_tokens"] != 0:
            raise RuntimeError("random initialization counters are not zero")

    stop_requested = {"value": False}

    def signal_handler(signum, frame):  # noqa: ARG001
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    milestones = sorted({int(value) for value in cfg.get("checkpoint_token_milestones", [])})
    start_time = time.time()
    train_start_tokens = counters["global_input_tokens"]
    last_checkpoint_path: Path | None = None
    last_checkpoint_tokens = -1

    with text_log_path.open("a", encoding="utf-8", buffering=1) as text_log, audit_path.open(
        "a", encoding="utf-8", buffering=1
    ) as audit_log:
        header = (
            f"device={device} dtype={args.dtype} fp32_params=true micro={micro_batch} "
            f"accum={grad_accum} true_step_sequences={expected_step_sequences} "
            f"true_step_tokens={expected_step_tokens} train_sequences={train_ds.sequence_count} "
            f"val_sequences={val_ds.sequence_count}"
        )
        print(header, flush=True)
        text_log.write(header + "\n")

        baseline_limit = args.max_val_seqs if args.max_val_seqs is not None else int(
            cfg.get("baseline_validation_sequences", 16)
        )
        baseline_loss, baseline_targets = evaluate(
            model,
            val_ds,
            device,
            args.dtype,
            args.val_batch_size,
            baseline_limit,
        )
        baseline_msg = (
            f"[baseline] loss={baseline_loss:.6f} ppl={math.exp(min(50.0, baseline_loss)):.3f} "
            f"targets={baseline_targets}"
        )
        print(baseline_msg, flush=True)
        text_log.write(baseline_msg + "\n")

        model.train()
        stop_reason = "limit"
        while (
            counters["global_step"] < max_steps
            and counters["global_input_tokens"] < max_tokens
            and not stop_requested["value"]
        ):
            elapsed = time.time() - start_time
            if args.stop_new_step_after_seconds > 0 and elapsed >= args.stop_new_step_after_seconds:
                stop_reason = "stop_new_step_after_seconds"
                break
            next_update = counters["global_step"] + 1
            lr = scheduler.prepare_update(next_update)
            optimizer.zero_grad(set_to_none=True)

            before_sequences = sampler.total_drawn
            step_input_tokens = 0
            step_target_tokens = 0
            step_sequences = 0
            step_loss_sum = 0.0
            micro_records: list[dict[str, Any]] = []

            for micro_index in range(grad_accum):
                indices = sampler.next_indices(micro_batch)
                input_ids, labels = train_ds.get_batch(indices)
                expected_shape = (micro_batch, seq_len)
                if tuple(input_ids.shape) != expected_shape or tuple(labels.shape) != expected_shape:
                    raise RuntimeError(
                        f"micro shape mismatch: input={tuple(input_ids.shape)} labels={tuple(labels.shape)} "
                        f"expected={expected_shape}"
                    )
                micro_input_tokens = int(input_ids.numel())
                micro_target_tokens = int((labels != -100).sum().item())
                if micro_input_tokens != micro_batch * seq_len:
                    raise RuntimeError("micro input token assertion failed")
                if micro_target_tokens != micro_batch * (seq_len - 1):
                    raise RuntimeError("micro target token assertion failed")

                input_ids = input_ids.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                with autocast_context(device, args.dtype):
                    logits = model(input_ids)
                    raw_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                    scaled_loss = raw_loss / grad_accum
                scaler.scale(scaled_loss).backward()

                step_input_tokens += int(input_ids.numel())
                step_target_tokens += micro_target_tokens
                step_sequences += int(input_ids.shape[0])
                step_loss_sum += float(raw_loss.detach().item()) * micro_target_tokens
                micro_records.append(
                    {
                        "micro_index": micro_index,
                        "input_shape": list(input_ids.shape),
                        "input_tokens": int(input_ids.numel()),
                        "target_tokens": micro_target_tokens,
                        "first_sequence_index": int(indices[0]),
                        "last_sequence_index": int(indices[-1]),
                    }
                )
                del logits, raw_loss, scaled_loss, input_ids, labels

            if step_sequences != expected_step_sequences:
                raise RuntimeError(f"step sequences {step_sequences} != {expected_step_sequences}")
            if step_input_tokens != expected_step_tokens:
                raise RuntimeError(f"step tokens {step_input_tokens} != {expected_step_tokens}")
            if sampler.total_drawn - before_sequences != expected_step_sequences:
                raise RuntimeError("data cursor did not advance by the true sequence count")

            if use_fp16:
                scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["grad_clip"])
            )
            grad_norm = float(grad_norm_tensor.item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.mark_completed(next_update)

            counters["global_step"] = next_update
            counters["optimizer_steps"] += 1
            counters["scheduler_steps"] += 1
            counters["global_input_tokens"] += step_input_tokens
            counters["global_target_tokens"] += step_target_tokens
            counters["global_sequences"] += step_sequences

            if counters["optimizer_steps"] != counters["global_step"]:
                raise RuntimeError("optimizer step counter mismatch")
            if counters["scheduler_steps"] != counters["global_step"]:
                raise RuntimeError("scheduler step counter mismatch")
            if scheduler.completed_steps != counters["global_step"]:
                raise RuntimeError("scheduler internal counter mismatch")
            if sampler.total_drawn != counters["global_sequences"]:
                raise RuntimeError("sampler/global sequence counter mismatch")

            step_loss = step_loss_sum / max(1, step_target_tokens)
            elapsed = time.time() - start_time
            new_tokens = counters["global_input_tokens"] - train_start_tokens
            if device.type == "cuda" and (
                next_update <= 2 or next_update % args.log_interval == 0 or args.audit_only
            ):
                torch.cuda.synchronize(device)
                elapsed = time.time() - start_time
            tokens_per_second = new_tokens / max(1e-9, elapsed)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )

            audit_record = {
                "step": next_update,
                "loss": step_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "step_input_tokens": step_input_tokens,
                "step_target_tokens": step_target_tokens,
                "step_sequences": step_sequences,
                "global_input_tokens": counters["global_input_tokens"],
                "global_target_tokens": counters["global_target_tokens"],
                "global_sequences": counters["global_sequences"],
                "optimizer_steps": counters["optimizer_steps"],
                "scheduler_steps": counters["scheduler_steps"],
                "sampler_epoch": sampler.epoch,
                "sampler_position": sampler.position,
                "sampler_total_drawn": sampler.total_drawn,
                "elapsed_seconds": elapsed,
                "true_tokens_per_second": tokens_per_second,
                "peak_cuda_memory_bytes": peak_memory,
                "micro_batches": micro_records if args.audit_only else None,
            }
            audit_log.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

            if next_update <= 2 or next_update % args.log_interval == 0 or args.audit_only:
                msg = (
                    f"[step {next_update}] loss={step_loss:.4f} "
                    f"true_tokens={counters['global_input_tokens']:,} "
                    f"true_sequences={counters['global_sequences']:,} lr={lr:.3e} "
                    f"grad_norm={grad_norm:.3f} true_tok/s={tokens_per_second:.1f} "
                    f"sampler=epoch{sampler.epoch}:pos{sampler.position}"
                )
                print(msg, flush=True)
                text_log.write(msg + "\n")

            if args.eval_interval > 0 and next_update % args.eval_interval == 0:
                val_loss, val_targets = evaluate(
                    model,
                    val_ds,
                    device,
                    args.dtype,
                    args.val_batch_size,
                    args.max_val_seqs,
                )
                msg = (
                    f"[step {next_update}] val_loss={val_loss:.6f} "
                    f"ppl={math.exp(min(50.0, val_loss)):.3f} targets={val_targets}"
                )
                print(msg, flush=True)
                text_log.write(msg + "\n")

            if not args.audit_only:
                crossed = [
                    threshold
                    for threshold in milestones
                    if threshold not in saved_milestones
                    and counters["global_input_tokens"] >= threshold
                ]
                if crossed:
                    for threshold in crossed:
                        saved_milestones.append(threshold)
                    checkpoint_path = out_dir / (
                        f"checkpoint_tokens_{counters['global_input_tokens']}.pt"
                    )
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        sampler,
                        counters,
                        cfg,
                        run_manifest,
                        saved_milestones,
                    )
                    last_checkpoint_path = checkpoint_path
                    last_checkpoint_tokens = counters["global_input_tokens"]
                    print(f"saved milestone checkpoint: {checkpoint_path.name}", flush=True)
                elif args.save_every_steps > 0 and next_update % args.save_every_steps == 0:
                    checkpoint_path = out_dir / f"checkpoint_step_{next_update}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        sampler,
                        counters,
                        cfg,
                        run_manifest,
                        saved_milestones,
                    )
                    last_checkpoint_path = checkpoint_path
                    last_checkpoint_tokens = counters["global_input_tokens"]

            if args.max_seconds > 0 and time.time() - start_time >= args.max_seconds:
                stop_reason = "max_seconds"
                break

        if stop_requested["value"]:
            stop_reason = "signal"

        final_val_loss, final_val_targets = evaluate(
            model,
            val_ds,
            device,
            args.dtype,
            args.val_batch_size,
            args.final_max_val_seqs,
        )
        final_summary = {
            "stop_reason": stop_reason,
            **counters,
            "final_val_loss": final_val_loss,
            "final_val_ppl": math.exp(min(50.0, final_val_loss)),
            "final_val_targets": final_val_targets,
            "elapsed_seconds": time.time() - start_time,
            "true_input_tokens_per_step": expected_step_tokens,
            "true_sequences_per_step": expected_step_sequences,
            "sampler": sampler.state_dict(),
            "random_init": args.resume is None,
            "resume": args.resume,
        }
        atomic_json_dump(final_summary, out_dir / "final_summary.json")
        print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
        text_log.write(json.dumps(final_summary, ensure_ascii=False) + "\n")

        if not args.no_final_checkpoint and not args.audit_only:
            if last_checkpoint_tokens == counters["global_input_tokens"] and last_checkpoint_path is not None:
                final_checkpoint = last_checkpoint_path
                (out_dir / "FINAL_CHECKPOINT.txt").write_text(
                    str(final_checkpoint) + "\n", encoding="utf-8"
                )
                print(f"final state already saved in: {final_checkpoint}", flush=True)
            else:
                final_checkpoint = out_dir / (
                    f"checkpoint_final_tokens_{counters['global_input_tokens']}.pt"
                )
                save_checkpoint(
                    final_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    sampler,
                    counters,
                    cfg,
                    run_manifest,
                    saved_milestones,
                )
                (out_dir / "FINAL_CHECKPOINT.txt").write_text(
                    str(final_checkpoint) + "\n", encoding="utf-8"
                )
                print(f"saved final checkpoint: {final_checkpoint}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
