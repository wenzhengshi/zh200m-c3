#!/usr/bin/env python3
"""A800 single-GPU pretraining loop for the 196M Chinese LM (budget-limited, resumable, SIGINT-safe)."""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="dir with train.bin / validation.bin")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=2850)
    p.add_argument("--max-tokens", type=int, default=747110400)
    p.add_argument("--max-seconds", type=float, default=14400.0)
    p.add_argument("--no-new-step-after", type=float, default=13800.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--speed-test", action="store_true", help="run N steps measuring throughput (no eval)")
    p.add_argument("--speed-test-steps", type=int, default=500)
    p.add_argument("--reserve-final-seconds", type=float, default=600.0)
    p.add_argument("--max-val-seqs", type=int, default=None,
                   help="cap evaluation sequences (full validation default; set small on CPU smoke)")
    return p.parse_args()


def safe_ppl(loss):
    return math.exp(loss) if loss < 700 else float("inf")


class Dataset:
    def __init__(self, path, seq_len, max_seqs=None):
        arr = np.memmap(path, dtype=np.int32, mode="r")
        self.tokens = arr
        self.seq_len = seq_len
        n = len(arr)
        seqs = n // seq_len
        if max_seqs:
            seqs = min(seqs, max_seqs)
        self.seqs = max(0, seqs)
        self.inputs = arr  # memmap over full stream

    def get(self, idx):
        start = idx * self.seq_len
        inp = self.tokens[start:start + self.seq_len]
        if start + self.seq_len + 1 <= len(self.tokens):
            lab = self.tokens[start + 1:start + self.seq_len + 1]
        else:
            lab = np.concatenate([self.tokens[start + 1:], np.array([-100], dtype=np.int32)])
        return inp.astype(np.int64), lab.astype(np.int64)

    def __len__(self):
        return self.seqs


class GradScalerDummy:
    """No-op scaler for bf16/fp32; stores empty state for checkpoint parity."""
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass
    def state_dict(self):
        return {}
    def load_state_dict(self, sd):
        pass


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, tokens, cfg, rng_state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": step,
            "trained_tokens": tokens,
            "config": cfg,
            "rng": rng_state,
        },
        tmp,
    )
    os.replace(tmp, str(path))


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["global_step"], ckpt["trained_tokens"], ckpt.get("config", {}), ckpt.get("rng", {})


def make_scheduler(optimizer, warmup, total_steps, peak_lr):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, ds, max_batches, device, dtype, seq_len):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(min(max_batches, len(ds))):
            inp, lab = ds.get(i)
            inp = torch.from_numpy(inp).unsqueeze(0).to(device)
            lab = torch.from_numpy(lab).unsqueeze(0).to(device)
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100
            )
            total += loss.item() * (lab != -100).sum().item()
            count += (lab != -100).sum().item()
    model.train()
    return total / count if count else float("nan")


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert cfg["parameter_count"] == 196153216
    assert cfg["vocab_size"] == 32000
    assert cfg["max_seq_len"] == 2048

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_fp16 = args.dtype == "fp16"

    model = build_model(cfg)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.to(device)

    micro = args.micro_batch
    accum = args.grad_accum or max(1, 262144 // (micro * cfg["max_seq_len"]))
    if args.grad_accum is None:
        assert micro * accum * cfg["max_seq_len"] == 262144
    eff_batch = micro * accum * cfg["max_seq_len"]
    print(f"device={device} dtype={args.dtype} micro={micro} accum={accum} eff_batch_tokens={eff_batch}", flush=True)

    peak_lr = cfg.get("peak_lr", 1.5e-3)
    warmup = cfg.get("warmup_steps", 200)
    max_steps = args.max_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.get("weight_decay", 0.1),
        fused=(device.startswith("cuda") and hasattr(torch.optim.AdamW, "fused")),
    )
    scheduler = make_scheduler(optimizer, warmup, max_steps, peak_lr)
    scaler = torch.amp.GradScaler(enabled=use_fp16) if use_fp16 else GradScalerDummy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    trained_tokens = 0
    if args.resume:
        step, trained_tokens, _, rng = load_checkpoint(
            Path(args.resume) / "checkpoint.pt", model, optimizer, scheduler, scaler, device
        )
        torch.set_rng_state(torch.ByteTensor(rng.get("torch", [])))
        print(f"resumed step={step} tokens={trained_tokens}", flush=True)

    train_ds = Dataset(Path(args.data_dir) / "train.bin", cfg["max_seq_len"])
    val_ds = Dataset(Path(args.data_dir) / "validation.bin", cfg["max_seq_len"])
    print(f"train seqs={len(train_ds)} val seqs={len(val_ds)}", flush=True)

    stop_flag = {"stop": False}

    def handler(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    logf = open(out_dir / "run.log", "a", encoding="utf-8")
    t0 = time.time()
    baseline = evaluate(model, val_ds, args.max_val_seqs or 8, device, dtype, cfg["max_seq_len"])
    logf.write(f"[step {step}] baseline val loss (8 seqs): {baseline:.4f}\n")
    print(f"baseline val loss: {baseline:.4f}", flush=True)

    max_tokens = args.max_tokens
    no_new_after = args.no_new_step_after
    reserve = args.reserve_final_seconds
    speed_only = args.speed_test
    speed_steps = args.speed_test_steps if speed_only else max_steps

    model.train()
    while step < max_steps and trained_tokens < max_tokens and not stop_flag["stop"]:
        if not speed_only and time.time() - t0 > no_new_after:
            logf.write(f"wall clock > {no_new_after}s, stopping new steps\n")
            break
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_n = 0
        for mb in range(accum):
            idx = (step * accum + mb) % len(train_ds)
            inp, lab = train_ds.get(idx)
            inp = torch.from_numpy(inp).unsqueeze(0).to(device)
            lab = torch.from_numpy(lab).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(inp)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100
                ) / accum
            scaler.scale(loss).backward()
            acc_loss += loss.item() * accum
            acc_n += (lab != -100).sum().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        trained_tokens += eff_batch
        if step % args.log_interval == 0 or step <= 2:
            elapsed = time.time() - t0
            msg = (
                f"[step {step}] loss={acc_loss:.4f} tokens={trained_tokens:,} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed:.0f}s "
                f"tok/s={trained_tokens/max(elapsed,1):.1f}"
            )
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
        if not speed_only and step % args.eval_interval == 0:
            vloss = evaluate(model, val_ds, args.max_val_seqs or 16, device, dtype, cfg["max_seq_len"])
            msg = f"[step {step}] fixed-val loss (16 seqs): {vloss:.4f} ppl={safe_ppl(vloss):.3f}"
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
        if speed_only and step >= speed_steps:
            break
        if stop_flag["stop"]:
            logf.write("SIGINT/SIGTERM received; saving checkpoint\n")

    # final checkpoint + full validation
    rng_state = {
        "torch": torch.get_rng_state().tolist(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    save_checkpoint(
        out_dir / "checkpoint.pt", model, optimizer, scheduler, scaler, step, trained_tokens, cfg, rng_state
    )
    final_val = evaluate(
        model, val_ds, args.max_val_seqs if args.max_val_seqs is not None else len(val_ds),
        device, dtype, cfg["max_seq_len"],
    ) if not speed_only else baseline
    msg = (
        f"[final] step={step} tokens={trained_tokens:,} full-val loss={final_val:.4f} "
        f"ppl={safe_ppl(final_val):.3f} elapsed={time.time()-t0:.0f}s"
    )
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.close()
    with (out_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "global_step": step,
                "trained_tokens": trained_tokens,
                "final_val_loss": final_val,
                "final_ppl": safe_ppl(final_val),
                "elapsed_seconds": time.time() - t0,
                "effective_batch_tokens": eff_batch,
                "stopped_by": "speed_test" if speed_only else ("signal" if stop_flag["stop"] else "limit"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
