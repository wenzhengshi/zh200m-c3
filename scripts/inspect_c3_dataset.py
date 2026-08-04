#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_bin(path: Path, seq_len: int, vocab_size: int, full_scan: bool) -> dict:
    size = path.stat().st_size
    if size % 4:
        raise RuntimeError(f"{path}: size not divisible by 4")
    tokens = np.memmap(path, dtype=np.uint32, mode="r")
    result = {
        "path": str(path),
        "bytes": size,
        "dtype": "uint32",
        "tokens": int(tokens.size),
        "full_sequences": int(tokens.size // seq_len),
        "remainder_tokens": int(tokens.size % seq_len),
    }
    if full_scan:
        minimum = vocab_size
        maximum = 0
        chunk = 16_000_000
        for start in range(0, tokens.size, chunk):
            view = tokens[start : start + chunk]
            minimum = min(minimum, int(view.min()))
            maximum = max(maximum, int(view.max()))
        result["min_token"] = minimum
        result["max_token"] = maximum
        result["all_tokens_in_vocab"] = maximum < vocab_size
    else:
        positions = np.unique(
            np.linspace(0, max(0, tokens.size - 1), num=min(100_000, tokens.size), dtype=np.int64)
        )
        sample = np.asarray(tokens[positions])
        result["sample_min_token"] = int(sample.min())
        result["sample_max_token"] = int(sample.max())
        result["sample_all_tokens_in_vocab"] = int(sample.max()) < vocab_size
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--full-token-range-scan", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    release = Path(args.data_release).resolve()
    training = release / "training_data"
    tokenizer = release / "tokenizer_frozen" / "tokenizer.model"
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8"))

    report = {
        "release": str(release),
        "tokenizer_sha256": sha256_file(tokenizer),
        "tokenizer_sha256_expected": cfg["tokenizer_sha256"],
        "tokenizer_match": sha256_file(tokenizer) == cfg["tokenizer_sha256"],
        "summary": summary,
        "files": {},
    }
    for name in ("train.bin", "validation.bin", "test.bin"):
        report["files"][name] = inspect_bin(
            training / name,
            int(cfg["max_seq_len"]),
            int(cfg["vocab_size"]),
            args.full_token_range_scan,
        )

    train = report["files"]["train.bin"]
    report["derived"] = {
        "configured_micro_batch": cfg["micro_batch_size"],
        "configured_grad_accum": cfg["gradient_accumulation_steps"],
        "true_sequences_per_step": cfg["micro_batch_size"]
        * cfg["gradient_accumulation_steps"],
        "true_input_tokens_per_step": cfg["micro_batch_size"]
        * cfg["gradient_accumulation_steps"]
        * cfg["max_seq_len"],
        "steps_for_one_train_pass": train["full_sequences"]
        // (cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
