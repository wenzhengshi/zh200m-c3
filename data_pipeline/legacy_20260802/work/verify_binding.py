#!/usr/bin/env python3
"""Binding integrity + train/val/test zero-overlap verification."""
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/cross_source").glob("training_bind_v1_*"))[-1]
print("dir:", D)

EOS = 2
PAD = 3
SEQ = 2048

train = np.memmap(D / "train.bin", dtype=np.int32, mode="r")
val = np.memmap(D / "validation.bin", dtype=np.int32, mode="r")
test = np.memmap(D / "test.bin", dtype=np.int32, mode="r")

print("train tokens:", len(train), "seqs:", len(train) // SEQ, "aligned:", len(train) % SEQ == 0)
print("val tokens:", len(val), "seqs:", len(val) // SEQ, "aligned:", len(val) % SEQ == 0)
print("test tokens:", len(test), "seqs:", len(test) // SEQ, "aligned:", len(test) % SEQ == 0)
print("train EOS count:", int(np.count_nonzero(train == EOS)))
print("train PAD count:", int(np.count_nonzero(train == PAD)))
print("val EOS count:", int(np.count_nonzero(val == EOS)), "test EOS count:", int(np.count_nonzero(test == EOS)))
print("token id range: min", int(train.min()), "max", int(train.max()))
assert int(train.max()) < 32000 and int(train.min()) >= 0
assert len(train) <= 747110400

# manifest vs stream
man_sum = 0
rows = 0
src_tok = Counter()
src_docs = Counter()
with gzip.open(D / "training_sample_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rows += 1
        man_sum += r["tokens"]
        src_tok[r["source"]] += r["tokens"]
        src_docs[r["source"]] += 1
print("manifest rows:", rows, "manifest tokens:", man_sum)
print("stream tokens:", len(train))
print("unmapped tail tokens (stream - manifest):", len(train) - man_sum)
print("per-source tokens:", dict(src_tok))
print("per-source docs:", dict(src_docs))

val_ids = set()
with gzip.open(D / "validation_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        val_ids.add(json.loads(line)["document_id"])
test_ids = set()
with gzip.open(D / "test_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        test_ids.add(json.loads(line)["document_id"])
train_ids = set()
with gzip.open(D / "training_sample_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        train_ids.add(json.loads(line)["document_id"])
print("train/val overlap:", len(train_ids & val_ids))
print("train/test overlap:", len(train_ids & test_ids))
print("val/test overlap:", len(val_ids & test_ids))
assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
print("ZERO-OVERLAP OK")
