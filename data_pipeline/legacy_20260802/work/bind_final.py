#!/usr/bin/env python3
"""Final training corpus binding: hash-based split -> per-source token-target sampling -> 2048 packing."""
import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
FINAL_TOK = os.environ["FINAL_TOK"]
TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
OUT = WS / f"outputs/cross_source/training_bind_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
TMP = Path("/tmp/training_bind_work")
TMP.mkdir(parents=True, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=FINAL_TOK)
EOS = sp.eos_id()
print("final tokenizer:", FINAL_TOK, "vocab:", sp.get_piece_size(), "eos:", EOS, flush=True)

TARGETS = {
    "wiki2019zh": 336199680,
    "webText2019zh": 351141888,
    "news2016zh": 29884416,
    "zhwikisource_classics": 7471104,
    "comment2019zh": 22413312,
}
CAP = 747110400
VAL_TOKENS = 4000000
TEST_TOKENS = 4000000
SEED = 20260802
SEQ_LEN = 2048

KEEP = WS / "outputs/cross_source/cross_source_dedup_v1_20260802_082519/cross_source_keep_list.jsonl.gz"


def load_cluster_map(path):
    m = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] in ("template_family", "repost_family"):
                cid = r.get("cluster_id") or r.get("kept_id") or f"fam_{hash(json.dumps(r, ensure_ascii=False))}"
                for doc in [r.get("kept_id")] + r.get("removed_ids", []) + r.get("docs", []):
                    if doc:
                        m[doc] = cid
    return m


cluster_map = {}
for cpath in (
    WS / "outputs/wiki2019zh/dedup_v1_20260802_001902/template_cluster_report.jsonl.gz",
    WS / "outputs/webText2019zh/dedup_v1_20260802_053022/duplicate_clusters.jsonl.gz",
    WS / "outputs/comment2019zh/dedup_v1_20260802_065014/duplicate_clusters.jsonl.gz",
):
    cluster_map.update(load_cluster_map(cpath))
print("cluster map size:", len(cluster_map), flush=True)

keep = {}
with gzip.open(KEEP, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        keep[r["document_id"]] = r["source"]
print("keep docs:", len(keep), flush=True)

by_source = defaultdict(list)
for uid, src in keep.items():
    by_source[src].append(uid)

# per-doc visible chars (from frozen manifests) for val/test allocation
chars_map = {}
for path, prefix, key in (
    (WS / "outputs/wiki2019zh/wiki2019zh_freeze_v2_20260802_004810/frozen_manifest.jsonl.gz", "wiki2019zh::", "document_id"),
    (WS / "outputs/webText2019zh/webText2019zh_freeze_v1_20260802_062134/frozen_manifest.jsonl.gz", "webText2019zh::", "document_id"),
    (WS / "outputs/comment2019zh/comment2019zh_freeze_v1_20260802_070511/frozen_manifest.jsonl.gz", "comment2019zh::", "document_id"),
):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chars_map[prefix + r[key]] = r["character_count"]
with (WS / "inputs/cleaned/news2016zh/news_tier_a_permanent.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        chars_map[f"news2016zh::{r['source_file']}:{r['block_index']}"] = r.get("visible_chars_permanent", 0)
with (WS / "inputs/cleaned/zhwikisource_classics/accepted_final.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        chars_map[f"zhwikisource_classics::{r['work']}/{r['title']}"] = r.get("visible_chars", 0)
print("chars_map size:", len(chars_map), flush=True)

split = {"val": {}, "test": {}, "train": {}}
for src, uids in by_source.items():
    share = TARGETS[src] / CAP
    val_chars_target = int(VAL_TOKENS * share * 1.55)
    test_chars_target = int(TEST_TOKENS * share * 1.55)
    ordered = sorted(uids, key=lambda u: hashlib.sha256(u.encode("utf-8")).hexdigest())
    val_set = set()
    acc = 0
    for u in ordered:
        val_set.add(u)
        acc += chars_map.get(u, 0)
        if acc >= val_chars_target:
            break
    test_set = set()
    acc = 0
    for u in ordered:
        if u in val_set:
            continue
        test_set.add(u)
        acc += chars_map.get(u, 0)
        if acc >= test_chars_target:
            break
    split["val"][src] = val_set
    split["test"][src] = test_set
    split["train"][src] = set(uids) - val_set - test_set
    print(src, "train:", len(split["train"][src]), "val:", len(val_set), "test:", len(test_set), flush=True)

FROZEN = {
    "news2016zh": ("inputs/cleaned/news2016zh", ["news_tier_a_permanent.jsonl"], lambda r: f"news2016zh::{r['source_file']}:{r['block_index']}"),
    "zhwikisource_classics": ("inputs/cleaned/zhwikisource_classics", ["accepted_final.jsonl"], lambda r: f"zhwikisource_classics::{r['work']}/{r['title']}"),
    "wiki2019zh": ("outputs/wiki2019zh", ["wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl"], lambda r: f"wiki2019zh::{r['document_id']}"),
    "webText2019zh": ("outputs/webText2019zh", ["webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", "webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl"], lambda r: f"webText2019zh::{r['document_id']}"),
    "comment2019zh": ("outputs/comment2019zh", ["comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", "comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl"], lambda r: f"comment2019zh::{r['document_id']}"),
}

# per-source train pools (bin + idx)
pool_f = {}
pool_idx = {}
pool_count = {}
for src in TARGETS:
    pool_f[src] = open(TMP / f"pool_{src}.bin", "wb")
    pool_idx[src] = open(TMP / f"pool_{src}.idx", "w", encoding="utf-8")
    pool_count[src] = 0

val_chunks = []
test_chunks = []
val_man = []
test_man = []
train_doc_info = {src: [] for src in TARGETS}

for src, (root, files, uid_key) in FROZEN.items():
    if src not in TARGETS:
        continue
    batch = []
    batch_ids = []
    for fname in files:
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uid = uid_key(r)
                if uid not in keep:
                    continue
                batch.append(r["text"])
                batch_ids.append(uid)
                if len(batch) >= 4000:
                    for uid, toks in zip(batch_ids, sp.encode(batch, out_type=int)):
                        arr = np.array(toks + [EOS], dtype=np.int32)
                        if uid in split["val"][src]:
                            val_chunks.append(arr)
                            val_man.append({"document_id": uid, "source": src, "tokens": len(arr)})
                        elif uid in split["test"][src]:
                            test_chunks.append(arr)
                            test_man.append({"document_id": uid, "source": src, "tokens": len(arr)})
                        else:
                            start = pool_f[src].tell()
                            pool_f[src].write(arr.tobytes())
                            pool_idx[src].write(f"{uid}\t{start}\t{len(arr)}\n")
                            train_doc_info[src].append((uid, len(arr)))
                            pool_count[src] += len(arr)
                    batch = []
                    batch_ids = []
    if batch:
        for uid, toks in zip(batch_ids, sp.encode(batch, out_type=int)):
            arr = np.array(toks + [EOS], dtype=np.int32)
            if uid in split["val"][src]:
                val_chunks.append(arr)
                val_man.append({"document_id": uid, "source": src, "tokens": len(arr)})
            elif uid in split["test"][src]:
                test_chunks.append(arr)
                test_man.append({"document_id": uid, "source": src, "tokens": len(arr)})
            else:
                start = pool_f[src].tell()
                pool_f[src].write(arr.tobytes())
                pool_idx[src].write(f"{uid}\t{start}\t{len(arr)}\n")
                train_doc_info[src].append((uid, len(arr)))
                pool_count[src] += len(arr)
    pool_f[src].close()
    pool_idx[src].close()
    print(src, "pool tokens:", pool_count[src], flush=True)

val_stream = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.int32)
test_stream = np.concatenate(test_chunks) if test_chunks else np.array([], dtype=np.int32)
print("val tokens:", len(val_stream), "test tokens:", len(test_stream), flush=True)

# train sampling with repetition per source
sampled = []  # (src, uid, length)
for src, target in TARGETS.items():
    docs = train_doc_info[src]
    if not docs:
        print("WARN empty pool", src, flush=True)
        continue
    round_no = 0
    acc = 0
    prev_uid = None
    prev_cluster = None
    while acc < target:
        rng = random.Random(SEED + round_no)
        order = docs[:]
        rng.shuffle(order)
        if prev_uid is not None and order and order[0][0] == prev_uid:
            order = order[1:] + order[:1]
        i = 0
        while i < len(order) and acc < target:
            uid, ln = order[i]
            cl = cluster_map.get(uid)
            if cl is not None and cl == prev_cluster:
                j = i + 1
                while j < len(order) and cluster_map.get(order[j][0]) == cl:
                    j += 1
                if j < len(order):
                    order[i], order[j] = order[j], order[i]
                else:
                    i += 1
                    continue
            sampled.append((src, uid, ln))
            acc += ln
            prev_uid = uid
            prev_cluster = cl
            i += 1
        round_no += 1
    print(src, "sampled:", acc, "rounds:", round_no, flush=True)

# free large structures before assembly (reduce memory/swap thrash)
import gc
del keep, by_source, chars_map, split, val_chunks, test_chunks
gc.collect()

# global deterministic doc-level shuffle
rg = random.Random(SEED + 1000)
rg.shuffle(sampled)

del train_doc_info

# read doc tokens from pool bins (cache idx)
pool_off = {}
for s in TARGETS:
    m = {}
    with (TMP / f"pool_{s}.idx").open("r", encoding="utf-8") as f:
        for line in f:
            uid, off, ln = line.rstrip("\n").split("\t")
            m[uid] = (int(off) // 4, int(ln))
    pool_off[s] = m

mmaps = {s: np.memmap(TMP / f"pool_{s}.bin", dtype=np.int32, mode="r") for s in TARGETS}
with open(OUT / "train.bin", "wb") as tf:
    for s, uid, ln in sampled:
        off, ln2 = pool_off[s][uid]
        tf.write(mmaps[s][off:off + ln2].tobytes())
total_written = (OUT / "train.bin").stat().st_size // 4
print("train stream tokens (before trim):", total_written, flush=True)

# trim to cap (CAP = 747110400 = 364800 * 2048, exact)
full_seqs = min(total_written, CAP) // SEQ_LEN
final_tokens = full_seqs * SEQ_LEN
with open(OUT / "train.bin", "r+b") as tf:
    tf.truncate(final_tokens * 4)
dropped = total_written - final_tokens
print("train seqs:", full_seqs, "tokens:", final_tokens, "dropped:", dropped, flush=True)

# walk the sampled order to build the manifest within final_tokens
src_actual_final = Counter()
consumed = 0
man_final = []
for s, uid, ln in sampled:
    if consumed + ln > final_tokens:
        break
    consumed += ln
    src_actual_final[s] += ln
    man_final.append({"document_id": uid, "source": s, "tokens": ln})
print("final per-source:", dict(src_actual_final), "total:", consumed, flush=True)

val_stream.astype(np.int32).tofile(OUT / "validation.bin")
test_stream.astype(np.int32).tofile(OUT / "test.bin")
with gzip.open(OUT / "training_sample_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in man_final:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "validation_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in val_man:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "test_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in test_man:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "stage": "training_bind_v1",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "final_tokenizer": FINAL_TOK,
    "final_tokenizer_sha256": hashlib.sha256(Path(FINAL_TOK).read_bytes()).hexdigest(),
    "max_train_tokens": CAP,
    "train_tokens_target_per_source": TARGETS,
    "train_tokens_actual_per_source": dict(sorted(src_actual_final.items())),
    "train_tokens_total_actual": consumed,
    "train_tokens_delta": consumed - CAP,
    "train_sequences": full_seqs,
    "sequence_length": SEQ_LEN,
    "dropped_partial_tokens": dropped,
    "validation": {"tokens": int(len(val_stream)), "documents": len(val_man), "sequences": len(val_stream) // SEQ_LEN},
    "test": {"tokens": int(len(test_stream)), "documents": len(test_man), "sequences": len(test_stream) // SEQ_LEN},
    "comment_ratio_actual": round(src_actual_final["comment2019zh"] / consumed * 100, 4) if consumed else 0,
    "seed": SEED,
    "note": "train/val/test按稳定文档哈希(sha256(uid))划分，严格不重叠；文档间插入真实EOS；2048连续打包；末尾残块丢弃；全局确定性打乱。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run.log").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
