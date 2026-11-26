#!/usr/bin/env python3
"""Full evaluation of one tokenizer on scopes A (eval set), B (frozen full), C (cross-source keep)."""
import gzip
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
TOK = os.environ.get("TOK", "c2")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
EVAL_DIR = AB / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

TOKENIZERS = {
    "old": "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model",
    "old_candidate": str(WS / "outputs/cross_source/tokenizer_ab_20260802_084554/candidate_bpe_32k.model"),
    "c1": str(AB / "c1_bpe_32k.model"),
    "c2": str(AB / "c2_bpe_32k.model"),
}
if TOK not in TOKENIZERS:
    raise SystemExit(f"unknown TOK {TOK}")

RESULT = EVAL_DIR / f"eval_{TOK}.json"
if RESULT.exists():
    print("already done:", RESULT)
    sys.exit(0)

MODEL = TOKENIZERS[TOK]
sp = spm.SentencePieceProcessor(model_file=MODEL)
unk_id = sp.unk_id()
CHUNK = 4000

REMOVED_NEWS = set()
with gzip.open(WS / "outputs/cross_source/cross_source_dedup_v1_20260802_082519/cross_source_removed_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"].startswith("news2016zh::"):
            REMOVED_NEWS.add(r["id"])
print("removed news ids:", len(REMOVED_NEWS))

FROZEN = {
    "news2016zh": ("inputs/cleaned/news2016zh", ["news_tier_a_permanent.jsonl"], lambda r: f"news2016zh::{r['source_file']}:{r['block_index']}"),
    "zhwikisource_classics": ("inputs/cleaned/zhwikisource_classics", ["accepted_final.jsonl"], lambda r: f"zhwikisource_classics::{r['work']}/{r['title']}"),
    "wiki2019zh": ("outputs/wiki2019zh", ["wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl"], lambda r: f"wiki2019zh::{r['document_id']}"),
    "webText2019zh": ("outputs/webText2019zh", ["webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", "webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl"], lambda r: f"webText2019zh::{r['document_id']}"),
    "comment2019zh": ("outputs/comment2019zh", ["comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", "comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl"], lambda r: f"comment2019zh::{r['document_id']}"),
}


def agg_texts(texts):
    enc = sp.encode(texts, out_type=int)
    acc = {
        "docs": 0, "chars": 0, "tokens": 0, "unk": 0, "docs_unk": 0, "byte_tokens": 0,
        "single_char_tokens": 0, "lengths": [], "docs_gt2048": 0, "seqs": 0, "last_util_sum": 0.0,
        "rt_mismatch": 0, "rt_ws_mismatch": 0, "rt_semantic_mismatch": 0,
    }
    for text, toks in zip(texts, enc):
        n = len(toks)
        acc["docs"] += 1
        acc["chars"] += len(text) - text.count("\n")
        acc["tokens"] += n
        u = sum(1 for t in toks if t == unk_id)
        acc["unk"] += u
        if u:
            acc["docs_unk"] += 1
        for t in toks:
            piece = sp.id_to_piece(t)
            if piece.startswith("<0x"):
                acc["byte_tokens"] += 1
            elif len(piece) == 1:
                acc["single_char_tokens"] += 1
        acc["lengths"].append(n)
        if n > 2048:
            acc["docs_gt2048"] += 1
        if n > 0:
            nch = math.ceil(n / 2048)
            acc["seqs"] += nch
            acc["last_util_sum"] += (n - (nch - 1) * 2048) / 2048.0
        dec = sp.decode(toks)
        norm = sp.Normalize(text)
        if dec != norm:
            acc["rt_mismatch"] += 1
            if "".join(dec.split()) != "".join(norm.split()):
                acc["rt_ws_mismatch"] += 1
        # SPM normalization semantics: spaces are U+2581 in normalized form, restored on decode.
        if dec.replace("\u2581", " ") != norm.replace("\u2581", " "):
            acc["rt_semantic_mismatch"] += 1
    return acc


def merge_acc(target, a):
    for k, v in a.items():
        if k == "lengths":
            target[k].extend(v)
        else:
            target[k] += v


def finalize(acc, label):
    lens = np.array(acc["lengths"], dtype=np.int64) if acc["lengths"] else np.array([], dtype=np.int64)

    def p(q):
        return int(np.percentile(lens, q)) if lens.size else 0

    return {
        "scope": label,
        "documents": acc["docs"],
        "visible_chars": acc["chars"],
        "tokens_without_eos": acc["tokens"],
        "tokens_with_one_eos_per_document": acc["tokens"] + acc["docs"],
        "unk_tokens": acc["unk"],
        "documents_with_unk": acc["docs_unk"],
        "unk_rate": round(acc["unk"] / acc["tokens"], 10) if acc["tokens"] else 0.0,
        "chars_per_token": round(acc["chars"] / acc["tokens"], 6) if acc["tokens"] else None,
        "mean_tokens_per_doc": round(float(lens.mean()), 4) if lens.size else 0,
        "p50": p(50), "p90": p(90), "p95": p(95), "p99": p(99),
        "max_tokens_per_doc": int(lens.max()) if lens.size else 0,
        "documents_gt_2048_tokens": acc["docs_gt2048"],
        "sequences_after_2048_chunking": acc["seqs"],
        "last_chunk_avg_utilization": round(acc["last_util_sum"] / acc["docs"], 6) if acc["docs"] else 0.0,
        "byte_fallback_tokens": acc["byte_tokens"],
        "byte_fallback_ratio": round(acc["byte_tokens"] / acc["tokens"], 10) if acc["tokens"] else 0.0,
        "single_char_tokens": acc["single_char_tokens"],
        "single_char_token_ratio": round(acc["single_char_tokens"] / acc["tokens"], 10) if acc["tokens"] else 0.0,
        "encode_decode_mismatch_exact": acc["rt_mismatch"],
        "encode_decode_mismatch_whitespace_insensitive": acc["rt_ws_mismatch"],
        "encode_decode_mismatch_normalized_semantics": acc["rt_semantic_mismatch"],
    }


result = {
    "tokenizer": TOK,
    "model": MODEL,
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "special_tokens": {
        "unk_id": sp.unk_id(), "unk_piece": sp.id_to_piece(sp.unk_id()),
        "bos_id": sp.bos_id(), "bos_piece": sp.id_to_piece(sp.bos_id()) if sp.bos_id() >= 0 else None,
        "eos_id": sp.eos_id(), "eos_piece": sp.id_to_piece(sp.eos_id()) if sp.eos_id() >= 0 else None,
        "pad_id": sp.pad_id(), "pad_piece": sp.id_to_piece(sp.pad_id()) if sp.pad_id() >= 0 else None,
    },
    "actual_vocab_size": sp.get_piece_size(),
    "vocab_single_char_pieces": sum(1 for i in range(sp.get_piece_size()) if len(sp.id_to_piece(i)) == 1 and not sp.id_to_piece(i).startswith("<")),
    "per_source": {"eval": {}, "full": {}},
    "scope_c": {},
}

# scope A: eval set
eval_rows = []
with gzip.open(AB / "samples/tokenizer_eval_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        eval_rows.append(json.loads(line))
texts = []
with (AB / "samples/tokenizer_eval.txt").open("r", encoding="utf-8") as f:
    for line in f:
        texts.append(line.rstrip("\n"))
assert len(texts) == len(eval_rows)
per_source_acc = {}
for row, text in zip(eval_rows, texts):
    per_source_acc.setdefault(row["source"], []).append(text)
for src, txts in sorted(per_source_acc.items()):
    result["per_source"]["eval"][src] = finalize(agg_texts(txts), f"eval:{src}")
print("eval set done", flush=True)

# scope B: frozen full per source
news_doc_tokens = {}
for src, (root, files, uid_key) in FROZEN.items():
    acc = {"docs": 0, "chars": 0, "tokens": 0, "unk": 0, "docs_unk": 0, "byte_tokens": 0,
           "single_char_tokens": 0, "lengths": [], "docs_gt2048": 0, "seqs": 0, "last_util_sum": 0.0,
            "rt_mismatch": 0, "rt_ws_mismatch": 0, "rt_semantic_mismatch": 0}
    batch_texts = []
    batch_ids = []
    for fname in files:
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uid = uid_key(r)
                batch_texts.append(r["text"])
                batch_ids.append(uid)
                if len(batch_texts) >= CHUNK:
                    merge_acc(acc, agg_texts(batch_texts))
                    if src == "news2016zh":
                        enc = sp.encode(batch_texts, out_type=int)
                        for uid, toks in zip(batch_ids, enc):
                            news_doc_tokens[uid] = len(toks)
                    batch_texts = []
                    batch_ids = []
    if batch_texts:
        merge_acc(acc, agg_texts(batch_texts))
        if src == "news2016zh":
            enc = sp.encode(batch_texts, out_type=int)
            for uid, toks in zip(batch_ids, enc):
                news_doc_tokens[uid] = len(toks)
    result["per_source"]["full"][src] = finalize(acc, f"full:{src}")
    print(src, "full done", flush=True)

# scope C: cross-source keep (B minus removed news)
news_full = result["per_source"]["full"]["news2016zh"]
removed_tokens = sum(news_doc_tokens[uid] for uid in REMOVED_NEWS if uid in news_doc_tokens)
others_tokens = sum(result["per_source"]["full"][s]["tokens_without_eos"] for s in result["per_source"]["full"] if s != "news2016zh")
others_docs = sum(result["per_source"]["full"][s]["documents"] for s in result["per_source"]["full"] if s != "news2016zh")
result["scope_c"] = {
    "note": "cross-source keep set = frozen full minus 1,539 removed news docs",
    "documents": others_docs + news_full["documents"] - len(REMOVED_NEWS),
    "tokens_without_eos": others_tokens + news_full["tokens_without_eos"] - removed_tokens,
    "removed_news_docs": len(REMOVED_NEWS),
    "removed_news_tokens": removed_tokens,
}
print("news docs tracked:", len(news_doc_tokens), "removed tokens:", removed_tokens, flush=True)
assert len(news_doc_tokens) == 12420
assert len(REMOVED_NEWS) == 1539

RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("saved", RESULT)
