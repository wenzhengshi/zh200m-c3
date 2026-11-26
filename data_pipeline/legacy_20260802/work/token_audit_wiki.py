#!/usr/bin/env python3
"""Exact SentencePiece token audit of the frozen wiki2019zh set."""
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sentencepiece as spm

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
FREEZE = sorted((WS / "outputs/wiki2019zh").glob("wiki2019zh_freeze_v2_*"))[-1]
OUT = FREEZE / f"token_audit_{TS}"
OUT.mkdir(exist_ok=False)

MODEL = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model"
sp = spm.SentencePieceProcessor(model_file=MODEL)
unk_id = sp.unk_id()
eos_id = sp.eos_id()
vocab = sp.get_piece_size()
print("vocab:", vocab, "unk_id:", unk_id, "eos_id:", eos_id)

CHUNK = 4000
per_tier = {"A": {"docs": 0, "chars": 0, "tokens": 0, "unk": 0}, "B": {"docs": 0, "chars": 0, "tokens": 0, "unk": 0}}
doc_token_lengths = defaultdict(list)
unk_examples = []
buckets = Counter()


def add_bucket(n):
    if n <= 63:
        buckets["00000_00063"] += 1
    elif n <= 127:
        buckets["00064_00127"] += 1
    elif n <= 255:
        buckets["00128_00255"] += 1
    elif n <= 511:
        buckets["00256_00511"] += 1
    elif n <= 1023:
        buckets["00512_01023"] += 1
    elif n <= 2047:
        buckets["01024_02047"] += 1
    elif n <= 4095:
        buckets["02048_04095"] += 1
    elif n <= 8191:
        buckets["04096_08191"] += 1
    else:
        buckets["08192_plus"] += 1


for name, tier in (("tier_a_frozen.jsonl", "A"), ("tier_b_frozen.jsonl", "B")):
    texts = []
    ids = []
    chars_batch = 0
    with (FREEZE / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            ids.append(r["document_id"])
            chars_batch += len(r["text"]) - r["text"].count("\n")
            if len(texts) >= CHUNK:
                enc = sp.encode(texts, out_type=int)
                for did, toks in zip(ids, enc):
                    n = len(toks)
                    u = sum(1 for t in toks if t == unk_id)
                    per_tier[tier]["docs"] += 1
                    per_tier[tier]["chars"] += 0  # set below
                    per_tier[tier]["tokens"] += n
                    per_tier[tier]["unk"] += u
                    doc_token_lengths[tier].append(n)
                    add_bucket(n)
                    if u and len(unk_examples) < 30:
                        unk_examples.append({"document_id": did, "unk_count": u})
                per_tier[tier]["chars"] += chars_batch
                texts = []
                ids = []
                chars_batch = 0
        if texts:
            enc = sp.encode(texts, out_type=int)
            for did, toks in zip(ids, enc):
                n = len(toks)
                u = sum(1 for t in toks if t == unk_id)
                per_tier[tier]["docs"] += 1
                per_tier[tier]["tokens"] += n
                per_tier[tier]["unk"] += u
                doc_token_lengths[tier].append(n)
                add_bucket(n)
                if u and len(unk_examples) < 30:
                    unk_examples.append({"document_id": did, "unk_count": u})
            per_tier[tier]["chars"] += chars_batch
    print(tier, "done", flush=True)

tot = {k: sum(v[k] for v in per_tier.values()) for k in ("docs", "chars", "tokens", "unk")}
all_lens = doc_token_lengths["A"] + doc_token_lengths["B"]


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))
    return vals[idx]


summary = {
    "stage": "token_audit",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "tokenizer": MODEL,
    "vocab_size": vocab,
    "unk_id": unk_id,
    "eos_id": eos_id,
    "input": str(FREEZE),
    "documents": tot["docs"],
    "visible_chars": tot["chars"],
    "tokens_without_eos": tot["tokens"],
    "tokens_with_one_eos_per_document": tot["tokens"] + tot["docs"],
    "unk_tokens": tot["unk"],
    "unk_rate": round(tot["unk"] / tot["tokens"], 10) if tot["tokens"] else 0.0,
    "chars_per_token": round(tot["chars"] / tot["tokens"], 6) if tot["tokens"] else None,
    "per_tier": per_tier,
    "document_token_stats": {
        "min": min(all_lens),
        "p50": pct(all_lens, 50),
        "p90": pct(all_lens, 90),
        "p95": pct(all_lens, 95),
        "p99": pct(all_lens, 99),
        "max": max(all_lens),
        "mean": round(statistics.mean(all_lens), 4),
    },
    "length_buckets": dict(sorted(buckets.items())),
    "unk_examples": unk_examples,
}
(OUT / "token_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with (OUT / "length_distribution.tsv").open("w", encoding="utf-8") as f:
    f.write("bucket\tdocuments\n")
    for k, v in sorted(buckets.items()):
        f.write(f"{k}\t{v}\n")

with (OUT / "unk_examples.jsonl").open("w", encoding="utf-8") as f:
    for e in unk_examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

log = [
    f"token_audit run.log",
    f"model: {MODEL}",
    f"docs: {tot['docs']:,} chars: {tot['chars']:,} tokens(no EOS): {tot['tokens']:,}",
    f"tokens(with EOS): {tot['tokens'] + tot['docs']:,} unk: {tot['unk']:,} unk_rate: {summary['unk_rate']}",
]
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
