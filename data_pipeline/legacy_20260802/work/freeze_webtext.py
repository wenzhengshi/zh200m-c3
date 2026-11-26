#!/usr/bin/env python3
"""Freeze webText2019zh deduplicated candidates + exact SentencePiece token audit."""
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sentencepiece as spm
import os

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
DS = os.environ.get("FREEZE_DATASET", "webText2019zh")
SRC = sorted((WS / f"outputs/{DS}").glob("curated_candidate_v1_*"))[-1]
DEDUP = sorted((WS / f"outputs/{DS}").glob("dedup_v1_*"))[-1]
OUT = WS / f"outputs/{DS}/{DS}_freeze_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

keep = set()
with gzip.open(DEDUP / "dedup_keep_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        keep.add(json.loads(line)["document_id"])
print("keep ids:", len(keep))

ta = (OUT / "tier_a_frozen.jsonl").open("w", encoding="utf-8")
tb = (OUT / "tier_b_frozen.jsonl").open("w", encoding="utf-8")
manifest = gzip.open(OUT / "frozen_manifest.jsonl.gz", "wt", encoding="utf-8")
counts = Counter()
chars = Counter()

for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B")):
    with (SRC / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["document_id"] not in keep:
                continue
            text = r["text"]
            if not text.strip():
                continue
            c = len(text) - text.count("\n")
            sh = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if tier == "A":
                ta.write(json.dumps({"document_id": r["document_id"], "text": text}, ensure_ascii=False) + "\n")
            else:
                tb.write(json.dumps({"document_id": r["document_id"], "text": text}, ensure_ascii=False) + "\n")
            manifest.write(json.dumps({"document_id": r["document_id"], "tier": tier, "character_count": c, "text_sha256": sh}, ensure_ascii=False) + "\n")
            counts[tier] += 1
            chars[tier] += c
ta.close(); tb.close(); manifest.close()

total = sum(counts.values())
summary = {
    "stage": f"{DS}_freeze_v1",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "total_documents": total,
    "tier_document_counts": dict(sorted(counts.items())),
    "tier_character_counts": {k: chars[k] for k in ("A", "B")},
    "review_isolated": 32262,
    "drop_archived": 18570,
    "dedup_removed": 85786,
    "frozen": True,
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "PERMANENT_FREEZE_NOTICE.txt").write_text(
    f"{DS} 已完成冻结 v1。\n禁止覆盖或修改本目录中的任何文件。\nREVIEW与DROP留档未进入本冻结集。\n",
    encoding="utf-8",
)
(OUT / "run.log").write_text(
    f"{DS}_freeze_v1 run.log\ncreated_at: {summary['created_at']}\nfrozen docs: {total:,}\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# ---- token audit ----
TA_OUT = OUT / f"token_audit_{TS}"
TA_OUT.mkdir()
MODEL = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model"
sp = spm.SentencePieceProcessor(model_file=MODEL)
unk_id = sp.unk_id()
per_tier = {"A": {"docs": 0, "chars": 0, "tokens": 0, "unk": 0}, "B": {"docs": 0, "chars": 0, "tokens": 0, "unk": 0}}
lengths = defaultdict(list)
buckets = Counter()
unk_examples = []
CHUNK = 4000


def add_bucket(n):
    for hi, key in ((63, "00000_00063"), (127, "00064_00127"), (255, "00128_00255"), (511, "00256_00511"),
                    (1023, "00512_01023"), (2047, "01024_02047"), (4095, "02048_04095"), (8191, "04096_08191")):
        if n <= hi:
            buckets[key] += 1
            return
    buckets["08192_plus"] += 1


for name, tier in (("tier_a_frozen.jsonl", "A"), ("tier_b_frozen.jsonl", "B")):
    texts, ids, cb = [], [], 0
    with (OUT / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            ids.append(r["document_id"])
            cb += len(r["text"]) - r["text"].count("\n")
            if len(texts) >= CHUNK:
                enc = sp.encode(texts, out_type=int)
                for did, toks in zip(ids, enc):
                    n = len(toks)
                    u = sum(1 for t in toks if t == unk_id)
                    per_tier[tier]["docs"] += 1
                    per_tier[tier]["tokens"] += n
                    per_tier[tier]["unk"] += u
                    lengths[tier].append(n)
                    add_bucket(n)
                    if u and len(unk_examples) < 30:
                        unk_examples.append({"document_id": did, "unk_count": u})
                per_tier[tier]["chars"] += cb
                texts, ids, cb = [], [], 0
        if texts:
            enc = sp.encode(texts, out_type=int)
            for did, toks in zip(ids, enc):
                n = len(toks)
                u = sum(1 for t in toks if t == unk_id)
                per_tier[tier]["docs"] += 1
                per_tier[tier]["tokens"] += n
                per_tier[tier]["unk"] += u
                lengths[tier].append(n)
                add_bucket(n)
                if u and len(unk_examples) < 30:
                    unk_examples.append({"document_id": did, "unk_count": u})
            per_tier[tier]["chars"] += cb
    print(tier, "tokenized", flush=True)

tot = {k: sum(v[k] for v in per_tier.values()) for k in ("docs", "chars", "tokens", "unk")}
all_lens = lengths["A"] + lengths["B"]
all_lens_sorted = sorted(all_lens)


def pct(p):
    if not all_lens_sorted:
        return None
    return all_lens_sorted[min(len(all_lens_sorted) - 1, int(round(p / 100 * (len(all_lens_sorted) - 1))))]


tok_summary = {
    "stage": "token_audit",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "tokenizer": MODEL,
    "vocab_size": sp.get_piece_size(),
    "unk_id": unk_id,
    "input": str(OUT),
    "documents": tot["docs"],
    "visible_chars": tot["chars"],
    "tokens_without_eos": tot["tokens"],
    "tokens_with_one_eos_per_document": tot["tokens"] + tot["docs"],
    "unk_tokens": tot["unk"],
    "unk_rate": round(tot["unk"] / tot["tokens"], 10) if tot["tokens"] else 0.0,
    "chars_per_token": round(tot["chars"] / tot["tokens"], 6) if tot["tokens"] else None,
    "per_tier": per_tier,
    "document_token_stats": {
        "min": min(all_lens), "p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
        "max": max(all_lens), "mean": round(statistics.mean(all_lens), 4),
    },
    "length_buckets": dict(sorted(buckets.items())),
    "unk_examples": unk_examples,
}
(TA_OUT / "token_audit_summary.json").write_text(json.dumps(tok_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (TA_OUT / "length_distribution.tsv").open("w", encoding="utf-8") as f:
    f.write("bucket\tdocuments\n")
    for k, v in sorted(buckets.items()):
        f.write(f"{k}\t{v}\n")
(TA_OUT / "unk_examples.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in unk_examples) + "\n", encoding="utf-8")
(TA_OUT / "run.log").write_text(
    f"token_audit run.log\ndocs: {tot['docs']:,} chars: {tot['chars']:,} tokens: {tot['tokens']:,} unk: {tot['unk']:,}\n",
    encoding="utf-8",
)
print(json.dumps(tok_summary, ensure_ascii=False, indent=2))
