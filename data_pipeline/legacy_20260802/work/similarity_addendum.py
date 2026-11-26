#!/usr/bin/env python3
"""Add exact 4-gram Jaccard similarity between removed docs and their kept docs."""
import gzip
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
MASK64 = (1 << 64) - 1
BASE = 131


def shingle_set(text, n=4):
    codes = [ord(c) for c in text]
    if len(codes) < n:
        return set()
    h = 0
    pw = 1
    for i in range(n):
        h = (h * BASE + codes[i]) & MASK64
        pw = (pw * BASE) & MASK64
    out = {h}
    for i in range(n, len(codes)):
        h = (h * BASE + codes[i] - codes[i - n] * pw) & MASK64
        out.add(h)
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def process(dedup_dir, source_files, uid_key, removed_name="dedup_removed_list.jsonl.gz"):
    removed = []
    with gzip.open(dedup_dir / removed_name, "rt", encoding="utf-8") as f:
        for line in f:
            removed.append(json.loads(line))
    print("removed rows:", len(removed), "dir:", dedup_dir.name)
    need = set()
    for r in removed:
        need.add(r["id"])
        need.add(r["kept_id"])
    texts = {}
    for fname in source_files:
        with fname.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uid = uid_key(r)
                if uid in need:
                    texts[uid] = r["text"]
    print("texts loaded:", len(texts))
    out_rows = []
    sim_sum = 0.0
    sim_n = 0
    for r in removed:
        if r["priority"] == 0 or r["priority"] == 1:
            sim = 1.0
        else:
            sa = shingle_set(texts[r["id"]])
            sb = shingle_set(texts[r["kept_id"]])
            sim = round(jaccard(sa, sb), 6)
            sim_sum += sim
            sim_n += 1
        out_rows.append({**r, "similarity": sim})
    out_name = "dedup_removed_similarity.jsonl.gz" if removed_name == "dedup_removed_list.jsonl.gz" else "cross_source_removed_similarity.jsonl.gz"
    with gzip.open(dedup_dir / out_name, "wt", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_p = dedup_dir / "summary.json"
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    summary["removed_similarity_addendum"] = {
        "file": out_name,
        "near_dup_pairs": sim_n,
        "near_dup_avg_similarity": round(sim_sum / sim_n, 6) if sim_n else None,
        "note": "exact 4-gram Jaccard between each removed doc and its kept doc",
    }
    summary_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("written:", len(out_rows), "avg near-dup sim:", round(sim_sum / sim_n, 6) if sim_n else None)


# webText
wt = sorted((WS / "outputs/webText2019zh").glob("curated_candidate_v1_*"))[-1]
process(
    sorted((WS / "outputs/webText2019zh").glob("dedup_v1_*"))[-1],
    [wt / "tier_a_clean.jsonl", wt / "tier_b_clean.jsonl"],
    lambda r: r["document_id"],
)

# cross-source
cs = sorted((WS / "outputs/cross_source").glob("cross_source_dedup_v1_*"))[-1]
news = WS / "inputs/cleaned/news2016zh/news_tier_a_permanent.jsonl"
process(
    cs,
    [news],
    lambda r: f"news2016zh::{r['source_file']}:{r['block_index']}",
    removed_name="cross_source_removed_list.jsonl.gz",
)
