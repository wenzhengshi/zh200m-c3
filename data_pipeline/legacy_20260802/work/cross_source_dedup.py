#!/usr/bin/env python3
"""Cross-source dedup across all frozen corpora (exact -> normalized -> near-dup + news repost report)."""
import gzip
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / f"outputs/cross_source/cross_source_dedup_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
TMP = Path("/tmp/cross_dedup_work")
TMP.mkdir(parents=True, exist_ok=True)

# source -> (root, tier files)
SOURCES = {
    "news2016zh": ("inputs/cleaned/news2016zh", ["news_tier_a_permanent.jsonl"]),
    "zhwikisource_classics": ("inputs/cleaned/zhwikisource_classics", ["accepted_final.jsonl"]),
    "wiki2019zh": ("outputs/wiki2019zh", ["wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl"]),
    "webText2019zh": ("outputs/webText2019zh", ["webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", "webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl"]),
    "comment2019zh": ("outputs/comment2019zh", ["comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", "comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl"]),
}
SRC_PRIORITY = {"news2016zh": 0, "zhwikisource_classics": 1, "wiki2019zh": 2, "webText2019zh": 3, "comment2019zh": 4}


def iter_frozen(source):
    root, files = SOURCES[source]
    for fname in files:
        if "tier_a" in fname or "permanent" in fname or "accepted_final" in fname:
            tier = "A"
        else:
            tier = "B"
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = r["text"]
                if source == "news2016zh":
                    uid = f"{source}::{r['source_file']}:{r['block_index']}"
                elif source == "zhwikisource_classics":
                    uid = f"{source}::{r['work']}/{r['title']}"
                else:
                    uid = f"{source}::{r['document_id']}"
                if not text.strip():
                    continue
                yield uid, tier, text

MASK64 = (1 << 64) - 1
BASE = 131
K = 32
BANDS = 8
ROWS = K // BANDS


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rolling_shingles(text, n=4):
    codes = [ord(c) for c in text]
    if len(codes) < n:
        return np.array([], dtype=np.uint64)
    h = 0
    pw = 1
    for i in range(n):
        h = (h * BASE + codes[i]) & MASK64
        pw = (pw * BASE) & MASK64
    out = [h]
    for i in range(n, len(codes)):
        h = (h * BASE + codes[i] - codes[i - n] * pw) & MASK64
        out.append(h)
    return np.array(out, dtype=np.uint64)


def bottom32(sh):
    if len(sh) <= K:
        return np.array(sorted(sh.tolist()), dtype=np.uint64)
    arr = (sh * 0x9E3779B97F4A7C15) & MASK64
    return np.sort(np.partition(arr, K - 1)[:K])


meta = {}  # uid -> (source, tier, chars, sha)
exact = defaultdict(list)
total = 0
for source in SOURCES:
    for uid, tier, text in iter_frozen(source):
        chars = len(text) - text.count("\n")
        sh = sha(text)
        meta[uid] = (source, tier, chars, sh)
        exact[sh].append(uid)
        total += 1
print("total docs:", total, flush=True)

removed = []
status = {}
clusters = []
survivor = set()


def winner_key(uid):
    src, tier, chars, _ = meta[uid]
    return (SRC_PRIORITY[src], 0 if tier == "A" else 1, -chars, uid)


for group in exact.values():
    if len(group) <= 1:
        survivor.add(group[0])
        continue
    group.sort(key=winner_key)
    kept = group[0]
    survivor.add(kept)
    for m in group[1:]:
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "cross_source_exact_duplicate", "priority": 0})
    clusters.append({"phase": "exact", "size": len(group), "kept_id": kept, "removed_ids": group[1:], "similarity": 1.0})
print("exact removed:", sum(1 for r in removed if r["priority"] == 0), "survivors:", len(survivor), flush=True)
del exact

# normalized (stream frozen files again, only for survivors)
norm_map = defaultdict(list)
for source in SOURCES:
    for uid, tier, text in iter_frozen(source):
        if uid in survivor:
            norm_map[sha(norm(text))].append(uid)
for group in norm_map.values():
    if len(group) <= 1:
        continue
    group.sort(key=winner_key)
    kept = group[0]
    for m in group[1:]:
        survivor.discard(m)
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "cross_source_normalized_duplicate", "priority": 1})
    clusters.append({"phase": "normalized", "size": len(group), "kept_id": kept, "removed_ids": group[1:], "similarity": 1.0})
print("normalized removed:", sum(1 for r in removed if r["priority"] == 1), "survivors:", len(survivor), flush=True)
del norm_map

# near-dup (>=0.85) via disk-backed MinHash for docs >= 200 chars
if not (TMP / "shingle_index.txt").exists() or not (TMP / "sketches.txt").exists():
    idx_lines = []
    sketch_lines = []
    shingle_f = open(TMP / "shingles.bin", "wb")
    for source in SOURCES:
        for uid, tier, text in iter_frozen(source):
            if uid not in survivor:
                continue
            if meta[uid][2] < 200:
                continue
            sh = rolling_shingles(text)
            if len(sh) < 4:
                continue
            sk = bottom32(sh)
            u = np.sort(np.unique(sh))
            offset = shingle_f.tell()
            shingle_f.write(u.tobytes())
            idx_lines.append(f"{uid}\t{offset}\t{len(u)}\n")
            sketch_lines.append(uid + "\t" + "\t".join(str(int(x)) for x in sk) + "\n")
    shingle_f.close()
    with (TMP / "shingle_index.txt").open("w", encoding="utf-8") as f:
        f.writelines(idx_lines)
    with (TMP / "sketches.txt").open("w", encoding="utf-8") as f:
        f.writelines(sketch_lines)
    print("sketched docs:", len(idx_lines), flush=True)
else:
    print("sketch checkpoint found", flush=True)

if not (TMP / "pairs_uniq.txt").exists():
    for b in range(BANDS):
        band_file = TMP / f"band_{b}.txt"
        with band_file.open("w", encoding="utf-8") as out:
            with (TMP / "sketches.txt").open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    uid = parts[0]
                    out.write("\t".join(parts[1 + b * ROWS:1 + (b + 1) * ROWS]) + "\t" + uid + "\n")
        subprocess.run(["sort", str(band_file), "-o", str(band_file)], check=True)
        pairs_f = open(TMP / "pairs_raw.txt", "a", encoding="utf-8")
        prev_key = None
        bucket = []
        with band_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                key = tuple(parts[:-1])
                uid = parts[-1]
                if key != prev_key:
                    if len(bucket) > 1:
                        bucket.sort()
                        if len(bucket) > 100:
                            bucket = bucket[:100]
                        for i in range(len(bucket)):
                            for j in range(i + 1, len(bucket)):
                                pairs_f.write(f"{bucket[i]}\t{bucket[j]}\n")
                    bucket = [uid]
                    prev_key = key
                else:
                    bucket.append(uid)
        if len(bucket) > 1:
            bucket.sort()
            if len(bucket) > 100:
                bucket = bucket[:100]
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    pairs_f.write(f"{bucket[i]}\t{bucket[j]}\n")
        pairs_f.close()
        band_file.unlink()
        print(f"band {b} done", flush=True)
    subprocess.run(["sort", "-u", str(TMP / "pairs_raw.txt"), "-o", str(TMP / "pairs_uniq.txt")], check=True)
print("pairs ready", flush=True)

shingle_index = {}
with (TMP / "shingle_index.txt").open("r", encoding="utf-8") as f:
    for line in f:
        uid, off, cnt = line.rstrip("\n").split("\t")
        shingle_index[uid] = (int(off), int(cnt))

if not (TMP / "sims.txt").exists():
    with open(TMP / "shingles.bin", "rb") as sf:
        with (TMP / "sims.txt").open("w", encoding="utf-8") as simf:
            with (TMP / "pairs_uniq.txt").open("r", encoding="utf-8") as pf:
                for line in pf:
                    a, b = line.rstrip("\n").split("\t")
                    oa, ca = shingle_index[a]
                    ob, cb = shingle_index[b]
                    sf.seek(oa)
                    sa = np.frombuffer(sf.read(ca * 8), dtype=np.uint64)
                    sf.seek(ob)
                    sb = np.frombuffer(sf.read(cb * 8), dtype=np.uint64)
                    inter = len(np.intersect1d(sa, sb))
                    union = ca + cb - inter
                    sim = inter / union if union else 0.0
                    if sim >= 0.55:
                        simf.write(f"{a}\t{b}\t{sim:.6f}\n")
    print("sims written", flush=True)
del shingle_index

sims = []
paired_ids = set()
with (TMP / "sims.txt").open("r", encoding="utf-8") as f:
    for line in f:
        a, b, s = line.rstrip("\n").split("\t")
        sims.append((a, b, float(s)))
        paired_ids.add(a)
        paired_ids.add(b)
print("paired ids:", len(paired_ids), "sims:", len(sims), flush=True)

all_ids = sorted(paired_ids)
id_index = {uid: i for i, uid in enumerate(all_ids)}
parent = list(range(len(all_ids)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[ry] = rx


for a, b, s in sims:
    if s >= 0.85:
        union(id_index[a], id_index[b])
groups = defaultdict(list)
for i, uid in enumerate(all_ids):
    groups[find(i)].append(uid)

near_removed = 0
news_reposts = []
for members in groups.values():
    if len(members) <= 1:
        continue
    members.sort(key=winner_key)
    kept = members[0]
    srcs = {meta[m][0] for m in members}
    for m in members[1:]:
        survivor.discard(m)
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "cross_source_near_duplicate", "priority": 2})
        near_removed += 1
    clusters.append({"phase": "near_dup", "size": len(members), "kept_id": kept, "removed_ids": members[1:], "similarity": None, "sources": sorted(srcs)})
    if "news2016zh" in srcs and len(srcs) > 1:
        news_reposts.append({"kept_id": kept, "cluster_size": len(members), "members": members, "sources": sorted(srcs)})
print("near-dup removed:", near_removed, "news repost clusters:", len(news_reposts), flush=True)

kept = []
for uid in survivor:
    src, tier, chars, sh = meta[uid]
    kept.append({"document_id": uid, "source": src, "tier": tier, "character_count": chars, "text_sha256": sh})
kept.sort(key=lambda x: x["document_id"])
with gzip.open(OUT / "cross_source_keep_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "cross_source_removed_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in removed:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with gzip.open(OUT / "cross_source_clusters.jsonl.gz", "wt", encoding="utf-8") as f:
    for c in clusters:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with gzip.open(OUT / "news_repost_report.jsonl.gz", "wt", encoding="utf-8") as f:
    for nr in news_reposts:
        f.write(json.dumps(nr, ensure_ascii=False) + "\n")

summary = {
    "stage": "cross_source_dedup_v1",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "documents_in_scope": total,
    "removed_reason_counts": dict(Counter(r["reason"] for r in removed)),
    "total_removed": len(removed),
    "documents_kept": len(kept),
    "news_repost_clusters": len(news_reposts),
    "per_source_kept": dict(Counter(k["source"] for k in kept)),
    "note": "跨来源去重：精确/标准化/近重复(>=0.85)；新闻转载簇仅报告不删除。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run.log").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
