#!/usr/bin/env python3
"""webText2019zh dedup v1 (disk-backed): exact -> normalized -> MinHash near-dup -> repost families."""
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
DS = os.environ.get("DEDUP_DATASET", "webText2019zh")
SRC = sorted((WS / f"outputs/{DS}").glob("curated_candidate_v1_*"))[-1]
OUT = WS / f"outputs/{DS}/dedup_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
TMP = Path(f"/tmp/{DS}_dedup_work")
TMP.mkdir(parents=True, exist_ok=True)

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


print("source:", SRC, "tmp:", TMP, flush=True)

# ---- phase 0/1: exact + normalized dedup (ids only, no text retention) ----
exact = defaultdict(list)
meta = {}  # id -> (tier, chars, sha)
survivor_set = set()

for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B")):
    with (SRC / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            if not text.strip():
                continue
            chars = len(text) - text.count("\n")
            sh = sha(text)
            meta[r["document_id"]] = (tier, chars, sh)
            exact[sh].append(r["document_id"])

removed = []
status = {}
clusters = []


def winner_key(did):
    t, c, _ = meta[did]
    return (0 if t == "A" else 1, -c, did)


for group in exact.values():
    if len(group) <= 1:
        survivor_set.add(group[0])
        continue
    group.sort(key=winner_key)
    kept = group[0]
    survivor_set.add(kept)
    for m in group[1:]:
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "exact_duplicate", "priority": 0})
    clusters.append({"phase": "exact", "size": len(group), "kept_id": kept, "removed_ids": group[1:], "similarity": 1.0})
print("phase0 exact removed:", sum(1 for r in removed), "survivors:", len(survivor_set), flush=True)
del exact

# normalized
norm_map = defaultdict(list)
with (SRC / "tier_a_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in survivor_set:
            norm_map[sha(norm(r["text"]))].append(r["document_id"])
with (SRC / "tier_b_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in survivor_set:
            norm_map[sha(norm(r["text"]))].append(r["document_id"])
for group in norm_map.values():
    if len(group) <= 1:
        continue
    group.sort(key=winner_key)
    kept = group[0]
    for m in group[1:]:
        survivor_set.discard(m)
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "normalized_duplicate", "priority": 1})
    clusters.append({"phase": "normalized", "size": len(group), "kept_id": kept, "removed_ids": group[1:], "similarity": 1.0})
print("phase1 normalized removed:", sum(1 for r in removed if r["priority"] == 1), "survivors:", len(survivor_set), flush=True)
del norm_map

# ---- phase 2a: build disk shingle store + sketches for survivors >= 200 chars ----
if not (TMP / "sketches.txt").exists():
    idx_lines = []
    sketch_lines = []
    shingle_f = open(TMP / "shingles.bin", "wb")
    for name in ("tier_a_clean.jsonl", "tier_b_clean.jsonl"):
        with (SRC / name).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                did = r["document_id"]
                if did not in survivor_set:
                    continue
                chars = meta[did][1]
                if chars < 200:
                    continue
                text = r["text"]
                sh = rolling_shingles(text)
                if len(sh) < 4:
                    continue
                sk = bottom32(sh)
                offset = shingle_f.tell()
                shingle_f.write(np.sort(np.unique(sh)).tobytes())
                count = len(np.unique(sh))
                idx_lines.append(f"{did}\t{offset}\t{count}\n")
                sketch_lines.append(did + "\t" + "\t".join(str(int(x)) for x in sk) + "\n")
    shingle_f.close()
    with (TMP / "shingle_index.txt").open("w", encoding="utf-8") as f:
        f.writelines(idx_lines)
    with (TMP / "sketches.txt").open("w", encoding="utf-8") as f:
        f.writelines(sketch_lines)
    print("sketched docs:", len(idx_lines), flush=True)
    del idx_lines, sketch_lines
else:
    print("sketches checkpoint found, skipping phase 2a", flush=True)

# ---- phase 2b: banded LSH via external sort ----
if not (TMP / "pairs_uniq.txt").exists():
    for b in range(BANDS):
        band_file = TMP / f"band_{b}.txt"
        with band_file.open("w", encoding="utf-8") as out:
            with (TMP / "sketches.txt").open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    did = parts[0]
                    key = tuple(parts[1 + b * ROWS:1 + (b + 1) * ROWS])
                    out.write("\t".join(key) + "\t" + did + "\n")
        subprocess.run(["sort", str(band_file), "-o", str(band_file)], check=True)
        pairs_f = open(TMP / "pairs_raw.txt", "a", encoding="utf-8")
        prev_key = None
        bucket = []
        with band_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                key = tuple(parts[:-1])
                did = parts[-1]
                if key != prev_key:
                    if len(bucket) > 1:
                        bucket.sort()
                        if len(bucket) > 100:
                            bucket = bucket[:100]
                        for i in range(len(bucket)):
                            for j in range(i + 1, len(bucket)):
                                pairs_f.write(f"{bucket[i]}\t{bucket[j]}\n")
                    bucket = [did]
                    prev_key = key
                else:
                    bucket.append(did)
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
else:
    print("pairs checkpoint found, skipping phase 2b", flush=True)
pair_count = sum(1 for _ in (TMP / "pairs_uniq.txt").open(encoding="utf-8"))
print("unique candidate pairs:", pair_count, flush=True)

# ---- phase 2c: exact Jaccard via disk shingle store ----
shingle_index = {}
with (TMP / "shingle_index.txt").open("r", encoding="utf-8") as f:
    for line in f:
        did, off, cnt = line.rstrip("\n").split("\t")
        shingle_index[did] = (int(off), int(cnt))

if not (TMP / "sims.txt").exists():
    with open(TMP / "shingles.bin", "rb") as sf:
        with (TMP / "sims.txt").open("w", encoding="utf-8") as simf:
            with (TMP / "pairs_uniq.txt").open("r", encoding="utf-8") as pf:
                for line in pf:
                    a, b = line.rstrip("\n").split("\t")
                    off_a, cnt_a = shingle_index[a]
                    off_b, cnt_b = shingle_index[b]
                    sf.seek(off_a)
                    sa = np.frombuffer(sf.read(cnt_a * 8), dtype=np.uint64)
                    sf.seek(off_b)
                    sb = np.frombuffer(sf.read(cnt_b * 8), dtype=np.uint64)
                    inter = len(np.intersect1d(sa, sb))
                    union = cnt_a + cnt_b - inter
                    sim = inter / union if union else 0.0
                    if sim >= 0.55:
                        simf.write(f"{a}\t{b}\t{sim:.6f}\n")
    print("sims written", flush=True)
else:
    print("sims checkpoint found, skipping phase 2c", flush=True)

# ---- phase 2d: union-find streaming ----
all_ids = sorted(survivor_set)
id_index = {did: i for i, did in enumerate(all_ids)}
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


sims = []
with (TMP / "sims.txt").open("r", encoding="utf-8") as f:
    for line in f:
        a, b, s = line.rstrip("\n").split("\t")
        s = float(s)
        sims.append((a, b, s))
        if s >= 0.85:
            union(id_index[a], id_index[b])

groups = defaultdict(list)
for i, did in enumerate(all_ids):
    groups[find(i)].append(did)

near_removed = 0
for members in groups.values():
    if len(members) <= 1:
        continue
    members.sort(key=winner_key)
    kept = members[0]
    for m in members[1:]:
        survivor_set.discard(m)
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "near_duplicate_repost", "priority": 2})
        near_removed += 1
    clusters.append({"phase": "near_dup", "size": len(members), "kept_id": kept, "removed_ids": members[1:], "similarity": None})
print("near_dup removed:", near_removed, flush=True)

# repost families (>=0.60), analysis only
parent2 = list(range(len(all_ids)))


def find2(x):
    while parent2[x] != x:
        parent2[x] = parent2[parent2[x]]
        x = parent2[x]
    return x


for a, b, s in sims:
    if s >= 0.60:
        ra, rb = find2(id_index[a]), find2(id_index[b])
        if ra != rb:
            parent2[rb] = ra
reposts = 0
rep_docs = 0
groups2 = defaultdict(list)
for i, did in enumerate(all_ids):
    groups2[find2(i)].append(did)
for members in groups2.values():
    if len(members) <= 1:
        continue
    members.sort(key=winner_key)
    rep_docs += len(members)
    reposts += 1
    clusters.append({"phase": "repost_family", "size": len(members), "kept_id": members[0], "removed_ids": members[1:], "similarity": None, "note": "analysis only"})
print("repost families:", reposts, "docs:", rep_docs, flush=True)

# ---- outputs ----
kept = []
for did in survivor_set:
    t, c, sh = meta[did]
    kept.append({"document_id": did, "tier": t, "character_count": c, "text_sha256": sh})
kept.sort(key=lambda x: x["document_id"])
with gzip.open(OUT / "dedup_keep_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "dedup_removed_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in removed:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with gzip.open(OUT / "duplicate_clusters.jsonl.gz", "wt", encoding="utf-8") as f:
    for c in clusters:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

summary = {
    "stage": "dedup_v1",
    "dataset": DS,
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "documents_in_scope": len(meta),
    "removed_reason_counts": dict(Counter(r["reason"] for r in removed)),
    "total_removed": len(removed),
    "documents_kept": len(kept),
    "characters_kept": sum(k["character_count"] for k in kept),
    "outputs": {"keep_list": "dedup_keep_list.jsonl.gz", "removed_list": "dedup_removed_list.jsonl.gz", "clusters": "duplicate_clusters.jsonl.gz"},
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "run.log").write_text(
    f"dedup_v1 run.log\ndataset: {DS}\ncreated_at: {summary['created_at']}\n"
    f"in scope: {len(meta):,}\nremoved: {summary['removed_reason_counts']}\nkept: {len(kept):,}\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
