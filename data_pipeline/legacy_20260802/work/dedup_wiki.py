#!/usr/bin/env python3
"""wiki2019zh candidate dedup: exact -> normalized -> MinHash near-dup -> template report."""
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import os

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
LIMIT = int(os.environ.get("DEDUP_LIMIT", "0"))
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
SRC = WS / "outputs/wiki2019zh/curated_candidate_v4_20260801_234934"
OUT = WS / f"outputs/wiki2019zh/dedup_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

# load quarantine ids
quarantine = set()
with (SRC / "residue_quarantine.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        quarantine.add(json.loads(line)["document_id"])
print("quarantine docs:", len(quarantine))

docs = []  # list of dicts: id, tier, text, chars
for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B")):
    with (SRC / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["document_id"] in quarantine:
                continue
            text = r["text"]
            docs.append(
                {
                    "id": r["document_id"],
                    "tier": tier,
                    "text": text,
                    "chars": len(text) - text.count("\n"),
                }
            )

print("docs in scope:", len(docs))
if LIMIT:
    docs = docs[:LIMIT]
    print("LIMITED to", len(docs))

doc_index = {d["id"]: d for d in docs}


def winner_key(d):
    # prefer A tier, then more chars, then lexicographically smaller id
    return (0 if d["tier"] == "A" else 1, -d["chars"], d["id"])


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", t)


status = {}  # doc id -> "kept"/"removed"
removed = []  # list of {id, kept_id, reason, priority}
clusters_out = []


def add_cluster(phase, kept, members, sim=None):
    clusters_out.append(
        {
            "phase": phase,
            "cluster_id": len(clusters_out),
            "size": len(members),
            "kept_id": kept,
            "kept_tier": doc_index[kept]["tier"],
            "removed_ids": [m for m in members if m != kept],
            "similarity": sim,
            "sha256": {mid: sha(doc_index[mid]["text"]) for mid in members},
        }
    )


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- phase 0: exact ----
exact = defaultdict(list)
for d in docs:
    exact[sha(d["text"])].append(d)
exact_clusters = 0
exact_removed = 0
for h, group in exact.items():
    if len(group) <= 1:
        continue
    group.sort(key=winner_key)
    kept = group[0]["id"]
    for d in group[1:]:
        status[d["id"]] = "removed"
        removed.append({"id": d["id"], "kept_id": kept, "reason": "exact_duplicate", "priority": 0})
        exact_removed += 1
    add_cluster("exact", kept, [d["id"] for d in group], sim=1.0)
    exact_clusters += 1

survivors = [d for d in docs if status.get(d["id"]) != "removed"]
print("phase0 exact: clusters", exact_clusters, "removed", exact_removed, "survivors", len(survivors))

# ---- phase 1: normalized (NFKC + remove whitespace) ----
norm_map = defaultdict(list)
for d in survivors:
    norm_map[sha(norm(d["text"]))].append(d)
norm_clusters = 0
norm_removed = 0
for h, group in norm_map.items():
    if len(group) <= 1:
        continue
    group.sort(key=winner_key)
    kept = group[0]["id"]
    for d in group[1:]:
        status[d["id"]] = "removed"
        removed.append({"id": d["id"], "kept_id": kept, "reason": "normalized_duplicate", "priority": 1})
        norm_removed += 1
    add_cluster("normalized", kept, [d["id"] for d in group], sim=1.0)
    norm_clusters += 1

survivors = [d for d in survivors if status.get(d["id"]) != "removed"]
print("phase1 normalized: clusters", norm_clusters, "removed", norm_removed, "survivors", len(survivors))

survivor_index = {d["id"]: d for d in survivors}

# ---- phase 2: MinHash near-dup (bottom-64, char 4-grams) ----
K = 64
BANDS = 16
ROWS = K // BANDS
P = (1 << 61) - 1
BASE = 131
MASK64 = (1 << 64) - 1


def rolling_shingles(text: str, n=4):
    """Polynomial rolling hash of char 4-grams -> list of ints (mod 2^64)."""
    codes = [ord(c) for c in text]
    if len(codes) < n:
        return []
    h = 0
    pow_base = 1
    for i in range(n):
        h = (h * BASE + codes[i]) & MASK64
        pow_base = (pow_base * BASE) & MASK64
    out = [h]
    for i in range(n, len(codes)):
        h = (h * BASE + codes[i] - codes[i - n] * pow_base) & MASK64
        out.append(h)
    return out


def bottom64(shingles):
    if len(shingles) <= K:
        return np.array(sorted(shingles), dtype=np.uint64)
    arr = np.array(shingles, dtype=np.uint64)
    # universal hash to randomize
    arr = (arr * 0x9E3779B97F4A7C15) & MASK64
    part = np.partition(arr, K - 1)[:K]
    return np.sort(part)


sketches = {}  # id -> list of 64 sorted ints (LSH only)
for d in survivors:
    if d["chars"] < 100:
        continue
    sh = rolling_shingles(d["text"])
    if not sh:
        continue
    sketches[d["id"]] = bottom64(sh)

print("docs with sketches:", len(sketches))

# LSH candidate pairs: process one band at a time to bound memory
seen_pairs = set()
BUCKET_MEMBER_CAP = 200
PAIR_CAP = 10_000_000
pair_cap_hit = False

for b in range(BANDS):
    buckets = defaultdict(list)
    for did, sk in sketches.items():
        key = tuple(int(x) for x in sk[b * ROWS:(b + 1) * ROWS])
        buckets[key].append(did)
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        members.sort()
        if len(members) > BUCKET_MEMBER_CAP:
            members = members[:BUCKET_MEMBER_CAP]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                if pair in seen_pairs:
                    continue
                if len(seen_pairs) >= PAIR_CAP:
                    pair_cap_hit = True
                    break
                seen_pairs.add(pair)
            if pair_cap_hit:
                break
        if pair_cap_hit:
            break
    if pair_cap_hit:
        break
    print(f"band {b} done, candidate pairs so far: {len(seen_pairs):,}", flush=True)

print("candidate pairs:", len(seen_pairs), "cap_hit:", pair_cap_hit)


# exact Jaccard for candidate pairs (lazy shingle-set cache)
shingle_cache = {}


SHINGLE_CACHE_MAX = 80_000


def shingle_set(did):
    s = shingle_cache.get(did)
    if s is None:
        d = survivor_index[did]
        s = set(rolling_shingles(d["text"]))
        shingle_cache[did] = s
        if len(shingle_cache) > SHINGLE_CACHE_MAX:
            shingle_cache.clear()
    return s


pair_sims = []
for a, b in seen_pairs:
    sa, sb = shingle_set(a), shingle_set(b)
    sim = len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0
    pair_sims.append((a, b, sim))
print("exact pair sims computed:", len(pair_sims), "cache size:", len(shingle_cache))


def union_find_clusters(pairs, threshold):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b, sim in pairs:
        if sim >= threshold:
            union(a, b)
    groups = defaultdict(list)
    for did in sketches:
        parent.setdefault(did, did)
        groups[find(did)].append(did)
    return groups


near_groups = union_find_clusters(pair_sims, 0.85)
near_clusters = 0
near_removed = 0
near_cluster_rows = []
for root, members in near_groups.items():
    if len(members) <= 1:
        continue
    members.sort(key=lambda x: winner_key(survivor_index[x]))
    kept = members[0]
    sims = [s for m in members[1:] for a, b, s in pair_sims if {a, b} == {kept, m}]
    avg_sim = round(sum(sims) / len(sims), 4) if sims else None
    for m in members[1:]:
        status[m] = "removed"
        removed.append({"id": m, "kept_id": kept, "reason": "near_duplicate", "priority": 2, "similarity": avg_sim})
        near_removed += 1
    near_clusters += 1
    near_cluster_rows.append(
        {
            "phase": "near_dup",
            "cluster_id": len(clusters_out) + len(near_cluster_rows),
            "size": len(members),
            "kept_id": kept,
            "removed_ids": members[1:],
            "similarity": avg_sim,
        }
    )

survivors = [d for d in survivors if status.get(d["id"]) != "removed"]
print("phase2 near-dup: clusters", near_clusters, "removed", near_removed, "survivors", len(survivors))

# ---- phase 3: template/family clusters (>=0.60, analysis only) ----
template_groups = union_find_clusters(pair_sims, 0.60)
template_rows = []
template_docs_involved = 0
for root, members in template_groups.items():
    if len(members) <= 1:
        continue
    members.sort(key=lambda x: winner_key(doc_index[x]))
    family_sims = [s for a, b, s in pair_sims if a in members and b in members]
    avg_fam_sim = round(sum(family_sims) / len(family_sims), 4) if family_sims else None
    kept = members[0]
    template_docs_involved += len(members)
    template_rows.append(
        {
            "phase": "template_family",
            "cluster_id": len(clusters_out) + 100000 + len(template_rows),
            "size": len(members),
            "kept_id": kept,
            "avg_similarity": avg_fam_sim,
            "docs": members,
            "chars_total": sum(doc_index[m]["chars"] for m in members),
            "note": "analysis only; template-generated families are not deleted at this threshold",
        }
    )

print("phase3 template families (>=0.60):", len(template_rows), "docs involved:", template_docs_involved)

# ---- outputs ----
kept_rows = []
for d in survivors:
    kept_rows.append(
        {
            "document_id": d["id"],
            "tier": d["tier"],
            "character_count": d["chars"],
            "text_sha256": sha(d["text"]),
        }
    )

with gzip.open(OUT / "dedup_keep_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in kept_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "dedup_removed_list.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in removed:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "duplicate_clusters.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in clusters_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in near_cluster_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with gzip.open(OUT / "template_cluster_report.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in template_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "stage": "dedup_v1",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "source": str(SRC),
    "documents_in_scope": len(docs),
    "quarantine_excluded": len(quarantine),
    "phase_exact": {"clusters": exact_clusters, "removed": exact_removed},
    "phase_normalized": {"clusters": norm_clusters, "removed": norm_removed},
    "phase_near_dup": {"clusters": near_clusters, "removed": near_removed, "threshold": 0.85},
    "phase_template_report": {
        "families": len(template_rows),
        "docs_involved": template_docs_involved,
        "threshold": 0.60,
        "note": "analysis only; no deletion at this threshold",
    },
    "total_removed": len(removed),
    "documents_kept": len(survivors),
    "characters_kept": sum(d["chars"] for d in survivors),
    "removed_reason_counts": dict(Counter(r["reason"] for r in removed)),
    "outputs": {
        "keep_list": "dedup_keep_list.jsonl.gz",
        "removed_list": "dedup_removed_list.jsonl.gz",
        "clusters": "duplicate_clusters.jsonl.gz",
        "template_report": "template_cluster_report.jsonl.gz",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = [
    f"dedup_v1 run.log",
    f"created_at: {summary['created_at']}",
    f"documents in scope: {len(docs):,}",
    f"exact: clusters={exact_clusters} removed={exact_removed}",
    f"normalized: clusters={norm_clusters} removed={norm_removed}",
    f"near_dup(>=0.85): clusters={near_clusters} removed={near_removed}",
    f"template families(>=0.60): {len(template_rows)} families, {template_docs_involved} docs (analysis only)",
    f"kept: {len(survivors):,} docs, {sum(d['chars'] for d in survivors):,} chars",
]
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
