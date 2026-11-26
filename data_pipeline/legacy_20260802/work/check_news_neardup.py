import gzip
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/cross_source").glob("cross_source_dedup_v1_*"))[-1]
print("dir:", D)
removed = []
with gzip.open(D / "cross_source_removed_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["reason"] == "cross_source_near_duplicate":
            removed.append(r)
print("near-dup removed:", len(removed))
print("sample removed:", json.dumps(removed[:3], ensure_ascii=False))

# cluster sizes
from collections import Counter
c = Counter()
with gzip.open(D / "cross_source_clusters.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"] == "near_dup":
            c[r["size"]] += 1
print("near-dup cluster size distribution:", dict(sorted(c.items())))

# show texts of one cluster
with gzip.open(D / "cross_source_clusters.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"] == "near_dup" and r["size"] >= 3:
            print("cluster:", json.dumps(r, ensure_ascii=False)[:500])
            break
