import gzip
import json

D = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/dedup_v1_20260802_001902"
n = 0
with gzip.open(D + "/duplicate_clusters.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["phase"] == "near_dup" and r["size"] >= 3:
            print(json.dumps(r, ensure_ascii=False)[:600])
            print()
            n += 1
            if n >= 3:
                break
if n == 0:
    print("no near_dup clusters with size >= 3")

# template family samples
print("---- template families ----")
with gzip.open(D + "/template_cluster_report.jsonl.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        if r["size"] >= 20:
            print(json.dumps(r, ensure_ascii=False)[:400])
            print()
            i += 1
            if i >= 3:
                break
