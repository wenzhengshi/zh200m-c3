import gzip, json
from collections import Counter

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837/document_manifest.jsonl.gz"
per = Counter()
chars = Counter()
n = 0
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        per[r["tier"]] += 1
        chars[r["tier"]] += r["character_count"]
        n += 1
print("rows:", n)
print("per-tier docs:", dict(per))
print("per-tier chars (manifest):", dict(chars))
print("total:", sum(chars.values()))
