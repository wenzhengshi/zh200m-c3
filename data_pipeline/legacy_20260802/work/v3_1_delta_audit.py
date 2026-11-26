#!/usr/bin/env python3
"""Per-document audit of the v3->v3.1 delta."""
import gzip
import json

M3 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/classification_manifest_v3.jsonl.gz"
M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"

v3 = {}
with gzip.open(M3, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        v3[r["document_id"]] = r

changed = []
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        old = v3[r["document_id"]]
        if old["tier"] != r["tier"] or old["primary_reason"] != r["primary_reason"]:
            changed.append((r["document_id"], old["tier"], old["primary_reason"], r["tier"], r["primary_reason"]))

print("docs with tier or primary-reason change:", len(changed))
from collections import Counter
tc = Counter((a, c) for _, a, _, c, _ in changed)
for k, v in sorted(tc.items()):
    print(" ", k, v)

# specific recovery sets check
pure_recover = [
    "wiki_237_string.txt_new.txt:0002044",  # 乘氏县
    "wiki_86_string.txt_new.txt:0000555",   # 2016年MTV电影大奖
    "wiki_163_string.txt_new.txt:0002453",  # 苏格兰PFA
    "wiki_103_string.txt_new.txt:0000768",  # 化学品列表
]
tail_recover = [
    "wiki_230_string.txt_new.txt:0000600",  # 佃农 (1,049 chars)
]
all_recover = set(pure_recover + tail_recover)
print("\nspot-check recovery docs:")
for did in pure_recover + tail_recover:
    old = v3[did]
    print(f"  {did}")
    print(f"    v3: tier={old['tier']} primary={old['primary_reason']} chars={old['character_count']}")
print("\n(others read from v3.1 manifest in main loop)")
