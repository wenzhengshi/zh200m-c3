import gzip, json
from collections import Counter

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/classification_manifest_v3.jsonl.gz"

focus = [
    "A->DROP", "A->REVIEW", "A->B",
    "B->DROP", "B->REVIEW", "B->A",
    "DROP->A", "DROP->B", "DROP->REVIEW",
    "REVIEW->A", "REVIEW->B", "REVIEW->DROP",
]
trans = {t: Counter() for t in focus}
total = 0
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = f"{r['v2_tier']}->{r['tier']}"
        if t in trans:
            trans[t][r["primary_reason"]] += 1
        total += 1

print("total_documents:", total)
for t in focus:
    c = trans[t]
    if not c:
        print(f"\n{t}: 0")
        continue
    print(f"\n{t}: {sum(c.values())}")
    for reason, n in c.most_common(15):
        print(f"  {reason}: {n}")
