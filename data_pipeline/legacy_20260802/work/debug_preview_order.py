import gzip
import json

PREVIEW = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/cleaning_preview_v2_20260801_230409/preview_all.jsonl.gz"
M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"

prev_ids = []
with gzip.open(PREVIEW, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        prev_ids.append(r["document_id"])
        if i >= 300:
            break

man_ids = []
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"].startswith("wiki_0_string.txt_new.txt") and r["repair_actions"]:
            man_ids.append(r["document_id"])
        if len(man_ids) >= 300:
            break

print("preview first 10:", prev_ids[:10])
print("manifest action first 10:", man_ids[:10])
if prev_ids[: len(man_ids)] == man_ids:
    print("MATCH for first", len(man_ids), "wiki_0 action docs")
else:
    for i, (a, b) in enumerate(zip(prev_ids, man_ids)):
        if a != b:
            print("first mismatch at", i, "preview=", a, "manifest=", b)
            break
