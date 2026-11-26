import gzip, json

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/classification_manifest_v3.jsonl.gz"

targets = {
    "wiki_160_string.txt_new.txt:0002528": "永康工业区车站 (v2: A->B possible_incomplete_tail)",
    "wiki_214_string.txt_new.txt:0005390": "七里岗乡 (v2: A->B possible_incomplete_tail)",
    "wiki_244_string.txt_new.txt:0003472": "黑嘴树鸭 (v2: A->B possible_incomplete_tail)",
    "wiki_251_string.txt_new.txt:0000087": "黑嘴树鸭相关第2篇 (v2: A->B possible_incomplete_tail)",
}
found = {}
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        did = r["document_id"]
        if did in targets:
            found[did] = r
            if len(found) == len(targets):
                break

for did, label in targets.items():
    r = found.get(did)
    if not r:
        print(did, label, "-> NOT FOUND")
        continue
    print(f"{did} | {label}")
    print(f"  v2_tier={r['v2_tier']} v3_tier={r['tier']} primary={r['primary_reason']}")
    print(f"  reasons={r['reasons']}")
    print(f"  repair_actions={r['repair_actions']}")
    print(f"  chars={r['character_count']} para={r['paragraph_count']}")
