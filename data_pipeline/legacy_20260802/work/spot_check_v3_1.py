import gzip, json

M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"
want = {
    "wiki_237_string.txt_new.txt:0002044": "乘氏县",
    "wiki_86_string.txt_new.txt:0000555": "2016年MTV电影大奖",
    "wiki_163_string.txt_new.txt:0002453": "苏格兰PFA最佳领队",
    "wiki_103_string.txt_new.txt:0000768": "化学品列表",
    "wiki_230_string.txt_new.txt:0000600": "佃农",
}
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in want:
            print(f"{want[r['document_id']]} | {r['document_id']}")
            print(f"  v3.1 tier={r['tier']} primary={r['primary_reason']} chars={r['character_count']} paras={r['paragraph_count']}")
            print(f"  reasons={r['reasons']}")
            print(f"  repair_actions={r['repair_actions']}")
            del want[r["document_id"]]
            if not want:
                break
