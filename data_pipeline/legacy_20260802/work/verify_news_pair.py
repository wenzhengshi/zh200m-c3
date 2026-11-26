import json

base = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/news2016zh_v22/news_freeze_v3_2/news_tier_a_permanent.jsonl"
pairs = [
    ("news2016zh_1737.txt:345", "news2016zh_1.txt:402"),
]
recs = {}
with open(base, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = f"{r['source_file']}:{r['block_index']}".replace("data/cpt_modern_v1/raw/news2016zh/", "")
        if key in (pairs[0][0], pairs[0][1]):
            recs[key] = r["text"]
        if len(recs) == 2:
            break
for k in (pairs[0][0], pairs[0][1]):
    print("====", k)
    print(recs[k][:400])
    print()
