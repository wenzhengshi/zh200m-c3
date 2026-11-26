import json

p = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/news2016zh_v22/news_freeze_v3_2/news_tier_a_permanent.jsonl"
with open(p, encoding="utf-8") as f:
    r = json.loads(f.readline())
print("news keys:", sorted(r.keys()))
print("id:", r.get("document_id") or r.get("id") or r.get("block_index"))
text = r.get("text") or r.get("body") or r.get("content")
print("text head:", repr(text)[:150] if text else None)

p2 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/zhwikisource_classics_v6/accepted_final.jsonl"
with open(p2, encoding="utf-8") as f:
    r2 = json.loads(f.readline())
print("wikisource keys:", sorted(r2.keys()))
print("wikisource id candidate:", r2.get("work"), r2.get("title"))
