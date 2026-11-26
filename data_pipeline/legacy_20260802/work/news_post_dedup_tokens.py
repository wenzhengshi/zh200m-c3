import gzip
import json

NEWS = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/news2016zh_v22/news_freeze_v3_2/news_tier_a_permanent.jsonl"
REMOVED = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/cross_source_dedup_v1_20260802_082519/cross_source_removed_list.jsonl.gz"

removed_news = set()
with gzip.open(REMOVED, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"].startswith("news2016zh::"):
            removed_news.add(r["id"])
print("removed news ids:", len(removed_news))

total_tokens = 0
total_docs = 0
removed_tokens = 0
with open(NEWS, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        uid = f"news2016zh::{r['source_file']}:{r['block_index']}"
        total_docs += 1
        total_tokens += r.get("token_count_permanent", 0)
        if uid in removed_news:
            removed_tokens += r.get("token_count_permanent", 0)
print("news frozen docs:", total_docs, "tokens:", total_tokens)
print("removed news tokens:", removed_tokens)
print("kept news docs:", total_docs - len(removed_news), "tokens:", total_tokens - removed_tokens)
