import gzip
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/webText2019zh").glob("curated_candidate_v1_*"))[-1]
print("dir:", D)
s = json.loads((D / "summary.json").read_text(encoding="utf-8"))
print("tiers:", s["tier_document_counts"])
print("cleaned:", s["documents_cleaned"])
empty = s["empty_after_cleaning"]
print("empty_after count:", len(empty))
for name in ("tier_a_clean.jsonl", "tier_b_clean.jsonl", "review.jsonl", "drop.jsonl"):
    n = sum(1 for _ in (D / name).open(encoding="utf-8"))
    print(name, n)
with gzip.open(D / "document_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    n = sum(1 for _ in f)
print("manifest rows:", n)
total = sum(s["tier_document_counts"].values())
print("tier sum:", total, "expected 4060616")
