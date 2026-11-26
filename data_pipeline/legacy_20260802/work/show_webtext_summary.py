import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/webText2019zh").glob("tier_dryrun_v1_*"))[-1]
s = json.loads((D / "summary_tier_dryrun_v1.json").read_text(encoding="utf-8"))
print("dir:", D)
print("total:", s["total_documents"], "chars:", s["total_characters"])
print("tiers:", s["tier_counts"])
print("tier chars:", s["tier_character_counts"])
assert sum(s["tier_counts"].values()) == s["total_documents"]
print("reconcile OK")
