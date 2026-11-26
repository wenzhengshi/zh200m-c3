import json, sys

for name, path in [
    ("v1", "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/audits/wiki2019zh_tier_dryrun_v1/summary_tier_dryrun_v1.json"),
    ("v2", "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/audits/wiki2019zh_tier_dryrun_v2/summary_tier_dryrun_v2.json"),
]:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    counts = s["tier_counts"]
    chars = s["tier_character_counts"]
    total = s["total_documents"]
    doc_sum = sum(counts.values())
    char_sum = sum(chars.values())
    print(f"{name}: doc_sum={doc_sum} total={total} match={doc_sum==total} char_sum={char_sum}")
    if "transition_counts" in s:
        tc = s["transition_counts"]
        print(f"{name}: transition_sum={sum(tc.values())} match_total={sum(tc.values())==total}")

with open("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/audits/wiki2019zh_tier_dryrun_v2/summary_tier_dryrun_v2.json", encoding="utf-8") as f:
    s2 = json.load(f)
reason_p = s2["primary_reason_counts"]
reason_o = s2["overlapping_reason_counts"]
print("v2 primary_reason_sum:", sum(reason_p.values()))
print("v2 overlapping_reason_sum:", sum(reason_o.values()))
repair = s2["repair_action_counts"]
print("v2 repair_action_sum:", sum(repair.values()))
