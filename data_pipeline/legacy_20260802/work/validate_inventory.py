import json, sys

p = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/takeover_inventory_v1_20260801_220145/summary.json"
with open(p, encoding="utf-8") as f:
    s = json.load(f)
print("summary.json valid JSON")
print("v3 executed:", s["v3_status"]["executed"])
print("symlinks:", s["symlink_verification"])
print("tier v2:", s["wiki2019zh_audits"]["tier_dryrun_v2"]["tier_counts"])
