import gzip
import json

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v2_20260801_232734/document_manifest.jsonl.gz"
ids = {
    "wiki_0_string.txt_new.txt:0000136",
    "wiki_0_string.txt_new.txt:0001253",
    "wiki_0_string.txt_new.txt:0001727",
    "wiki_0_string.txt_new.txt:0001813",
    "wiki_102_string.txt_new.txt:0000698",
}
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in ids:
            print(r["document_id"], "| primary:", r["primary_reason"])
            print("  actions:", r["repair_actions"])
            print("  reasons:", r["reasons"])
