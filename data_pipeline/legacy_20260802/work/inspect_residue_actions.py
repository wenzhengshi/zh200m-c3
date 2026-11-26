import gzip
import json
import re
from pathlib import Path

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837/document_manifest.jsonl.gz"
ids = {
    "wiki_0_string.txt_new.txt:0001615",
    "wiki_122_string.txt_new.txt:0001340",
    "wiki_102_string.txt_new.txt:0000698",
    "wiki_124_string.txt_new.txt:0001415",
    "wiki_33_string.txt_new.txt:0002378",
    "wiki_202_string.txt_new.txt:0004827",
    "wiki_21_string.txt_new.txt:0002094",
}
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in ids:
            print(r["document_id"], "| tier:", r["tier"], "| primary:", r["primary_reason"])
            print("  actions:", r["repair_actions"])

# raw wiki match in wiki_102:0000698
OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837")
with (OUT / "tier_b_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] == "wiki_102_string.txt_new.txt:0000698":
            text = r["text"]
            for m in re.finditer(r"\{\{|\}\}|\[\[|\]\]|<\s*/?\s*ref\b", text, re.I):
                s = max(0, m.start() - 60)
                e = min(len(text), m.end() + 60)
                print("WIKI RESIDUE:", repr(text[s:e]))
            break
