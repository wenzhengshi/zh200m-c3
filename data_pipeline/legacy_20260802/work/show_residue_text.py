import json
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837")
want = {
    "wiki_124_string.txt_new.txt:0001415": "url",
    "wiki_202_string.txt_new.txt:0004827": "url",
    "wiki_33_string.txt_new.txt:0002378": "url",
    "wiki_122_string.txt_new.txt:0001340": "empty_slots",
    "wiki_21_string.txt_new.txt:0002094": "empty_slots",
    "wiki_0_string.txt_new.txt:0001615": "empty_language",
    "wiki_102_string.txt_new.txt:0000698": "raw_wiki",
}
seen = set()
with (OUT / "tier_b_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in want:
            print("====", r["document_id"], "|", want[r["document_id"]])
            print(r["text"][:600])
            print()
            seen.add(r["document_id"])
            if len(seen) == len(want):
                break
