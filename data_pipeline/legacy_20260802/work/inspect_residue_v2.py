import json
import re
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v2_20260801_232734")
RAW = re.compile(r"\{\{|\}\}|\[\[|\]\]|<\s*/?\s*ref\b", re.I)
SLOT = re.compile(r"(?:[（(]\s*[，,；;、/|·\s]{0,12}[）)]|《\s*》|〈\s*〉|「\s*」|『\s*』)")

shown_wiki = 0
shown_slot = 0
with (OUT / "tier_b_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        text = r["text"]
        if shown_wiki < 8 and RAW.search(text):
            m = RAW.search(text)
            s = max(0, m.start() - 50)
            e = min(len(text), m.end() + 50)
            print("WIKI:", r["document_id"], "|", repr(text[s:e]))
            shown_wiki += 1
        if shown_slot < 8 and SLOT.search(text):
            m = SLOT.search(text)
            s = max(0, m.start() - 30)
            e = min(len(text), m.end() + 30)
            print("SLOT:", r["document_id"], "|", repr(text[s:e]))
            shown_slot += 1
        if shown_wiki >= 8 and shown_slot >= 8:
            break
