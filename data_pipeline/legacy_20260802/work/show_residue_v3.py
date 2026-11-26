import json
import re
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v3_20260801_233900")
SLOT = re.compile(r"(?:[（(]\s*[，,；;、/|·\s]{0,12}[）)]|《\s*》|〈\s*〉|「\s*」|『\s*』)")
LANG = re.compile(r"(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]?\s*[，,；;）)]")

slot_ids = {"wiki_126_string.txt_new.txt:0001554", "wiki_33_string.txt_new.txt:0000173", "wiki_37_string.txt_new.txt:0002307"}
lang_ids = {"wiki_0_string.txt_new.txt:0001615", "wiki_84_string.txt_new.txt:0002337"}
shown_slot = 0
shown_lang = 0
with (OUT / "tier_b_clean.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        did = r["document_id"]
        if did in slot_ids and shown_slot < 3:
            m = SLOT.search(r["text"])
            s = max(0, m.start() - 40)
            e = min(len(r["text"]), m.end() + 40)
            print("SLOT:", did, "|", repr(r["text"][s:e]))
            shown_slot += 1
        if did in lang_ids and shown_lang < 2:
            m = LANG.search(r["text"])
            s = max(0, m.start() - 40)
            e = min(len(r["text"]), m.end() + 40)
            print("LANG:", did, "|", repr(r["text"][s:e]))
            shown_lang += 1
        if shown_slot >= 3 and shown_lang >= 2:
            break
