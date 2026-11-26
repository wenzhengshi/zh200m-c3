#!/usr/bin/env python3
"""List the remaining hard-residue docs and write the quarantine/review list."""
import json
import re
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v4_20260801_234934")
SLOT = re.compile(r"(?:[（(]\s*[，,；;、/|·\s]{0,12}[）)]|《\s*》|〈\s*〉|「\s*」|『\s*』)")
LANG = re.compile(
    r"(?:[（(]\s*(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]?\s*[，,；;）)]"
    r"|(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]\s*[，,；;）)])"
)

rows = []
for name in ("tier_a_clean.jsonl", "tier_b_clean.jsonl"):
    with (OUT / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            kinds = []
            if LANG.search(r["text"]):
                kinds.append("empty_language_field")
            if SLOT.search(r["text"]):
                kinds.append("empty_slots")
            if kinds:
                for k in kinds:
                    m = LANG.search(r["text"]) if k == "empty_language_field" else SLOT.search(r["text"])
                    s = max(0, m.start() - 60)
                    e = min(len(r["text"]), m.end() + 60)
                    rows.append(
                        {
                            "document_id": r["document_id"],
                            "tier": "B",
                            "residue_type": k,
                            "context": r["text"][s:e],
                        }
                    )

print("quarantine rows:", len(rows))
for row in rows:
    print(f"  {row['residue_type']} | {row['document_id']} | {row['context'][:80]!r}")

with (OUT / "residue_quarantine.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["residue_quarantine"] = {
    "file": "residue_quarantine.jsonl",
    "documents": len(rows),
    "note": "A层硬残留=0；B层20篇硬残留按规则13进入人工复核清单，冻结时排除，不扩大分类规则。",
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("quarantine written")
