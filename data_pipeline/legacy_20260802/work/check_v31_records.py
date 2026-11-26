import gzip
import json
import re
from pathlib import Path

M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"
ids = {"wiki_0_string.txt_new.txt:0001615", "wiki_126_string.txt_new.txt:0001554"}
recs = {}
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in ids:
            recs[r["document_id"]] = r
            if len(recs) == len(ids):
                break

EMPTY_LANG = re.compile(
    r"(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]?\s*[，,；;）)]"
)
EMPTY_SLOT = re.compile(
    r"(?:[（(]\s*[，,；;、/|·\s]{0,12}[）)]|《\s*》|〈\s*〉|「\s*」|『\s*』)"
)

INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")


def iter_documents(path):
    paragraphs = []
    idx = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if not line.strip():
                if paragraphs:
                    idx += 1
                    yield idx, paragraphs
                    paragraphs = []
            else:
                paragraphs.append(line)
    if paragraphs:
        idx += 1
        yield idx, paragraphs


for path in sorted(INPUT.iterdir()):
    for idx, paras in iter_documents(path):
        did = f"{path.name}:{idx:07d}"
        if did in ids:
            text = "\n".join(paras)
            print("DOC:", did)
            print("  v3.1 actions:", recs[did]["repair_actions"])
            print("  v3.1 reasons:", recs[did]["reasons"])
            print("  LANG regex matches:", [m.group() for m in EMPTY_LANG.finditer(text)])
            print("  SLOT regex matches:", [m.group() for m in EMPTY_SLOT.finditer(text)])
            print("  first 120:", text[:120].replace("\n", " / "))
            print()
