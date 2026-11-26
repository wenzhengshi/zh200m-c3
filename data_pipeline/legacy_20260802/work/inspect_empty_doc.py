import gzip
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
CAND = WS / "outputs/wiki2019zh/curated_candidate_v4_20260801_234934"
DID = "wiki_108_string.txt_new.txt:0003962"

with gzip.open(CAND / "document_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] == DID:
            print("candidate manifest:", json.dumps(r, ensure_ascii=False)[:800])
            break

with (CAND / "cleaning_log.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] == DID:
            print("cleaning log:", json.dumps(r, ensure_ascii=False)[:600])
            break

# raw text from input
import re
path = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh/wiki_108_string.txt_new.txt")
paragraphs = []
idx = 0
with path.open("r", encoding="utf-8", errors="replace") as f:
    cur = []
    for raw in f:
        line = raw.rstrip("\r\n")
        if not line.strip():
            if cur:
                idx += 1
                if idx == 3962:
                    paragraphs = cur
                    break
                cur = []
        else:
            cur.append(line)
print("raw paragraphs:", len(paragraphs))
print("raw text:", json.dumps("\n".join(paragraphs), ensure_ascii=False)[:600])
