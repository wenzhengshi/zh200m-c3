#!/usr/bin/env python3
"""Quantify definite_incomplete_tail trigger branches in v3.1."""
import gzip
import json
import re
from collections import Counter
from pathlib import Path

M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"
INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
CHART_STUB_RE = re.compile(r"^(?:人口变化图示|人口變化圖示|人口变化表|人口變化表|历年人口|歷年人口)\s*$")
DECLARATIVE_RE = re.compile(r"(?:是|为|属于|位于|生存于|成立于|出生于|指|包括|设立于|列出)")
TRAILING_FRAGMENT_RE = re.compile(r"(?:以及|并且|而且|或者|以及其|的|和|与|及|为|是|在|于|由)$")
END_PUNCTUATION = set("。！？；.!?;”’」』）》】")

tail_ids = set()
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["primary_reason"] == "definite_incomplete_tail":
            tail_ids.add(r["document_id"])


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


def count_commas(s):
    s = DIGIT_COMMA.sub("", s)
    return s.count("，") + s.count("、") + s.count(",")


branches = Counter()
d_samples = []
for path in sorted(INPUT.iterdir()):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        did = f"{rel}:{idx:07d}"
        if did not in tail_ids:
            continue
        last = paragraphs[-1].strip()
        if last.endswith(("，", ",", "、", "；", ";")):
            branches["ends_with_comma_mark"] += 1
        elif TRAILING_FRAGMENT_RE.search(last):
            branches["ends_with_connector"] += 1
        elif CHART_STUB_RE.fullmatch(last.strip()):
            branches["chart_stub"] += 1
        else:
            n = count_commas(last)
            if n >= 2 and not DECLARATIVE_RE.search(last):
                branches["no_final_punct_2plus_commas"] += 1
                if len(d_samples) < 40:
                    d_samples.append((did, last))
            else:
                branches["other_unclassified"] += 1

print("definite_incomplete_tail total:", len(tail_ids))
for k, v in branches.most_common():
    print(f"  {k}: {v}")
print("\nno_final_punct_2plus_commas samples:")
for did, last in d_samples:
    print(f"  {did}")
    print(f"    {last[:120]}")
