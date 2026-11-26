#!/usr/bin/env python3
"""Precise impact of the two v3.1 fixes (list-keyword scope; digit-comma)."""
import gzip
import json
import re
from pathlib import Path

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/classification_manifest_v3.jsonl.gz"
INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
LIST_TERM = re.compile(r"(?:列表|名单|名單|一览|一覽|目录|目錄|年表|作品表|集数列表|集數列表)")

# Load manifest records for the two populations
pure_list_ids = set()
tail_ids = set()
recs = {}
with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["primary_reason"] == "pure_list_or_heading_stub":
            pure_list_ids.add(r["document_id"])
            recs.setdefault("pure", {})[r["document_id"]] = r
        if r["primary_reason"] == "definite_incomplete_tail":
            tail_ids.add(r["document_id"])
            recs.setdefault("tail", {})[r["document_id"]] = r


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


pure_kw_first = 0
pure_kw_first_ids = []
pure_kw_later_ids = []
tail_flip = 0
tail_flip_ids = []

for path in sorted(INPUT.iterdir()):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        did = f"{rel}:{idx:07d}"
        if did in pure_list_ids:
            first = paragraphs[0].strip()
            if LIST_TERM.search(first):
                pure_kw_first += 1
                pure_kw_first_ids.append(did)
            else:
                pure_kw_later_ids.append(did)
        if did in tail_ids:
            last = paragraphs[-1].strip()
            stripped = DIGIT_COMMA.sub("", last)
            comma2 = stripped.count("，") + stripped.count("、") + stripped.count(",")
            if DIGIT_COMMA.search(last) and comma2 < 2:
                tail_flip += 1
                tail_flip_ids.append(did)

print("pure_list total:", len(pure_list_ids))
print("  keyword in FIRST paragraph (would stay DROP under fix):", pure_kw_first)
print("  keyword NOT in first paragraph (would recover from pure_list):", len(pure_kw_later_ids))

print("\ndefinite_incomplete_tail total:", len(tail_ids))
print("  digit-comma AND <2 commas after strip (would recover):", tail_flip)

import sys
if "--list" in sys.argv:
    print("\npure_list recover ids:")
    for d in sorted(pure_kw_later_ids):
        r = recs["pure"][d]
        print(f"  {d} v2={r['v2_tier']} chars={r['character_count']} paras={r['paragraph_count']}")
    print("\ntail flip ids:")
    for d in sorted(tail_flip_ids):
        r = recs["tail"][d]
        print(f"  {d} v2={r['v2_tier']} chars={r['character_count']} paras={r['paragraph_count']}")
