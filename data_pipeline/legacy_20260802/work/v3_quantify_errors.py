#!/usr/bin/env python3
"""Quantify v3 misclassification candidates from the verified manifest."""
import gzip
import json
import re
from collections import Counter
from pathlib import Path

M = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/classification_manifest_v3.jsonl.gz"

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")

pure_list_stats = Counter()
pure_list_a_drop = []
def_incomplete_total = 0
targets = {}

with gzip.open(M, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pr = r["primary_reason"]
        chars = r["character_count"]
        paras = r["paragraph_count"]
        if pr == "pure_list_or_heading_stub":
            key = ("P1" if paras == 1 else "P2+", "C>=60" if chars >= 60 else "C<60")
            pure_list_stats[key] += 1
            if r["v2_tier"] == "A":
                pure_list_a_drop.append((r["document_id"], chars, paras))
        if pr == "definite_incomplete_tail":
            def_incomplete_total += 1
            targets[r["document_id"]] = None

print("pure_list_or_heading_stub primary total:", sum(pure_list_stats.values()))
for k, v in sorted(pure_list_stats.items()):
    print("  ", k, v)

print("\nA->DROP pure_list docs (need text-level review):", len(pure_list_a_drop))
for did, chars, paras in pure_list_a_drop:
    print("  ", did, "chars=", chars, "paras=", paras)

print("\ndefinite_incomplete_tail primary total:", def_incomplete_total)

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


matched = 0
digit_comma_docs = 0
comma2_after_strip = 0
for path in sorted(INPUT.iterdir()):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        did = f"{rel}:{idx:07d}"
        if did not in targets:
            continue
        matched += 1
        last = paragraphs[-1].strip()
        if DIGIT_COMMA.search(last):
            digit_comma_docs += 1
        stripped = DIGIT_COMMA.sub("", last)
        if stripped.count("，") + stripped.count("、") + stripped.count(",") >= 2:
            comma2_after_strip += 1
    if matched >= len(targets):
        break

print("definite_incomplete_tail docs rescanned:", matched, "of", len(targets))
print("  last paragraph has digit-comma pattern:", digit_comma_docs)
print("  last paragraph still >=2 commas after digit-comma removal:", comma2_after_strip)
