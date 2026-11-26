#!/usr/bin/env python3
"""Rebuild the candidate document_manifest with correct per-doc stats (no-newline chars, final SHA)."""
import gzip
import json
from collections import Counter
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837")
M31 = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz")

import sys
sys.path.insert(0, "/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work")
from wiki_cleaner_core import sha256_hex

# load classifier manifest
clf = {}
with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        clf[r["document_id"]] = r

def rows(name):
    with (OUT / name).open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def char_count(text: str) -> int:
    return len(text) - text.count("\n")

per_tier_docs = Counter()
per_tier_chars = Counter()
manifest_out = []
errors = []

for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B"), ("review.jsonl", "REVIEW"), ("drop.jsonl", "DROP")):
    for row in rows(name):
        did = row["document_id"]
        text = row["text"]
        c = clf[did]
        if c["tier"] != tier:
            errors.append(f"tier mismatch {did}")
            continue
        ccount = char_count(text)
        final_sha = sha256_hex(text)
        if tier in ("A", "REVIEW", "DROP"):
            if final_sha != c["text_sha256"]:
                errors.append(f"raw sha mismatch {did}")
        manifest_out.append(
            {
                "document_id": did,
                "source_file": c["source_file"],
                "document_index_in_file": c["document_index_in_file"],
                "tier": tier,
                "primary_reason": c["primary_reason"],
                "reasons": c["reasons"],
                "repair_actions": c["repair_actions"],
                "text_sha256": final_sha,
                "character_count": ccount,
            }
        )
        per_tier_docs[tier] += 1
        per_tier_chars[tier] += ccount

print("errors:", len(errors))
for e in errors[:20]:
    print(" ", e)
print("docs:", dict(per_tier_docs))
print("chars (no-newline):", dict(per_tier_chars))
total = sum(per_tier_chars.values())
print("total:", total)
print("expected A/REVIEW/DROP (v3.1):", 235415779, 21102526, 2522786)

# expected totals per classifier (A/REVIEW/DROP unchanged text)
assert per_tier_docs == {"A": 522329, "B": 414991, "REVIEW": 15858, "DROP": 65514}
assert per_tier_chars["A"] == 235415779
assert per_tier_chars["REVIEW"] == 21102526
assert per_tier_chars["DROP"] == 2522786
assert not errors

with gzip.open(OUT / "document_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in manifest_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["tier_character_counts"] = {k: per_tier_chars[k] for k in ("A", "B", "REVIEW", "DROP")}
summary["input_character_total_reconciled"] = 427473417
summary["output_character_total"] = total
summary["character_count_convention"] = (
    "character_count = paragraph-length sum (newlines excluded). "
    "input total reconciled with full_audit_v1 (427,473,417); output total reflects B cleaning."
)
summary["note"] += " document_manifest rebuilt with corrected per-doc stats (REVIEW/DROP rows were affected by a stale-length bug in the first pass)."
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = (OUT / "run.log").read_text(encoding="utf-8")
log += "\n[correction] document_manifest.jsonl.gz rebuilt: character_count = no-newline paragraph sum; text_sha256 = final stored text;\n"
log += f"[correction] per-tier chars: {dict(per_tier_chars)}; output total {total:,}; input reconciled {427473417:,}.\n"
(OUT / "run.log").write_text(log, encoding="utf-8")
print("manifest rebuilt")
