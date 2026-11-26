#!/usr/bin/env python3
"""Correct candidate character counts to the no-newline convention (full-audit compatible)."""
import gzip
import json
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v1_20260801_230837")


def doc_char_count(text: str) -> int:
    return len(text) - text.count("\n")


per_tier = {"A": 0, "B": 0, "REVIEW": 0, "DROP": 0}
counts = {"A": 0, "B": 0, "REVIEW": 0, "DROP": 0}

manifest_rows = []
with gzip.open(OUT / "document_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        manifest_rows.append(json.loads(line))

def stream_rows(name):
    with (OUT / name).open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

text_counts = {}
for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B"), ("review.jsonl", "REVIEW"), ("drop.jsonl", "DROP")):
    for row in stream_rows(name):
        did = row["document_id"]
        c = doc_char_count(row["text"])
        text_counts[did] = c
        per_tier[tier] += c
        counts[tier] += 1

total_old = 0
for row in manifest_rows:
    total_old += row["character_count"]
    row["character_count"] = text_counts[row["document_id"]]

total_new = sum(per_tier.values())
print("old manifest char sum:", total_old)
print("new manifest char sum:", total_new)
print("expected (full audit):", 427473417)
assert total_new == 427473417, "reconciliation failed"
assert counts == {"A": 522329, "B": 414991, "REVIEW": 15858, "DROP": 65514}

with gzip.open(OUT / "document_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for row in manifest_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["tier_character_counts"] = per_tier
summary["character_count_convention"] = (
    "character_count = sum of paragraph lengths, excluding newline separators; "
    "reconciled with full_audit_v1 total_characters = 427,473,417"
)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = (OUT / "run.log").read_text(encoding="utf-8")
log += "\n[correction] character_count redefined as visible chars (paragraph-length sum, no newlines);\n"
log += f"[correction] manifest + summary updated; new char sum = {total_new:,} == full_audit total.\n"
(OUT / "run.log").write_text(log, encoding="utf-8")
print("done")
