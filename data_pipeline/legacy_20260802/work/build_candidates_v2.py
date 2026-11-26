#!/usr/bin/env python3
"""Generate wiki2019zh curated candidate v2 (cleaner core v2; correct stats; cross-check vs preview v3)."""
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work")
from wiki_cleaner_core import apply_actions, sha256_hex

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / f"outputs/wiki2019zh/curated_candidate_v4_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

INPUT_DIR = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")
M31 = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz")
PREVIEW = sorted(
    (WS / "outputs/wiki2019zh").glob("cleaning_preview_v5_*")
)[-1] / "preview_all.jsonl.gz"
print("preview source:", PREVIEW)

tier_a = (OUT / "tier_a_clean.jsonl").open("w", encoding="utf-8")
tier_b = (OUT / "tier_b_clean.jsonl").open("w", encoding="utf-8")
review_f = (OUT / "review.jsonl").open("w", encoding="utf-8")
drop_f = (OUT / "drop.jsonl").open("w", encoding="utf-8")
clean_log = (OUT / "cleaning_log.jsonl").open("w", encoding="utf-8")
manifest_f = gzip.open(OUT / "document_manifest.jsonl.gz", "wt", encoding="utf-8")

counts = Counter()
chars = Counter()
action_docs = 0
clean_log_entries = 0
crosscheck = 0
crosscheck_fail = 0

manifest = gzip.open(M31, "rt", encoding="utf-8")
preview = gzip.open(PREVIEW, "rt", encoding="utf-8")


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


def visible_chars(text: str) -> int:
    return len(text) - text.count("\n")


files_done = 0
for path in sorted(INPUT_DIR.iterdir()):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        line = manifest.readline()
        if not line:
            raise RuntimeError("manifest premature end")
        rec = json.loads(line)
        did = f"{rel}:{idx:07d}"
        if rec["document_id"] != did:
            raise RuntimeError(f"alignment mismatch: {did}")

        tier = rec["tier"]
        actions = rec["repair_actions"]
        raw_text = "\n".join(paragraphs)
        raw_sha = rec["text_sha256"]
        if sha256_hex(raw_text) != raw_sha:
            raise RuntimeError(f"SHA mismatch on {did}")

        if actions:
            prow = json.loads(preview.readline())
            crosscheck += 1
            if prow["document_id"] != did:
                raise RuntimeError(f"preview alignment mismatch: {did}")

        if tier == "A":
            text = raw_text
            final_sha = raw_sha
            counts["A"] += 1
            chars["A"] += visible_chars(raw_text)
            tier_a.write(json.dumps({"document_id": did, "text": raw_text}, ensure_ascii=False) + "\n")
        elif tier == "B":
            if actions:
                cleaned, notes = apply_actions(list(paragraphs), actions, rec)
                csha = sha256_hex(cleaned)
                if csha != prow["text_sha256_after"]:
                    crosscheck_fail += 1
                    raise RuntimeError(f"cleaned SHA mismatch vs preview: {did}")
                clean_log.write(
                    json.dumps(
                        {
                            "document_id": did,
                            "tier": "B",
                            "primary_reason": rec["primary_reason"],
                            "repair_actions": actions,
                            "cleaning_notes": notes,
                            "char_before": prow["char_before"],
                            "char_after": prow["char_after"],
                            "text_sha256_before": raw_sha,
                            "text_sha256_after": csha,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                clean_log_entries += 1
                action_docs += 1
                text = cleaned
                final_sha = csha
            else:
                text = raw_text
                final_sha = raw_sha
            counts["B"] += 1
            chars["B"] += visible_chars(text)
            tier_b.write(json.dumps({"document_id": did, "text": text}, ensure_ascii=False) + "\n")
        elif tier == "REVIEW":
            counts["REVIEW"] += 1
            chars["REVIEW"] += visible_chars(raw_text)
            review_f.write(json.dumps({"document_id": did, "text": raw_text}, ensure_ascii=False) + "\n")
        elif tier == "DROP":
            counts["DROP"] += 1
            chars["DROP"] += visible_chars(raw_text)
            drop_f.write(json.dumps({"document_id": did, "text": raw_text}, ensure_ascii=False) + "\n")
        else:
            raise RuntimeError(f"unknown tier {tier}")

        manifest_f.write(
            json.dumps(
                {
                    "document_id": did,
                    "source_file": rel,
                    "document_index_in_file": idx,
                    "tier": tier,
                    "primary_reason": rec["primary_reason"],
                    "reasons": rec["reasons"],
                    "repair_actions": actions,
                    "text_sha256": final_sha,
                    "character_count": visible_chars(text),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    files_done += 1
    if files_done % 50 == 0:
        print(f"files={files_done} counts={dict(counts)}", flush=True)

if manifest.readline():
    raise RuntimeError("manifest extra records")
if preview.readline():
    raise RuntimeError("preview has extra records")

manifest.close()
preview.close()
tier_a.close()
tier_b.close()
review_f.close()
drop_f.close()
clean_log.close()
manifest_f.close()

total = sum(counts.values())
assert counts == {"A": 522329, "B": 414991, "REVIEW": 15858, "DROP": 65514}, counts
assert chars["A"] == 235415779, chars
assert chars["REVIEW"] == 21102526, chars
assert chars["DROP"] == 2522786, chars
output_total = sum(chars.values())

summary = {
    "stage": "curated_candidate_v4",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "classifier": "tier_dryrun_v3_1",
    "cleaner_core": "wiki_cleaner_core v4",
    "supersedes": "curated_candidate_v3_20260801_233900 (cascade empty-slot residues after language-field/email removal)",
    "input_directory": str(INPUT_DIR),
    "total_documents": total,
    "tier_document_counts": dict(sorted(counts.items())),
    "tier_character_counts": {k: chars[k] for k in ("A", "B", "REVIEW", "DROP")},
    "input_character_total_reconciled": 427473417,
    "output_character_total": output_total,
    "documents_cleaned_with_actions": action_docs,
    "cleaning_log_entries": clean_log_entries,
    "preview_crosscheck_documents": crosscheck,
    "preview_crosscheck_failures": crosscheck_fail,
    "outputs": {
        "tier_a_clean": "tier_a_clean.jsonl",
        "tier_b_clean": "tier_b_clean.jsonl",
        "review": "review.jsonl",
        "drop": "drop.jsonl",
        "cleaning_log": "cleaning_log.jsonl",
        "manifest": "document_manifest.jsonl.gz",
    },
    "note": "A与清洗后的B分文件保存，未混合；REVIEW独立隔离；DROP留档；字符数=段落长度和（不含换行）。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = [
    f"curated_candidate_v4 run.log",
    f"created_at: {summary['created_at']}",
    f"total_documents: {total:,}",
    f"tier counts: {dict(sorted(counts.items()))}",
    f"tier chars (no-newline): {dict(sorted(chars.items()))}",
    f"output char total: {output_total:,}",
    f"documents cleaned: {action_docs:,}",
    f"preview crosscheck: {crosscheck:,} docs, {crosscheck_fail} failures",
]
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
