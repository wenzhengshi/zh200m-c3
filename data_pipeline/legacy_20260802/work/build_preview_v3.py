#!/usr/bin/env python3
"""Generate wiki2019zh cleaning preview v3 (cleaner core v2: in-body URLs, malformed ref, quote slots)."""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work")
from wiki_cleaner_core import apply_actions, sha256_hex

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / f"outputs/wiki2019zh/cleaning_preview_v5_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
SAMPLES_DIR = OUT / "preview_samples"
SAMPLES_DIR.mkdir()

INPUT_DIR = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")
M31 = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz")


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


class Reservoir:
    def __init__(self, limit: int, seed: int):
        import random
        self.limit = limit
        self.rng = random.Random(seed)
        self.seen = 0
        self.items = []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(item)
            return
        index = self.rng.randrange(self.seen)
        if index < self.limit:
            self.items[index] = item


SEED = 20260801
action_samples = defaultdict(lambda: Reservoir(30, SEED))
stats = Counter()
action_docs = Counter()
char_before_total = 0
char_after_total = 0
noop_count = 0
previewed = 0
tail_kept = 0
tail_kept_ids = []
alignment_failures = 0

manifest_f = gzip.open(M31, "rt", encoding="utf-8")
preview_f = gzip.open(OUT / "preview_all.jsonl.gz", "wt", encoding="utf-8")

for path in sorted(INPUT_DIR.iterdir()):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        old_line = manifest_f.readline()
        if not old_line:
            raise RuntimeError("manifest premature end")
        rec = json.loads(old_line)
        did = f"{rel}:{idx:07d}"
        if rec["document_id"] != did:
            raise RuntimeError(f"alignment mismatch {did} vs {rec['document_id']}")
        actions = rec["repair_actions"]
        if not actions:
            continue
        original_text = "\n".join(paragraphs)
        before = rec["character_count"]
        text_sha_before = rec["text_sha256"]
        if sha256_hex(original_text) != text_sha_before:
            alignment_failures += 1
            raise RuntimeError(f"SHA mismatch on {did}")
        cleaned_text, notes = apply_actions(list(paragraphs), actions, rec)
        after = sum(len(p) for p in cleaned_text.split("\n")) if cleaned_text else 0
        text_sha_after = sha256_hex(cleaned_text)
        char_before_total += before
        char_after_total += after
        previewed += 1
        if cleaned_text == original_text:
            noop_count += 1
        if any(n.startswith("tail_missing_final_period") for n in notes):
            tail_kept += 1
            tail_kept_ids.append((did, rec["primary_reason"]))
        for a in actions:
            action_docs[a] += 1
        row = {
            "document_id": did,
            "tier": rec["tier"],
            "primary_reason": rec["primary_reason"],
            "reasons": rec["reasons"],
            "repair_actions": actions,
            "cleaning_notes": notes,
            "char_before": before,
            "char_after": after,
            "text_sha256_before": text_sha_before,
            "text_sha256_after": text_sha_after,
            "original_text": original_text,
            "cleaned_text": cleaned_text,
        }
        preview_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for a in actions:
            action_samples[a].add(row)
    stats["files"] += 1
    if stats["files"] % 20 == 0:
        print(f"processed files={stats['files']} previewed={previewed:,}", flush=True)

if manifest_f.readline():
    raise RuntimeError("manifest has extra records")
manifest_f.close()
preview_f.close()

for action, res in sorted(action_samples.items()):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", action)
    with (SAMPLES_DIR / f"action_{safe}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_documents_with_action={action_docs[action]}\n")
        f.write(f"sample_count={len(res.items)}\n")
        f.write(f"seed={SEED}\n\n")
        for i, row in enumerate(res.items, start=1):
            f.write("=" * 100 + "\n")
            f.write(f"sample={i}\n")
            f.write(f"document_id={row['document_id']}\n")
            f.write(f"tier={row['tier']}\n")
            f.write(f"primary_reason={row['primary_reason']}\n")
            f.write(f"repair_actions={json.dumps(row['repair_actions'], ensure_ascii=False)}\n")
            f.write(f"cleaning_notes={json.dumps(row['cleaning_notes'], ensure_ascii=False)}\n")
            f.write(f"char_before={row['char_before']} char_after={row['char_after']}\n")
            f.write(f"text_sha256_before={row['text_sha256_before']}\n")
            f.write(f"text_sha256_after={row['text_sha256_after']}\n")
            f.write("-" * 100 + "\n")
            f.write("原始文本：\n" + row["original_text"][:1200] + "\n")
            f.write("-" * 100 + "\n")
            f.write("清洗后文本：\n" + row["cleaned_text"][:1200] + "\n\n")

with (OUT / "tail_kept_for_review.jsonl").open("w", encoding="utf-8") as f:
    for did, reason in tail_kept_ids:
        f.write(json.dumps({"document_id": did, "primary_reason": reason}, ensure_ascii=False) + "\n")

summary = {
    "stage": "cleaning_preview_v5",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "classifier": "tier_dryrun_v3_1",
    "cleaner_core": "wiki_cleaner_core v4 (v3 + final cascade empty-slot cleanup for all action docs)",
    "supersedes": "cleaning_preview_v4_20260801_233653",
    "input_directory": str(INPUT_DIR),
    "preview_documents_with_actions": previewed,
    "alignment_failures": alignment_failures,
    "characters_before": char_before_total,
    "characters_after": char_after_total,
    "characters_removed": char_before_total - char_after_total,
    "noop_documents": noop_count,
    "tail_kept_for_review": tail_kept,
    "action_document_counts": dict(sorted(action_docs.items())),
    "note": "预览只审计清洗动作，不生成正式训练文件；REVIEW未进入任何清洗路径。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = [
    f"cleaning_preview_v5 run.log",
    f"created_at: {summary['created_at']}",
    f"classifier: tier_dryrun_v3_1",
    f"previewed documents: {previewed:,}",
    f"characters before: {char_before_total:,}",
    f"characters after: {char_after_total:,}",
    f"no-op documents: {noop_count:,}",
    f"tail kept for review: {tail_kept:,}",
    f"alignment/SHA failures: {alignment_failures}",
    "",
    "per-action document counts:",
]
for a, n in sorted(action_docs.items()):
    log.append(f"  {a}: {n:,}")
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
