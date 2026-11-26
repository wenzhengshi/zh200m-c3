#!/usr/bin/env python3
"""Freeze wiki2019zh deduplicated candidates into wiki2019zh_freeze_v1."""
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
SRC = WS / "outputs/wiki2019zh/curated_candidate_v4_20260801_234934"
DEDUP = WS / "outputs/wiki2019zh/dedup_v1_20260802_001902"
OUT = WS / f"outputs/wiki2019zh/wiki2019zh_freeze_v2_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

# keep ids
keep = set()
with gzip.open(DEDUP / "dedup_keep_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        keep.add(json.loads(line)["document_id"])
print("keep ids:", len(keep))

ta = (OUT / "tier_a_frozen.jsonl").open("w", encoding="utf-8")
tb = (OUT / "tier_b_frozen.jsonl").open("w", encoding="utf-8")
manifest = gzip.open(OUT / "frozen_manifest.jsonl.gz", "wt", encoding="utf-8")

counts = Counter()
chars = Counter()
total = 0
excluded_empty = []

for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B")):
    with (SRC / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["document_id"] not in keep:
                continue
            text = r["text"]
            if not text.strip():
                excluded_empty.append(r["document_id"])
                continue
            c = len(text) - text.count("\n")
            if c == 0:
                excluded_empty.append(r["document_id"])
                continue
            sh = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if tier == "A":
                ta.write(json.dumps({"document_id": r["document_id"], "text": text}, ensure_ascii=False) + "\n")
            else:
                tb.write(json.dumps({"document_id": r["document_id"], "text": text}, ensure_ascii=False) + "\n")
            manifest.write(
                json.dumps(
                    {
                        "document_id": r["document_id"],
                        "tier": tier,
                        "character_count": c,
                        "text_sha256": sh,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[tier] += 1
            chars[tier] += c
            total += 1

ta.close()
tb.close()
manifest.close()

assert total == 936077 - len(excluded_empty), (total, len(excluded_empty))
summary = {
    "stage": "wiki2019zh_freeze_v2",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "source_candidates": str(SRC),
    "source_dedup": str(DEDUP),
    "total_documents": total,
    "tier_document_counts": dict(sorted(counts.items())),
    "tier_character_counts": {k: chars[k] for k in ("A", "B")},
    "review_isolated_documents": 15858,
    "drop_archived_documents": 65514,
    "quarantined_residue_documents": 20,
    "dedup_removed_documents": 1223,
    "excluded_empty_after_cleaning": excluded_empty,
    "frozen": True,
    "note": "A与B分文件冻结，REVIEW未进入；v2在v1基础上排除清洗后为空的文档（trim删除了唯一段落）；冻结条件全部满足。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "PERMANENT_FREEZE_NOTICE.txt").write_text(
    "wiki2019zh 已完成冻结 v2。\n禁止覆盖或修改本目录中的任何文件。\nREVIEW(15,858)、残留复核(20)、清洗后为空文档未进入本冻结集。\n",
    encoding="utf-8",
)
with (OUT / "freeze_excluded_empty.jsonl").open("w", encoding="utf-8") as f:
    for did in excluded_empty:
        f.write(json.dumps({"document_id": did, "reason": "empty_after_cleaning"}) + "\n")
log = [
    f"wiki2019zh_freeze_v2 run.log",
    f"created_at: {summary['created_at']}",
    f"frozen docs: {total:,} (A {counts['A']:,} / B {counts['B']:,})",
    f"frozen chars: {chars['A'] + chars['B']:,}",
    f"excluded: REVIEW {15858:,}, DROP archived {65514:,}, residue quarantine {20}, dedup removed {1223:,}, empty-after-cleaning {len(excluded_empty)}",
]
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
