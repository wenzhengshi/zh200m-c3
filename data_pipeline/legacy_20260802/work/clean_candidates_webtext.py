#!/usr/bin/env python3
"""webText2019zh cleaning preview + curated candidates v1 (one pass)."""
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
SRC = WS / "outputs/webText2019zh/tier_dryrun_v1_20260802_010603"
INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/webText2019zh")
PREVIEW = WS / f"outputs/webText2019zh/cleaning_preview_v1_{TS}"
CAND = WS / f"outputs/webText2019zh/curated_candidate_v1_{TS}"
PREVIEW.mkdir(parents=True, exist_ok=False)
CAND.mkdir(parents=True, exist_ok=False)
(PREVIEW / "preview_samples").mkdir()

URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{5,10}(?!\d)")
WECHAT_RE = re.compile(r"(?:wxid[_\-][A-Za-z0-9_\-]+|微信号\s*[:：]?\s*[A-Za-z0-9_\-]{5,20})")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
AD_LINE_RE = re.compile(r"(?:广告|推广|促销|特价|优惠券|点击购买|立即购买|限时抢购|免费领取)")
NAV_LINE_RE = re.compile(r"(?:首页|导航|友情链接|网站地图|版权所有|ICP备|免责声明|关于我们|联系我们|设为首页|收藏本站)")
CONTACT_LINE_RE = re.compile(r"(?:联系电话|联系方式|联系地址|手机号|微信号|邮箱[:：]|e?-?mail[:：])", re.I)


def sha(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_doc(paragraphs, actions):
    notes = []
    p = list(paragraphs)
    if "remove_phone" in actions:
        p = [PHONE_RE.sub("", x) for x in p]
    if "remove_qq" in actions:
        p = [QQ_RE.sub("", x) for x in p]
    if "remove_wechat" in actions:
        p = [WECHAT_RE.sub("", x) for x in p]
    if "remove_idcard" in actions:
        p = [IDCARD_RE.sub("", x) for x in p]
    if "remove_email" in actions:
        p = [EMAIL_RE.sub("", x) for x in p]
    if "remove_url" in actions:
        p = [URL_RE.sub("", x) for x in p]
    out = []
    for x in p:
        line = x.strip()
        if not line:
            continue
        if "remove_ad_lines" in actions and AD_LINE_RE.search(line) and len(line) < 80:
            notes.append("ad_line:dropped")
            continue
        if "remove_nav_footer_lines" in actions and NAV_LINE_RE.search(line) and len(line) < 80:
            notes.append("nav_footer_line:dropped")
            continue
        if "remove_contact_lines" in actions and CONTACT_LINE_RE.search(line) and len(line) < 120:
            notes.append("contact_line:dropped")
            continue
        out.append(line)
    return "\n".join(out), notes


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
    def __init__(self, limit, seed=20260802):
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
        i = self.rng.randrange(self.seen)
        if i < self.limit:
            self.items[i] = item


samples = defaultdict(lambda: Reservoir(30))
manifest_f = gzip.open(SRC / "classification_manifest_v1.jsonl.gz", "rt", encoding="utf-8")
preview_f = gzip.open(PREVIEW / "preview_all.jsonl.gz", "wt", encoding="utf-8")
ta = (CAND / "tier_a_clean.jsonl").open("w", encoding="utf-8")
tb = (CAND / "tier_b_clean.jsonl").open("w", encoding="utf-8")
rv = (CAND / "review.jsonl").open("w", encoding="utf-8")
dp = (CAND / "drop.jsonl").open("w", encoding="utf-8")
clog = (CAND / "cleaning_log.jsonl").open("w", encoding="utf-8")
cman = gzip.open(CAND / "document_manifest.jsonl.gz", "wt", encoding="utf-8")

counts = Counter()
chars = Counter()
action_docs = 0
previewed = 0
empty_after = []

for fi, path in enumerate(sorted(INPUT.iterdir()), start=1):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        line = manifest_f.readline()
        if not line:
            raise RuntimeError("manifest premature end")
        rec = json.loads(line)
        did = f"{rel}:{idx:07d}"
        if rec["document_id"] != did:
            raise RuntimeError(f"alignment: {did}")
        tier = rec["tier"]
        actions = rec["repair_actions"]
        raw = "\n".join(paragraphs)
        raw_sha = rec["text_sha256"]
        assert sha(raw) == raw_sha, f"sha mismatch {did}"
        ccount = sum(len(p) for p in paragraphs)
        if tier == "A":
            text, final_sha = raw, raw_sha
            counts["A"] += 1
            chars["A"] += ccount
            ta.write(json.dumps({"document_id": did, "text": raw}, ensure_ascii=False) + "\n")
        elif tier == "B":
            if actions:
                cleaned, notes = clean_doc(paragraphs, actions)
                csha = sha(cleaned)
                after = sum(len(x) for x in cleaned.split("\n")) if cleaned else 0
                previewed += 1
                action_docs += 1
                samples[tuple(actions)].add(
                    {"document_id": did, "actions": actions, "notes": notes, "before": ccount, "after": after,
                     "sha_before": raw_sha, "sha_after": csha, "original": raw[:1200], "cleaned": cleaned[:1200]}
                )
                preview_f.write(
                    json.dumps(
                        {
                            "document_id": did,
                            "tier": "B",
                            "primary_reason": rec["primary_reason"],
                            "repair_actions": actions,
                            "cleaning_notes": notes,
                            "char_before": ccount,
                            "char_after": after,
                            "text_sha256_before": raw_sha,
                            "text_sha256_after": csha,
                            "original_text": raw,
                            "cleaned_text": cleaned,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                clog.write(
                    json.dumps(
                        {
                            "document_id": did,
                            "primary_reason": rec["primary_reason"],
                            "repair_actions": actions,
                            "cleaning_notes": notes,
                            "char_before": ccount,
                            "char_after": after,
                            "text_sha256_before": raw_sha,
                            "text_sha256_after": csha,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if not cleaned.strip():
                    empty_after.append(did)
                text, final_sha = cleaned, csha
            else:
                text, final_sha = raw, raw_sha
            counts["B"] += 1
            chars["B"] += sum(len(x) for x in text.split("\n"))
            tb.write(json.dumps({"document_id": did, "text": text}, ensure_ascii=False) + "\n")
        elif tier == "REVIEW":
            counts["REVIEW"] += 1
            chars["REVIEW"] += ccount
            rv.write(json.dumps({"document_id": did, "text": raw}, ensure_ascii=False) + "\n")
        else:
            counts["DROP"] += 1
            chars["DROP"] += ccount
            dp.write(json.dumps({"document_id": did, "text": raw}, ensure_ascii=False) + "\n")
        cman.write(
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
                    "character_count": sum(len(x) for x in text.split("\n")) if tier in ("A", "B") else ccount,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if fi % 100 == 0 or fi == 896:
        print(f"files={fi} counts={dict(counts)} previewed={previewed:,}", flush=True)

manifest_f.close()
preview_f.close()
ta.close(); tb.close(); rv.close(); dp.close(); clog.close(); cman.close()

for actions, res in sorted(samples.items(), key=lambda kv: -len(kv[0])):
    safe = "_".join(actions)[:80]
    with (PREVIEW / "preview_samples" / f"action_{safe}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_documents_with_action_set={res.seen}\n\n")
        for i, s in enumerate(res.items, start=1):
            f.write(f"sample={i} document_id={s['document_id']}\n")
            f.write(f"actions={json.dumps(s['actions'], ensure_ascii=False)} notes={json.dumps(s['notes'], ensure_ascii=False)}\n")
            f.write(f"char_before={s['before']} char_after={s['after']}\n")
            f.write("原始文本：\n" + s["original"] + "\n---\n清洗后文本：\n" + s["cleaned"] + "\n\n")

total = sum(counts.values())
summary = {
    "stage": "cleaning_and_candidates_v1",
    "dataset": "webText2019zh",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "total_documents": total,
    "tier_document_counts": dict(sorted(counts.items())),
    "tier_character_counts": {k: chars[k] for k in ("A", "B", "REVIEW", "DROP")},
    "documents_cleaned": action_docs,
    "empty_after_cleaning": empty_after,
    "preview_dir": str(PREVIEW),
    "candidate_dir": str(CAND),
    "note": "A与B分离；REVIEW(32,262)隔离；DROP(18,570)留档；PII(手机/QQ/微信/身份证/邮箱)已抹除，广告/导航/联系方式行已删除。",
}
(CAND / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(PREVIEW / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for d, fname in ((CAND, "run.log"), (PREVIEW, "run.log")):
    (d / fname).write_text(
        f"webText cleaning+candidates v1 run.log\ncreated_at: {summary['created_at']}\n"
        f"docs: {total:,} A={counts['A']:,} B={counts['B']:,} REVIEW={counts['REVIEW']:,} DROP={counts['DROP']:,}\n"
        f"cleaned docs: {action_docs:,}\n",
        encoding="utf-8",
    )
print(json.dumps(summary, ensure_ascii=False, indent=2))
