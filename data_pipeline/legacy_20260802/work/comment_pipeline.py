#!/usr/bin/env python3
"""comment2019zh pipeline v1: full audit + tier classification + cleaning + candidates (one pass)."""
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/comment2019zh")
AUD = WS / f"outputs/comment2019zh/full_audit_v1_{TS}"
CAND = WS / f"outputs/comment2019zh/curated_candidate_v1_{TS}"
AUD.mkdir(parents=True, exist_ok=False)
CAND.mkdir(parents=True, exist_ok=False)
(AUD / "samples").mkdir()

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{5,10}(?!\d)")
WECHAT_RE = re.compile(r"(?:wxid[_\-][A-Za-z0-9_\-]+|微信号\s*[:：]?\s*[A-Za-z0-9_\-]{5,20})")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
AD_KW = ["广告", "推广", "促销", "特价", "优惠券", "点击购买", "立即购买", "限时抢购", "免费领取", "扫码", "加微信", "加QQ", "淘宝", "天猫", "京东", "包邮"]
SPAM_KW = ["刷单", "好评返现", "返现", "复制这条", "点赞", "关注我", "求关注"]
PROFANITY = ["他妈", "去死", "傻逼", "傻B", "脑残", "垃圾货", "骗子", "坑人", "不要脸", "尼玛", "草泥马", "滚蛋", "畜生", "白痴", "贱人", "狗屎"]
TEMPLATE_REPLIES = ["谢谢分享", "感谢分享", "收藏了", "mark", "路过", "顶一个", "学习了", "沙发", "板凳"]


def count_symbols(text):
    return sum(1 for c in text if unicodedata.category(c).startswith(("P", "S")))


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


audit_counts = Counter()
reservoirs = defaultdict(lambda: Reservoir(50))
ta = (CAND / "tier_a_clean.jsonl").open("w", encoding="utf-8")
tb = (CAND / "tier_b_clean.jsonl").open("w", encoding="utf-8")
rv = (CAND / "review.jsonl").open("w", encoding="utf-8")
dp = (CAND / "drop.jsonl").open("w", encoding="utf-8")
clog = (CAND / "cleaning_log.jsonl").open("w", encoding="utf-8")
cman = gzip.open(CAND / "document_manifest.jsonl.gz", "wt", encoding="utf-8")
tier_counts = Counter()
tier_chars = Counter()
primary = Counter()
action_counts = Counter()
total = 0
total_chars = 0
cleaned_docs = 0
empty_after = []

for fi, path in enumerate(sorted(INPUT.iterdir()), start=1):
    rel = path.name
    for idx, paragraphs in iter_documents(path):
        total += 1
        text = "\n".join(paragraphs)
        chars = sum(len(p) for p in paragraphs)
        total_chars += chars
        compact = "".join(c for c in text if not c.isspace())
        chinese = len(CHINESE_RE.findall(compact))
        chinese_ratio = chinese / len(compact) if compact else 0.0
        symbol_ratio = count_symbols(compact) / len(compact) if compact else 0.0
        did = f"{rel}:{idx:07d}"
        sample = {"document_id": did, "chars": chars, "text": text[:300]}

        # audit
        if PHONE_RE.search(text):
            audit_counts["phone"] += 1; reservoirs["phone"].add(sample)
        if QQ_RE.search(text):
            audit_counts["qq"] += 1; reservoirs["qq"].add(sample)
        if WECHAT_RE.search(text):
            audit_counts["wechat"] += 1; reservoirs["wechat"].add(sample)
        if IDCARD_RE.search(text):
            audit_counts["id_card"] += 1; reservoirs["id_card"].add(sample)
        if EMAIL_RE.search(text):
            audit_counts["email"] += 1; reservoirs["email"].add(sample)
        if URL_RE.search(text):
            audit_counts["url"] += 1; reservoirs["url"].add(sample)
        if sum(1 for k in AD_KW if k in text):
            audit_counts["ad_keyword"] += 1; reservoirs["ad_keyword"].add(sample)
        if sum(1 for k in SPAM_KW if k in text):
            audit_counts["spam_keyword"] += 1; reservoirs["spam_keyword"].add(sample)
        if sum(1 for k in PROFANITY if k in text):
            audit_counts["profanity"] += 1; reservoirs["profanity"].add(sample)
        if sum(1 for k in TEMPLATE_REPLIES if k in text):
            audit_counts["template_reply"] += 1; reservoirs["template_reply"].add(sample)
        emoji_n = len(EMOJI_RE.findall(text))
        if emoji_n >= 5:
            audit_counts["emoji_stacked"] += 1; reservoirs["emoji_stacked"].add(sample)
        if chinese == 0 and compact and symbol_ratio >= 0.6:
            audit_counts["punctuation_only"] += 1; reservoirs["punctuation_only"].add(sample)
        if chars < 20:
            audit_counts["very_short_lt20"] += 1; reservoirs["very_short_lt20"].add(sample)
        if chars < 80:
            audit_counts["short_20_79"] += 1
        if CONTROL_RE.search(text) or "\ufffd" in text:
            audit_counts["control_or_replacement"] += 1
        if len(compact) >= 50 and chinese_ratio < 0.15:
            audit_counts["low_chinese_ratio"] += 1; reservoirs["low_chinese_ratio"].add(sample)

        # classify
        hard, review, soft, actions = [], [], [], []
        pii = len(PHONE_RE.findall(text)) + len(QQ_RE.findall(text)) + len(WECHAT_RE.findall(text)) + len(IDCARD_RE.findall(text))
        ad_n = sum(1 for k in AD_KW if k in text)
        spam_n = sum(1 for k in SPAM_KW if k in text)
        prof_n = sum(1 for k in PROFANITY if k in text)
        templ_n = sum(1 for k in TEMPLATE_REPLIES if k in text)

        if chars < 20:
            hard.append("uninformative_very_short")
        if chinese == 0 and compact and symbol_ratio >= 0.6:
            hard.append("punctuation_only")
        if prof_n >= 2 or (prof_n == 1 and chars < 60):
            hard.append("profanity_heavy")
        if spam_n >= 2:
            hard.append("spam")
        if pii >= 2:
            review.append("privacy_heavy")
        if ad_n >= 2:
            review.append("ad_heavy")
        if templ_n and chars < 40:
            hard.append("template_reply_only")
        if emoji_n >= 5:
            review.append("emoji_stacked")

        if pii:
            soft.append("privacy_cleanable")
            if PHONE_RE.search(text): actions.append("remove_phone")
            if QQ_RE.search(text): actions.append("remove_qq")
            if WECHAT_RE.search(text): actions.append("remove_wechat")
            if IDCARD_RE.search(text): actions.append("remove_idcard")
        if EMAIL_RE.search(text):
            soft.append("email_cleanable"); actions.append("remove_email")
        if URL_RE.search(text):
            soft.append("url_cleanable"); actions.append("remove_url")
        if ad_n == 1:
            soft.append("ad_line_cleanable"); actions.append("remove_ad_lines")
        if chars < 80:
            soft.append("short_document")
        if chars >= 50 and chinese_ratio < 0.5:
            soft.append("low_chinese_ratio")
        if chars >= 50 and symbol_ratio >= 0.25:
            soft.append("high_symbol_ratio")

        hard = list(dict.fromkeys(hard)); review = list(dict.fromkeys(review)); soft = list(dict.fromkeys(soft)); actions = list(dict.fromkeys(actions))
        reasons = hard + review + soft
        if hard:
            tier, pr = "DROP", hard[0]
        elif review:
            tier, pr = "REVIEW", review[0]
        elif chars >= 80 and chinese_ratio >= 0.5 and symbol_ratio < 0.25 and not soft and not actions:
            tier, pr = "A", "strict_clean"
        else:
            tier, pr = "B", soft[0] if soft else "usable_nonstandard"
        if not reasons:
            reasons = [pr]
        tier_counts[tier] += 1
        tier_chars[tier] += chars
        primary[pr] += 1
        for a in actions:
            action_counts[a] += 1
        raw_sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

        # clean
        cleaned, notes = None, []
        if tier == "B" and actions:
            p = list(paragraphs)
            if "remove_phone" in actions: p = [PHONE_RE.sub("", x) for x in p]
            if "remove_qq" in actions: p = [QQ_RE.sub("", x) for x in p]
            if "remove_wechat" in actions: p = [WECHAT_RE.sub("", x) for x in p]
            if "remove_idcard" in actions: p = [IDCARD_RE.sub("", x) for x in p]
            if "remove_email" in actions: p = [EMAIL_RE.sub("", x) for x in p]
            if "remove_url" in actions: p = [URL_RE.sub("", x) for x in p]
            out = []
            for x in p:
                line = x.strip()
                if not line:
                    continue
                if "remove_ad_lines" in actions and any(k in line for k in AD_KW) and len(line) < 60:
                    notes.append("ad_line:dropped")
                    continue
                out.append(line)
            cleaned = "\n".join(out)
            csha = hashlib.sha256(cleaned.encode("utf-8", errors="replace")).hexdigest()
            clog.write(json.dumps({"document_id": did, "primary_reason": pr, "repair_actions": actions, "cleaning_notes": notes,
                                   "char_before": chars, "char_after": sum(len(x) for x in cleaned.split("\n")) if cleaned else 0,
                                   "text_sha256_before": raw_sha, "text_sha256_after": csha}, ensure_ascii=False) + "\n")
            cleaned_docs += 1
            if not cleaned.strip():
                empty_after.append(did)
        elif tier == "B":
            cleaned = text

        final_text = cleaned if tier == "B" else text
        final_sha = hashlib.sha256(final_text.encode("utf-8", errors="replace")).hexdigest()
        if tier == "A":
            ta.write(json.dumps({"document_id": did, "text": text}, ensure_ascii=False) + "\n")
        elif tier == "B":
            tb.write(json.dumps({"document_id": did, "text": final_text}, ensure_ascii=False) + "\n")
        elif tier == "REVIEW":
            rv.write(json.dumps({"document_id": did, "text": text}, ensure_ascii=False) + "\n")
        else:
            dp.write(json.dumps({"document_id": did, "text": text}, ensure_ascii=False) + "\n")
        cman.write(json.dumps({"document_id": did, "source_file": rel, "document_index_in_file": idx, "tier": tier,
                               "primary_reason": pr, "reasons": reasons, "repair_actions": actions,
                               "text_sha256": final_sha, "character_count": len(final_text) - final_text.count("\n") if tier in ("A", "B") else chars},
                              ensure_ascii=False) + "\n")
    if fi % 100 == 0 or fi == 687:
        print(f"files={fi} docs={total:,} tiers={dict(tier_counts)}", flush=True)

ta.close(); tb.close(); rv.close(); dp.close(); clog.close(); cman.close()

for name, res in sorted(reservoirs.items()):
    with (AUD / "samples" / f"{name}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_candidates={audit_counts[name]}\n\n")
        for i, s in enumerate(res.items, start=1):
            f.write(f"sample={i} document_id={s['document_id']} chars={s['chars']}\n{s['text']}\n\n")

audit_summary = {
    "stage": "full_audit_v1", "dataset": "comment2019zh",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "input_directory": str(INPUT), "file_count": 687,
    "total_documents": total, "total_characters": total_chars,
    "candidate_counts": dict(sorted(audit_counts.items())),
    "tier_counts": dict(sorted(tier_counts.items())),
    "tier_character_counts": dict(sorted(tier_chars.items())),
    "primary_reason_counts": dict(sorted(primary.items())),
    "repair_action_counts": dict(sorted(action_counts.items())),
}
(AUD / "summary.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(AUD / "run.log").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
(CAND / "summary.json").write_text(json.dumps({
    "stage": "curated_candidate_v1", "dataset": "comment2019zh",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "total_documents": total,
    "tier_document_counts": dict(sorted(tier_counts.items())),
    "tier_character_counts": dict(sorted(tier_chars.items())),
    "documents_cleaned": cleaned_docs,
    "empty_after_cleaning": empty_after,
    "note": "评论数据风险最高；A/B分离，REVIEW隔离，DROP留档；PII已抹除。",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(CAND / "run.log").write_text(f"comment2019zh candidates v1\ncreated_at: {audit_summary['created_at']}\n", encoding="utf-8")
print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
