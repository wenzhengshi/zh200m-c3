#!/usr/bin/env python3
"""webText2019zh tier dry-run v1 classifier (no source modification)."""
import gzip
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{5,10}(?!\d)")
WECHAT_RE = re.compile(r"(?:wxid[_\-][A-Za-z0-9_\-]+|微信号\s*[:：]?\s*[A-Za-z0-9_\-]{5,20})")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
AD_KW = ["广告", "推广", "促销", "特价", "优惠券", "点击购买", "立即购买", "限时抢购", "免费领取", "扫码", "加盟", "代理招商"]
NAV_KW = ["首页", "导航", "友情链接", "网站地图", "版权所有", "ICP备", "免责声明", "关于我们", "联系我们", "设为首页", "收藏本站"]
CONTACT_KW = ["联系电话", "联系方式", "联系地址", "手机号", "微信号", "邮箱：", "E-mail", "e-mail"]
TEMPLATE_MARKERS = ["{$", "{%", "{{if", "{{loop", "<!--", "<div", "<span", "class=\"", "style=\""]


def count_symbols(text):
    return sum(1 for c in text if unicodedata.category(c).startswith(("P", "S")))


def classify(paragraphs):
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    text = "\n".join(paragraphs)
    compact = "".join(c for c in text if not c.isspace())
    chars = sum(len(p) for p in paragraphs)
    chinese = len(CHINESE_RE.findall(compact))
    latin = len(LATIN_RE.findall(compact))
    chinese_ratio = chinese / len(compact) if compact else 0.0
    symbol_ratio = count_symbols(compact) / len(compact) if compact else 0.0
    control = CONTROL_RE.search(text) is not None
    replacement = "\ufffd" in text
    url_count = len(URL_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    phone_count = len(PHONE_RE.findall(text))
    qq_count = len(QQ_RE.findall(text))
    wechat_count = len(WECHAT_RE.findall(text))
    idcard_count = len(IDCARD_RE.findall(text))
    ad_hits = sum(1 for k in AD_KW if k in text)
    nav_hits = sum(1 for k in NAV_KW if k in text)
    contact_hits = sum(1 for k in CONTACT_KW if k in text)
    template_hits = sum(1 for k in TEMPLATE_MARKERS if k in text)
    pii_count = phone_count + qq_count + wechat_count + idcard_count + email_count

    hard = []
    review = []
    soft = []
    actions = []

    if chars < 20:
        hard.append("uninformative_very_short")
    elif chars < 60 and chinese_ratio < 0.30:
        hard.append("short_low_chinese_junk")
    if chars >= 60 and chinese_ratio < 0.15:
        hard.append("garbled_or_low_chinese")
    if control or replacement:
        hard.append("control_or_replacement_characters")
    if template_hits and chars < 150:
        hard.append("template_no_content")

    if ad_hits >= 3:
        review.append("ad_heavy")
    if template_hits and chars >= 150:
        review.append("template_marker")
    if latin >= 50 and chinese < 0.20 and chars >= 100:
        review.append("machine_translation_or_foreign")
    if symbol_ratio >= 0.35 and chars >= 50:
        review.append("extremely_high_symbol_ratio")
    if pii_count >= 3:
        review.append("privacy_heavy")

    if pii_count:
        soft.append("privacy_cleanable")
        if phone_count:
            actions.append("remove_phone")
        if qq_count:
            actions.append("remove_qq")
        if wechat_count:
            actions.append("remove_wechat")
        if idcard_count:
            actions.append("remove_idcard")
        if email_count:
            actions.append("remove_email")
    if url_count:
        soft.append("url_cleanable")
        actions.append("remove_url")
    if nav_hits:
        soft.append("nav_footer_cleanable")
        actions.append("remove_nav_footer_lines")
    if ad_hits and ad_hits < 3:
        soft.append("ad_line_cleanable")
        actions.append("remove_ad_lines")
    if contact_hits:
        soft.append("contact_line_cleanable")
        actions.append("remove_contact_lines")
    if chars < 80:
        soft.append("short_document")
    if chars >= 50 and chinese_ratio < 0.50:
        soft.append("low_chinese_ratio")
    if chars >= 50 and symbol_ratio >= 0.25:
        soft.append("high_symbol_ratio")

    hard = list(dict.fromkeys(hard))
    review = list(dict.fromkeys(review))
    soft = list(dict.fromkeys(soft))
    actions = list(dict.fromkeys(actions))
    reasons = hard + review + soft

    if hard:
        tier = "DROP"
        primary = hard[0]
    elif review:
        tier = "REVIEW"
        primary = review[0]
    elif chars >= 80 and chinese_ratio >= 0.50 and symbol_ratio < 0.25 and not soft and not actions:
        tier = "A"
        primary = "strict_clean"
    else:
        tier = "B"
        primary = soft[0] if soft else "usable_nonstandard"
    if not reasons:
        reasons = [primary]
    return {
        "tier": tier,
        "primary_reason": primary,
        "reasons": reasons,
        "repair_actions": actions,
        "metrics": {
            "character_count": chars,
            "paragraph_count": len(paragraphs),
            "chinese_ratio": round(chinese_ratio, 6),
            "symbol_ratio": round(symbol_ratio, 6),
            "control_count": 1 if control else 0,
            "replacement_count": 1 if replacement else 0,
            "url_count": url_count,
            "phone_count": phone_count,
            "qq_count": qq_count,
            "wechat_count": wechat_count,
            "idcard_count": idcard_count,
            "email_count": email_count,
            "ad_hits": ad_hits,
            "nav_hits": nav_hits,
            "contact_hits": contact_hits,
            "template_hits": template_hits,
            "pii_count": pii_count,
        },
    }


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


def main():
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("usage: classify_webtext_v1.py <INPUT_DIR> <OUTPUT_DIR>")
    input_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    rng = random.Random(20260802)

    tier_counts = Counter()
    tier_chars = Counter()
    primary = Counter()
    overlapping = Counter()
    actions_c = Counter()
    total = 0
    total_chars = 0
    samples = defaultdict(list)

    def reservoir_add(key, item, limit=50):
        lst = samples[key]
        lst.append(item)
        if len(lst) > limit:
            lst.pop(rng.randrange(len(lst)))

    with gzip.open(out_dir / "classification_manifest_v1.jsonl.gz", "wt", encoding="utf-8") as mf:
        files = sorted(input_dir.iterdir())
        for fi, path in enumerate(files, start=1):
            rel = path.name
            for idx, paragraphs in iter_documents(path):
                res = classify(paragraphs)
                did = f"{rel}:{idx:07d}"
                text = "\n".join(paragraphs)
                text_sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                mf.write(
                    json.dumps(
                        {
                            "document_id": did,
                            "source_file": rel,
                            "document_index_in_file": idx,
                            "tier": res["tier"],
                            "primary_reason": res["primary_reason"],
                            "reasons": res["reasons"],
                            "repair_actions": res["repair_actions"],
                            "text_sha256": text_sha,
                            **res["metrics"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total += 1
                total_chars += res["metrics"]["character_count"]
                tier_counts[res["tier"]] += 1
                tier_chars[res["tier"]] += res["metrics"]["character_count"]
                primary[res["primary_reason"]] += 1
                for r in set(res["reasons"]):
                    overlapping[r] += 1
                for a in set(res["repair_actions"]):
                    actions_c[a] += 1
                reservoir_add(("tier", res["tier"]), {"document_id": did, **res, "text": text[:500]})
                reservoir_add(("reason", res["primary_reason"]), {"document_id": did, **res, "text": text[:500]})
            if fi % 100 == 0 or fi == len(files):
                print(f"[{fi}/{len(files)}] docs={total:,}", flush=True)

    for (kind, key), items in samples.items():
        if kind == "tier":
            safe = f"tier_{key}"
        else:
            safe = f"reason_{re.sub(r'[^A-Za-z0-9_.-]+', '_', key)}"
        with (samples_dir / f"{safe}_sample.txt").open("w", encoding="utf-8") as f:
            f.write(f"total_candidates={tier_counts[key] if kind == 'tier' else primary[key]}\n")
            for i, s in enumerate(items, start=1):
                f.write(f"sample={i} document_id={s['document_id']} tier={s['tier']} primary={s['primary_reason']}\n")
                f.write("reasons=" + json.dumps(s["reasons"], ensure_ascii=False) + "\n")
                f.write("actions=" + json.dumps(s["repair_actions"], ensure_ascii=False) + "\n")
                f.write("metrics=" + json.dumps(s["metrics"], ensure_ascii=False, sort_keys=True) + "\n")
                f.write(s["text"] + "\n\n")

    summary = {
        "dataset": "webText2019zh",
        "version": "tier_dryrun_v1",
        "input_directory": str(input_dir),
        "file_count": len(files),
        "total_documents": total,
        "total_characters": total_chars,
        "tier_counts": dict(sorted(tier_counts.items())),
        "tier_character_counts": dict(sorted(tier_chars.items())),
        "primary_reason_counts": dict(sorted(primary.items())),
        "overlapping_reason_counts": dict(sorted(overlapping.items())),
        "repair_action_counts": dict(sorted(actions_c.items())),
        "note": "Dry-run only. No source document modified.",
        "seed": 20260802,
    }
    (out_dir / "summary_tier_dryrun_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "tier_counts_v1.tsv").open("w", encoding="utf-8") as f:
        f.write("tier\tdocuments\tcharacters\n")
        for t in ("A", "B", "REVIEW", "DROP"):
            f.write(f"{t}\t{tier_counts[t]}\t{tier_chars[t]}\n")
    with (out_dir / "reason_counts_v1.tsv").open("w", encoding="utf-8") as f:
        f.write("reason\tprimary_count\toverlapping_count\n")
        for r in sorted(set(primary) | set(overlapping)):
            f.write(f"{r}\t{primary[r]}\t{overlapping[r]}\n")
    with (out_dir / "repair_action_counts_v1.tsv").open("w", encoding="utf-8") as f:
        f.write("repair_action\tdocuments\n")
        for a, n in sorted(actions_c.items()):
            f.write(f"{a}\t{n}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
