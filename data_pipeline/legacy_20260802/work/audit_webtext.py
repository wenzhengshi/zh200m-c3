#!/usr/bin/env python3
"""Full audit of webText2019zh (read-only): risk-candidate statistics + samples."""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
INPUT = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/webText2019zh")
OUT = WS / f"outputs/webText2019zh/full_audit_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
SAMPLES = OUT / "samples"
SAMPLES.mkdir()

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{5,10}(?!\d)")
WECHAT_RE = re.compile(r"(?:wxid[_\-][A-Za-z0-9_\-]+|微信号\s*[:：]?\s*[A-Za-z0-9_\-]{5,20})")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
AD_KEYWORDS = [
    "广告", "推广", "促销", "特价", "优惠券", "点击购买", "立即购买", "淘宝", "天猫", "京东", "包邮",
    "限时抢购", "免费领取", "扫码", "二维码", "客服", "加盟", "代理招商",
]
NAV_KEYWORDS = [
    "首页", "导航", "友情链接", "网站地图", "版权", "ICP备", "版权所有", "免责声明",
    "关于我们", "联系我们", "网站首页", "设为首页", "收藏本站",
]
CONTACT_KEYWORDS = ["联系电话", "联系方式", "邮箱：", "E-mail", "地址：", "联系地址", "手机号"]
TEMPLATE_MARKERS = ["{$", "{%", "{{if", "{{loop", "<!--", "class=\"", "style=\"", "<div", "<span"]
ADSAMPLE_KEYWORDS = ["广告", "推广", "营销", "SEO", "百度", "竞价"]


def count_symbols(text):
    return sum(1 for c in text if unicodedata.category(c).startswith(("P", "S")))


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


counts = Counter()
file_count = 0
total_docs = 0
total_chars = 0
total_lines = 0
reservoirs = defaultdict(lambda: Reservoir(50))

for path in sorted(INPUT.iterdir()):
    for idx, paragraphs in iter_documents(path):
        total_docs += 1
        text = "\n".join(paragraphs)
        chars = sum(len(p) for p in paragraphs)
        total_chars += chars
        total_lines += len(paragraphs)
        compact = "".join(ch for ch in text if not ch.isspace())
        chinese = len(CHINESE_RE.findall(compact))
        latin = len(LATIN_RE.findall(compact))
        chinese_ratio = chinese / len(compact) if compact else 0.0
        latin_ratio = latin / len(compact) if compact else 0.0
        symbol_ratio = count_symbols(compact) / len(compact) if compact else 0.0
        sample = {"document_id": f"{path.name}:{idx:07d}", "chars": chars, "text": text[:400]}

        if CONTROL_RE.search(text):
            counts["control_characters"] += 1
            reservoirs["control_characters"].add(sample)
        if "\ufffd" in text:
            counts["unicode_replacement"] += 1
            reservoirs["unicode_replacement"].add(sample)
        if EMAIL_RE.search(text):
            counts["email"] += 1
            reservoirs["email"].add(sample)
        if URL_RE.search(text):
            counts["url"] += 1
            reservoirs["url"].add(sample)
        if PHONE_RE.search(text):
            counts["phone"] += 1
            reservoirs["phone"].add(sample)
        if QQ_RE.search(text):
            counts["qq_number"] += 1
            reservoirs["qq_number"].add(sample)
        if WECHAT_RE.search(text):
            counts["wechat"] += 1
            reservoirs["wechat"].add(sample)
        if IDCARD_RE.search(text):
            counts["id_card"] += 1
            reservoirs["id_card"].add(sample)
        ad_hits = sum(1 for k in AD_KEYWORDS if k in text)
        if ad_hits:
            counts["ad_keyword"] += 1
            reservoirs["ad_keyword"].add(sample)
        nav_hits = sum(1 for k in NAV_KEYWORDS if k in text)
        if nav_hits:
            counts["nav_keyword"] += 1
            reservoirs["nav_keyword"].add(sample)
        if any(k in text for k in CONTACT_KEYWORDS):
            counts["contact_info"] += 1
            reservoirs["contact_info"].add(sample)
        if any(k in text for k in TEMPLATE_MARKERS):
            counts["template_marker"] += 1
            reservoirs["template_marker"].add(sample)
        if len(compact) >= 50 and chinese_ratio < 0.15:
            counts["extremely_low_chinese_ratio"] += 1
            reservoirs["extremely_low_chinese_ratio"].add(sample)
        if len(compact) >= 50 and latin_ratio >= 0.5:
            counts["high_foreign_ratio"] += 1
            reservoirs["high_foreign_ratio"].add(sample)
        if len(compact) >= 50 and symbol_ratio >= 0.35:
            counts["extremely_high_symbol_ratio"] += 1
            reservoirs["extremely_high_symbol_ratio"].add(sample)
        if chars < 20:
            counts["very_short_lt20"] += 1
            reservoirs["very_short_lt20"].add(sample)
        if chars < 80:
            counts["short_20_79"] += 1
        if chars > 10000:
            counts["very_long_gt10000"] += 1
        if len(paragraphs) == 1:
            counts["single_paragraph"] += 1
    file_count += 1
    if file_count % 100 == 0:
        print(f"files={file_count} docs={total_docs:,}", flush=True)

for name, res in sorted(reservoirs.items()):
    with (SAMPLES / f"{name}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_candidates={counts[name]}\nseed=20260802\n\n")
        for i, s in enumerate(res.items, start=1):
            f.write(f"sample={i} document_id={s['document_id']} chars={s['chars']}\n")
            f.write(s["text"] + "\n\n")

summary = {
    "stage": "full_audit_v1",
    "dataset": "webText2019zh",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "input_directory": str(INPUT),
    "file_count": file_count,
    "total_documents": total_docs,
    "total_characters": total_chars,
    "total_paragraphs": total_lines,
    "candidate_counts": dict(sorted(counts.items())),
    "note": "候选类别为可重叠启发式统计，不是最终删除结论。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
log = [
    f"full_audit_v1 run.log",
    f"files: {file_count}, docs: {total_docs:,}, chars: {total_chars:,}, paragraphs: {total_lines:,}",
    "candidate counts: " + json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
]
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
