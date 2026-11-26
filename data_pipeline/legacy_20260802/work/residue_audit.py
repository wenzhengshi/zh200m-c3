#!/usr/bin/env python3
"""Hard-residue audit of curated candidate v1 (tier A + B texts)."""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/curated_candidate_v4_20260801_234934")
AUD = OUT / f"residue_audit_{TS}"
AUD.mkdir(exist_ok=False)
SAMPLES = AUD / "samples"
SAMPLES.mkdir()

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
FORMULA_RE = re.compile(r"\bformula_\d+\b", re.I)
RAW_WIKI_RE = re.compile(r"\{\{|\}\}|\[\[|\]\]|<\s*/?\s*ref\b", re.I)
SOFT_HTML_RE = re.compile(r"<\s*br\s*/?\s*>|<\s*/?\s*div\b[^>]*>", re.I)
HARD_HTML_RE = re.compile(
    r"<\s*/?\s*(?:table|tr|td|th|ol|li|onlyinclude|noinclude|includeonly)\b"
    r"|!\s*colspan\s*=|\balign\s*=",
    re.I,
)
VARIANT_RE = re.compile(r"-\{[^{}\n]{0,1000}\}-")
EMPTY_SLOT_RE = re.compile(
    r"(?:[（(]\s*[，,；;、/|·\s]{0,12}[）)]|《\s*》|〈\s*〉|「\s*」|『\s*』)"
)
EMPTY_LANGUAGE_FIELD_RE = re.compile(
    r"(?:[（(]\s*(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]?\s*[，,；;）)]"
    r"|(?:英语|英文|德语|法语|日语|拉丁语|西班牙语|俄语)\s*[：:]\s*[，,；;）)])"
)
PIPE_TABLE_RE = re.compile(r"(?:\|\|.*){2,}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
EXT_NUM_HOLE_RE = re.compile(
    r"(?:总面积\s*(?:为|是)?\s*(?:平方公里|平方千米)"
    r"|年\s*时\s*的?\s*人口\s*(?:为|是)?\s*人"
    r"|人口\s*(?:为|是)?\s*人(?:[。，；;]|$)"
    r"|坐标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|座标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|海拔高度\s*(?:为|是)?\s*(?:米|公尺))"
)
SEV_HOLE_RE = re.compile(
    r"(?:广告标语是\s*[。；;]"
    r"|(?:本名|原名|英文名|外文名)\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:编剧|导演|主演|作者)\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:并)?由\s*(?:执导|导演|主演)(?=[，。；;]|$)"
    r"|参见主条目\s*[：:]\s*(?:和)?\s*$)"
)
MISSING_SUBJECT_RE = re.compile(
    r"(?:^是(?!否|非|由此)|^为\s*(?:由|一个|一名|一种|一款|一部|下辖|位于|发行|发售|上映|播出))"
)


def count_symbols(text):
    import unicodedata
    return sum(1 for c in text if unicodedata.category(c).startswith(("P", "S")))


class Reservoir:
    def __init__(self, limit, seed=20260801):
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


def detect(text, first_paragraph):
    """Return dict of detector -> True, with category."""
    compact = "".join(ch for ch in text if not ch.isspace())
    chinese = len(CHINESE_RE.findall(compact))
    latin = len(LATIN_RE.findall(compact))
    chinese_ratio = chinese / len(compact) if compact else 0.0
    latin_ratio = latin / len(compact) if compact else 0.0
    symbol_ratio = count_symbols(compact) / len(compact) if compact else 0.0
    hard = {
        "control_characters": bool(CONTROL_RE.search(text)),
        "unicode_replacement": "\ufffd" in text,
        "formula_placeholder": bool(FORMULA_RE.search(text)),
        "raw_wiki_markup": bool(RAW_WIKI_RE.search(text)),
        "hard_html": bool(HARD_HTML_RE.search(text)),
        "empty_slots": bool(EMPTY_SLOT_RE.search(text)),
        "empty_language_field": bool(EMPTY_LANGUAGE_FIELD_RE.search(text)),
        "email": bool(EMAIL_RE.search(text)),
        "url": bool(URL_RE.search(text)),
        "pipe_table": bool(PIPE_TABLE_RE.search(text) or text.count("||") >= 2),
    }
    soft = {
        "soft_html": bool(SOFT_HTML_RE.search(text)),
        "wiki_variant_markup": bool(VARIANT_RE.search(text)),
        "low_chinese_ratio": len(compact) >= 50 and chinese_ratio < 0.50,
        "extremely_low_chinese_ratio": len(compact) >= 50 and chinese_ratio < 0.15,
        "high_symbol_ratio": len(compact) >= 50 and symbol_ratio >= 0.25,
        "extremely_high_symbol_ratio": len(compact) >= 50 and symbol_ratio >= 0.35,
        "short_document_lt20": len(text) < 20,
        "short_document_lt80": len(text) < 80,
        "long_document_gt50000": len(text) > 50000,
        "long_document_gt100000": len(text) > 100000,
        "high_foreign_ratio": latin_ratio >= 0.5 and len(compact) >= 100,
    }
    heur = {
        "missing_subject": bool(MISSING_SUBJECT_RE.search(first_paragraph)),
        "numeric_hole": bool(EXT_NUM_HOLE_RE.search(text)),
        "severe_hole": bool(SEV_HOLE_RE.search(text)),
    }
    return hard, soft, heur


counts_hard = Counter()
counts_soft = Counter()
counts_heur = Counter()
per_doc_hard = Counter()
hard_a = []
hard_b = []
reservoirs = defaultdict(lambda: Reservoir(30))
doc_total = 0

for name, tier in (("tier_a_clean.jsonl", "A"), ("tier_b_clean.jsonl", "B")):
    with (OUT / name).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_total += 1
            text = row["text"]
            first = text.split("\n", 1)[0] if text else ""
            hard, soft, heur = detect(text, first)
            for k, v in hard.items():
                if v:
                    counts_hard[k] += 1
                    per_doc_hard[(tier, k)] += 1
                    reservoirs[k].add((tier, row["document_id"]))
                    if tier == "A":
                        hard_a.append((k, row["document_id"]))
                    else:
                        hard_b.append((k, row["document_id"]))
            for k, v in soft.items():
                if v:
                    counts_soft[k] += 1
            for k, v in heur.items():
                if v:
                    counts_heur[k] += 1
                    reservoirs["heur_" + k].add((tier, row["document_id"]))

summary = {
    "stage": "residue_audit",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "documents_scanned": doc_total,
    "tier": "A+B (candidates; REVIEW/DROP excluded by design)",
    "hard_error_counts": dict(sorted(counts_hard.items())),
    "soft_acceptable_counts": dict(sorted(counts_soft.items())),
    "heuristic_review_counts": dict(sorted(counts_heur.items())),
    "hard_by_tier": {f"{t}::{k}": v for (t, k), v in sorted(per_doc_hard.items())},
    "hard_document_ids_A": [did for _, did in hard_a],
    "note": "硬残留不得通过忽略清零；A中出现硬残留视为分类器漏判，B中硬残留计入复核。",
}
(AUD / "residue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

for k, res in sorted(reservoirs.items()):
    if k.startswith("heur_"):
        continue
    with (AUD / "samples" / f"hard_{k}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_documents={counts_hard[k]}\n")
        for i, (tier, did) in enumerate(res.items, start=1):
            f.write(f"sample={i} tier={tier} document_id={did}\n")

for k in ("heur_missing_subject", "heur_numeric_hole", "heur_severe_hole"):
    with (AUD / "samples" / f"{k}_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_documents={counts_heur[k.split('_',1)[1]]}\n")
        for i, (tier, did) in enumerate(reservoirs[k].items, start=1):
            f.write(f"sample={i} tier={tier} document_id={did}\n")

log = [
    "residue audit run.log",
    f"documents scanned (A+B): {doc_total:,}",
    "hard errors: " + json.dumps(dict(sorted(counts_hard.items())), ensure_ascii=False),
    "soft: " + json.dumps(dict(sorted(counts_soft.items())), ensure_ascii=False),
    "heuristic: " + json.dumps(dict(sorted(counts_heur.items())), ensure_ascii=False),
]
(AUD / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
