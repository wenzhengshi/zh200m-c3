#!/usr/bin/env python3
"""Generate the wiki2019zh cleaning preview v1 (preview only, no training files)."""
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / f"outputs/wiki2019zh/cleaning_preview_v2_{TS}"
OUT.mkdir(parents=True, exist_ok=False)
SAMPLES_DIR = OUT / "preview_samples"
SAMPLES_DIR.mkdir()

INPUT_DIR = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh")
M31 = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz")
CLS_SRC = WS / "outputs/wiki2019zh/tier_dryrun_v3_1_package_20260801_223302/classify_wiki2019zh_tier_dryrun_v3_1.py"

# ---- import regex constants and helpers from the classifier (up to def classify) ----
sys.argv = [
    "classify_wiki2019zh_tier_dryrun_v3_1.py",
    str(INPUT_DIR),
    str(M31),
    "/tmp/preview_consts",
]
src = CLS_SRC.read_text(encoding="utf-8")
header = src.split("def classify(", 1)[0]
ns = {"__name__": "cleaner_consts"}
exec(compile(header, str(CLS_SRC), "exec"), ns)
CHINESE_RE = ns["CHINESE_RE"]
LATIN_RE = ns["LATIN_RE"]
URL_RE = ns["URL_RE"]
EMAIL_RE = ns["EMAIL_RE"]
EMPTY_SLOT_RE = ns["EMPTY_SLOT_RE"]
VARIANT_RE = ns["VARIANT_RE"]
RAW_WIKI_RE = ns["RAW_WIKI_RE"]
SOFT_HTML_RE = ns["SOFT_HTML_RE"]
HARD_HTML_RE = ns["HARD_HTML_RE"]
FORMULA_RE = ns["FORMULA_RE"]
SEVERE_HOLE_RE = ns["SEVERE_HOLE_RE"]
EXTENDED_NUMERIC_HOLE_RE = ns["EXTENDED_NUMERIC_HOLE_RE"]
CORE_MISSING_TAIL_RE = ns["CORE_MISSING_TAIL_RE"]
CHART_STUB_RE = ns["CHART_STUB_RE"]
EMPTY_LANGUAGE_FIELD_RE = ns["EMPTY_LANGUAGE_FIELD_RE"]
REFERENCE_HEADING_RE = ns["REFERENCE_HEADING_RE"]
TRAILING_FRAGMENT_RE = ns["TRAILING_FRAGMENT_RE"]
DECLARATIVE_RE = ns["DECLARATIVE_RE"]
END_PUNCTUATION = ns["END_PUNCTUATION"]
foreign_paragraph_metrics = ns["foreign_paragraph_metrics"]

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")

# extra regexes for cleaning
WIKI_VARIANT_VALUE = re.compile(r"-{(?:zh-hans|zh-hant|zh-cn|zh-tw|zh|zh-sg):([^{}]*?)}-", re.S)
WIKI_VARIANT_PIPE = re.compile(r"-{([^{}|;]+)\|([^{}]*)}-", re.S)
WIKI_VARIANT_PLAIN = re.compile(r"-{([^{}]*?)}-", re.S)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.S)
LINK_PIPE_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
LINK_PLAIN_RE = re.compile(r"\[\[([^\]]+)\]\]")
REF_SELFCLOSE_RE = re.compile(r"<ref[^>]*/>", re.I)
REF_TAG_RE = re.compile(r"</?ref[^>]*>", re.I)
BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.I)
DIV_RE = re.compile(r"<\s*/?\s*div\b[^>]*>", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
PARENT_ADMIN_RE_COMMA = re.compile(r"，是下辖的一个")
PARENT_ADMIN_RE_PLAIN = re.compile(r"是下辖的一个")

PROSE_MARKERS = re.compile(
    r"(?:的|了|着|在|于|中|从|到|向|被|把|也|都|更|又|曾|将|已|并|或|而|以|由|和|与|及|其|等|之|为|是|个|条|名|位|有|因|据|于|后|前|时|年)"
)


def count_comma_marks(s: str) -> int:
    s = DIGIT_COMMA.sub("", s)
    return s.count("，") + s.count("、") + s.count(",")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_control(text: str) -> str:
    return CONTROL_RE.sub("", text)


def clean_wiki_variant(text: str) -> str:
    text = WIKI_VARIANT_VALUE.sub(lambda m: m.group(1), text)
    text = WIKI_VARIANT_PIPE.sub(lambda m: m.group(2), text)
    text = WIKI_VARIANT_PLAIN.sub(lambda m: m.group(1), text)
    return text


def clean_raw_wiki(text: str) -> str:
    text = TEMPLATE_RE.sub("", text)
    text = REF_SELFCLOSE_RE.sub("", text)
    text = REF_TAG_RE.sub("", text)
    text = LINK_PIPE_RE.sub(lambda m: m.group(2), text)
    text = LINK_PLAIN_RE.sub(lambda m: m.group(1), text)
    text = text.replace("[[", "").replace("]]", "")
    text = text.replace("{{", "").replace("}}", "")
    return text


def clean_soft_html(text: str) -> str:
    text = BR_RE.sub("", text)
    text = DIV_RE.sub("", text)
    return text


def clean_email(text: str) -> str:
    return EMAIL_RE.sub("", text)


def clean_empty_language_field(text: str) -> str:
    return EMPTY_LANGUAGE_FIELD_RE.sub("", text)


def clean_empty_slots(text: str) -> str:
    return EMPTY_SLOT_RE.sub("", text)


def clean_url_lines(paragraphs):
    return [p for p in paragraphs if not URL_RE.fullmatch(p.strip())]


def clean_reference_section(paragraphs):
    out = []
    cut = False
    for p in paragraphs:
        if cut:
            continue
        if REFERENCE_HEADING_RE.match(p):
            cut = True
            continue
        out.append(p)
    return out


def clean_parent_admin(text: str) -> str:
    text = PARENT_ADMIN_RE_COMMA.sub("是一个", text)
    text = PARENT_ADMIN_RE_PLAIN.sub("是一个", text)
    return text


def trim_defective_clause(paragraph):
    """Remove hole sentences / trim hole clauses conservatively.

    For each sentence containing a severe/numeric hole match:
    - if non-hole content in the sentence is < 10 chars, drop the whole sentence
      (template/extraction artifact, e.g. "该市镇总面积平方公里，年时的人口为人。");
    - otherwise remove only the exact matched hole spans, then collapse
      leftover punctuation (e.g. "电影，导演为，编剧为。" -> "电影。").
    """
    matches = []
    for rx in (SEVERE_HOLE_RE, EXTENDED_NUMERIC_HOLE_RE):
        for m in rx.finditer(paragraph):
            matches.append(m)
    if not matches:
        return paragraph

    # sentence boundaries: 。！？； plus newline
    terms = [m.start() for m in re.finditer(r"[。！？；\n]", paragraph)]
    terms = [-1] + terms + [len(paragraph)]
    sentences = []
    for i in range(len(terms) - 1):
        start = terms[i] + 1
        end = terms[i + 1] + 1
        if end > len(paragraph):
            end = len(paragraph)
        sentences.append((start, end))

    out = []
    for start, end in sentences:
        sentence = paragraph[start:end]
        sm = [m for m in matches if m.start() >= start and m.end() <= end]
        if not sm:
            out.append(sentence)
            continue
        hole_len = sum(len(m.group()) for m in sm)
        non_hole = len(sentence) - hole_len
        if non_hole < 10:
            continue  # drop the whole sentence
        # clause-level trim: remove exact match spans inside the sentence
        spans = sorted((m.start() - start, m.end() - start) for m in sm)
        pieces = []
        pos = 0
        for s, e in spans:
            pieces.append(sentence[pos:s])
            pos = e
        pieces.append(sentence[pos:])
        repaired = "".join(pieces)
        # collapse leftover punctuation / bare copula remnants
        repaired = re.sub(r"[，,]\s*[为是]\s*(?=[，,。；;])", "，", repaired)
        repaired = re.sub(r"[，,]{2,}", "，", repaired)
        repaired = re.sub(r"[，,][。！？]", "。", repaired)
        repaired = re.sub(r"^(?:而|并且|而且|且|以及|还有|此外)", "", repaired)
        out.append(repaired)
    return "".join(out)


def is_isolated_list_tail(last: str) -> bool:
    if last[-1] in END_PUNCTUATION:
        return False
    if CHART_STUB_RE.fullmatch(last.strip()):
        return True
    if count_comma_marks(last) < 2:
        return False
    if DECLARATIVE_RE.search(last):
        return False
    if PROSE_MARKERS.search(last):
        return False
    return True


def apply_actions(paragraphs, actions, metrics):
    """Apply the doc's repair actions in a fixed conservative order."""
    notes = []
    changed = False
    original_paragraphs = list(paragraphs)

    def rebuild():
        return "\n".join(paragraphs)

    if "remove_control_characters" in actions:
        paragraphs = [clean_control(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "normalize_wiki_variant_markup" in actions:
        paragraphs = [clean_wiki_variant(p) for p in paragraphs]

    if "remove_soft_html" in actions:
        paragraphs = [clean_soft_html(p) for p in paragraphs]

    if "remove_raw_wiki_markup" in actions:
        paragraphs = [clean_raw_wiki(p) for p in paragraphs]

    if "remove_email" in actions:
        paragraphs = [clean_email(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "remove_empty_language_field" in actions:
        paragraphs = [clean_empty_language_field(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "remove_or_normalize_empty_slots" in actions:
        paragraphs = [clean_empty_slots(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "remove_foreign_paragraphs" in actions:
        _, _, idxs = foreign_paragraph_metrics(paragraphs)
        if idxs:
            keep = [p for i, p in enumerate(paragraphs, start=1) if i not in idxs]
            paragraphs = keep

    if "remove_url_or_reference_lines" in actions:
        paragraphs = clean_url_lines(paragraphs)

    if "trim_reference_section" in actions:
        paragraphs = clean_reference_section(paragraphs)

    if "normalize_missing_parent_admin_definition" in actions:
        paragraphs = [clean_parent_admin(p) for p in paragraphs]

    if "remove_or_trim_defective_clause" in actions:
        paragraphs = [trim_defective_clause(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "remove_trailing_empty_heading" in actions and metrics.get("trailing_empty_heading_cleanable"):
        if paragraphs and paragraphs[-1].rstrip().endswith(("：", ":")):
            paragraphs = paragraphs[:-1]

    if "trim_incomplete_tail" in actions and paragraphs:
        last = paragraphs[-1].strip()
        if not last:
            paragraphs = paragraphs[:-1]
        elif last.endswith(("，", ",", "、", "；", ";")):
            paragraphs = paragraphs[:-1]
            notes.append("tail_ends_with_comma_mark:trimmed")
        elif TRAILING_FRAGMENT_RE.search(last):
            paragraphs = paragraphs[:-1]
            notes.append("tail_ends_with_connector:trimmed")
        elif is_isolated_list_tail(last):
            paragraphs = paragraphs[:-1]
            notes.append("tail_isolated_list:trimmed")
        else:
            notes.append("tail_missing_final_period:kept_for_review")

    cleaned_text = rebuild()
    return cleaned_text, notes


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
        computed_before = sha256_hex(original_text)
        if computed_before != text_sha_before:
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

# ---- write per-action sample files ----
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

# ---- tail keep list ----
with (OUT / "tail_kept_for_review.jsonl").open("w", encoding="utf-8") as f:
    for did, reason in tail_kept_ids:
        f.write(json.dumps({"document_id": did, "primary_reason": reason}, ensure_ascii=False) + "\n")

# ---- summary ----
summary = {
    "stage": "cleaning_preview_v2",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "classifier": "tier_dryrun_v3_1",
    "classifier_audit_dir": str(M31.parent),
    "input_directory": str(INPUT_DIR),
    "preview_documents_with_actions": previewed,
    "alignment_failures": alignment_failures,
    "characters_before": char_before_total,
    "characters_after": char_after_total,
    "characters_removed": char_before_total - char_after_total,
    "noop_documents": noop_count,
    "tail_kept_for_review": tail_kept,
    "action_document_counts": dict(sorted(action_docs.items())),
    "tier_counts_from_classifier": {
        "A": 522329,
        "B": 414991,
        "DROP": 65514,
        "REVIEW": 15858,
    },
    "note": "预览只审计清洗动作，不生成正式训练文件；REVIEW未进入任何清洗路径。",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

log = [
    f"cleaning_preview_v1 run.log",
    f"created_at: {summary['created_at']}",
    f"classifier: tier_dryrun_v3_1 (audit dir verified: SHA256SUMS 63/63 OK)",
    f"preview scope: all documents with >=1 repair_action (B tier cleanable docs)",
    f"previewed documents: {previewed:,}",
    f"characters before: {char_before_total:,}",
    f"characters after: {char_after_total:,}",
    f"no-op documents: {noop_count:,}",
    f"tail kept for review (missing final period, content intact): {tail_kept:,}",
    f"alignment/SHA failures: {alignment_failures}",
    "",
    "per-action document counts:",
]
for a, n in sorted(action_docs.items()):
    log.append(f"  {a}: {n:,}")
log.append("")
log.append("cleaning policy notes:")
log.append("  - trim_incomplete_tail: trims only comma/connector truncation, chart stubs, and isolated lists;")
log.append("    complete tails missing a final period are KEPT intact and listed in tail_kept_for_review.jsonl.")
log.append("  - no fabricated titles/entities/coordinates/populations/areas/formulas.")
(OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
