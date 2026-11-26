#!/usr/bin/env python3
"""Shared cleaning core for wiki2019zh (identical logic to cleaning_preview_v2)."""
import re

URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
EMPTY_SLOT_RE = re.compile(
    r"(?:"
    r"[（(]\s*[，,；;、/|·\s]{0,12}[）)]"
    r"|《\s*》|〈\s*〉|「\s*」|『\s*』"
    r")"
)
SEVERE_HOLE_RE = re.compile(
    r"(?:"
    r"广告标语是\s*[。；;]"
    r"|(?:本名|原名|英文名|外文名)"
    r"\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:编剧|导演|主演|作者)"
    r"\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:并)?由\s*(?:执导|导演|主演)(?=[，。；;]|$)"
    r"|参见主条目\s*[：:]\s*(?:和)?\s*$"
    r")"
)
EXTENDED_NUMERIC_HOLE_RE = re.compile(
    r"(?:"
    r"总面积\s*(?:为|是)?\s*(?:平方公里|平方千米)"
    r"|年\s*时\s*的?\s*人口\s*(?:为|是)?\s*人"
    r"|人口\s*(?:为|是)?\s*人(?:[。，；;]|$)"
    r"|坐标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|座标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|海拔高度\s*(?:为|是)?\s*(?:米|公尺)"
    r")"
)
CHART_STUB_RE = re.compile(
    r"^(?:人口变化图示|人口變化圖示|人口变化表|人口變化表|历年人口|歷年人口)\s*$"
)
EMPTY_LANGUAGE_FIELD_RE = re.compile(
    r"(?:英语|英文|德语|法语|日语|"
    r"拉丁语|西班牙语|俄语)\s*[：:]?\s*[，,；;）)]"
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:参考文献|參考文獻|参考资料|參考資料|"
    r"外部链接|外部連結|相关条目|相關條目|"
    r"参见|參見)\s*[:：]?\s*$"
)
TRAILING_FRAGMENT_RE = re.compile(
    r"(?:以及|并且|而且|或者|以及其|的|和|与|及|为|是|在|于|由)$"
)
DECLARATIVE_RE = re.compile(
    r"(?:是|为|属于|位于|生存于|成立于|出生于|指|包括|设立于|列出)"
)
END_PUNCTUATION = set("。！？；.!?;”’」』）》】")

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
WIKI_VARIANT_VALUE = re.compile(r"-{(?:zh-hans|zh-hant|zh-cn|zh-tw|zh|zh-sg):([^{}]*?)}-", re.S)
WIKI_VARIANT_PIPE = re.compile(r"-{([^{}|;]+)\|([^{}]*)}-", re.S)
WIKI_VARIANT_PLAIN = re.compile(r"-{([^{}]*?)}-", re.S)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.S)
LINK_PIPE_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
LINK_PLAIN_RE = re.compile(r"\[\[([^\]]+)\]\]")
REF_SELFCLOSE_RE = re.compile(r"<ref[^>]*/>", re.I)
REF_TAG_RE = re.compile(r"</?ref[^>]*>", re.I)
REF_ANY_RE = re.compile(r"<[^>]*ref[^>]*>", re.I)
NOWIKI_TAG_RE = re.compile(r"</?nowiki[^>]*>", re.I)
INS_TAG_RE = re.compile(r"</?ins\b[^>]*>", re.I)
BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.I)
DIV_RE = re.compile(r"<\s*/?\s*div\b[^>]*>", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
PARENT_ADMIN_RE_COMMA = re.compile(r"，是下辖的一个")
PARENT_ADMIN_RE_PLAIN = re.compile(r"是下辖的一个")
EMPTY_SLOT_QUOTE_RE = re.compile(r'[（(]\s*[\u2018\u2019\u201c\u201d\x27\x22]{1,2}\s*[）)]')


def foreign_paragraph_metrics(paragraphs):
    CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    LATIN_RE = re.compile(r"[A-Za-z]")
    count = 0
    characters = 0
    indexes = []
    for index, paragraph in enumerate(paragraphs):
        compact = "".join(ch for ch in paragraph if not ch.isspace())
        if len(compact) < 120:
            continue
        chinese = len(CHINESE_RE.findall(compact))
        latin = len(LATIN_RE.findall(compact))
        chinese_ratio = chinese / len(compact)
        latin_ratio = latin / len(compact)
        if chinese_ratio < 0.18 and latin_ratio >= 0.55:
            count += 1
            characters += len(compact)
            indexes.append(index + 1)
    return count, characters, indexes


def count_comma_marks(s: str) -> int:
    s = DIGIT_COMMA.sub("", s)
    return s.count("，") + s.count("、") + s.count(",")


def sha256_hex(text: str) -> str:
    import hashlib
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
    text = REF_ANY_RE.sub("", text)
    text = re.sub(r"<ref\b[^>\n]{0,300}", "", text, flags=re.I)
    text = re.sub(r"</ref\b", "", text, flags=re.I)
    text = NOWIKI_TAG_RE.sub("", text)
    text = INS_TAG_RE.sub("", text)
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
    text = EMPTY_SLOT_RE.sub("", text)
    text = EMPTY_SLOT_QUOTE_RE.sub("", text)
    return text


def clean_url_lines(paragraphs):
    out = []
    for p in paragraphs:
        p = URL_RE.sub("", p)
        if p.strip():
            out.append(p)
    return out


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
    """Remove hole sentences / trim hole clauses conservatively (see preview rules)."""
    matches = []
    for rx in (SEVERE_HOLE_RE, EXTENDED_NUMERIC_HOLE_RE):
        for m in rx.finditer(paragraph):
            matches.append(m)
    if not matches:
        return paragraph
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
            continue
        spans = sorted((m.start() - start, m.end() - start) for m in sm)
        pieces = []
        pos = 0
        for s, e in spans:
            pieces.append(sentence[pos:s])
            pos = e
        pieces.append(sentence[pos:])
        repaired = "".join(pieces)
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
    PROSE_MARKERS = re.compile(
        r"(?:的|了|着|在|于|中|从|到|向|被|把|也|都|更|又|曾|将|已|并|或|而|以|由|和|与|及|其|等|之|为|是|个|条|名|位|有|因|据|后|前|时|年)"
    )
    if PROSE_MARKERS.search(last):
        return False
    return True


def apply_actions(paragraphs, actions, metrics):
    """Apply the doc's repair actions in a fixed conservative order."""
    notes = []

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
            paragraphs = [p for i, p in enumerate(paragraphs, start=1) if i not in idxs]

    if "remove_url_or_reference_lines" in actions:
        paragraphs = clean_url_lines(paragraphs)
        paragraphs = [clean_empty_slots(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if p.strip()]

    if "trim_reference_section" in actions:
        paragraphs = clean_reference_section(paragraphs)

    if "normalize_missing_parent_admin_definition" in actions:
        paragraphs = [clean_parent_admin(p) for p in paragraphs]

    if "remove_or_trim_defective_clause" in actions:
        paragraphs = [trim_defective_clause(p) for p in paragraphs]
        paragraphs = [clean_empty_slots(p) for p in paragraphs]
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

    # final mechanical cascade cleanup: clause-level removals can leave empty slots
    # (e.g. removing "英语：" from "（英语：，）" leaves "（，）").
    paragraphs = [clean_empty_slots(p) for p in paragraphs]
    paragraphs = [p for p in paragraphs if p.strip()]

    return "\n".join(paragraphs), notes
