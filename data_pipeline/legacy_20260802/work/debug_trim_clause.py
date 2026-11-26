import re

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


def trim_defective_clause(paragraph):
    matches = []
    for rx in (SEVERE_HOLE_RE, EXTENDED_NUMERIC_HOLE_RE):
        for m in rx.finditer(paragraph):
            matches.append(m)
    if not matches:
        return paragraph
    spans = []
    for m in matches:
        s = m.start()
        e = m.end()
        start = paragraph.rfind("。", 0, s)
        start = paragraph.rfind("！", 0, s) if paragraph.rfind("！", 0, s) > start else start
        start = paragraph.rfind("？", 0, s) if paragraph.rfind("？", 0, s) > start else start
        start = -1 if start == -1 else start + 1
        end = paragraph.find("。", e)
        if end == -1:
            end = paragraph.find("！", e)
        if end == -1:
            end = paragraph.find("？", e)
        end = len(paragraph) if end == -1 else end + 1
        spans.append((start, end))
    spans.sort()
    merged = []
    for sp in spans:
        if merged and sp[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], sp[1]))
        else:
            merged.append(sp)
    parts = []
    pos = 0
    for s, e in merged:
        parts.append(paragraph[pos:s])
        pos = e
    parts.append(paragraph[pos:])
    return "".join(parts)


p1 = "点湖村的座标为，而该地的平均海拔高度为666米（即2185英尺）。"
p3 = "该市镇总面积平方公里，年时的人口为人。"
for p in (p1, p3):
    print("INPUT :", repr(p))
    print("OUTPUT:", repr(trim_defective_clause(p)))
    print("matches:", [(m.group(), m.start(), m.end()) for rx in (SEVERE_HOLE_RE, EXTENDED_NUMERIC_HOLE_RE) for m in rx.finditer(p)])
    print()
