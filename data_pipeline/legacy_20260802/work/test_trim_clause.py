import re

EXT = re.compile(
    r"(?:总面积\s*(?:为|是)?\s*(?:平方公里|平方千米)"
    r"|年\s*时\s*的?\s*人口\s*(?:为|是)?\s*人"
    r"|人口\s*(?:为|是)?\s*人(?:[。，；;]|$)"
    r"|坐标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|座标\s*(?:为|是)?\s*(?:[，,。；;]|而|$)"
    r"|海拔高度\s*(?:为|是)?\s*(?:米|公尺))"
)
SEV = re.compile(
    r"(?:广告标语是\s*[。；;]"
    r"|(?:本名|原名|英文名|外文名)\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:编剧|导演|主演|作者)\s*(?:为|是|：|:)\s*(?=[，。；;])"
    r"|(?:并)?由\s*(?:执导|导演|主演)(?=[，。；;]|$)"
    r"|参见主条目\s*[：:]\s*(?:和)?\s*$)"
)


def trim(p):
    matches = []
    for rx in (SEV, EXT):
        for m in rx.finditer(p):
            matches.append(m)
    if not matches:
        return p
    terms = [m.start() for m in re.finditer(r"[。！？；\n]", p)]
    terms = [-1] + terms + [len(p)]
    sents = []
    for i in range(len(terms) - 1):
        s = terms[i] + 1
        e = terms[i + 1] + 1
        if e > len(p):
            e = len(p)
        sents.append((s, e))
    out = []
    for s, e in sents:
        sent = p[s:e]
        sm = [m for m in matches if m.start() >= s and m.end() <= e]
        if not sm:
            out.append(sent)
            continue
        hole = sum(len(m.group()) for m in sm)
        non = len(sent) - hole
        if non < 10:
            continue
        spans = sorted((m.start() - s, m.end() - s) for m in sm)
        pieces = []
        pos = 0
        for a, b in spans:
            pieces.append(sent[pos:a])
            pos = b
        pieces.append(sent[pos:])
        r = "".join(pieces)
        r = re.sub(r"[，,]\s*[为是]\s*(?=[，,。；;])", "，", r)
        r = re.sub(r"[，,]{2,}", "，", r)
        r = re.sub(r"[，,][。！？]", "。", r)
        r = re.sub(r"^(?:而|并且|而且|且|以及|还有|此外)", "", r)
        out.append(r)
    return "".join(out)


tests = [
    ("点湖村的座标为，而该地的平均海拔高度为666米（即2185英尺）。", "clause-trim"),
    ("该市镇总面积平方公里，年时的人口为人。", "sentence-drop"),
    ("该陨坑中心月面坐标为，直径52.66公里，深约2.38公里。", "clause-trim"),
    ('《瞎趴姊妹》（""）是2015年美国喜剧电影，导演为，编剧为。', "clause-trim"),
    ("该地共人口136人，而该地的面积约为210.20平方千米。", "no-hole"),
]
for t, label in tests:
    print(label, "|", repr(t), "->", repr(trim(t)))
