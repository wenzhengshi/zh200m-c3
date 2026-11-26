import re
from pathlib import Path

EMPTY_SLOT_RE = re.compile(
    r"(?:"
    r"[（(]\s*[，,；;、/|·\s]{0,12}[）)]"
    r"|《\s*》|〈\s*〉|「\s*」|『\s*』"
    r")"
)

path = Path("/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh/wiki_0_string.txt_new.txt")
paragraphs = []
idx = 0
with path.open("r", encoding="utf-8", errors="replace") as f:
    cur = []
    for raw in f:
        line = raw.rstrip("\r\n")
        if not line.strip():
            if cur:
                idx += 1
                if idx == 136:
                    paragraphs = cur
                    break
                cur = []
        else:
            cur.append(line)

text = "\n".join(paragraphs)
print("RAW TEXT:")
print(text[:400])
print()
print("EMPTY_SLOT_RE matches:", [m.group() for m in EMPTY_SLOT_RE.finditer(text)])
print("paren contexts:", [repr(text[s-8:e+8]) for s, e in [(m.start(), m.end()) for m in re.finditer(r"[（(]", text)][:8]])
