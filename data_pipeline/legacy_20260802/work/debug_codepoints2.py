path = "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh/wiki_230_string.txt_new.txt"
paragraphs = []
idx = 0
with open(path, encoding="utf-8", errors="replace") as f:
    cur = []
    for raw in f:
        line = raw.rstrip("\r\n")
        if not line.strip():
            if cur:
                idx += 1
                if idx == 600:
                    paragraphs = cur
                    break
                cur = []
        else:
            cur.append(line)

last = paragraphs[-1].strip()
print("len:", len(last))
for i, ch in enumerate(last):
    if ch in ",，、":
        print(f"index={i} cp={ord(ch):04X} prev={repr(last[i-1])} ({ord(last[i-1]):04X}) next={repr(last[i+1])} ({ord(last[i+1]):04X})")
# print all digit-like chars
import unicodedata
digit_like = [ch for ch in last if unicodedata.category(ch) == "Nd"]
print("digit-like chars:", [(ch, hex(ord(ch))) for ch in digit_like])
