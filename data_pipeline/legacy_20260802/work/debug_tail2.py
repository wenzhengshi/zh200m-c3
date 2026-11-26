import gzip
import json
import re

M31 = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_20260801_223312/classification_manifest_v3_1.jsonl.gz"
did = "wiki_230_string.txt_new.txt:0000600"

with gzip.open(M31, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] == did:
            print("v3.1 record tier:", r["tier"], "primary:", r["primary_reason"])
            print("metric definite_incomplete_tail:", r.get("definite_incomplete_tail"))
            print("metric trailing_empty_heading:", r.get("trailing_empty_heading"))
            break

# fetch the document text from the input file
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
print("\nlast paragraph:", repr(last))

DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")


def count_comma_marks(s):
    s = DIGIT_COMMA.sub("", s)
    return s.count("，") + s.count("、") + s.count(",")


CHART_STUB_RE = re.compile(
    r"^(?:人口变化图示|人口變化圖示|人口变化表|人口變化表|历年人口|歷年人口)\s*$"
)
DECLARATIVE_RE = re.compile(
    r"(?:是|为|属于|位于|生存于|成立于|出生于|指|包括|设立于|列出)"
)
TRAILING_FRAGMENT_RE = re.compile(
    r"(?:以及|并且|而且|或者|以及其|的|和|与|及|为|是|在|于|由)$"
)
END_PUNCTUATION = set("。！？；.!?;”’」』）》】")

print("count_comma_marks:", count_comma_marks(last))
print("ends with comma:", last.endswith(("，", ",", "、", "；", ";")), "| colon:", last.endswith(("：", ":")))
print("TRAILING_FRAGMENT search:", bool(TRAILING_FRAGMENT_RE.search(last)))
print("last char in END_PUNCTUATION:", last[-1] in END_PUNCTUATION)
print("CHART_STUB fullmatch:", bool(CHART_STUB_RE.fullmatch(last.strip())))
print("DECLARATIVE search:", bool(DECLARATIVE_RE.search(last)))

# v3 original logic for comparison
print("\nv3 logic (raw ascii comma count):", sum(last.count(m) for m in ("，", ",", "、")))
