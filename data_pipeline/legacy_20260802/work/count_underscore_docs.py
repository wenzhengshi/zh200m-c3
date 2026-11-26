import json
from collections import Counter
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
FROZEN = {
    "news2016zh": (WS / "inputs/cleaned/news2016zh/news_tier_a_permanent.jsonl", lambda r: True),
    "zhwikisource_classics": (WS / "inputs/cleaned/zhwikisource_classics/accepted_final.jsonl", lambda r: True),
    "wiki2019zh": [
        (WS / "outputs/wiki2019zh/wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", lambda r: True),
        (WS / "outputs/wiki2019zh/wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl", lambda r: True),
    ],
    "webText2019zh": [
        (WS / "outputs/webText2019zh/webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", lambda r: True),
        (WS / "outputs/webText2019zh/webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl", lambda r: True),
    ],
    "comment2019zh": [
        (WS / "outputs/comment2019zh/comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", lambda r: True),
        (WS / "outputs/comment2019zh/comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl", lambda r: True),
    ],
}

counts = Counter()
total = Counter()
for src, spec in FROZEN.items():
    files = spec if isinstance(spec, list) else [spec]
    for path, _ in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                total[src] += 1
                if "\u2581" in r["text"]:
                    counts[src] += 1
    print(src, "docs_with_U2581:", counts[src], "total:", total[src], flush=True)
print("TOTAL docs_with_U2581:", sum(counts.values()), "of", sum(total.values()))
