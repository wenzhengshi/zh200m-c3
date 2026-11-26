import gzip
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/cross_source").glob("training_bind_v1_*"))[-1]
print("dir:", D)
tb = (D / "train.bin").stat().st_size // 4
print("train.bin tokens:", tb)
n = 0
tok = 0
with gzip.open(D / "training_sample_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        n += 1
        tok += r["tokens"]
print("manifest rows:", n, "manifest tokens:", tok)
from collections import Counter
src = Counter()
with gzip.open(D / "training_sample_manifest.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        src[r["source"]] += r["tokens"]
print("per-source manifest:", dict(src))
