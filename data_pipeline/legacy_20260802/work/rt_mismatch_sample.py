import json
from pathlib import Path

import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
sp = spm.SentencePieceProcessor(model_file=str(AB / "c2_bpe_32k.model"))

FROZEN = [
    (WS / "inputs/cleaned/news2016zh/news_tier_a_permanent.jsonl", "news"),
    (WS / "inputs/cleaned/zhwikisource_classics/accepted_final.jsonl", "classics"),
    (WS / "outputs/wiki2019zh/wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki_a"),
]
found = 0
for path, label in FROZEN:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            ids = sp.encode(text, out_type=int)
            dec = sp.decode(ids)
            norm = sp.Normalize(text)
            if dec != norm:
                found += 1
                print("====", label)
                print("original :", repr(text[:120]))
                print("normalize:", repr(norm[:120]))
                print("decode   :", repr(dec[:120]))
                if found >= 5:
                    break
        if found >= 5:
            break
print("samples:", found)
