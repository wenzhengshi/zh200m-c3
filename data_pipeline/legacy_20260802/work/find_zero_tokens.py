import json
from pathlib import Path

import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
FREEZE = sorted((WS / "outputs/wiki2019zh").glob("wiki2019zh_freeze_v1_*"))[-1]
sp = spm.SentencePieceProcessor(model_file="/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model")

zeros = []
for name in ("tier_a_frozen.jsonl", "tier_b_frozen.jsonl"):
    with (FREEZE / name).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = sp.encode(r["text"], out_type=int)
            if len(toks) == 0:
                zeros.append((r["document_id"], len(r["text"]), repr(r["text"][:80])))
            if len(zeros) > 20:
                break
        if len(zeros) > 20:
            break

print("zero-token docs found:", len(zeros))
for z in zeros:
    print(" ", z)
