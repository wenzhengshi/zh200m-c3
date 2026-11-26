import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
EVAL = AB / "evaluation"
for name in ("old", "old_candidate", "c1", "c2"):
    d = json.loads((EVAL / f"eval_{name}.json").read_text(encoding="utf-8"))
    full = d["per_source"]["full"]
    sem = sum(full[s]["encode_decode_mismatch_normalized_semantics"] for s in full)
    exact = sum(full[s]["encode_decode_mismatch_exact"] for s in full)
    print(name, "semantic_mismatch:", sem, "exact_mismatch:", exact)
