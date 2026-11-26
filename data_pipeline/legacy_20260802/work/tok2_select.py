#!/usr/bin/env python3
"""Compare the four tokenizer evaluations and apply the acceptance/selection rules."""
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
EVAL = AB / "evaluation"

data = {}
for name in ("old", "old_candidate", "c1", "c2"):
    data[name] = json.loads((EVAL / f"eval_{name}.json").read_text(encoding="utf-8"))

SOURCES = ["news2016zh", "zhwikisource_classics", "wiki2019zh", "webText2019zh", "comment2019zh"]

rows = []
for name, d in data.items():
    full = d["per_source"]["full"]
    row = {
        "tokenizer": name,
        "vocab_size": d["actual_vocab_size"],
        "full_unk": sum(full[s]["unk_tokens"] for s in SOURCES),
        "full_docs_unk": sum(full[s]["documents_with_unk"] for s in SOURCES),
        "full_tokens": sum(full[s]["tokens_without_eos"] for s in SOURCES),
        "full_byte_tokens": sum(full[s]["byte_fallback_tokens"] for s in SOURCES),
        "full_byte_ratio": round(
            sum(full[s]["byte_fallback_tokens"] for s in SOURCES)
            / max(1, sum(full[s]["tokens_without_eos"] for s in SOURCES)), 8),
        "rt_mismatch_semantic": sum(full[s]["encode_decode_mismatch_normalized_semantics"] for s in SOURCES),
        "rt_mismatch_exact": sum(full[s]["encode_decode_mismatch_exact"] for s in SOURCES),
        "scope_c_tokens": d["scope_c"]["tokens_without_eos"],
        "per_source_tokens": {s: full[s]["tokens_without_eos"] for s in SOURCES},
        "per_source_unk": {s: full[s]["unk_tokens"] for s in SOURCES},
        "single_char_pieces": d["vocab_single_char_pieces"],
        "special_tokens": d["special_tokens"],
    }
    rows.append(row)

for r in rows:
    print(json.dumps(r, ensure_ascii=False))

# selection rules
old_tokens = {s: next(x for x in rows if x["tokenizer"] == "old")["per_source_tokens"][s] for s in SOURCES}
report = {"comparison": rows}
for name in ("c1", "c2"):
    r = next(x for x in rows if x["tokenizer"] == name)
    report[name] = {
        "unk_zero": r["full_unk"] == 0,
        "vocab_32000": r["vocab_size"] == 32000,
        "byte_ratio": r["full_byte_ratio"],
        "byte_ratio_under_0_5pct": r["full_byte_ratio"] < 0.005,
        "encode_decode_pass": r["rt_mismatch_semantic"] == 0,
        "source_inflation_over_5pct": {
            s: round((r["per_source_tokens"][s] / old_tokens[s] - 1) * 100, 3)
            for s in SOURCES
        },
        "any_source_inflated_over_5pct": any(
            (r["per_source_tokens"][s] / old_tokens[s] - 1) > 0.05 for s in SOURCES
        ),
    }

c1 = report["c1"]
c2 = report["c2"]
if not c1["unk_zero"]:
    if not c2["unk_zero"]:
        report["decision"] = {"winner": None, "reason": "c1 与 c2 全量 UNK>0，均淘汰"}
    else:
        gates = {
            "c2_vocab_32000": c2["vocab_32000"],
            "c2_byte_ratio_under_0_5pct": c2["byte_ratio_under_0_5pct"],
            "c2_no_inflation_over_5pct": not c2["any_source_inflated_over_5pct"],
            "c2_encode_decode_pass": c2["encode_decode_pass"],
        }
        if all(gates.values()):
            report["decision"] = {"winner": "c2", "reason": "c1 淘汰(UNK>0)；c2 全量 UNK=0 且通过全部自动定案门槛"}
        else:
            report["decision"] = {
                "winner": None,
                "reason": "c1 淘汰(UNK>0)；c2 全量 UNK=0，但未通过自动定案门槛（见 gates），按规则保留报告并停止等待决策",
                "gates": gates,
            }
elif not c2["unk_zero"]:
    report["decision"] = {"winner": "c1", "reason": "c2 full UNK>0 -> eliminated"}
else:
    if c2["byte_ratio_under_0_5pct"] and not c2["any_source_inflated_over_5pct"] and c2["encode_decode_pass"]:
        c1_tok = next(x for x in rows if x["tokenizer"] == "c1")["full_tokens"]
        c2_tok = next(x for x in rows if x["tokenizer"] == "c2")["full_tokens"]
        winner = "c2" if c2_tok <= c1_tok else "c1"
        report["decision"] = {"winner": winner,
                              "reason": "both UNK=0; byte<0.5%; no >5% inflation; fewer full tokens wins"}
    else:
        report["decision"] = {
            "winner": None,
            "reason": "c2 fails auto-finalize gate (byte ratio or inflation or rt); keep report and stop for user decision",
            "c2_byte_ratio_ok": c2["byte_ratio_under_0_5pct"],
            "c2_inflation_ok": not c2["any_source_inflated_over_5pct"],
            "c2_rt_ok": c2["encode_decode_pass"],
        }

(AB / "tokenizer_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\nDECISION:", json.dumps(report["decision"], ensure_ascii=False, indent=2))
