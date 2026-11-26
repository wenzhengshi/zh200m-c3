#!/usr/bin/env python3
"""Freeze the adopted C2 tokenizer into tokenizer_freeze_v1."""
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
OUT = WS / f"outputs/cross_source/tokenizer_freeze_v1_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

src_model = AB / "c2_bpe_32k.model"
shutil.copy(src_model, OUT / "tokenizer.model")
shutil.copy(AB / "c2_bpe_32k.vocab", OUT / "tokenizer.vocab")
shutil.copy(AB / "samples/tokenizer_train_manifest.jsonl.gz", OUT / "training_sample_manifest.jsonl.gz")
shutil.copy(AB / "samples/tokenizer_eval_manifest.jsonl.gz", OUT / "tokenizer_eval_manifest.jsonl.gz")
shutil.copy(AB / "samples/sample_manifest.json", OUT / "sample_manifest.json")
shutil.copy(AB / "tokenizer_ab_v2_report.md", OUT / "tokenizer_ab_report.md")

sha256 = hashlib.sha256(src_model.read_bytes()).hexdigest()

# evaluation report: aggregate eval jsons + selection
eval_data = {}
for name in ("old", "old_candidate", "c1", "c2"):
    eval_data[name] = json.loads((AB / "evaluation" / f"eval_{name}.json").read_text(encoding="utf-8"))
selection = json.loads((AB / "tokenizer_selection.json").read_text(encoding="utf-8"))
eval_report = {"evaluations": eval_data, "selection": selection,
               "note": "用户决策：接受并采用C2（全量UNK=0、vocab=32000、规范化语义编解码0不一致、各源无>5%膨胀、总Token较旧模型−25.5%）。"}
(OUT / "evaluation_report.json").write_text(json.dumps(eval_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

c2 = eval_data["c2"]
full_corpus_token_counts = {
    "tokenizer": "c2 (final, frozen)",
    "final_tokenizer_sha256": sha256,
    "vocab_size": 32000,
    "per_source_full": {
        s: {
            "documents": c2["per_source"]["full"][s]["documents"],
            "tokens_without_eos": c2["per_source"]["full"][s]["tokens_without_eos"],
            "unk_tokens": c2["per_source"]["full"][s]["unk_tokens"],
            "unk_rate": c2["per_source"]["full"][s]["unk_rate"],
            "byte_fallback_tokens": c2["per_source"]["full"][s]["byte_fallback_tokens"],
            "docs_gt_2048": c2["per_source"]["full"][s]["documents_gt_2048_tokens"],
            "seqs_2048": c2["per_source"]["full"][s]["sequences_after_2048_chunking"],
        }
        for s in c2["per_source"]["full"]
    },
    "scope_c_cross_source_keep": {
        "documents": c2["scope_c"]["documents"],
        "tokens_without_eos": c2["scope_c"]["tokens_without_eos"],
    },
    "note": "B=五源冻结全量；C=跨来源去重后保留集。",
}
(OUT / "full_corpus_token_counts.json").write_text(json.dumps(full_corpus_token_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

config = {
    "final_tokenizer": "c2 (adopted by user decision)",
    "final_tokenizer_sha256": sha256,
    "vocab_size": 32000,
    "context_length": 2048,
    "model_type": "BPE",
    "character_coverage": 0.9995,
    "byte_fallback": True,
    "normalization_rule_name": "identity",
    "add_dummy_prefix": False,
    "remove_extra_whitespaces": False,
    "split_by_unicode_script": True,
    "split_by_whitespace": True,
    "split_by_number": True,
    "split_digits": True,
    "max_sentencepiece_length": 16,
    "unk_id": 0, "unk_piece": "<unk>",
    "bos_id": 1, "bos_piece": "<s>",
    "eos_id": 2, "eos_piece": "</s>",
    "pad_id": 3, "pad_piece": "<pad>",
    "user_defined_symbols": ["<|system|>", "<|user|>", "<|assistant|>", "<|n|>", "<|tab|>"],
    "training_sample": {"docs": 606100, "chars": 215915328, "seed": 20260802,
                        "manifest": "training_sample_manifest.jsonl.gz"},
    "eval_sample": {"docs": 46057, "chars": 18291998, "manifest": "tokenizer_eval_manifest.jsonl.gz"},
    "notes": [
        "random_init模型将使用该Tokenizer；best_model.pt与该Tokenizer无初始化关系。",
        "旧 tokenizer/zh_bpe_32k.model 仅作为对照保留，未被覆盖。",
    ],
}
(OUT / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

(OUT / "run.log").write_text(
    f"tokenizer_freeze_v1 run.log\ncreated_at: {datetime.now(timezone(timedelta(hours=8))).isoformat()}\n"
    f"final_tokenizer=c2 final_tokenizer_sha256={sha256}\n"
    f"user decision: accept C2 (byte ratio 3.72% accepted by user)\n"
    f"source: {AB}\n",
    encoding="utf-8",
)
print("freeze dir:", OUT)
print("sha256:", sha256)
