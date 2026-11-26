#!/usr/bin/env python3
"""Tokenizer round-2: create ab_v2 dir, record existing config, build train/eval samples."""
import gzip
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
OUT = WS / f"outputs/cross_source/tokenizer_ab_v2_{TS}"
SAMPLES = OUT / "samples"
SAMPLES.mkdir(parents=True, exist_ok=False)

# ---- 1) existing tokenizer config (parsed from zh_bpe_32k.model proto, verified) ----
existing_config = {
    "source_model": "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model",
    "source_model_sha256": "af0c6373494745206027dcebb44985a4c5a80670d81db8d9c87ec4c2aa510f50",
    "parse_method": "minimal protobuf wire-format decoder over serialized_model_proto (no assumptions)",
    "model_type": "BPE",
    "vocab_size": 32000,
    "actual_vocab_size": 32000,
    "character_coverage": 1.0,
    "byte_fallback": True,
    "normalization_rule_name": "identity",
    "add_dummy_prefix": False,
    "remove_extra_whitespaces": False,
    "split_by_unicode_script": True,
    "split_by_whitespace": True,
    "split_by_number": True,
    "split_digits": True,
    "max_sentencepiece_length": 16,
    "max_sentence_length": 16384,
    "shuffle_input_sentence": False,
    "unk_id": 0, "unk_piece": "<unk>",
    "bos_id": 1, "bos_piece": "<s>",
    "eos_id": 2, "eos_piece": "</s>",
    "pad_id": 3, "pad_piece": "<pad>",
    "user_defined_symbols": ["<|system|>", "<|user|>", "<|assistant|>", "<|n|>", "<|tab|>"],
    "trainer_field49_bool": 1,
    "trainer_field49_note": "trainer_spec field 49 = 1（wire解析可见，但无官方字段名映射，训练API未暴露，无法通过参数复制；其余字段均已复制）",
    "note": "该模型以identity归一化、无dummy prefix、无extra whitespace移除训练；byte_fallback=true、coverage=1.0。",
}
(OUT / "existing_tokenizer_config.json").write_text(json.dumps(existing_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ---- 2) sampling plan ----
PLAN = {
    "news2016zh": (9000, 1881),
    "zhwikisource_classics": (1100, 176),
    "wiki2019zh": (280000, 20000),
    "webText2019zh": (280000, 20000),
    "comment2019zh": (36000, 4000),
}
SEED = 20260802

# ---- 3) load cross-source keep list, group ids by source ----
keep_ids = {}
with gzip.open(WS / "outputs/cross_source/cross_source_dedup_v1_20260802_082519/cross_source_keep_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        src = r["source"]
        keep_ids.setdefault(src, []).append(r["document_id"])
print("keep counts:", {k: len(v) for k, v in keep_ids.items()})

# ---- 4) seeded split per source ----
train_ids = {}
eval_ids = {}
for src, (ntrain, neval) in PLAN.items():
    ids = keep_ids[src]
    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    train_ids[src] = set(shuffled[:ntrain])
    eval_ids[src] = set(shuffled[ntrain:ntrain + neval])
    print(f"{src}: train={ntrain} eval={neval}")

# ---- 5) stream frozen files, write train/eval text + manifest ----
FROZEN = {
    "news2016zh": ("inputs/cleaned/news2016zh", ["news_tier_a_permanent.jsonl"], lambda r: f"news2016zh::{r['source_file']}:{r['block_index']}"),
    "zhwikisource_classics": ("inputs/cleaned/zhwikisource_classics", ["accepted_final.jsonl"], lambda r: f"zhwikisource_classics::{r['work']}/{r['title']}"),
    "wiki2019zh": ("outputs/wiki2019zh", ["wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl"], lambda r: f"wiki2019zh::{r['document_id']}"),
    "webText2019zh": ("outputs/webText2019zh", ["webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", "webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl"], lambda r: f"webText2019zh::{r['document_id']}"),
    "comment2019zh": ("outputs/comment2019zh", ["comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", "comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl"], lambda r: f"comment2019zh::{r['document_id']}"),
}

train_f = (SAMPLES / "tokenizer_train.txt").open("w", encoding="utf-8")
eval_f = (SAMPLES / "tokenizer_eval.txt").open("w", encoding="utf-8")
train_man = gzip.open(SAMPLES / "tokenizer_train_manifest.jsonl.gz", "wt", encoding="utf-8")
eval_man = gzip.open(SAMPLES / "tokenizer_eval_manifest.jsonl.gz", "wt", encoding="utf-8")
src_counts = Counter()
src_chars = Counter()
eval_src_counts = Counter()
eval_src_chars = Counter()
written = 0

for src, (root, files, uid_key) in FROZEN.items():
    for fname in files:
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uid = uid_key(r)
                if uid in train_ids.get(src, set()):
                    text = r["text"]
                    chars = len(text) - text.count("\n")
                    line_text = text.replace("\n", " ")
                    train_f.write(line_text + "\n")
                    train_man.write(json.dumps({"document_id": uid, "source": src, "chars": chars}, ensure_ascii=False) + "\n")
                    src_counts[src] += 1
                    src_chars[src] += chars
                    written += 1
                elif uid in eval_ids.get(src, set()):
                    text = r["text"]
                    chars = len(text) - text.count("\n")
                    line_text = text.replace("\n", " ")
                    eval_f.write(line_text + "\n")
                    eval_man.write(json.dumps({"document_id": uid, "source": src, "chars": chars}, ensure_ascii=False) + "\n")
                    eval_src_counts[src] += 1
                    eval_src_chars[src] += chars
    print(src, "done, train:", src_counts[src], "eval:", eval_src_counts[src], flush=True)

train_f.close(); eval_f.close(); train_man.close(); eval_man.close()

manifest = {
    "stage": "tokenizer_round2_samples",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "seed": SEED,
    "basis": "cross_source_dedup_v1_20260802_082519 keep list (11,142,268)",
    "plan_docs": PLAN,
    "train_source_counts": dict(sorted(src_counts.items())),
    "train_source_character_counts": dict(sorted(src_chars.items())),
    "eval_source_counts": dict(sorted(eval_src_counts.items())),
    "eval_source_character_counts": dict(sorted(eval_src_chars.items())),
    "train_total_docs": sum(src_counts.values()),
    "train_total_chars": sum(src_chars.values()),
    "eval_total_docs": sum(eval_src_counts.values()),
    "eval_total_chars": sum(eval_src_chars.values()),
    "comment_train_share_docs": round(src_counts["comment2019zh"] / sum(src_counts.values()) * 100, 3),
    "comment_train_share_chars": round(src_chars["comment2019zh"] / sum(src_chars.values()) * 100, 3),
    "note": "训练与评测样本严格分离（按source内seed=20260802洗牌后前缀切分，ID不重叠）；classics取86%训练/14%评测（全集仅1,276，无法同时100%训练且独立评测，按高覆盖处理并记录）。",
}
(SAMPLES / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("OUT:", OUT)
