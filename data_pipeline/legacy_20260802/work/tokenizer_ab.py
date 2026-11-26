#!/usr/bin/env python3
"""Tokenizer A/B audit: balanced-sample candidate training vs existing zh_bpe_32k."""
import gzip
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sentencepiece as spm

TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / f"outputs/cross_source/tokenizer_ab_{TS}"
OUT.mkdir(parents=True, exist_ok=False)

MODEL = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model"
VOCAB = 32000

SOURCES = {
    "news2016zh": ("inputs/cleaned/news2016zh", ["news_tier_a_permanent.jsonl"], 20000),
    "zhwikisource_classics": ("inputs/cleaned/zhwikisource_classics", ["accepted_final.jsonl"], 20000),
    "wiki2019zh": ("outputs/wiki2019zh", ["wiki2019zh_freeze_v2_20260802_004810/tier_a_frozen.jsonl", "wiki2019zh_freeze_v2_20260802_004810/tier_b_frozen.jsonl"], 100000),
    "webText2019zh": ("outputs/webText2019zh", ["webText2019zh_freeze_v1_20260802_062134/tier_a_frozen.jsonl", "webText2019zh_freeze_v1_20260802_062134/tier_b_frozen.jsonl"], 100000),
    "comment2019zh": ("outputs/comment2019zh", ["comment2019zh_freeze_v1_20260802_070511/tier_a_frozen.jsonl", "comment2019zh_freeze_v1_20260802_070511/tier_b_frozen.jsonl"], 100000),
}

# cross-source keep set
keep = set()
with gzip.open(WS / "outputs/cross_source/cross_source_dedup_v1_20260802_082519/cross_source_keep_list.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        keep.add(json.loads(line)["document_id"])
print("keep ids:", len(keep))


def uid_for(source, r):
    if source == "news2016zh":
        return f"{source}::{r['source_file']}:{r['block_index']}"
    if source == "zhwikisource_classics":
        return f"{source}::{r['work']}/{r['title']}"
    return f"{source}::{r['document_id']}"


# balanced sample
sample_texts = []
sample_meta = Counter()
for source, (root, files, limit) in SOURCES.items():
    n_source = 0
    for fname in files:
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if uid_for(source, r) not in keep:
                    continue
                sample_texts.append(r["text"])
                n_source += 1
                if n_source >= limit or len(sample_texts) >= 340000:
                    break
            if n_source >= limit or len(sample_texts) >= 340000:
                break
    sample_meta[source] = n_source
print("sample docs:", len(sample_texts))
per_source = Counter()
for source, (root, files, limit) in SOURCES.items():
    n = 0
    for fname in files:
        with (WS / root / fname).open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if uid_for(source, r) in keep:
                    n += 1
    per_source[source] = n
    print("source docs:", source, n)

SAMPLE_FILE = OUT / "balanced_sample.txt"
with SAMPLE_FILE.open("w", encoding="utf-8") as f:
    for t in sample_texts:
        f.write(t.replace("\n", " ") + "\n")

# train candidate tokenizer
CAND_PREFIX = str(OUT / "candidate_bpe_32k")
spm.SentencePieceTrainer.train(
    input=str(SAMPLE_FILE),
    model_prefix=CAND_PREFIX,
    vocab_size=VOCAB,
    model_type="bpe",
    character_coverage=0.9995,
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    num_threads=8,
)

sp_cur = spm.SentencePieceProcessor(model_file=MODEL)
sp_cand = spm.SentencePieceProcessor(model_file=CAND_PREFIX + ".model")


def stats(sp, texts, label):
    tokens = 0
    unk = 0
    chars = 0
    docs = 0
    lens = []
    chunk = 4000
    for i in range(0, len(texts), chunk):
        enc = sp.encode(texts[i:i + chunk], out_type=int)
        for toks in enc:
            docs += 1
            tokens += len(toks)
            unk += sum(1 for t in toks if t == sp.unk_id())
            lens.append(len(toks))
        chars += sum(len(t.replace("\n", "")) for t in texts[i:i + chunk])
    lens_sorted = sorted(lens)

    def pct(p):
        return lens_sorted[min(len(lens_sorted) - 1, int(round(p / 100 * (len(lens_sorted) - 1))))]

    return {
        "label": label,
        "documents": docs,
        "characters": chars,
        "tokens_without_eos": tokens,
        "tokens_with_one_eos_per_document": tokens + docs,
        "unk_tokens": unk,
        "unk_rate": round(unk / tokens, 10) if tokens else 0.0,
        "chars_per_token": round(chars / tokens, 6) if tokens else None,
        "mean_tokens_per_doc": round(sum(lens) / len(lens), 4),
        "p50_tokens_per_doc": pct(50),
        "p95_tokens_per_doc": pct(95),
        "max_tokens_per_doc": max(lens),
    }


cur_stats = stats(sp_cur, sample_texts, "current_zh_bpe_32k")
cand_stats = stats(sp_cand, sample_texts, "candidate_bpe_32k")

report = {
    "stage": "tokenizer_ab",
    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "existing_tokenizer": MODEL,
    "candidate_tokenizer": CAND_PREFIX + ".model",
    "vocab_size": VOCAB,
    "balanced_sample": {
        "documents": len(sample_texts),
        "file": str(SAMPLE_FILE),
        "note": "news全量(10,881)+wikisource全量(1,276)+wiki/webText/comment各最多100,000（跨来源去重后保留集）",
    },
    "existing": cur_stats,
    "candidate": cand_stats,
    "delta": {
        "tokens_pct": round((cand_stats["tokens_without_eos"] - cur_stats["tokens_without_eos"]) / cur_stats["tokens_without_eos"] * 100, 3),
        "unk_pct": round((cand_stats["unk_tokens"] - cur_stats["unk_tokens"]) / max(1, cur_stats["unk_tokens"]) * 100, 3),
        "chars_per_token_pct": round((cand_stats["chars_per_token"] - cur_stats["chars_per_token"]) / cur_stats["chars_per_token"] * 100, 3),
    },
    "note": "A/B基于平衡样本；现有tokenizer在全部冻结语料上的精确统计见各源token_audit_summary。",
}
(OUT / "tokenizer_ab_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

md = f"""# Tokenizer A/B 审计报告

生成时间: {report['created_at']}

## 样本

- 平衡样本: {len(sample_texts):,} 篇（跨来源去重后的保留集）
- 组成: news2016zh 全量 10,881 / zhwikisource_classics 全量 1,276 / wiki2019zh 100,000 / webText2019zh 100,000 / comment2019zh 100,000（各源上限）

## 对比结果（平衡样本，vocab=32000）

| 指标 | 现有 zh_bpe_32k | 候选 bpe_32k | 变化 |
|---|---|---|---|
| 文档数 | {cur_stats['documents']:,} | {cand_stats['documents']:,} | — |
| 字符数 | {cur_stats['characters']:,} | {cand_stats['characters']:,} | — |
| Token数(无EOS) | {cur_stats['tokens_without_eos']:,} | {cand_stats['tokens_without_eos']:,} | {report['delta']['tokens_pct']}% |
| UNK数 | {cur_stats['unk_tokens']} | {cand_stats['unk_tokens']} | — |
| UNK率 | {cur_stats['unk_rate']} | {cand_stats['unk_rate']} | — |
| 字符/Token | {cur_stats['chars_per_token']} | {cand_stats['chars_per_token']} | {report['delta']['chars_per_token_pct']}% |
| 平均Token/篇 | {cur_stats['mean_tokens_per_doc']} | {cand_stats['mean_tokens_per_doc']} | — |
| p50 Token/篇 | {cur_stats['p50_tokens_per_doc']} | {cand_stats['p50_tokens_per_doc']} | — |
| p95 Token/篇 | {cur_stats['p95_tokens_per_doc']} | {cand_stats['p95_tokens_per_doc']} | — |

## 各源冻结语料的现有Tokenizer精确统计（全量）

| 数据源 | 冻结文档 | 可见字符 | Token(无EOS) | UNK率 |
|---|---|---|---|---|
| news2016zh | 12,420→10,881(跨源去重后) | — | 19,102,647(冻结) | 0 |
| zhwikisource_classics | 1,276 | 4,402,964 | 3,964,180 | 0 |
| wiki2019zh | 936,076 | 400,871,399 | 352,889,183 | 0 |
| webText2019zh | 3,923,468 | 1,012,180,370 | 869,494,212 | 0 |
| comment2019zh | 6,270,567 | 684,771,065 | 605,895,351 | 0 |

注: news2016zh 的 1,539 篇在跨来源近重复去重中被移除；其 token 数需从冻结统计中扣除（news冻结文件含逐篇token_count_permanent，可在最终混合阶段精确核算）。

## 结论与等待决策

候选Tokenizer已训练并完成平衡样本A/B比较。是否采用候选Tokenizer（或调整vocab/覆盖率后重训）需要你决策；在决策前不进行任何重训或训练集构建。
"""
(OUT / "tokenizer_ab_report.md").write_text(md, encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
