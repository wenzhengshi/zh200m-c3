#!/usr/bin/env python3
"""Consolidate overall pipeline status from verified summaries into outputs/."""
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
OUT = WS / "outputs"

# gather key facts
wiki_freeze = sorted((OUT / "wiki2019zh").glob("wiki2019zh_freeze_v2_*"))[-1]
web_freeze = sorted((OUT / "webText2019zh").glob("webText2019zh_freeze_v1_*"))[-1]
com_freeze = sorted((OUT / "comment2019zh").glob("comment2019zh_freeze_v1_*"))[-1]
cs = sorted((OUT / "cross_source").glob("cross_source_dedup_v1_*"))[-1]
ab = sorted((OUT / "cross_source").glob("tokenizer_ab_*"))[-1]

wiki_sum = json.loads((wiki_freeze / "summary.json").read_text(encoding="utf-8"))
wiki_tok = json.loads(sorted((wiki_freeze).glob("token_audit_*"))[-1].joinpath("token_audit_summary.json").read_text(encoding="utf-8"))
web_sum = json.loads((web_freeze / "summary.json").read_text(encoding="utf-8"))
web_tok = json.loads(sorted(web_freeze.glob("token_audit_*"))[-1].joinpath("token_audit_summary.json").read_text(encoding="utf-8"))
com_sum = json.loads((com_freeze / "summary.json").read_text(encoding="utf-8"))
com_tok = json.loads(sorted(com_freeze.glob("token_audit_*"))[-1].joinpath("token_audit_summary.json").read_text(encoding="utf-8"))
cs_sum = json.loads((cs / "summary.json").read_text(encoding="utf-8"))
ab_sum = json.loads((ab / "tokenizer_ab_report.json").read_text(encoding="utf-8"))

web_cand_sum = json.loads(sorted((OUT / "webText2019zh").glob("curated_candidate_v1_*"))[-1].joinpath("summary.json").read_text(encoding="utf-8"))
com_cand_sum = json.loads(sorted((OUT / "comment2019zh").glob("curated_candidate_v1_*"))[-1].joinpath("summary.json").read_text(encoding="utf-8"))
wiki_cand_sum = json.loads(sorted((OUT / "wiki2019zh").glob("curated_candidate_v4_*"))[-1].joinpath("summary.json").read_text(encoding="utf-8"))

overall = {
    "generated_at": "2026-08-02T09:10:00+08:00",
    "sources": {
        "news2016zh": {"frozen": True, "docs": 12420, "tokens": 19102647, "unk_rate": 0.0,
                       "note": "已有冻结v3.2，跨来源去重后保留10,881篇/16,620,802 tokens"},
        "zhwikisource_classics": {"frozen": True, "docs": 1276, "tokens": 3964180, "unk_rate": 0.0,
                                  "note": "v6已执行，finalize_report+token_audit+web_residue_audit均验证"},
        "wiki2019zh": {"frozen": True, "docs": wiki_sum["total_documents"], "chars": sum(wiki_sum["tier_character_counts"].values()),
                       "tokens": wiki_tok["tokens_without_eos"], "unk_rate": wiki_tok["unk_rate"], "dir": str(wiki_freeze)},
        "webText2019zh": {"frozen": True, "docs": web_sum["total_documents"], "chars": sum(web_sum["tier_character_counts"].values()),
                          "tokens": web_tok["tokens_without_eos"], "unk_rate": web_tok["unk_rate"], "dir": str(web_freeze)},
        "comment2019zh": {"frozen": True, "docs": com_sum["total_documents"], "chars": sum(com_sum["tier_character_counts"].values()),
                          "tokens": com_tok["tokens_without_eos"], "unk_rate": com_tok["unk_rate"], "dir": str(com_freeze)},
    },
    "cross_source_dedup": {
        "dir": str(cs),
        "in_scope": cs_sum["documents_in_scope"],
        "removed": cs_sum["total_removed"],
        "kept": cs_sum["documents_kept"],
        "reasons": cs_sum["removed_reason_counts"],
        "news_repost_clusters": cs_sum["news_repost_clusters"],
        "final_total_tokens_existing_tokenizer": ab_sum["existing_tokenizer_full_corpus_post_cross_source_dedup"]["tokens_without_eos"],
    },
    "tokenizer_ab": {
        "dir": str(ab),
        "sample_docs": ab_sum["balanced_sample"]["documents"],
        "existing": ab_sum["existing"],
        "candidate": ab_sum["candidate"],
        "delta": ab_sum["delta"],
        "status": "AWAITING_USER_DECISION",
    },
    "review_registers": {
        "wiki2019zh": {
            "review_isolated": wiki_cand_sum["tier_document_counts"]["REVIEW"],
            "residue_quarantine": wiki_cand_sum.get("residue_quarantine", {}).get("documents"),
            "tail_kept_for_review": len([l for l in (sorted((OUT / "wiki2019zh").glob("cleaning_preview_v5_*"))[-1] / "tail_kept_for_review.jsonl").open(encoding="utf-8")]),
            "drop_archived": wiki_cand_sum["tier_document_counts"]["DROP"],
        },
        "webText2019zh": {
            "review_isolated": web_cand_sum["tier_document_counts"]["REVIEW"],
            "drop_archived": web_cand_sum["tier_document_counts"]["DROP"],
            "empty_after_cleaning": len(web_cand_sum.get("empty_after_cleaning", [])),
        },
        "comment2019zh": {
            "review_isolated": com_cand_sum["tier_document_counts"]["REVIEW"],
            "drop_archived": com_cand_sum["tier_document_counts"]["DROP"],
            "empty_after_cleaning": len(com_cand_sum.get("empty_after_cleaning", [])),
        },
    },
    "pending_decisions": ["最终Tokenizer选择（现有zh_bpe_32k / 候选bpe_32k / 调整参数重训）", "最终混合比例设计（comment严格限比）", "训练集构建（决策后执行，不启动训练）"],
}

(OUT / "overall_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

md = """# 0.2B 中文LM 预训练语料清洗总体状态

生成时间: 2026-08-02 09:10 (+08:00)

## 1. 冻结状态（现有 zh_bpe_32k 精确统计，UNK 均=0）

| 数据源 | 冻结文档 | 可见字符 | Token(无EOS) | 冻结目录 |
|---|---|---|---|---|
| news2016zh | 12,420（跨源后10,881） | — | 19,102,647（跨源后16,620,802） | 已有v3.2（只读） |
| zhwikisource_classics | 1,276 | 4,402,964 | 3,964,180 | v6（只读） |
| wiki2019zh | {docs:,} | {chars:,} | {tokens:,} | {wdir} |
| webText2019zh | {wdocs:,} | {wchars:,} | {wtokens:,} | {wdir2} |
| comment2019zh | {cdocs:,} | {cchars:,} | {ctokens:,} | {cdir} |

## 2. 跨来源去重

- 输入 11,143,807 → 删除 {csremoved:,}（全部为 news2016zh 内部近重复转载，精确/标准化跨源重复为0）→ 保留 **{cskept:,} 篇**
- 跨源去重后全量 Token（现有tokenizer）：**{tottok:,}**

## 3. Tokenizer A/B（等待决策）

平衡样本 {sampdocs:,} 篇，vocab=32,000：

| 指标 | 现有 zh_bpe_32k | 候选 bpe_32k |
|---|---|---|
| Token（无EOS） | {etok:,} | {ctok:,}（{dtok}%） |
| UNK率 | {eunk} | {cunk}（{dunk}%） |
| 字符/Token | {echr} | {cchr}（{dchr}%） |

候选压缩率更高但出现 UNK；**待你决策后**才进入最终混合比例设计与训练集构建。

## 4. 人工复核登记（不阻塞冻结；REVIEW 已全部隔离）

- wiki2019zh：REVIEW {wreview:,} / DROP留档 {wdrop:,} / 残留复核 {wq} / 尾部缺句号保留 {wtail:,}
- webText2019zh：REVIEW {wbrev:,} / DROP留档 {wbdrop:,} / 清洗后为空 {wbempty}
- comment2019zh：REVIEW {crev:,} / DROP留档 {cdrop:,} / 清洗后为空 {cempty}

## 5. 待决策/待办

1. 最终 Tokenizer 选择（现有 / 候选 / 重训参数调整）
2. 最终混合比例设计（comment 严格限比）
3. 训练集构建与装订（决策后执行；**不启动训练**）

## 6. 完整性说明

- 全部阶段目录已生成 SHA256SUMS.txt 并通过自校验。
- webText/cross_source 去重的 保留↔删除 精确相似度已补充（addendum 文件）。
- 未修改任何原始/冻结数据；best_model.pt 未触碰；未启动训练。
""".format(
    docs=wiki_sum["total_documents"], chars=sum(wiki_sum["tier_character_counts"].values()), tokens=wiki_tok["tokens_without_eos"],
    wdir=wiki_freeze.name,
    wdocs=web_sum["total_documents"], wchars=sum(web_sum["tier_character_counts"].values()), wtokens=web_tok["tokens_without_eos"],
    wdir2=web_freeze.name,
    cdocs=com_sum["total_documents"], cchars=sum(com_sum["tier_character_counts"].values()), ctokens=com_tok["tokens_without_eos"],
    cdir=com_freeze.name,
    csremoved=cs_sum["total_removed"], cskept=cs_sum["documents_kept"],
    tottok=ab_sum["existing_tokenizer_full_corpus_post_cross_source_dedup"]["tokens_without_eos"],
    sampdocs=ab_sum["balanced_sample"]["documents"],
    etok=ab_sum["existing"]["tokens_without_eos"], ctok=ab_sum["candidate"]["tokens_without_eos"], dtok=ab_sum["delta"]["tokens_pct"],
    eunk=ab_sum["existing"]["unk_rate"], cunk=ab_sum["candidate"]["unk_rate"], dunk=round(ab_sum["candidate"]["unk_rate"] * 100, 4),
    echr=ab_sum["existing"]["chars_per_token"], cchr=ab_sum["candidate"]["chars_per_token"], dchr=ab_sum["delta"]["chars_per_token_pct"],
    wreview=wiki_cand_sum["tier_document_counts"]["REVIEW"], wdrop=wiki_cand_sum["tier_document_counts"]["DROP"],
    wq=wiki_cand_sum.get("residue_quarantine", {}).get("documents", "n/a"),
    wtail=len([l for l in (sorted((OUT / "wiki2019zh").glob("cleaning_preview_v5_*"))[-1] / "tail_kept_for_review.jsonl").open(encoding="utf-8")]),
    wbrev=web_cand_sum["tier_document_counts"]["REVIEW"], wbdrop=web_cand_sum["tier_document_counts"]["DROP"],
    wbempty=len(web_cand_sum.get("empty_after_cleaning", [])),
    crev=com_cand_sum["tier_document_counts"]["REVIEW"], cdrop=com_cand_sum["tier_document_counts"]["DROP"],
    cempty=len(com_cand_sum.get("empty_after_cleaning", [])),
)
(OUT / "OVERALL_STATUS.md").write_text(md, encoding="utf-8")
print("written:", OUT / "OVERALL_STATUS.md")
