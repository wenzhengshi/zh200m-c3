#!/usr/bin/env python3
"""Assemble the read-only takeover inventory for wiki2019zh (and workspace state)."""
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta

WS = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace"
OUT = os.path.join(
    WS,
    "outputs/wiki2019zh/takeover_inventory_v1_20260801_220145",
)
os.makedirs(OUT, exist_ok=False)

now = datetime(2026, 8, 1, 22, 1, 45, tzinfo=timezone(timedelta(hours=8)))

summary = {
    "inventory": {
        "version": "takeover_inventory_v1",
        "created_at": now.isoformat(),
        "workspace": WS,
        "read_only": True,
        "note": "只读接管盘点。未修改任何输入、冻结语料、审计目录或模型权重。",
    },
    "symlink_verification": {
        "total": 16,
        "valid": 16,
        "broken": 0,
        "detail": "SOURCE_PATHS.tsv 中全部16条软链接目标均存在（已验证）。",
    },
    "model_and_tokenizer": {
        "tokenizer": {
            "path": WS + "/reference/zh_bpe_32k.model",
            "sha256": "af0c6373494745206027dcebb44985a4c5a80670d81db8d9c87ec4c2aa510f50",
            "expected_sha256": "af0c6373494745206027dcebb44985a4c5a80670d81db8d9c87ec4c2aa510f50",
            "match": True,
        },
        "best_model_pt": {
            "path": "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/best_model.pt",
            "sha256": "716c61175c9903c23c1c8033485de3123014fe6c6edd06da38d4ec12a22a7e41",
            "expected_sha256": "716c61175c9903c23c1c8033485de3123014fe6c6edd06da38d4ec12a22a7e41",
            "match": True,
            "size_bytes": 784683963,
            "touched": False,
        },
    },
    "canonical_archives": {
        "wiki2019zh_corpus.zip": {
            "path": WS + "/inputs/archives/wiki2019zh_corpus.zip",
            "sha256": "b42e010ffd111de1a010e43db043122ce55ffdd2f93ab4b7712d96b7d2043070",
            "expected_sha256": "b42e010ffd111de1a010e43db043122ce55ffdd2f93ab4b7712d96b7d2043070",
            "match": True,
            "zip_members": 255,
            "extracted_files": 255,
            "zip_vs_extracted": "fully_identical (full_audit_v1 input_comparison_summary.json, 已验证)",
        },
        "webText2019zh_corpus.zip": {
            "path": WS + "/inputs/archives/webText2019zh_corpus.zip",
            "sha256": "a9bb7549ad1d537e09f532eb4930dd7b105968f71bbb90ea24f3f884cdeb4141",
            "expected_sha256": None,
            "match": "no_reference_provided",
            "note": "任务未给出该ZIP的已知SHA256；工作区内未找到参考值。",
        },
        "comment2019zh_corpus.zip": {
            "path": WS + "/inputs/archives/comment2019zh_corpus.zip",
            "sha256": "5252766c2947c5b7edd054d44edc9e55650efb8ac1095a4664e6e77bf71f8643",
            "expected_sha256": None,
            "match": "no_reference_provided",
            "note": "任务未给出该ZIP的已知SHA256；工作区内未找到参考值。",
        },
    },
    "cleaned_corpora": {
        "news2016zh": {
            "resolved_path": "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/news2016zh_v22/news_freeze_v3_2",
            "sha256sums_check": "ALL_OK (6/6, 已验证)",
            "tier_a_documents": 12420,
            "final_quarantine_documents": 790,
            "exact_body_duplicates_removed": 704,
            "exact_body_duplicate_components": 591,
            "remaining_technical_residue_in_permanent_tier_a": 0,
            "tokenizer_audit": {
                "permanent_tokens": 19102647,
                "permanent_unk_tokens": 0,
                "permanent_unk_rate": 0.0,
            },
            "freeze_notice_present": True,
            "char_count_in_freeze_summary": None,
            "note": "冻结摘要未记录总字符数；按盘点规则不重新扫描语料，标记为未记录。",
        },
        "zhwikisource_classics": {
            "resolved_path": "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/zhwikisource_classics_v6",
            "finalize_report_present": True,
            "accepted_documents": 1276,
            "accepted_visible_chars": 4402964,
            "accepted_sha256": "b7099b01b420e00aec04dc98d59638ee7524f6a00c77b81a09e7010deda62ffc",
            "finalize_report_declared_sha256": "b7099b01b420e00aec04dc98d59638ee7524f6a00c77b81a09e7010deda62ffc",
            "accepted_hash_match": True,
            "quarantined_documents": 49,
            "quarantined_sha256": "33449e791a72c126e892179b23f613f9127e767c9a4fae3a29fe4396527ea6b8",
            "finalize_report_declared_quarantined_sha256": "33449e791a72c126e892179b23f613f9127e767c9a4fae3a29fe4396527ea6b8",
            "quarantined_hash_match": True,
            "trimmed_documents": 12,
            "manual_review_documents": 44,
            "token_audit_present": True,
            "token_audit": {
                "documents": 1276,
                "visible_chars": 4402964,
                "tokens_without_eos": 3964180,
                "tokens_with_one_eos_per_document": 3965456,
                "unk_tokens": 0,
                "unk_rate": 0.0,
                "hard_defect_documents": 0,
                "non_sentence_ending_documents": 30,
                "note": "hard_defects=0 已满足；30个非句尾文档为诗行/标题性质，属可人工解释项（audit summary 已说明）。",
            },
            "web_residue_audit_present": True,
            "web_residue_audit": {
                "documents_with_any_issue": 3,
                "issue_counts": {"annotation_heading": 1, "suspected_dangling_poem_title": 2},
                "hard_residue": 0,
                "note": "3个issue均为启发式候选（审计脚本明确注明不可直接批量删除），非硬残留。",
            },
            "sha256_completeness": "PARTIAL",
            "sha256_completeness_note": "v6目录内无SHA256SUMS.txt；finalize_report.json仅含2个输出文件的SHA256（均已复算匹配）。建议后续补全全目录SHA256SUMS.txt。",
            "freeze_notice_present": False,
            "freeze_notice_note": "v6目录无PERMANENT_FREEZE_NOTICE.txt；是否已正式冻结需人工确认（文件本身只读使用）。",
            "v6_executed": True,
            "v6_executed_evidence": "accepted_final.jsonl/finalize_report.json/quarantined.jsonl/manual_review_decisions.tsv/trim_log.jsonl/token_audit/web_residue_audit 均存在且行数与摘要一致（已验证）。",
        },
    },
    "wiki2019zh_audits": {
        "full_audit_v1": {
            "path": WS + "/audits/wiki2019zh_full_audit_v1",
            "sha256sums_check": "ALL_OK (25/25, 已验证)",
            "file_count": 255,
            "total_documents": 1018692,
            "total_physical_lines": 10666086,
            "total_blank_lines": 1018957,
            "total_paragraphs": 9647129,
            "total_characters": 427473417,
            "total_utf8_replacement_characters": 27,
            "total_control_characters": 5264,
        },
        "targeted_error_audit_v1": {
            "path": WS + "/audits/wiki2019zh_targeted_error_audit_v1",
            "sha256sums_check": "ALL_OK (4/4, 已验证)",
            "total_documents_verified": 1018692,
            "categories_sampled": 9,
            "sample_per_category": 30,
        },
        "tier_dryrun_v1": {
            "path": WS + "/audits/wiki2019zh_tier_dryrun_v1",
            "sha256sums_check": "ALL_OK (32/32, 已验证)",
            "manifest_lines": 1018692,
            "total_documents": 1018692,
            "tier_counts": {"A": 572058, "B": 347570, "DROP": 87119, "REVIEW": 11945},
            "tier_character_counts": {"A": 265011928, "B": 128983165, "DROP": 7616399, "REVIEW": 25861925},
            "doc_sum_matches_total": True,
            "char_sum": 427473417,
            "char_sum_matches_full_audit": True,
        },
        "tier_dryrun_v2": {
            "path": WS + "/audits/wiki2019zh_tier_dryrun_v2",
            "sha256sums_check": "ALL_OK (50/50, 已验证)",
            "manifest_lines": 1018692,
            "total_documents": 1018692,
            "tier_counts": {"A": 515829, "B": 382852, "DROP": 89374, "REVIEW": 30637},
            "tier_character_counts": {"A": 230381520, "B": 152389771, "DROP": 4156051, "REVIEW": 40546075},
            "doc_sum_matches_total": True,
            "char_sum": 427473417,
            "char_sum_matches_full_audit": True,
            "transition_sum": 1018692,
            "transition_sum_matches_total": True,
            "primary_reason_sum": 1018692,
            "primary_reason_sum_matches_total": True,
            "compact_transition_review_sha256_check": "MATCH (e9edff46ffa83f4e6044758532ef7522037e1c709cd16ddfb2f4be655e815418)",
            "v1_to_v2_key_transitions": {
                "A->DROP": 4957,
                "A->REVIEW": 12310,
                "B->DROP": 7109,
                "B->REVIEW": 5791,
                "DROP->A": 53,
                "DROP->B": 6520,
                "REVIEW->A": 350,
                "REVIEW->B": 2297,
            },
        },
    },
    "v3_status": {
        "executed": False,
        "audit_dir_exists": "audits/wiki2019zh_tier_dryrun_v3" + " -> 不存在（已验证）",
        "script_exists": "scripts/wiki2019zh_tier_dryrun_v3_fixed" + " -> 不存在（已验证）",
        "whole_home_search": "find /home/xmkjsoft -iname '*tier_dryrun_v3*' / '*dryrun_v3*' 均无结果（已验证）",
        "evidence": [
            "outputs/wiki2019zh 目录为空，不存在任何v3产物",
            "audits/ 下仅4个软链接：full_audit_v1 / targeted_error_audit_v1 / tier_dryrun_v1 / tier_dryrun_v2",
            "scripts/ 下仅2个软链接：cpt_modern_v1_all / wiki2019zh_tier_dryrun_v2_fixed",
            "logs/build_workspace.log 明确记录：未找到 wiki2019zh tier dry-run v3",
        ],
        "conclusion": "v3尚未执行，且不存在v3脚本；不能、也无从重跑v3。下一步必须先编写v3规则脚本（修复v2已知系统误判），在新目录中干跑。",
    },
    "known_v2_systematic_error_requirements": [
        "1. 将正常「主演」「导演」「作者」相关句子误判为语义空穴 —— v2脚本 SEVERE_HOLE_RE 含 (?:编剧|导演|主演|作者) 模式（已确认代码存在）。",
        "2. 因单一坐标/面积/人口字段缺失删除整篇有价值地理词条 —— v2脚本 EXTENDED_NUMERIC_HOLE_RE 含 总面积/人口/坐标/海拔 模式（已确认代码存在）。",
        "3. 正文完整但尾部仅「下辖以下地区：」时误删整篇 —— v2脚本 SEVERE_HOLE_RE 含 下辖以下地区 模式（已确认代码存在）。",
        "4. 将「某街道，是下辖的一个乡级行政区单位。」错误放入严格干净A —— v2脚本 MISSING_SUBJECT_RE/COMPLETE_SHORT_RE 相关（已确认代码存在）。",
        "5. 仅因末尾缺句号大量命中「尾部不完整」 —— v2 primary reason possible_incomplete_tail=22658, overlapping=57341；抽样样本（永康工业区车站/七里岗乡/黑嘴树鸭）证实整篇完整仅缺末句号被判 possible_incomplete_tail（已人工复核样例）。",
    ],
    "pending_inputs": {
        "wiki2019zh": {"resolved_path": "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh", "file_count": 255, "document_count_verified": 1018692},
        "webText2019zh": {"resolved_path": "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/webText2019zh", "file_count": 896, "document_count": "未盘点（任务未要求；后续正式处理前统计）"},
        "comment2019zh": {"resolved_path": "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/comment2019zh", "file_count": 687, "document_count": "未盘点（任务未要求；后续正式处理前统计）"},
    },
    "outputs_state": {
        "outputs/wiki2019zh": "empty",
        "outputs/webText2019zh": "empty",
        "outputs/comment2019zh": "empty",
        "outputs/cross_source": "empty",
        "note": "本盘点目录为 outputs/wiki2019zh 下首个目录。",
    },
    "verification_classification": {
        "verified": [
            "16条软链接有效性",
            "tokenizer SHA256 与已知值一致",
            "best_model.pt SHA256 与已知值一致（未触碰）",
            "wiki2019zh canonical ZIP SHA256 与已知值一致（b42e010f...）",
            "news2016zh 冻结目录 SHA256SUMS 6/6 通过，行数与摘要一致（12420/790/704）",
            "zhwikisource v6 accepted/quarantined SHA256 与 finalize_report 声明一致，行数与摘要一致（1276/49/12/44）",
            "wiki2019zh 4个既有审计 SHA256SUMS 全部通过（25/25、4/4、32/32、50/50）",
            "v1/v2 manifest 均为 1,018,692 行，Tier/转移/主因计数全部对账一致",
            "compact_transition_review_v2.txt SHA256 匹配",
            "v3 未执行且无脚本（多重证据）",
        ],
        "unverified": [
            "webText2019zh / comment2019zh canonical ZIP 无参考SHA256（仅记录实测值）",
            "zhwikisource v6 无全目录SHA256SUMS.txt（部分覆盖：finalize_report 2个哈希已匹配）",
            "news2016zh 冻结字符数未记录（摘要无此字段，未重新扫描）",
            "zhwikisource v6 是否正式「冻结」无冻结通知（需人工确认）",
            "webText2019zh / comment2019zh 文档数（后续处理阶段再精确统计）",
        ],
        "inferred": [
            "zhwikisource v6 token_audit / web_residue_audit 生成时间晚于 accepted_final.jsonl，推断是在v6完成后运行（目录时间戳 Jul 31 21:35 vs 21:36/21:46-48）",
        ],
    },
}

with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------- INPUT_INVENTORY.tsv ----------------
rows = [
    ["entry", "type", "resolved_target", "sha256", "status"],
    [
        "audits/wiki2019zh_full_audit_v1",
        "symlink->audit",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_full_audit_v1_20260801_192933",
        "SHA256SUMS 25/25 OK",
        "verified",
    ],
    [
        "audits/wiki2019zh_targeted_error_audit_v1",
        "symlink->audit",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_targeted_error_audit_v1_20260801_201355",
        "SHA256SUMS 4/4 OK",
        "verified",
    ],
    [
        "audits/wiki2019zh_tier_dryrun_v1",
        "symlink->audit",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v1_20260801_195347",
        "SHA256SUMS 32/32 OK; manifest 1,018,692 行",
        "verified",
    ],
    [
        "audits/wiki2019zh_tier_dryrun_v2",
        "symlink->audit",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v2_20260801_205142",
        "SHA256SUMS 50/50 OK; manifest 1,018,692 行",
        "verified",
    ],
    [
        "inputs/archives/wiki2019zh_corpus.zip",
        "symlink->canonical_zip",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/wiki2019zh_corpus.zip",
        "b42e010ffd111de1a010e43db043122ce55ffdd2f93ab4b7712d96b7d2043070",
        "verified (与已知值一致)",
    ],
    [
        "inputs/archives/webText2019zh_corpus.zip",
        "symlink->canonical_zip",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/webText2019zh_corpus.zip",
        "a9bb7549ad1d537e09f532eb4930dd7b105968f71bbb90ea24f3f884cdeb4141",
        "recorded (无参考值)",
    ],
    [
        "inputs/archives/comment2019zh_corpus.zip",
        "symlink->canonical_zip",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/comment2019zh_corpus.zip",
        "5252766c2947c5b7edd054d44edc9e55650efb8ac1095a4664e6e77bf71f8643",
        "recorded (无参考值)",
    ],
    [
        "inputs/cleaned/news2016zh",
        "symlink->frozen",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/news2016zh_v22/news_freeze_v3_2",
        "SHA256SUMS 6/6 OK",
        "verified; 12,420 保留 / 790 隔离 / 704 精确重复移除 / 19,102,647 tokens / UNK 0",
    ],
    [
        "inputs/cleaned/zhwikisource_classics",
        "symlink->curated_v6",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/curated/zhwikisource_classics_v6",
        "accepted=b7099b01b420e00aec04dc98d59638ee7524f6a00c77b81a09e7010deda62ffc; quarantined=33449e791a72c126e892179b23f613f9127e767c9a4fae3a29fe4396527ea6b8",
        "verified vs finalize_report; 1276 docs / 3,964,180 tokens / UNK 0 / hard_defects 0; 无SHA256SUMS.txt(部分覆盖)",
    ],
    [
        "inputs/pending/wiki2019zh",
        "symlink->extracted",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh",
        "zip已验证; 255 files",
        "verified; 1,018,692 文档",
    ],
    [
        "inputs/pending/webText2019zh",
        "symlink->extracted",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/webText2019zh",
        "zip=a9bb7549ad1d537e09f532eb4930dd7b105968f71bbb90ea24f3f884cdeb4141",
        "recorded; 896 files; 文档数待统计",
    ],
    [
        "inputs/pending/comment2019zh",
        "symlink->extracted",
        "/home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/comment2019zh",
        "zip=5252766c2947c5b7edd054d44edc9e55650efb8ac1095a4664e6e77bf71f8643",
        "recorded; 687 files; 文档数待统计",
    ],
    [
        "reference/zh_bpe_32k.model",
        "symlink->tokenizer",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model",
        "af0c6373494745206027dcebb44985a4c5a80670d81db8d9c87ec4c2aa510f50",
        "verified (与已知值一致)",
    ],
    [
        "reference/model.json",
        "symlink->config",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/configs/model.json",
        "n/a",
        "read (196.15M 参数 / 2048 ctx / 32K vocab)",
    ],
    [
        "scripts/cpt_modern_v1_all",
        "symlink->scripts",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1",
        "n/a",
        "read (历史脚本与报告目录)",
    ],
    [
        "scripts/wiki2019zh_tier_dryrun_v2_fixed",
        "symlink->scripts",
        "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/wiki2019zh_tier_dryrun_v2_fixed",
        "n/a",
        "read (v2脚本+runner, 无v3)",
    ],
]
with open(os.path.join(OUT, "INPUT_INVENTORY.tsv"), "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")


# ---------------- run.log ----------------
log_lines = [
    "takeover_inventory_v1 run.log",
    "created_at: " + now.isoformat(),
    "mode: READ-ONLY (未写入/修改 inputs, audits 目标, tokenizer, best_model.pt)",
    "",
    "== 步骤1: 读取工作区说明文件 ==",
    "STATUS.md / CODEX_TASK.md / SOURCE_PATHS.tsv / TREE.txt 已完整阅读",
    "说明文件结论与本地实际情况一致性: STATUS.md 称 'v3正在准备或执行'；本地实测 v3 不存在(见步骤5)，说明文件为计划性描述，以本地为准。",
    "",
    "== 步骤2: 软链接有效性 ==",
    "SOURCE_PATHS.tsv 16条: 全部 OK (target 存在)",
    "",
    "== 步骤3: 参考文件与权重 ==",
    "reference/zh_bpe_32k.model SHA256 = af0c6373...510f50, 与已知值一致: OK",
    "best_model.pt SHA256 = 716c6117...a7e41, 与已知值一致: OK (仅只读哈希, 未触碰)",
    "reference/model.json: 196,153,216 参数描述 / vocab 32000 / max_seq_len 2048 / train_tokens 1525238705 (记录)",
    "",
    "== 步骤4: canonical ZIP ==",
    "wiki2019zh_corpus.zip SHA256 = b42e010f...43070, 与已知值一致: OK",
    "webText2019zh_corpus.zip SHA256 = a9bb7549...4141 (无参考值, 仅记录)",
    "comment2019zh_corpus.zip SHA256 = 5252766c...8643 (无参考值, 仅记录)",
    "",
    "== 步骤5: 已冻结/已清洗语料校验 ==",
    "news2016zh (news_freeze_v3_2): SHA256SUMS.txt 6/6 OK;",
    "  tier_a 行数=12420 (摘要 permanent_tier_a_documents=12420 一致)",
    "  final_quarantine 行数=790 (摘要 790 一致); exact_body_duplicates 行数=704 (摘要 704 一致)",
    "  tokenizer_audit: 19,102,647 tokens / UNK 0 / UNK率 0.0 (摘要记录, 未重跑)",
    "  PERMANENT_FREEZE_NOTICE.txt 存在",
    "zhwikisource_classics (v6):",
    "  accepted_final.jsonl 行数=1276, SHA256=b7099b01...62ffc == finalize_report 声明: OK",
    "  quarantined.jsonl 行数=49, SHA256=33449e79...ea6b8 == finalize_report 声明: OK",
    "  quarantined_manual_review 29 / trim_log 12 / manual_review_decisions 45行(44条+表头): 与摘要一致",
    "  token_audit: 3,964,180 tokens(无EOS) / 3,965,456(含EOS) / UNK 0 / hard_defects 0 / 非句尾30",
    "  web_residue_audit: 3个启发式issue文档(annotation_heading=1, suspected_dangling_poem_title=2), 非硬残留; 硬残留=0",
    "  缺口: v6目录无 SHA256SUMS.txt; 无 PERMANENT_FREEZE_NOTICE.txt (冻结状态需人工确认)",
    "",
    "== 步骤6: wiki2019zh 既有审计校验 ==",
    "full_audit_v1: SHA256SUMS 25/25 OK; 255文件/1,018,692文档/427,473,417字符(摘要)",
    "targeted_error_audit_v1: SHA256SUMS 4/4 OK; 9类 x 30样本",
    "tier_dryrun_v1: SHA256SUMS 32/32 OK; manifest 1,018,692行; A=572,058 B=347,570 DROP=87,119 REVIEW=11,945; 字符和=427,473,417",
    "tier_dryrun_v2: SHA256SUMS 50/50 OK; manifest 1,018,692行; A=515,829 B=382,852 DROP=89,374 REVIEW=30,637; 字符和=427,473,417",
    "  v1->v2 转移矩阵和=1,018,692; primary_reason 和=1,018,692; compact_transition_review_v2.txt SHA256 匹配",
    "  v2 已知系统误判5类: 已对照脚本确认相关正则存在; 抽样样本证实『仅缺末句号→possible_incomplete_tail』(永康工业区车站/七里岗乡/黑嘴树鸭)",
    "",
    "== 步骤7: v3 判定 ==",
    "audits/wiki2019zh_tier_dryrun_v3: 不存在",
    "scripts/wiki2019zh_tier_dryrun_v3_fixed: 不存在",
    "find /home/xmkjsoft -iname '*tier_dryrun_v3*' / '*dryrun_v3*': 无结果",
    "logs/build_workspace.log: '未找到 wiki2019zh tier dry-run v3'",
    "结论: v3 未执行, 无脚本可读/可跑; 下一步必须先编写v3脚本(修复v2已知误判)再干跑",
    "",
    "== 步骤8: 盘点产物 ==",
    "本目录: outputs/wiki2019zh/takeover_inventory_v1_20260801_220145/",
    "包含: summary.json, VERIFICATION_REPORT.md, INPUT_INVENTORY.tsv, run.log, SHA256SUMS.txt",
    "未生成任何训练文件; 未修改任何输入; 未重跑旧版本。",
]
with open(os.path.join(OUT, "run.log"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(log_lines) + "\n")


# ---------------- VERIFICATION_REPORT.md ----------------
md = """# wiki2019zh 接管盘点报告 (takeover_inventory_v1)

生成时间: 2026-08-01T22:01:45+08:00
工作区: `/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace`
性质: **只读盘点**。未修改任何输入、冻结语料、审计目录、tokenizer 或模型权重；未生成训练文件；未重跑旧版本。

## 0. 结论摘要

1. 工作区结构完整，16 条软链接全部有效。
2. tokenizer 与 `best_model.pt` 的 SHA256 均与任务给定的已知值一致（权重未触碰）。
3. wiki2019zh canonical ZIP SHA256 与已知值一致；ZIP 255 个成员与解压目录 255 个文件完全一致。
4. news2016zh 冻结 v3.2 全部 SHA256 通过，文档数对账一致（12,420 / 790 / 704）。
5. zhwikisource v6 已正式执行：`accepted_final.jsonl` 存在且哈希与 finalize_report 一致（1,276 文档）；Token 精确统计已完成（3,964,180，UNK=0，硬缺陷=0）；网页残留审计为 3 个启发式候选（非硬残留）。
6. wiki2019zh 全部 4 个既有审计 SHA256 通过，Manifest/计数/转移矩阵全部对账一致。
7. **v3 未执行，且不存在 v3 脚本** —— 无法"审查既有 v3 规则"，必须先新建 v3 规则脚本并干跑。
8. 下一步可直接执行的命令见第 9 节（创建 v3 脚本 → 语法检查 → 新目录干跑 → 转移矩阵与高风险抽样）。

## 1. 软链接验证

`SOURCE_PATHS.tsv` 共 16 条，逐一 `test -e` 检查：**16/16 有效，0 损坏**。

## 2. 模型与 Tokenizer

| 对象 | SHA256 实测 | 已知值 | 结果 |
|---|---|---|---|
| tokenizer/zh_bpe_32k.model | `af0c6373...510f50` | `af0c6373...510f50` | 一致 |
| best_model.pt (784,683,963 B) | `716c6117...a7e41` | `716c6117...a7e41` | 一致，未触碰 |

`reference/model.json` 记录：196.15M 参数、vocab 32,000、max_seq_len 2,048。

## 3. canonical ZIP

| ZIP | 实测 SHA256 | 判定 |
|---|---|---|
| wiki2019zh_corpus.zip | `b42e010f...43070` | 与已知值一致；255 成员 = 255 解压文件，fully identical（full_audit_v1 比对，已验证） |
| webText2019zh_corpus.zip | `a9bb7549...4141` | 已记录，任务未给参考值 |
| comment2019zh_corpus.zip | `5252766c...8643` | 已记录，任务未给参考值 |

## 4. 已冻结语料

### news2016zh (news_freeze_v3_2)

- SHA256SUMS.txt：**6/6 通过**。
- `news_tier_a_permanent.jsonl` 行数 = **12,420**（摘要一致）；`final_quarantine_v3_2.jsonl` = **790**；`exact_body_duplicates_v3_2.jsonl` = **704**。
- 冻结摘要 token 审计：**19,102,647 tokens，UNK=0，UNK率=0.0**（记录于摘要，本次未重跑）。
- 硬残留（permanent tier A）= 0（摘要记录）。
- 存在 `PERMANENT_FREEZE_NOTICE.txt`。
- 注：摘要未记录总字符数，按盘点规则不重新扫描，标记"未记录"。

### zhwikisource_classics (v6)

- **v6 已正式执行**：`accepted_final.jsonl`、`finalize_report.json`、`quarantined.jsonl`、`quarantined_manual_review.jsonl`、`trim_log.jsonl`、`manual_review_decisions.tsv`、`token_audit/`、`web_residue_audit/` 全部存在。
- `accepted_final.jsonl`：1,276 行；SHA256 = `b7099b01...62ffc`，与 finalize_report 声明一致。
- `quarantined.jsonl`：49 行；SHA256 = `33449e79...ea6b8`，与 finalize_report 声明一致。
- 人工复核：44 篇（KEEP 3 / TRIM 12 / QUARANTINE 29），与文件行数一致。
- **Token 精确统计已完成**：3,964,180 tokens（无 EOS）、3,965,456（含 EOS）、UNK=0、UNK率=0.0、hard_defect_documents=0、非句尾文档 30（诗行/标题性质，审计摘要已注明可人工解释）。
- **网页残留审计**：3 个 issue 文档（annotation_heading=1、suspected_dangling_poem_title=2）——审计脚本明确注明为启发式候选、不可直接批量删除，**硬残留=0**。
- 缺口：v6 目录内**无 SHA256SUMS.txt**（finalize_report 仅覆盖 2 个输出文件，均已匹配）；**无冻结通知文件**。建议补全 SHA256SUMS.txt，并人工确认是否视为正式冻结。

## 5. wiki2019zh 既有审计

| 审计 | SHA256SUMS | Manifest 行数 | 关键计数 |
|---|---|---|---|
| full_audit_v1 | 25/25 OK | — | 255 文件 / 1,018,692 文档 / 427,473,417 字符 |
| targeted_error_audit_v1 | 4/4 OK | — | 9 类 × 30 样本 |
| tier_dryrun_v1 | 32/32 OK | 1,018,692 | A=572,058 B=347,570 DROP=87,119 REVIEW=11,945 |
| tier_dryrun_v2 | 50/50 OK | 1,018,692 | A=515,829 B=382,852 DROP=89,374 REVIEW=30,637 |

- v1/v2 的 Tier 文档数之和 = 1,018,692；字符数之和 = 427,473,417（与 full_audit 一致）。
- v1→v2 转移矩阵之和 = 1,018,692；v2 primary_reason 之和 = 1,018,692。
- `compact_transition_review_v2.txt` SHA256 与其 .sha256 文件匹配。
- 重点关注转移：A→DROP 4,957、A→REVIEW 12,310、B→DROP 7,109、B→REVIEW 5,791、DROP→A 53、DROP→B 6,520、REVIEW→A 350、REVIEW→B 2,297。

## 6. v3 状态判定（以本地为准）

**v3 未执行。** 证据：

- `audits/wiki2019zh_tier_dryrun_v3` 不存在；`scripts/wiki2019zh_tier_dryrun_v3_fixed` 不存在。
- 全盘搜索 `*tier_dryrun_v3*` / `*dryrun_v3*` 无结果。
- `logs/build_workspace.log` 明确记录"未找到 wiki2019zh tier dry-run v3"。
- `outputs/wiki2019zh` 为空。

因此不存在可"审查/复跑"的 v3 脚本；下一步必须先**新建 v3 规则脚本**（基于 v2 修复下述已知系统误判），再做语法检查、新目录干跑、转移矩阵与高风险抽样。

## 7. v2 已知系统误判（v3 必须修复，任务已列明，代码已核实）

| # | 误判 | v2 代码证据 |
|---|---|---|
| 1 | 正常"主演/导演/作者"句子被当语义空穴 | `SEVERE_HOLE_RE` 含 `(?:编剧\|导演\|主演\|作者)` 模式 |
| 2 | 单字段（坐标/面积/人口）缺失即删整篇 | `EXTENDED_NUMERIC_HOLE_RE` 覆盖 总面积/人口/坐标/海拔 |
| 3 | 正文完整、尾部仅"下辖以下地区："被整删 | `SEVERE_HOLE_RE` 含 `下辖以下地区[：:]$` |
| 4 | "某街道，是下辖的一个乡级行政区单位。"误入严格干净 A | `MISSING_SUBJECT_RE` / `COMPLETE_SHORT_RE` 相关判定 |
| 5 | 仅因末尾缺句号命中"尾部不完整" | `possible_incomplete_tail` primary=22,658 / overlapping=57,341；抽样样本（永康工业区车站、七里岗乡、黑嘴树鸭）证实整篇完整仅缺末句号被降级 |

抽样样本已人工复核（见 v2 `samples/` 与 `compact_transition_review_v2.txt`，哈希已校验）。

## 8. 其他数据源现状

- pending/webText2019zh：896 个文件，文档数待正式处理阶段统计。
- pending/comment2019zh：687 个文件，文档数待正式处理阶段统计。
- outputs/webText2019zh、outputs/comment2019zh、outputs/cross_source：均为空。

## 9. 下一步（下一条可直接执行的命令）

v3 脚本不存在，因此下一步是**创建 v3 分类脚本并在新目录干跑**（不会覆盖 v1/v2，也不会修改任何输入）：

```bash
cd /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace

# 1) 编写 v3 脚本（建议基于 v2 脚本 + 修复第 7 节 5 类误判）
# 2) Python 语法检查
python3 -m py_compile outputs/wiki2019zh/tier_dryrun_v3_<时间戳>/classify_wiki2019zh_tier_dryrun_v3.py

# 3) 新目录全量干跑（输出到 outputs/wiki2019zh/，不触碰 audits/ 既有目录）
python3 outputs/wiki2019zh/tier_dryrun_v3_<时间戳>/classify_wiki2019zh_tier_dryrun_v3.py \
  /home/xmkjsoft/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh \
  audits/wiki2019zh_tier_dryrun_v2/classification_manifest_v2.jsonl.gz \
  outputs/wiki2019zh/tier_dryrun_v3_<时间戳>
```

干跑后必须产出：v2→v3 转移矩阵、高风险转移抽样（重点覆盖 A→DROP、A→REVIEW、B→DROP、B→REVIEW、DROP→A、DROP→B、REVIEW→A、REVIEW→B 及任务列明的 10 个重点原因）、SHA256SUMS.txt、summary.json、run.log，并先给分类误判分析与清洗预览，再决定是否需要 v4。

## 10. 已验证 / 未验证 / 推测 清单

- **已验证**：软链接 16/16；tokenizer 与 best_model.pt 哈希；wiki ZIP 哈希与 ZIP/解压一致性；news2016zh 全部 SHA256 与行数；zhwikisource v6 关键哈希与行数、token 审计、残留审计；wiki2019zh 4 个审计的全部 SHA256、manifest 行数、Tier/转移/主因计数对账；v3 不存在。
- **未验证**：webText/comment ZIP 参考哈希（任务未提供）；zhwikisource v6 全目录 SHA256SUMS（缺失）；news2016zh 冻结字符数（摘要未记录，未重扫）；zhwikisource v6 冻结状态（无冻结通知）；webText/comment 文档数（未盘点）。
- **推测**：zhwikisource token_audit / web_residue_audit 是在 v6 finalize 之后运行（目录时间戳 21:35 vs 21:36/21:46-48），推断其审计对象即当前 accepted_final.jsonl。
"""
with open(os.path.join(OUT, "VERIFICATION_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write(md)


# ---------------- SHA256SUMS.txt ----------------
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


entries = []
for name in ["summary.json", "VERIFICATION_REPORT.md", "INPUT_INVENTORY.tsv", "run.log"]:
    entries.append((sha256_file(os.path.join(OUT, name)), name))
entries.sort()
with open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
    for h, name in entries:
        f.write(f"{h}  {name}\n")

print("inventory written to:", OUT)
for h, name in entries:
    print(f"  {h}  {name}")
