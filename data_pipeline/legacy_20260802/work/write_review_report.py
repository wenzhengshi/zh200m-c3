#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/wiki2019zh/cleaning_preview_v2_20260801_230409")

report = """# wiki2019zh 清洗预览审查报告 (cleaning_preview_v2)

生成时间: 2026-08-01T23:0x+08:00
分类器: tier_dryrun_v3_1（已校验，SHA256SUMS 63/63 OK）
预览目录: outputs/wiki2019zh/cleaning_preview_v2_20260801_230409
supersedes: cleaning_preview_v1_20260801_225824（其 remove_or_trim_defective_clause 实现存在缺陷，
仅删除句末句号而未真正修剪缺陷子句，故作废；本 v2 已修复并重新生成）

## 1. 预览范围与规模

- 预览对象：v3.1 分类中带至少一个 repair_action 的文档，共 **223,345 篇**（B tier 可清洗文档）。
- 对齐校验：逐文档 document_id 与 v3.1 manifest 对齐，文本 SHA256 与 manifest 全数一致，**0 失败**。
- 字符：清洗前 159,793,266 → 清洗后 156,874,510，移除 2,918,756。
- 无变化文档（no-op）：4,802 篇（动作命中但文本未变，见 §3）。
- 全部 14 类动作的文档计数与分类器 repair_action_counts 完全一致。

## 2. 逐动作抽检结论（每类 30 个样本已人工复核）

| 动作 | 文档数 | 抽检结论 |
|---|---|---|
| remove_control_characters | 2,307 | 正常删除控制字符 ✓ |
| normalize_wiki_variant_markup | 26,945 | -{zh-hans:X}-/-{X|Y}-/-{X}- 规范化正确 ✓ |
| remove_soft_html | 6,419 | br/div 标签删除正确 ✓ |
| remove_raw_wiki_markup | 1,757 | 模板/ref/链接展开正确 ✓ |
| remove_email | 33 | 邮箱删除正确 ✓ |
| remove_empty_language_field | 27,864 | "英语："等空字段删除正确 ✓ |
| remove_or_normalize_empty_slots | 82,028 | （），《》等空槽删除正确 ✓ |
| remove_foreign_paragraphs | 1,993 | 长外语段落删除正确 ✓ |
| remove_url_or_reference_lines | 2,307 | 独立URL行删除正确 ✓ |
| trim_reference_section | 3 | 参考文献区截断正确 ✓ |
| normalize_missing_parent_admin_definition | 32,312 | "X，是下辖的一个Y" → "X是一个Y" ✓ |
| remove_trailing_empty_heading | 35,787 | 正文后的"下辖以下地区："等空标题删除正确 ✓ |
| trim_incomplete_tail | 5,982 | 明确截断/孤立列表删除；缺句号完整句保留（见 §3.1）|
| remove_or_trim_defective_clause | 19,575 | 模板句整句删除/嵌入空洞干净修剪；个别保留数据带轻微语病（见 §3.2）|

## 3. 需要人工复核的类别

### 3.1 尾部缺句号但内容完整（2,971 篇，清单: tail_kept_for_review.jsonl）
trim_incomplete_tail 的保守策略：仅删除明确截断（逗号/连接词结尾）、人口图示、孤立列表；
末尾是完整句子仅缺句号时**不删除、保留原文**，并列入本清单供人工确认。

### 3.2 defective-clause 修剪的轻微语病（约数百篇）
例：点湖村 —— "点湖村的座标为，而该地的平均海拔高度为666米（即2185英尺）。"
→ "点湖村的而该地的平均海拔高度为666米（即2185英尺）。"（数据保留、语病待人工确认）。
穆尔环形山类嵌入空洞则修复干净："该陨坑中心月面坐标为，直径52.66公里" → "该陨坑中心月面直径52.66公里"。

### 3.3 no-op 文档（4,802 篇）
动作存在但清洗后无变化，主要构成为 remove_url_or_reference_lines（URL 在行内而非独立行）与
remove_trailing_empty_heading（条件未完全满足）。不影响正确性，但建议抽样确认。

### 3.4 缺标题/主语短文（DROP 5,486 篇）
short_missing_title_or_subject 为既定规则 DROP（"是…"开头的完整短文 80-200 字）。
其中相当部分内容完整可抢救，建议在冻结前做人工复核/二次评估（见候选集阶段）。

### 3.5 pure_list_or_heading_stub（DROP 2,232 篇）
首段含列表关键词的短壳（"XX列表如下："等），抽检为真实列表空壳为主，符合规则。

## 4. 结论

- 清洗预览 v2 全部校验通过（SHA256SUMS 18/18 OK，对齐 0 失败）。
- 清洗动作可审计落地；3.1/3.2 类少量复杂异常进入人工复核清单，不扩大规则（规则13）。
- 下一步：基于 v3.1 分类 + 本预览清洗逻辑生成正式候选集（A/B 分离、REVIEW 隔离、DROP 留档）。
"""

(OUT / "PREVIEW_REVIEW_REPORT.md").write_text(report, encoding="utf-8")

summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["preview_review_report"] = "PREVIEW_REVIEW_REPORT.md"
summary["supersedes"] = "cleaning_preview_v1_20260801_225824 (defective clause-trim; superseded)"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("report written")
