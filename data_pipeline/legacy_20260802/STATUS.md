# Codex 中文语料清洗总工作区

生成时间：2026-08-01 21:48:17 +0800

项目根目录：

```
/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime
```

工作区：

```
/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
```

## 目录说明

- `inputs/cleaned/news2016zh`
  - 新闻语料已清洗并冻结。
  - 只读使用，禁止在目标目录原地修改。

- `inputs/cleaned/zhwikisource_classics`
  - 维基文库古籍已完成主体清洗和人工复核收尾。
  - 仍应检查最新v6是否已正式执行、硬残留是否为0、最终Token是否已统计。
  - 禁止原地修改。

- `inputs/pending/wiki2019zh`
  - 中文维基百科，共1,018,692个文档。
  - 已完成输入验证、全量审计、Tier v1/v2；v3正在准备或执行。
  - 尚未正式清洗和冻结。

- `inputs/pending/webText2019zh`
  - 尚待完整审计、清洗、去重和冻结。

- `inputs/pending/comment2019zh`
  - 尚待完整审计、清洗、去重和冻结。
  - 评论数据预计风险最高，最终占比应严格限制。

- `audits/`
  - 已完成的审计结果和分类Manifest。

- `scripts/`
  - 已有清洗、审计、冻结脚本。
  - `cpt_modern_v1_all`指向现有整个处理目录。

- `reference/`
  - 当前SentencePiece tokenizer和模型配置。

- `outputs/`
  - Codex所有新脚本、清洗候选、日志、报告、Manifest和SHA256必须写到这里。
  - 不得写回`inputs`指向的原始或冻结目录。

## 硬规则

1. 不得覆盖或修改任何冻结语料。
2. 不得修改canonical ZIP和原始解压目录。
3. 不得覆盖：
   `/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/best_model.pt`
4. 所有新处理必须写入`outputs`下的新版本目录。
5. 每轮必须保留：
   - 输入路径及SHA256
   - 输出SHA256
   - keep/drop/review清单
   - 删除或修改原因
   - 清洗前后字符数
   - 精确文档数
6. REVIEW不能直接进入训练集。
7. 最终冻结前必须做：
   - 硬残留审计
   - 完全重复去重
   - 近似重复/转载检测
   - SentencePiece精确Token统计
8. 不得根据估算填写文档数或Token数。

## 当前推荐处理顺序

1. 完成wiki2019zh v3规则复核
2. 生成wiki2019zh清洗预览
3. 正式生成wiki2019zh清洗候选
4. 清洗后残留审计
5. wiki2019zh内部去重并冻结
6. webText2019zh
7. comment2019zh
8. 跨来源去重
9. Tokenizer复用/重训A/B审计
10. 最终混合比例设计
