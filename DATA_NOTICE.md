# 训练数据合规说明（重要）

本模型的训练语料**不能随本仓库分发**，原因如下：

## 1. CLUECorpusSmall 四源（约 99%）

webText2019zh、wiki2019zh、news2016zh、comment2019zh 均来自
[CLUECorpus2020 / CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020)。
官方数据申请条款明确要求：

> 将使用语料研究目的和用途，计划、研究机构和申请者介绍，发送到邮箱，
> **并承诺不向第三方提供**。

因此，无论是原始语料还是经清洗/重打包的衍生版本，都不得直接分发。

## 2. 维基文库古籍（约 1%）

来自维基文库，文本按 CC BY-SA 授权：可分发，但必须署名并以相同协议共享。
如需随包发布，请从 Wikimedia 官方 dump 重新导出并附署名与协议说明。

## 3. 本仓库包含什么

- `data/training_sample_manifest.jsonl.gz`：训练集使用的文档清单
  （`source::document_id` 与 token 数，**不含正文**）
- 数据清洗/组织流程说明（见下方复现步骤）

## 4. 如何复现训练集

1. 按 CLUE 官方流程（CLUEbenchmark@163.com）申请下载 CLUECorpusSmall；
2. 下载维基文库古籍（Wikimedia dump）；
3. 按既有清洗流程（去重、质量过滤、EOS 插入、2048 打包、确定性打乱）生成
   `train.bin` / `validation.bin` / `test.bin`；
4. 使用 `training/train_c3_a800.py` 训练。

清洗与打包脚本的详细说明请参考仓库内脚本与 manifest 字段定义。
