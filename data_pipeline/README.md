# Zh200M-C3 数据清洗与构建流程

本目录保存 196M 中文基础模型正式预训练前使用的中文语料清洗、审计、去重、冻结、Tokenizer 评估、采样、数据拼接和最终校验流程。

目标是保留完整的历史处理路径，使数据构建过程可追溯、可审计。这里只发布代码、说明和校验信息，不发布原始正文、清洗后的正文或 `train.bin`；这些材料仍受上游授权、版权和个人信息风险约束。

## 目录

- `legacy_20260802/`
  - 2026-08-01 至 2026-08-02 的原始工作目录快照。
  - `STATUS.md`：当时的项目状态。
  - `SOURCE_PATHS.tsv`：输入路径和软链接目标。
  - `TREE.txt`：当时工作区的路径结构。
  - `work/`：清洗、审计、去重、冻结、Tokenizer 评估、拼接和校验脚本。
- `MANIFEST.sha256`
  - 本目录内 146 个原始文本文件和脚本的 SHA-256 清单。

上传时排除了 `__pycache__` 和 `.pyc` 编译缓存。历史探索脚本、调试脚本和最终脚本均原样保留，以便审计完整迭代过程。

## 主要阶段

| 阶段 | 代表脚本 | 作用 |
|---|---|---|
| 环境与输入审计 | `check_env.py`、`check_deps2.py`、`inspect_frozen_formats.py` | 检查依赖、输入格式、软链接和冻结状态 |
| 输入盘点 | `build_inventory.py`、`validate_inventory.py` | 建立文档和文件级输入清单 |
| `wiki2019zh` 清洗 | `wiki_cleaner_core.py`、`build_candidates_v2.py`、`build_preview_v3.py` | 规则清洗、候选生成和预览审查 |
| `wiki2019zh` 冻结 | `freeze_wiki.py`、`token_audit_wiki.py` | 文档级冻结、精确 Token 统计和审计 |
| `webText2019zh` 清洗 | `audit_webtext.py`、`classify_webtext_v1.py`、`clean_candidates_webtext.py` | 全量审计、分类、清洗候选和误判抽样 |
| `webText2019zh` 去重冻结 | `dedup_webtext.py`、`freeze_webtext.py` | 内部去重、冻结和统计 |
| `comment2019zh` 清洗 | `comment_pipeline.py` | 评论语料审计、清洗和冻结 |
| 新闻辅助审计 | `check_news_neardup.py`、`news_post_dedup_tokens.py`、`verify_news_pair.py` | 新闻近似重复和 Token 校验 |
| 跨来源去重 | `cross_source_dedup.py`、`similarity_addendum.py` | 来源之间完全重复和近似重复处理 |
| Tokenizer A/B | `tokenizer_ab.py`、`tok2_build_samples.py`、`tok2_select.py`、`tok2_train.py`、`tok2_eval.py` | 固定样本、训练并评估候选 Tokenizer |
| 最终采样与拼接 | `bind_final.py`、`rebuild_manifest.py` | 按冻结 Tokenizer、来源目标和确定性洗牌拼接训练集 |
| 最终校验 | `verify_binding.py`、`verify_hashes.sh`、`final_checks.sh` | 校验 Token、EOS、来源计数、文档重叠和 SHA-256 |

`work/zh200m/` 中还保留了模型结构、CPU 推理、导出和基准测试的历史脚本。正式训练入口位于仓库的 `training/train_c3_a800.py`。

## 正式数据参数

```text
训练 Token 数            747,110,400
训练完整序列数           364,800
上下文长度               2,048
优化步数                 2,850
有效批量                 262,144 Token/步
manifest 完整文档 Token  747,110,276
末尾未列入 manifest      124
validation               4,112,358 Token / 19,491 文档
test                     4,118,503 Token / 19,662 文档
```

五个来源的实测 Token 数：

| 来源 | Token 数 |
|---|---:|
| `webText2019zh` | 351,140,474 |
| `wiki2019zh` | 336,198,878 |
| `news2016zh` | 29,885,015 |
| `comment2019zh` | 22,413,226 |
| `zhwikisource_classics` | 7,472,683 |
| 合计 | 747,110,276 |

核心校验值：

```text
train.bin SHA-256
d2a6b96aff03a8f57262db37479cbcc0b057ff1edcb3aa9472ad44cd20a99852

validation.bin SHA-256
22493574e3f613b282ec084cbd169b573fd7bfec8d855057438b420420cda8f9

test.bin SHA-256
8e245f2f4dae6c5a4cc92efa319039e2c1e9ceffb414df9254c554da55ab21d0

冻结 tokenizer.model SHA-256
bea16a0e5f5d07307fd471f6aaf6e8c457589b5cc7e71486f0beca124a1cecb0
```

## 最终拼接路径

历史记录中的正式执行方式如下：

```bash
source /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/.venv_core_audit/bin/activate

F=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/tokenizer_freeze_v1_20260802_160659
export FINAL_TOK="$F/tokenizer.model"

python3 -u work/bind_final.py
python3 -u work/verify_binding.py
```

历史输出目录：

```text
/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/training_bind_v1_20260802_194220
```

这些脚本保留了当时的 WSL 绝对路径，直接在其他机器运行前必须修改输入、输出和虚拟环境路径。运行前应先备份输入，并始终写入新的时间戳输出目录。

## 数据发布边界

本流程对应的原始语料来自多个上游项目，许可证和再分发条件并不统一。仓库只公开构建代码与校验信息，不代表对原始正文、`train.bin` 或派生数据授予额外许可。第三方语料的署名、许可证、相同方式共享、隐私和删除请求应由实际数据发布者逐项处理。详见仓库根目录的 `DATA_NOTICE.md`。

## 完整性校验

在本目录执行：

```bash
sha256sum -c MANIFEST.sha256
```

该清单校验的是本目录保存的脚本与说明文件，不包含未上传的语料和模型二进制。
