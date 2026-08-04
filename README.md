# Zh200M-C3 · 196M 中文 Decoder-only Transformer 基础模型

从零随机初始化预训练的中文基础语言模型（非微调、非对齐）。冻结自有 32K 中文
SentencePiece BPE Tokenizer，组织约 7.47 亿 Token 中文语料，在 A800 上以 BF16
完成 2,850 步预训练（完整一个 epoch）。

> ⚠️ 本模型使用**非标准 RoPE 布局**（`repeat_interleave(2)` + 前/后半 `chunk` 旋转），
> 与 LLaMA 默认布局不兼容。转 HF / GGUF 时必须保留此实现，否则输出错误。

## 快速开始

```bash
pip install -r requirements.txt
```

权重下载：[Release v0.2 · zh200m-c3-196m-fp32.pt](https://github.com/wenzhengshi/train_model/releases/tag/v0.2)
（FP32，784MB，SHA-256 `e876f400ef08ef75eefe13b10efd1012b010fb5d3567f97c5e19fc772b7a58d0`）

下载后：

```bash
# 本地 CPU / GPU 交互式生成（KV Cache + 实时逐 Token 输出）
python inference/interactive_c3_747m_kvcache.py --weights zh200m-c3-196m-fp32.pt
```

也可以直接用任意 checkpoint（完整训练态）：

```bash
python inference/interactive_c3_747m_kvcache.py --checkpoint checkpoint_tokens_747110400.pt
```

## 模型结构

| 项目 | 配置 |
| --- | --- |
| 参数量 | 196,153,216（≈0.196B） |
| 结构 | Decoder-only Transformer |
| 词表 | 32,000（自有中文 BPE，冻结） |
| 上下文 | 2,048 |
| 隐藏维度 | 896 |
| 层数 | 20 |
| 注意力 | GQA：Q 头 14 / KV 头 2，head_dim 64 |
| FFN | 2,432（SwiGLU） |
| 归一化 | RMSNorm |
| 位置编码 | RoPE（非标准布局，见上） |
| 其他 | 输入/输出权重共享，无 bias |

## 训练配置

| 项目 | 值 |
| --- | --- |
| 设备 | NVIDIA A800-SXM4-80GB |
| 精度 | BF16（参数 FP32） |
| Micro Batch × 梯度累积 | 4 × 32 |
| 有效批量 | 262,144 Token/步 |
| 峰值学习率 | 0.0015（200 步 warmup，cosine 衰减到 0.1×） |
| Weight Decay | 0.1 |
| 梯度裁剪 | 1.0 |
| 训练步数 | 5,700（2 个 epoch，各 2,850 步） |
| 训练 Token | 1,494,220,800（两个完整 epoch） |
| 最终验证 loss | 3.7121（PPL 40.94，全量验证集） |

## 训练数据（合规说明）

训练语料来自：webText2019zh、wiki2019zh、news2016zh、comment2019zh
（**CLUECorpusSmall，研究用途，官方要求“不向第三方提供”**）以及维基文库古籍
（CC BY-SA）。

**本仓库不包含语料正文**，只包含数据清单（`data/` 下的 manifest）与复现说明。
如需复现训练集，请自行按 CLUE 官方流程申请下载后运行清洗脚本，详见
[DATA_NOTICE.md](DATA_NOTICE.md)。

## 目录

```text
model/      模型定义（含非标准 RoPE）
training/   可审计训练脚本
configs/    训练配置
tokenizer/  冻结中文 Tokenizer
inference/  CPU/GPU 推理与 Web 界面
data/       训练数据清单（不含正文）
```

## 许可

- 代码与模型权重：MIT License
- 数据：见 [DATA_NOTICE.md](DATA_NOTICE.md)

## 状态

- v0.2（当前）：2 个 epoch（14.94 亿 Token）完成，最终验证 loss 3.7121
  （PPL 40.94），相比 1 个 epoch（3.8295 / 46.04）提升约 11%
- v0.1：1 个 epoch（747.1M Token），验证 loss 3.8295（PPL 46.04）

本模型是**基础模型**：能续写中文，但不是聊天/指令/问答模型，事实可靠性有限。
