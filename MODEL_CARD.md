# Model Card · Zh200M-C3

## 概述

- **模型**：196,153,216 参数的 Decoder-only Transformer，中文基础语言模型
- **训练方式**：完全随机初始化预训练（不加载任何旧 checkpoint）
- **Tokenizer**：自有中文 SentencePiece BPE（词表 32K），全程冻结
- **训练数据**：约 14.94 亿 Token 中文（CLUECorpusSmall 四源 + 维基文库古籍，2 个 epoch）
- **训练硬件**：NVIDIA A800-SXM4-80GB × 1，BF16

## 架构细节

| 参数 | 值 |
| --- | --- |
| vocab_size | 32000 |
| max_seq_len | 2048 |
| dim | 896 |
| n_layer | 20 |
| n_head / n_kv_head | 14 / 2（GQA） |
| head_dim | 64 |
| ffn_dim | 2432 |
| rope_theta | 10000.0 |
| rms_norm_eps | 1e-5 |
| custom_rope_layout | `repeat_interleave_2_plus_front_back_rotate_half` |

⚠️ **非标准 RoPE**：旋转实现为
`repeat_interleave(2)` + 前/后半 `chunk` 旋转，与 LLaMA 默认
`cat((freqs, freqs)) + rotate_half` 布局**不相等**。任何格式转换必须保留本实现。

## 训练详情

- 数据：364,800 条 × 2048 序列；文档间插入真实 EOS；全局确定性打乱（seed 20260802）
- 优化：AdamW（β=(0.9, 0.95), eps=1e-8），weight decay 0.1，grad clip 1.0
- 学习率：峰值 0.0015，200 步 warmup，cosine 衰减至 0.1×
- 有效批量：262,144 Token/步；共 5,700 步 / 1,494,220,800 Token（2 个 epoch）
- 最终验证（全量 4,108,329 targets）：loss 3.7121，PPL 40.94
  （1 个 epoch 时为 loss 3.8295 / PPL 46.04）

## 数据来源与许可

| 来源 | 占比 | 许可/限制 |
| --- | --- | --- |
| webText2019zh | ~47% | CLUECorpusSmall：研究用途，**不得向第三方提供** |
| wiki2019zh | ~45% | 同上（源自 CLUE 分发） |
| news2016zh | ~4% | 同上 |
| comment2019zh | ~3% | 同上 |
| 维基文库古籍 | ~1% | CC BY-SA |

语料正文不随本仓库分发。复现方法见 DATA_NOTICE.md。

## 预期用途与限制

**可以做**：中文文本续写、语言规律探测、作为继续预训练/SFT 的起点。

**不适合**：未经 SFT/对齐直接充当聊天助手；不保证事实正确性；生成内容需人工审核。

## 本地运行参考

Intel i5-4460 / 无 CUDA / PyTorch CPU FP32：约 2~3 Token/s，KV Cache 仅 2 个
KV 头（约 40 MiB）。
