# 196M 模型架构与 CPU 推理兼容性审计

生成时间: 2026-08-02（Tokenizer 第二轮评测期间）
依据: `configs/model.json`（参考配置）与 196,153,216 参数目标。

## 1. 参考配置读取（不得假设）

`reference/model.json` 实际内容：

| 字段 | 值 |
|---|---|
| name | zhtext-a800-200m |
| vocab_size | 32000 |
| max_seq_len | 2048 |
| n_layer | 20 |
| d_model | 896 |
| n_head | 14 |
| n_kv_head | 2 |
| ffn_hidden_size | 2432 |
| rope_theta | 10000.0 |
| rms_norm_eps | 1e-05 |
| dropout | 0.0 |
| tie_embeddings | true |
| native_gqa | true |

## 2. 与 LLaMA/GGUF 兼容架构逐项对照

| 要求 | 配置/实现 | 结论 |
|---|---|---|
| RMSNorm | rms_norm_eps=1e-5，无bias | 兼容 |
| RoPE | rope_theta=10000，head_dim=896/14=64 | 兼容 |
| SwiGLU | ffn_hidden_size=2432（按 3×d×f 核算） | 兼容（见 §3 参数核算） |
| GQA | n_kv_head=2 | 兼容 |
| 无bias Linear | 实现全部 bias=False | 兼容 |
| Embedding 与 LM Head 共享 | tie_embeddings=true | 兼容 |
| vocab_size=32000 / ctx=2048 | 一致 | 兼容 |

## 3. 参数量核算（必须等于 196,153,216）

SwiGLU MLP：gate/up/down 三个投影，每层 3×896×2432 = 6,537,216。

- Embedding（共享）：32,000 × 896 = 28,672,000
- 每层 Attention（GQA）：q 896×896 + k 896×128 + v 896×128 + o 896×896 = 1,835,008
- 每层 MLP（SwiGLU）：3×896×2432 = 6,537,216
- 每层 RMSNorm：2×896 = 1,792
- 层合计：8,374,016 × 20 = 167,480,320
- 最终 Norm：896
- **合计：28,672,000 + 167,480,320 + 896 = 196,153,216 ✓**

> 注：若把 ffn_hidden_size=2432 当成普通双层 MLP（2×d×f），总参数只有 152,571,776；
> 只有 SwiGLU（3×d×f）才能精确命中 196,153,216。因此参考配置隐含 SwiGLU。

## 4. 结论：方案选择

- 方案1（保留架构、自研 CPU 推理）：可行但重复造轮子，无法直接获得 llama.cpp 生态
  （GGUF/量化/CPU 内核/Spring Boot sidecar）。
- 方案2（从零训练前采用 LLaMA 兼容架构）：**参考配置本就是 LLaMA 兼容结构**，
  无需改动架构即可满足 GGUF 导出要求。

**默认选择方案2**：按 §2 结构实现，`model.py` 内建参数断言。

## 5. 正式训练前断言（写入 train.py）

- parameter_count == 196153216
- vocab_size == 32000
- context_length == 2048
- random_init == true
- init_checkpoint == null

## 6. GGUF 映射

| PyTorch 张量 | GGUF 张量名 |
|---|---|
| embed_tokens.weight | token_embd.weight |
| norm.weight | output_norm.weight |
| embed_tokens.weight（共享） | output.weight |
| layers.i.attn_norm.weight | blk.i.attn_norm.weight |
| layers.i.attn.q/k/v/o_proj.weight | blk.i.attn_q/k/v/output.weight |
| layers.i.ffn_norm.weight | blk.i.ffn_norm.weight |
| layers.i.mlp.gate/up/down_proj.weight | blk.i.ffn_gate/up/down.weight |
