# 196M 中文 LM 训练与 CPU 推理（A800 单卡 / llama.cpp）

## 架构

LLaMA 兼容：RMSNorm + RoPE + SwiGLU + GQA(2) + tied embedding + 无 bias。
参数断言：`parameter_count == 196153216`（`model.py` 内建）。

## 正式 4 小时训练命令（A800 单卡）

```bash
cd /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
source .venv_core_audit/bin/activate

python3 outputs/cross_source/zh200m_cpu_inference_v1_<ts>/train.py \
  --config outputs/cross_source/zh200m_cpu_inference_v1_<ts>/config/training_config.json \
  --data-dir outputs/cross_source/training_bind_v1_<ts> \
  --out-dir outputs/cross_source/zh200m_run_v1_<ts>/ckpt \
  --micro-batch 4 --grad-accum 32 --dtype bf16 --device cuda
```

预算：max_train_tokens=747,110,400；effective_batch_tokens=262,144；
max_optimizer_steps=2,850；max_wall_clock_seconds=14,400；
13,800s 后不再开始新步；SIGINT/SIGTERM 安全保存。

微批测速：`--micro-batch 8 --grad-accum 16` 或 `--micro-batch 16 --grad-accum 8`。
500 步测速（不正式执行）：`--speed-test --speed-test-steps 500 --micro-batch 4 --grad-accum 32`。

断点续训：`--resume <out-dir>/ckpt`。

## 导出与量化（训练后）

```bash
python3 export_gguf.py --checkpoint ckpt/checkpoint.pt --config config/training_config.json \
  --sp-model /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model --out model-f16.gguf
python3 export_onnx.py --checkpoint ckpt/checkpoint.pt --config config/training_config.json \
  --out-fp32 model_fp32.onnx --out-int8 model_int8.onnx

llama-quantize model-f16.gguf model-q8_0.gguf Q8_0
llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

## CPU 基准

```bash
bash benchmark_cpu.sh model-q5_k_m.gguf bench_out <llamacpp_dir>
python3 parse_bench.py bench_out/bench_raw.log bench_out
```

## 质量验证（训练后）

`validate_quant.py` 对原始/F16/Q8/Q5/Q4/ONNX-INT8 计算 val loss、perplexity、
固定提示词续写、逐步一致率、logit 误差、乱码率与重复率（不可执行前不承诺结果）。

## 关键约束

- 禁止加载/覆盖 `best_model.pt`；random_init=true；init_checkpoint=null；global_step=0。
- 训练前必须先通过 smoke_test.py（2 步 + GGUF 导出 + llama.cpp CPU 加载生成验证）。
- 本目录只含程序与报告；正式 4 小时训练未启动。
