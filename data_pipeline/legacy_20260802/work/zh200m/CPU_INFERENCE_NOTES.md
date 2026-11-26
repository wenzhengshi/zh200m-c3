# CPU 推理环境说明

## 本沙箱实测发现（2026-08-02）

1. llama.cpp b10223（预编译与本地 CMake 构建均复现）：`llama-cli` 在本 WSL 沙箱（4 核、
   7GB）的 decode 阶段卡死（wait_woken，SIGTERM 无效）；`llama-server`、`llama-bench`、
   `llama-tokenize`、`llama-quantize` 均正常。已在目标机器上优先使用 `llama-server`；
   目标 A800/其他 CPU 上可再验证 llama-cli 是否可用。
2. 手写 GGUF（gguf python 库）在 llama.cpp decode 时出现 f16/f32 混乘 abort；
   **改用官方 `convert_hf_to_gguf.py`（HF→GGUF）后加载与生成完全正常**。
   因此导出流程固定为：checkpoint → hf_export.py（safetensors+config+tokenizer.model）
   → convert_hf_to_gguf.py → llama-quantize。
3. 烟雾测试路径：train 2 步 → checkpoint → HF 导出 → F16 GGUF → llama-server 加载 →
   编码 → 生成 ≥1 token → Token ID 范围校验 → KV cache 复用校验 → 记录时延。
4. 本机基准（Q5_K_M，196M）：1 线程 pp2048≈47.8 tok/s、tg≈36.9 tok/s；2 线程 tg≈67–73 tok/s；
   4 线程受后台任务挤占 tg≈2–4 tok/s（无效数据，以目标机正式基准为准）。
5. 训练与推理程序均未启动正式 4 小时预训练；`best_model.pt` 未触碰。
