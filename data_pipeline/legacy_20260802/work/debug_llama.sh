#!/usr/bin/env bash
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604
GGML_LOG_LEVEL=1 LLAMA_LOG_LEVEL=1 /home/xmkjsoft/llamacpp/llama-b10223/llama-cli \
  -m "$D/smoke/model-f16.gguf" -p "你" -n 1 --temp 0 -t 2 -c 128 --no-display-prompt \
  > /tmp/llama_debug.txt 2>&1
echo "rc=$?"
grep -E "llama_model_load|n_embd|n_head|n_layer|rope|freq|tensor|vocab|arch|head_count|ffn|kv" /tmp/llama_debug.txt | head -40
