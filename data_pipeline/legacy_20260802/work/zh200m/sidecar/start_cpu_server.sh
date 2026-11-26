#!/usr/bin/env bash
set -euo pipefail
# llama.cpp 本地推理 sidecar（LLaMA server）
LLAMA_DIR="${LLAMA_DIR:-/home/xmkjsoft/llamacpp/llama-b10223}"
MODEL="${MODEL:-/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1/model-q5_k_m.gguf}"
THREADS="${THREADS:-4}"
CTX="${CTX:-2048}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  -t "$THREADS" \
  -c "$CTX" \
  --host "$HOST" \
  --port "$PORT" \
  --no-webui
