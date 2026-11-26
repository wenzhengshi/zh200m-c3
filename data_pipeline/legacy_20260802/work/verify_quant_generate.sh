#!/usr/bin/env bash
set -u
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604
B=/home/xmkjsoft/llama.cpp-src/build/bin
PORT=18082
for M in model-q8_0.gguf model-q5_k_m.gguf model-q4_k_m.gguf; do
  $B/llama-server -m "$D/smoke/$M" -t 2 -c 256 --port $PORT --no-webui > /tmp/srv_$M.log 2>&1 &
  SRV=$!
  for i in $(seq 1 30); do
    sleep 1
    H=$(curl -s http://127.0.0.1:$PORT/health 2>/dev/null)
    case "$H" in *ok*) break;; esac
  done
  R=$(curl -s -X POST http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d '{"prompt":"新华社报道，","n_predict":3,"temperature":0,"cache_prompt":true,"stream":false,"return_tokens":true}')
  echo "$M -> $R" | head -c 400
  echo
  kill $SRV 2>/dev/null
  sleep 1
done
