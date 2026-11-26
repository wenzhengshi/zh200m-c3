#!/usr/bin/env bash
set -u
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604
B=/home/xmkjsoft/llama.cpp-src/build/bin
PORT=18085
PROMPT="新华社报道，北京和上海今天发布了最新的天气预报，提醒市民注意防寒保暖。"
OUT="$D/bench/ttft_contexts.json"
echo "[" > "$OUT"
FIRST=1
for C in 128 512 1024 2048; do
  "$B/llama-server" -m "$D/smoke/model-q5_k_m.gguf" -t 2 -c "$C" --port $PORT --no-webui > /tmp/srv_ttft.log 2>&1 &
  SRV=$!
  for i in $(seq 1 40); do
    sleep 1
    H=$(curl -s http://127.0.0.1:$PORT/health 2>/dev/null)
    case "$H" in *ok*) break;; esac
  done
  R=$(curl -s -X POST http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":8,\"temperature\":0,\"cache_prompt\":false,\"stream\":false}")
  T=$(python3 -c "import json,sys; d=json.load(sys.stdin); t=d.get('timings',{}); print(json.dumps({'ctx':$C,'prompt_ms':t.get('prompt_ms'),'predicted_per_token_ms':t.get('predicted_per_token_ms'),'predicted_per_second':t.get('predicted_per_second')}))" <<< "$R")
  if [ "$FIRST" = "1" ]; then echo "$T" >> "$OUT"; FIRST=0; else echo ",$T" >> "$OUT"; fi
  echo "ctx=$C $T"
  kill $SRV 2>/dev/null
  sleep 1
done
echo "]" >> "$OUT"
