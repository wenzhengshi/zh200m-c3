#!/usr/bin/env bash
set -u
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604
B=/home/xmkjsoft/llama.cpp-src/build/bin
PORT=18084
MODEL="$D/smoke/model-q5_k_m.gguf"
CTX=2048
T0=$(date +%s.%N)
"$B/llama-server" -m "$MODEL" -t 2 -c "$CTX" --port $PORT --no-webui > /tmp/srv_rss.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do
  sleep 1
  H=$(curl -s http://127.0.0.1:$PORT/health 2>/dev/null)
  case "$H" in *ok*) break;; esac
done
T1=$(date +%s.%N)
LOAD_MS=$(echo "($T1 - $T0) * 1000" | bc -l)
RSS_KB=$(grep VmHWM /proc/$SRV/status 2>/dev/null | awk '{print $2}')
# first completion timing (TTFT approx)
curl -s -X POST http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" \
  -d '{"prompt":"新华社报道，","n_predict":16,"temperature":0,"cache_prompt":false,"stream":false}' > /tmp/srv_rss_resp.json 2>/dev/null
kill $SRV 2>/dev/null
echo "load_ms=$LOAD_MS"
echo "peak_rss_kb=$RSS_KB"
python3 - <<'EOF'
import json
d = json.load(open("/tmp/srv_rss_resp.json"))
t = d.get("timings", {})
print("ttft_ms=", round(t.get("prompt_ms", 0) + t.get("predicted_per_token_ms", 0), 2))
print("prompt_ms=", t.get("prompt_ms"), "predicted_per_second=", t.get("predicted_per_second"))
EOF
# analytic KV cache: n_layer=20, n_kv=2, head_dim=64, ctx=2048, f16 (2 bytes)
echo "kv_cache_mb_analytic_f16=$(python3 -c "print(20*2*64*2048*2/1048576)")"
