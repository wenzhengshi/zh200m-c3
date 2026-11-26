#!/usr/bin/env bash
set -u
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604
B=/home/xmkjsoft/llama.cpp-src/build/bin
OUT="$D/bench"
: > "$OUT/bench_raw_clean.log"
for T in 1 2 4; do
  for P in 128 512 1024 2048; do
    echo "### threads=$T prompt=$P" >> "$OUT/bench_raw_clean.log"
    timeout 300 "$B/llama-bench" -m "$D/smoke/model-q5_k_m.gguf" -t "$T" -p "$P" -n 32 -r 1 \
      >> "$OUT/bench_raw_clean.log" 2>&1 || echo "FAILED t=$T p=$P" >> "$OUT/bench_raw_clean.log"
  done
done
echo "clean bench done"
