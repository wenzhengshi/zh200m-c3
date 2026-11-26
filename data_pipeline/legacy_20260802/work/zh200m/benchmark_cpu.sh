#!/usr/bin/env bash
set -u
MODEL="${1:?model gguf}"
OUT="${2:?out dir}"
LL="${3:-/home/xmkjsoft/llamacpp/llama-b10223}"
mkdir -p "$OUT"
LOG="$OUT/bench_raw.log"
: > "$LOG"
for T in 1 2 4; do
  for C in 128 512 1024 2048; do
    echo "### threads=$T ctx=$C" >> "$LOG"
    "$LL/llama-bench" -m "$MODEL" -t "$T" -c "$C" -p 128 -n 64 -r 1 >> "$LOG" 2>&1
  done
done
echo "raw benchmark saved to $LOG"
