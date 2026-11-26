#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?checkpoint}"
CONFIG="${2:?config}"
SP="${3:?sp model}"
OUT="${4:?out dir}"
LL="${5:-/home/xmkjsoft/llamacpp/llama-b10223}"
CONVERT="${6:-/home/xmkjsoft/llama.cpp-src/convert_hf_to_gguf.py}"
mkdir -p "$OUT"
HF="$OUT/hf_model"
python3 "$(dirname "$0")/hf_export.py" --checkpoint "$CKPT" --config "$CONFIG" --sp-model "$SP" --out-dir "$HF"
python3 "$CONVERT" --outfile "$OUT/model-f16.gguf" --outtype f16 "$HF"
python3 "$(dirname "$0")/export_onnx.py" --checkpoint "$CKPT" --config "$CONFIG" --out-fp32 "$OUT/model_fp32.onnx" --out-int8 "$OUT/model_int8.onnx"
"$LL/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-q8_0.gguf" Q8_0
"$LL/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-q5_k_m.gguf" Q5_K_M
"$LL/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-q4_k_m.gguf" Q4_K_M
echo "exports done in $OUT"
