#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "=== PWD ==="
pwd

for d in audits inputs scripts reference outputs logs; do
  echo
  echo "=== $d ==="
  find "$d" -maxdepth 2 -printf "%y %p -> %l\n" 2>/dev/null | head -120
done

echo
echo "=== outputs/wiki2019zh ==="
find outputs/wiki2019zh -maxdepth 2 -printf "%y %p -> %l\n" 2>/dev/null | head -80

echo
echo "=== logs ==="
ls -la logs/ 2>/dev/null

echo
echo "=== inputs/pending/wiki2019zh file count ==="
find -L inputs/pending/wiki2019zh -maxdepth 1 -type f | wc -l

echo
echo "=== tokenizer SHA256 (reference) ==="
sha256sum reference/zh_bpe_32k.model 2>/dev/null

echo
echo "=== model.json ==="
cat reference/model.json 2>/dev/null
