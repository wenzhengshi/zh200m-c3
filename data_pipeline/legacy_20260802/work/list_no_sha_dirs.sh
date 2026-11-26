#!/usr/bin/env bash
cd /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs || exit 1
find . -maxdepth 3 -type d | sort | while read -r d; do
  if [ -d "$d" ] && [ ! -f "$d/SHA256SUMS.txt" ]; then
    echo "NO_SHA $d"
  fi
done
