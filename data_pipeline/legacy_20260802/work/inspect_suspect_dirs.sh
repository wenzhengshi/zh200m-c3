#!/usr/bin/env bash
cd /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs || exit 1
for d in \
  cross_source/cross_source_dedup_v1_20260802_071149 \
  cross_source/cross_source_dedup_v1_20260802_071227 \
  cross_source/tokenizer_ab_20260802_084324 \
  cross_source/tokenizer_ab_20260802_084449 \
  wiki2019zh/dedup_v1_20260802_001042 \
  wiki2019zh/dedup_v1_20260802_001153 \
  wiki2019zh/cleaning_preview_v1_20260801_225803 \
  wiki2019zh/cleaning_preview_v1_20260801_225814 ; do
  echo "==== $d"
  ls -la "$d" 2>/dev/null | head -8
done
