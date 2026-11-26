#!/usr/bin/env bash
cd /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs || exit 1
echo "--- key dirs ---"
ls -d wiki2019zh/wiki2019zh_freeze_v2_* \
  webText2019zh/webText2019zh_freeze_v1_* \
  comment2019zh/comment2019zh_freeze_v1_* \
  cross_source/cross_source_dedup_v1_* \
  cross_source/tokenizer_ab_* 2>/dev/null
echo "--- SHA presence ---"
for d in \
  wiki2019zh/wiki2019zh_freeze_v2_20260802_004810 \
  webText2019zh/webText2019zh_freeze_v1_20260802_062134 \
  comment2019zh/comment2019zh_freeze_v1_20260802_070511 \
  cross_source/cross_source_dedup_v1_20260802_082519 \
  cross_source/tokenizer_ab_20260802_084554 ; do
  printf "%s: " "$d"
  if [ -f "$d/SHA256SUMS.txt" ]; then echo OK; else echo MISSING; fi
done
echo "--- tokenizer report ---"
ls cross_source/tokenizer_ab_20260802_084554/ | head -8
