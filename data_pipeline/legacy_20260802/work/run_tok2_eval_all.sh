#!/usr/bin/env bash
set -u
source /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/.venv_core_audit/bin/activate
SCRIPT=/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work/tok2_eval.py
for TOK in old old_candidate c1 c2; do
  echo "RUN $TOK" >> /tmp/tok2_eval_v2.log
  TOK="$TOK" python3 -u "$SCRIPT" >> /tmp/tok2_eval_v2.log 2>&1
  echo "DONE $TOK" >> /tmp/tok2_eval_v2.log
done
echo "ALL_DONE" >> /tmp/tok2_eval_v2.log
