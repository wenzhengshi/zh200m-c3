#!/usr/bin/env bash
set -u
D=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052

for f in "$@"; do
  echo "FILE: $f"
  awk '
    BEGIN { samples = 0 }
    /^total_candidates=/ { print; next }
    /^sample=/ { samples++; if (samples > 12) exit }
    { print }
  ' "$D/$f" | tr "=" "-"
  echo
done
