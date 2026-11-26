#!/usr/bin/env bash
set -euo pipefail
OUT=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs

# remove empty/partial crashed dirs (validated absolute paths inside outputs)
rm -rf "$OUT/cross_source/cross_source_dedup_v1_20260802_071149"
rm -rf "$OUT/cross_source/cross_source_dedup_v1_20260802_071227"
rm -rf "$OUT/cross_source/tokenizer_ab_20260802_084324"
rm -rf "$OUT/cross_source/tokenizer_ab_20260802_084449"
rm -rf "$OUT/wiki2019zh/dedup_v1_20260802_001042"
rm -rf "$OUT/wiki2019zh/dedup_v1_20260802_001153"
rm -rf "$OUT/wiki2019zh/cleaning_preview_v1_20260801_225803"
rm -rf "$OUT/wiki2019zh/cleaning_preview_v1_20260801_225814"
rm -rf "$OUT/webText2019zh/dedup_v1_20260802_013130"
echo "removed crashed dirs"

GEN=/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work/gen_sha256sums.sh
for d in \
  webText2019zh/full_audit_v1_20260802_005157 \
  webText2019zh/tier_dryrun_v1_20260802_010603 \
  webText2019zh/cleaning_preview_v1_20260802_012519 \
  webText2019zh/curated_candidate_v1_20260802_012519 \
  comment2019zh/full_audit_v1_20260802_062730 \
  comment2019zh/curated_candidate_v1_20260802_062730 \
  wiki2019zh/cleaning_preview_v1_20260801_225824 \
  wiki2019zh/curated_candidate_v1_20260801_230837 \
  wiki2019zh/curated_candidate_v2_20260801_232734 \
  wiki2019zh/curated_candidate_v3_20260801_233900 \
  wiki2019zh/curated_candidate_v4_20260801_234934 \
  wiki2019zh/tier_dryrun_v3_1_package_20260801_223302 \
  wiki2019zh/wiki2019zh_freeze_v1_20260802_004344 ; do
  echo "==== $d"
  bash "$GEN" "$OUT/$d" | tail -2
done
echo "ALL DONE"
