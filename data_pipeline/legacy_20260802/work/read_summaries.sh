#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "########## news2016zh summary ##########"
cat inputs/cleaned/news2016zh/summary_news_freeze_v3_2.json
echo
echo "########## news2016zh SHA256SUMS ##########"
cat inputs/cleaned/news2016zh/SHA256SUMS.txt
echo
echo "########## news2016zh freeze notice ##########"
cat inputs/cleaned/news2016zh/PERMANENT_FREEZE_NOTICE.txt

echo
echo "########## zhwikisource finalize_report.json ##########"
cat inputs/cleaned/zhwikisource_classics/finalize_report.json
echo
echo "########## zhwikisource token_audit ##########"
ls -la inputs/cleaned/zhwikisource_classics/token_audit/
for f in inputs/cleaned/zhwikisource_classics/token_audit/*; do echo "--- $f ---"; cat "$f"; echo; done
echo
echo "########## zhwikisource web_residue_audit ##########"
ls -la inputs/cleaned/zhwikisource_classics/web_residue_audit/
for f in inputs/cleaned/zhwikisource_classics/web_residue_audit/*; do echo "--- $f ---"; head -50 "$f"; echo; done
