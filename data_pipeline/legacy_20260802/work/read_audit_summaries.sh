#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "########## v1 rules ##########"
cat audits/wiki2019zh_tier_dryrun_v1/RULES_TIER_DRYRUN_V1.txt
echo
echo "########## v1 summary ##########"
cat audits/wiki2019zh_tier_dryrun_v1/summary_tier_dryrun_v1.json
echo
echo "########## v1 tier counts ##########"
cat audits/wiki2019zh_tier_dryrun_v1/tier_counts.tsv
echo
echo "########## v1 reason counts ##########"
cat audits/wiki2019zh_tier_dryrun_v1/reason_counts.tsv
echo
echo "########## v2 rules ##########"
cat audits/wiki2019zh_tier_dryrun_v2/RULES_TIER_DRYRUN_V2.txt
echo
echo "########## v2 summary ##########"
cat audits/wiki2019zh_tier_dryrun_v2/summary_tier_dryrun_v2.json
echo
echo "########## v2 tier counts ##########"
cat audits/wiki2019zh_tier_dryrun_v2/tier_counts_v2.tsv
echo
echo "########## v2 reason counts ##########"
cat audits/wiki2019zh_tier_dryrun_v2/reason_counts_v2.tsv
echo
echo "########## v2 repair action counts ##########"
cat audits/wiki2019zh_tier_dryrun_v2/repair_action_counts_v2.tsv
echo
echo "########## v1->v2 transition counts ##########"
cat audits/wiki2019zh_tier_dryrun_v2/transition_counts_v1_to_v2.tsv
