#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "########## full_audit_v1 summary ##########"
cat audits/wiki2019zh_full_audit_v1/summary_full_audit_v1.json
echo
echo "########## full_audit_v1 input comparison ##########"
cat audits/wiki2019zh_full_audit_v1/input_comparison_summary.json
echo
echo "########## targeted_error summary ##########"
cat audits/wiki2019zh_targeted_error_audit_v1/summary_targeted_error_audit_v1.json
echo
echo "########## input zip verification files ##########"
cat audits/wiki2019zh_tier_dryrun_v1/input_zip_verification.txt
cat audits/wiki2019zh_tier_dryrun_v2/input_zip_verification.txt
echo
echo "########## v2 run.log ##########"
cat audits/wiki2019zh_tier_dryrun_v2/run.log
echo
echo "########## compact transition review v2 hash check ##########"
cat audits/wiki2019zh_tier_dryrun_v2/compact_transition_review_v2.txt.sha256
echo "computed: $(sha256sum audits/wiki2019zh_tier_dryrun_v2/compact_transition_review_v2.txt | cut -d' ' -f1)"
echo
echo "########## build_workspace.log ##########"
cat logs/build_workspace.log
echo
echo "########## cpt_modern_v1_all top-level ##########"
ls -la scripts/cpt_modern_v1_all/ | head -80
