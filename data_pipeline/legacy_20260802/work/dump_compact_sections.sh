#!/usr/bin/env bash
R=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052/compact_transition_review_v3.txt
sed -n "$1,$2p" "$R" | tr "=" "-"
