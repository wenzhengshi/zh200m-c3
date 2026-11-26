#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "########## v2 script header (docstring) ##########"
head -60 audits/wiki2019zh_tier_dryrun_v2/classify_wiki2019zh_tier_dryrun_v2.py

echo
echo "########## v2 script: tail-incomplete logic ##########"
grep -n -A 6 -B 2 "possible_incomplete_tail\|incomplete_tail\|句号\|missing_punct" audits/wiki2019zh_tier_dryrun_v2/classify_wiki2019zh_tier_dryrun_v2.py | head -80

echo
echo "########## v2 script: 主演/导演/作者/街道 logic ##########"
grep -n "主演\|导演\|作者\|乡级行政\|行政区划\|下辖" audits/wiki2019zh_tier_dryrun_v2/classify_wiki2019zh_tier_dryrun_v2.py | head -40

echo
echo "########## outputs dirs empty check ##########"
for d in outputs/wiki2019zh outputs/webText2019zh outputs/comment2019zh outputs/cross_source; do
  echo "$d: $(ls -A "$d" | wc -l) entries"
done

echo
echo "########## v2 samples: possible_incomplete_tail (first 12) ##########"
head -40 audits/wiki2019zh_tier_dryrun_v2/samples/reason_possible_incomplete_tail_sample.txt

echo
echo "########## v2 samples: transition A->B (first 12) ##########"
head -40 audits/wiki2019zh_tier_dryrun_v2/samples/transition_A_to_B_sample.txt
