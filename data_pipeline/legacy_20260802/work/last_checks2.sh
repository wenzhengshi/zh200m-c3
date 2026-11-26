#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "########## scripts/wiki2019zh_tier_dryrun_v2_fixed contents ##########"
ls -la scripts/wiki2019zh_tier_dryrun_v2_fixed/ 2>/dev/null

echo
echo "########## v2 manifest first record ##########"
gzip -dc audits/wiki2019zh_tier_dryrun_v2/classification_manifest_v2.jsonl.gz | head -1

echo
echo "########## pending webText/comment file counts (read-only) ##########"
echo "webText2019zh files: $(find -L inputs/pending/webText2019zh -maxdepth 1 -type f | wc -l)"
echo "comment2019zh files: $(find -L inputs/pending/comment2019zh -maxdepth 1 -type f | wc -l)"

echo
echo "########## reference hash search (targeted, non-corpus paths) ##########"
grep -rn "a9bb7549ad1d537e09f532eb4930dd7b105968f71bbb90ea24f3f884cdeb4141\|5252766c2947c5b7edd054d44edc9e55650efb8ac1095a4664e6e77bf71f8643" \
  /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/audits \
  /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/cpt_modern_v1/*.md \
  /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace 2>/dev/null | head -10
echo "(done)"

echo
echo "########## v2 script rule block lines 60-140 ##########"
sed -n '60,140p' audits/wiki2019zh_tier_dryrun_v2/classify_wiki2019zh_tier_dryrun_v2.py
