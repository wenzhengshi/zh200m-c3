#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

check_sha256sums() {
  local dir="$1"
  local name="$2"
  echo "########## verify $name ##########"
  if [ -f "$dir/SHA256SUMS.txt" ]; then
    ( cd "$dir" && sha256sum -c SHA256SUMS.txt ) 2>&1
  else
    echo "NO SHA256SUMS.txt in $dir"
  fi
  echo
}

check_sha256sums audits/wiki2019zh_full_audit_v1 "full_audit_v1"
check_sha256sums audits/wiki2019zh_targeted_error_audit_v1 "targeted_error_audit_v1"
check_sha256sums audits/wiki2019zh_tier_dryrun_v1 "tier_dryrun_v1"
check_sha256sums audits/wiki2019zh_tier_dryrun_v2 "tier_dryrun_v2"
check_sha256sums inputs/cleaned/news2016zh "news2016zh freeze v3_2"

echo "########## manifest line counts ##########"
for m in audits/wiki2019zh_tier_dryrun_v1/classification_manifest.jsonl.gz \
         audits/wiki2019zh_tier_dryrun_v2/classification_manifest_v2.jsonl.gz; do
  echo "$m: $(gzip -dc "$m" | wc -l)"
done

echo
echo "########## news tier_a line count ##########"
echo "news_tier_a_permanent.jsonl: $(wc -l < inputs/cleaned/news2016zh/news_tier_a_permanent.jsonl)"
echo "final_quarantine_v3_2.jsonl: $(wc -l < inputs/cleaned/news2016zh/final_quarantine_v3_2.jsonl)"
echo "exact_body_duplicates_v3_2.jsonl: $(wc -l < inputs/cleaned/news2016zh/exact_body_duplicates_v3_2.jsonl)"

echo
echo "########## zhwikisource v6 line counts + hashes ##########"
echo "accepted_final.jsonl: $(wc -l < inputs/cleaned/zhwikisource_classics/accepted_final.jsonl)"
echo "quarantined.jsonl: $(wc -l < inputs/cleaned/zhwikisource_classics/quarantined.jsonl)"
echo "quarantined_manual_review.jsonl: $(wc -l < inputs/cleaned/zhwikisource_classics/quarantined_manual_review.jsonl)"
echo "trim_log.jsonl: $(wc -l < inputs/cleaned/zhwikisource_classics/trim_log.jsonl)"
echo "manual_review_decisions.tsv: $(wc -l < inputs/cleaned/zhwikisource_classics/manual_review_decisions.tsv)"
sha256sum inputs/cleaned/zhwikisource_classics/accepted_final.jsonl \
          inputs/cleaned/zhwikisource_classics/quarantined.jsonl \
          inputs/cleaned/zhwikisource_classics/quarantined_manual_review.jsonl \
          inputs/cleaned/zhwikisource_classics/trim_log.jsonl \
          inputs/cleaned/zhwikisource_classics/manual_review_decisions.tsv

echo
echo "########## best_model.pt (read-only) ##########"
ls -la /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/best_model.pt
sha256sum /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/best_model.pt

echo
echo "########## v3 search (whole home, targeted names) ##########"
find /home/xmkjsoft -iname "*tier_dryrun_v3*" 2>/dev/null | head -40
find /home/xmkjsoft -iname "*dryrun_v3*" 2>/dev/null | head -40
echo "(empty = no v3 artifacts)"
