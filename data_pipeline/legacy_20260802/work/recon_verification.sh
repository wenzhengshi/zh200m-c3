#!/usr/bin/env bash
set -u
WS=/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace
cd "$WS" || exit 1

echo "=== symlink target existence ==="
while IFS=$'\t' read -r entry target; do
  if [ -e "$target" ]; then
    echo "OK   $entry -> $target"
  else
    echo "BROKEN $entry -> $target"
  fi
done < <(tail -n +2 SOURCE_PATHS.tsv)

echo
echo "=== news2016zh freeze dir ==="
ls -la inputs/cleaned/news2016zh/

echo
echo "=== zhwikisource_classics_v6 dir ==="
ls -la inputs/cleaned/zhwikisource_classics/

echo
echo "=== audits wiki2019zh_tier_dryrun_v1 ==="
ls -la audits/wiki2019zh_tier_dryrun_v1/

echo
echo "=== audits wiki2019zh_tier_dryrun_v2 ==="
ls -la audits/wiki2019zh_tier_dryrun_v2/

echo
echo "=== audits full_audit_v1 ==="
ls -la audits/wiki2019zh_full_audit_v1/

echo
echo "=== audits targeted_error_audit_v1 ==="
ls -la audits/wiki2019zh_targeted_error_audit_v1/

echo
echo "=== search for v3 anywhere ==="
find /home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime -iname "*v3*" -maxdepth 6 2>/dev/null | head -60

echo
echo "=== canonical ZIPs ==="
sha256sum inputs/archives/wiki2019zh_corpus.zip inputs/archives/webText2019zh_corpus.zip inputs/archives/comment2019zh_corpus.zip 2>/dev/null

echo
echo "=== pending wiki sample files ==="
ls -la inputs/pending/wiki2019zh/ | head -12
