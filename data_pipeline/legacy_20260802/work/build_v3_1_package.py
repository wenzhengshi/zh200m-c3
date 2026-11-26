#!/usr/bin/env python3
"""Build the minimal v3.1 fix package from the verified v3 classifier."""
import hashlib
import os
from datetime import datetime, timezone, timedelta

SRC = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/wiki2019zh_tier_dryrun_v3_fixed/classify_wiki2019zh_tier_dryrun_v3.py"
WS = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace"
TS = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
PKG = os.path.join(WS, f"outputs/wiki2019zh/tier_dryrun_v3_1_package_{TS}")
os.makedirs(PKG, exist_ok=False)

with open(SRC, encoding="utf-8") as f:
    code = f.read()

# ---- minimal edits ----
edits = [
    # version string
    ('"version": "tier_dryrun_v3"', '"version": "tier_dryrun_v3_1"'),
    # usage text
    (
        '"用法：classify_wiki2019zh_tier_dryrun_v3.py "\n        "<INPUT_DIR> <V2_MANIFEST.jsonl.gz> <OUTPUT_DIR>"',
        '"用法：classify_wiki2019zh_tier_dryrun_v3_1.py "\n        "<INPUT_DIR> <PREV_MANIFEST.jsonl.gz> <OUTPUT_DIR>"',
    ),
    (
        '    raise SystemExit(f"v1 manifest不存在：{V2_MANIFEST}")',
        '    raise SystemExit(f"输入manifest不存在：{V2_MANIFEST}")',
    ),
    # add list_term_first right after list_term definition
    (
        "    list_term = bool(LIST_TERM_RE.search(text[:300]))\n",
        "    list_term = bool(LIST_TERM_RE.search(text[:300]))\n"
        "    list_term_first = bool(LIST_TERM_RE.search(first))\n",
    ),
    # pure_list rule: keyword must be in the first paragraph
    (
        "    elif (\n        list_term\n        and character_count < 120\n        and not first_has_substance\n        and not trailing_empty_heading_cleanable\n    ):",
        "    elif (\n        list_term_first\n        and character_count < 120\n        and not first_has_substance\n        and not trailing_empty_heading_cleanable\n    ):",
    ),
    # digit-comma-safe tail comma counting
    (
        '                        sum(last.count(mark) for mark in ("，", ",", "、")) >= 2',
        "                        count_comma_marks(last) >= 2",
    ),
    # output file names
    ("classification_manifest_v3.jsonl.gz", "classification_manifest_v3_1.jsonl.gz"),
    ("summary_tier_dryrun_v3.json", "summary_tier_dryrun_v3_1.json"),
    ("tier_counts_v3.tsv", "tier_counts_v3_1.tsv"),
    ("transition_counts_v2_to_v3.tsv", "transition_counts_v3_to_v3_1.tsv"),
    ("reason_counts_v3.tsv", "reason_counts_v3_1.tsv"),
    ("repair_action_counts_v3.tsv", "repair_action_counts_v3_1.tsv"),
]
for old, new in edits:
    if old not in code:
        raise SystemExit(f"PATTERN NOT FOUND:\n{old}")
    code = code.replace(old, new, 1)

# add helper after TRAILING_FRAGMENT_RE block
anchor = 'TRAILING_FRAGMENT_RE = re.compile(\n    r"(?:以及|并且|而且|或者|以及其|的|和|与|及|为|是|在|于|由)$"\n)\n'
helper = anchor + (
    "\n"
    "def count_comma_marks(s: str) -> int:\n"
    '    """Count comma-like marks, ignoring ASCII commas inside digit groups (e.g. 27,760)."""\n'
    '    s = re.sub(r"(?<=\\d),(?=\\d)", "", s)\n'
    '    return s.count("，") + s.count("、") + s.count(",")\n'
)
if anchor not in code:
    raise SystemExit("TRAILING_FRAGMENT_RE anchor not found")
code = code.replace(anchor, helper, 1)

classifier_path = os.path.join(PKG, "classify_wiki2019zh_tier_dryrun_v3_1.py")
with open(classifier_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(code)

# ---- RULES ----
rules = """wiki2019zh tier dry-run v3.1

基于已校验的v3干跑与高风险转移审计，做两处最小规则修正（不重构分类器整体设计）：

修正1: pure_list_or_heading_stub 收窄触发范围
- v3 在 text[:300] 内任意位置命中"列表/名单/年表/一览/目录/作品表"即触发，
  导致完整短词条（首段无特定系词）被误杀（A->DROP 73篇全部为多段落完整短文）。
- v3.1 要求关键词必须出现在"首段"，仅首段带列表关键词的短壳才判 pure_list_or_heading_stub。
- 预期回收: 183 篇（其中73篇来自v3的A->DROP，95篇来自B，15篇来自DROP）。

修正2: definite_incomplete_tail 的逗号计数排除数字千分位逗号
- v3 将数字内部的 ASCII 逗号（如 27,760）计入">=2个逗号"启发，
  导致末尾仅缺句号的完整长文被误判为截断（已确认43篇）。
- v3.1 先剥离数字组内逗号（(?<=\\d),(?=\\d)）再计数。
- 预期回收: 43 篇完整文。

其余规则与v3完全一致：
1. “导演、主演、作者”正常位于句尾时不判语义空洞；仅实体确实缺失时命中。
2. 坐标/面积/人口局部缺字段：有正文可保留进B并记录局部清理；损坏密集进REVIEW；无可抢救才DROP。
3. “面积和人口皆未知”是完整陈述。
4. 正文有效、末尾仅“下辖以下地区：”等空标题进B，只删末尾标题。
5. 公式/定义后内容缺失继续REVIEW。
6. “某街道，是下辖的一个乡镇级行政单位”进B并规范化，不进A。
7. 缺标题或主语：短文DROP、长文REVIEW。
8. 尾部残缺只识别逗号/连接词截断、孤立列表、人口图示，不因单纯缺句号批量命中。

本轮只干跑分类，不修改源数据，不生成训练文件。
"""
with open(os.path.join(PKG, "RULES_TIER_DRYRUN_V3_1.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(rules)

# ---- README ----
readme = f"""v3.1 修复包使用说明

来源: 基于已校验的 v3 干跑结果（wiki2019zh_tier_dryrun_v3_20260801_221052，SHA256SUMS 69/69 OK）
      与高风险转移审计发现的两处系统误判。
位置: outputs/wiki2019zh/tier_dryrun_v3_1_package_{TS}/

执行:
  bash run_wiki2019zh_tier_dryrun_v3_1.sh

脚本会:
- 验证 canonical ZIP SHA256 (b42e010f...);
- 验证 v3 审计目录全部 SHA256 与 manifest gzip 完整性;
- py_compile v3.1 分类器;
- 对 1,018,692 篇执行 v3.1 全量干跑，输出到新时间戳目录
  data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_<时间戳>;
- 硬校验总词条数 / Tier和 / 转移矩阵和 = 1,018,692;
- 生成 v3->v3.1 转移矩阵、高风险转移样本、重点原因样本、compact复核报告;
- 生成并自校验 SHA256SUMS;
- 复制复核报告到 Windows 下载目录。

不修改原始语料，不生成正式训练文件，不覆盖 v1/v2/v3。
"""
with open(os.path.join(PKG, "README.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(readme)

# ---- runner ----
runner = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/jicreate_chinese_lm_0.2b_runtime"
INPUT="$HOME/modern_chinese_data/cluecorpus_small/extracted/wiki2019zh"
ZIP="$HOME/modern_chinese_data/cluecorpus_small/wiki2019zh_corpus.zip"

V3_DIR="$ROOT/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_20260801_221052"
V3_MANIFEST="$V3_DIR/classification_manifest_v3.jsonl.gz"

EXPECTED_ZIP_SHA256="b42e010ffd111de1a010e43db043122ce55ffdd2f93ab4b7712d96b7d2043070"

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
CLASSIFIER="$SCRIPT_DIR/classify_wiki2019zh_tier_dryrun_v3_1.py"
RULES="$SCRIPT_DIR/RULES_TIER_DRYRUN_V3_1.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/data/cpt_modern_v1/audits/wiki2019zh_tier_dryrun_v3_1_${{RUN_ID}}"

cd "$ROOT"
source .venv_core_audit/bin/activate

mkdir -p "$OUT_DIR/samples"

echo "============================================================"
echo "1. 验证输入、v3结果和v3.1脚本"
echo "============================================================"

printf '%s  %s\\n' \\
  "$EXPECTED_ZIP_SHA256" \\
  "$ZIP" |
sha256sum -c - |
tee "$OUT_DIR/input_zip_verification.txt"

(
  cd "$V3_DIR"
  sha256sum -c SHA256SUMS.txt
)

gzip -t "$V3_MANIFEST"
python3 -m py_compile "$CLASSIFIER"

echo "v3 manifest完整性：通过"
echo "v3.1 Python语法检查：通过"

cp "$CLASSIFIER" "$OUT_DIR/classify_wiki2019zh_tier_dryrun_v3_1.py"
cp "$RULES" "$OUT_DIR/RULES_TIER_DRYRUN_V3_1.txt"

echo
echo "============================================================"
echo "2. 执行v3.1全量干跑"
echo "============================================================"

python3 -u \\
  "$OUT_DIR/classify_wiki2019zh_tier_dryrun_v3_1.py" \\
  "$INPUT" \\
  "$V3_MANIFEST" \\
  "$OUT_DIR" |
tee "$OUT_DIR/run.log"

echo
echo "============================================================"
echo "3. 验证v3.1结果"
echo "============================================================"

python3 - "$OUT_DIR/summary_tier_dryrun_v3_1.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

expected = 1_018_692
actual = data["total_documents"]
if actual != expected:
    raise SystemExit(f"总词条数异常：{{actual}} != {{expected}}")
tier_total = sum(data["tier_counts"].values())
if tier_total != expected:
    raise SystemExit(f"Tier总数异常：{{tier_total}} != {{expected}}")
transition_total = sum(data["transition_counts"].values())
if transition_total != expected:
    raise SystemExit(f"转移矩阵总数异常：{{transition_total}} != {{expected}}")
print("v3.1总词条数检查：通过")
print(f"total_documents={{actual}}")
print(f"tier_total={{tier_total}}")
print(f"transition_total={{transition_total}}")
PY

gzip -t "$OUT_DIR/classification_manifest_v3_1.jsonl.gz"
echo "v3.1 manifest gzip完整性：通过"

echo
echo "============================================================"
echo "4. 生成紧凑人工复核报告"
echo "============================================================"

REPORT="$OUT_DIR/compact_transition_review_v3_1.txt"
: > "$REPORT"

{{
  echo "wiki2019zh v3.1 高风险转移人工复核"
  echo "v3_1_dir=$OUT_DIR"
  echo

  echo "================ Tier统计 ================"
  cat "$OUT_DIR/tier_counts_v3_1.tsv"
  echo

  echo "================ v3到v3.1转移统计 ================"
  cat "$OUT_DIR/transition_counts_v3_to_v3_1.tsv"
  echo

  echo "================ 清理动作统计 ================"
  cat "$OUT_DIR/repair_action_counts_v3_1.tsv"
  echo

  for NAME in \\
    A_to_DROP \\
    A_to_REVIEW \\
    A_to_B \\
    B_to_A \\
    B_to_DROP \\
    B_to_REVIEW \\
    DROP_to_A \\
    DROP_to_B \\
    DROP_to_REVIEW \\
    REVIEW_to_A \\
    REVIEW_to_B \\
    REVIEW_to_DROP
  do
    FILE="$OUT_DIR/samples/transition_${{NAME}}_sample.txt"
    echo
    echo "######################################################################"
    echo "transition=$NAME"
    echo "######################################################################"
    if [ -f "$FILE" ]; then
      awk '
        BEGIN {{ samples = 0 }}
        /^sample=/ {{
          samples++
          if (samples > 8) exit
        }}
        {{ print }}
      ' "$FILE"
    else
      echo "没有该转移样本。"
    fi
  done

  for REASON in \\
    pure_list_or_heading_stub \\
    definite_incomplete_tail \\
    local_semantic_or_numeric_hole_cleanable \\
    dense_or_core_semantic_numeric_hole \\
    missing_parent_admin_definition_cleanable \\
    substantive_body_with_empty_tail_heading \\
    core_content_missing_after_heading \\
    list_intro_without_list \\
    short_missing_title_or_subject \\
    long_missing_title_or_subject
  do
    FILE="$OUT_DIR/samples/reason_${{REASON}}_sample.txt"
    echo
    echo "######################################################################"
    echo "reason=$REASON"
    echo "######################################################################"
    if [ -f "$FILE" ]; then
      awk '
        BEGIN {{ samples = 0 }}
        /^sample=/ {{
          samples++
          if (samples > 6) exit
        }}
        {{ print }}
      ' "$FILE"
    else
      echo "没有该原因样本。"
    fi
  done
}} > "$REPORT"

echo
echo "============================================================"
echo "5. 生成并验证SHA256SUMS"
echo "============================================================"

(
  cd "$OUT_DIR"
  find . \\
    -type f \\
    ! -name SHA256SUMS.txt \\
    -printf '%P\\n' |
  sort |
  xargs -r sha256sum \\
    > SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt
)

sha256sum "$REPORT" |
tee "$REPORT.sha256"

echo
echo "============================================================"
echo "6. 核心结果"
echo "============================================================"

cat "$OUT_DIR/tier_counts_v3_1.tsv"
echo
cat "$OUT_DIR/transition_counts_v3_to_v3_1.tsv"
echo
cat "$OUT_DIR/repair_action_counts_v3_1.tsv"

DOWNLOAD_DIR="/mnt/c/Users/xmkjsoft/Downloads"
if [ -d "$DOWNLOAD_DIR" ]; then
  cp "$REPORT" "$DOWNLOAD_DIR/"
  cp "$REPORT.sha256" "$DOWNLOAD_DIR/"
  echo
  echo "复核报告已复制到："
  echo 'C:\\Users\\xmkjsoft\\Downloads\\compact_transition_review_v3_1.txt'
fi

echo
echo "============================================================"
echo "结果目录"
echo "============================================================"
echo "$OUT_DIR"
"""
runner_path = os.path.join(PKG, "run_wiki2019zh_tier_dryrun_v3_1.sh")
with open(runner_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(runner)
os.chmod(runner_path, 0o755)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("package dir:", PKG)
for name in sorted(os.listdir(PKG)):
    p = os.path.join(PKG, name)
    print(" ", sha256_file(p), name)
