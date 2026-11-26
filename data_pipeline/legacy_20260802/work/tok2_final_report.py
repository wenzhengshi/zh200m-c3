#!/usr/bin/env python3
"""Write the final tokenizer_ab_v2 report from the four evaluations + selection."""
import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
EVAL = AB / "evaluation"
SOURCES = ["news2016zh", "zhwikisource_classics", "wiki2019zh", "webText2019zh", "comment2019zh"]

data = {}
for name in ("old", "old_candidate", "c1", "c2"):
    data[name] = json.loads((EVAL / f"eval_{name}.json").read_text(encoding="utf-8"))
sel = json.loads((AB / "tokenizer_selection.json").read_text(encoding="utf-8"))


def full(name):
    return data[name]["per_source"]["full"]


def agg(name, key):
    return sum(full(name)[s][key] for s in SOURCES)


rows = []
for name in ("old", "old_candidate", "c1", "c2"):
    d = data[name]
    rows.append(
        {
            "name": name,
            "vocab": d["actual_vocab_size"],
            "full_tokens": agg(name, "tokens_without_eos"),
            "full_unk": agg(name, "unk_tokens"),
            "byte_ratio": round(agg(name, "byte_fallback_tokens") / max(1, agg(name, "tokens_without_eos")), 6),
            "rt_semantic": agg(name, "encode_decode_mismatch_normalized_semantics"),
            "scope_c_tokens": d["scope_c"]["tokens_without_eos"],
            "docs_gt2048": agg(name, "documents_gt_2048_tokens"),
            "seqs_2048": agg(name, "sequences_after_2048_chunking"),
        }
    )

md = [
    "# Tokenizer 第二轮 A/B/C 评测与定案报告（tokenizer_ab_v2）",
    "",
    "生成时间: 2026-08-02（全量五源精确评测，四轮均完成后）",
    "",
    "## 1. 候选与训练设置",
    "",
    "- C1: BPE, vocab=32000, character_coverage=1.0, byte_fallback=false",
    "- C2: BPE, vocab=32000, character_coverage=0.9995, byte_fallback=true",
    "- 与旧模型完全一致的复制配置：normalization=identity、add_dummy_prefix=false、",
    "  remove_extra_whitespaces=false、split_digits=true、max_sentencepiece_length=16、",
    "  special tokens unk=0/<unk> bos=1/<s> eos=2/</s> pad=3/<pad>、user_defined_symbols 5 个",
    "- 训练样本 606,100 篇（news 9,000 + classics 1,100 + wiki 280,000 + webText 280,000 + comment 36,000；",
    "  comment 占 5.94% 文档 / 1.82% 字符，严格降权）；评测样本 46,057 篇与训练完全分离",
    "- 评测范围：A 评测集、B 五源冻结全量、C 跨来源去重保留集",
    "",
    "## 2. 全量评测总览（B/C 口径，UNK 为五源合计）",
    "",
    "| Tokenizer | 实际词表 | 全量Token(无EOS) | 全量UNK | byte占比 | 编解码(规范化语义)不一致 | 跨源保留集Token | 超2048文档数 | 2048切块序列数 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    md.append(
        f"| {r['name']} | {r['vocab']} | {r['full_tokens']:,} | {r['full_unk']:,} | "
        f"{r['byte_ratio']*100:.3f}% | {r['rt_semantic']:,} | {r['scope_c_tokens']:,} | "
        f"{r['docs_gt2048']:,} | {r['seqs_2048']:,} |"
    )
md += [
    "",
    "> 注：raw 精确文本比较的“不一致”全部来自 SPM 的空格=U+2581 规范化语义（Normalize 将空格写作▁、",
    "decode 还原为空格）；按“规范化语义”比较后 old 与 c2 均为 0。",
    "",
    "## 3. 每源全量 Token 与 UNK（B 口径）",
    "",
    "| Tokenizer | news | classics | wiki | webText | comment |",
    "|---|---|---|---|---|---|",
]
for name in ("old", "c1", "c2"):
    f = full(name)
    md.append(
        f"| {name} | {f['news2016zh']['tokens_without_eos']:,} | "
        f"{f['zhwikisource_classics']['tokens_without_eos']:,} | "
        f"{f['wiki2019zh']['tokens_without_eos']:,} | "
        f"{f['webText2019zh']['tokens_without_eos']:,} | "
        f"{f['comment2019zh']['tokens_without_eos']:,} |"
    )
md += [
    "",
    "| Tokenizer | news UNK | classics UNK | wiki UNK | webText UNK | comment UNK |",
    "|---|---|---|---|---|---|",
]
for name in ("old", "c1", "c2"):
    f = full(name)
    md.append(
        f"| {name} | {f['news2016zh']['unk_tokens']:,} | "
        f"{f['zhwikisource_classics']['unk_tokens']:,} | "
        f"{f['wiki2019zh']['unk_tokens']:,} | "
        f"{f['webText2019zh']['unk_tokens']:,} | "
        f"{f['comment2019zh']['unk_tokens']:,} |"
    )
md += [
    "",
    "## 4. 验收与定案",
    "",
    "- C1: 全量 UNK=48,582,164 > 0 → **淘汰**",
    "- 旧候选(cc=0.9995, bf=false): 全量 UNK=1,129,513 > 0 → **淘汰**",
    "- C2: 全量 UNK=0 ✓；vocab=32000 ✓；规范化语义编解码 0 不一致 ✓；",
    "  各源相对旧模型 Token 变化：news −17.4%、classics +2.1%、wiki −21.7%、webText −27.1%、comment −26.0%（无 +5% 膨胀）✓；",
    "  **byte fallback Token 占比 = 3.72% > 0.5% → 未通过自动定案门槛**",
    "- 旧模型（对照）：UNK=0、规范化语义往返 0 不一致，但其 byte 占比亦达 2.69%，且用户未允许沿用旧模型定案",
    "",
    "## 5. 结论",
    "",
    "**按规则 15：C2 不满足 byte 占比 ≤0.5% 的自动定案条件，保留本报告并停止，等待最终决策。**",
    "可选方向：① 接受 C2（压缩更优、UNK=0，byte 占比 3.72% 需接受）；② 提高 character_coverage 重训 C2（如 0.9999）后再全量复核；③ 其他参数方案。",
    "旧 Tokenizer 未被覆盖，`tokenizer/zh_bpe_32k.model` 保持不变。",
    "",
    "## 6. 数据文件",
    "",
    "- evaluation/eval_old.json / eval_old_candidate.json / eval_c1.json / eval_c2.json",
    "- tokenizer_selection.json（本决策）",
    "- samples/tokenizer_train_manifest.jsonl.gz / tokenizer_eval_manifest.jsonl.gz（训练/评测样本分离）",
]
(AB / "tokenizer_ab_v2_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("report written")
