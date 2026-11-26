import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_*"))[-1]
print("dir:", D)

final_totals = {
    "documents": 11142268,
    "tokens_without_eos": 1848863728,
    "unk_rate": 0.0,
    "per_source": {
        "news2016zh": {"docs": 10881, "tokens": 16620802},
        "zhwikisource_classics": {"docs": 1276, "tokens": 3964180},
        "wiki2019zh": {"docs": 936076, "tokens": 352889183},
        "webText2019zh": {"docs": 3923468, "tokens": 869494212},
        "comment2019zh": {"docs": 6270567, "tokens": 605895351},
    },
    "note": "跨来源去重后保留集，使用现有zh_bpe_32k精确统计；UNK=0。",
}

rp = D / "tokenizer_ab_report.json"
r = json.loads(rp.read_text(encoding="utf-8"))
r["existing_tokenizer_full_corpus_post_cross_source_dedup"] = final_totals
rp.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

md = (D / "tokenizer_ab_report.md").read_text(encoding="utf-8")
md += """
## 跨来源去重后的全量精确统计（现有 zh_bpe_32k，UNK=0）

| 数据源 | 文档 | Token(无EOS) |
|---|---|---|
| news2016zh | 10,881 | 16,620,802 |
| zhwikisource_classics | 1,276 | 3,964,180 |
| wiki2019zh | 936,076 | 352,889,183 |
| webText2019zh | 3,923,468 | 869,494,212 |
| comment2019zh | 6,270,567 | 605,895,351 |
| **合计** | **11,142,268** | **1,848,863,728** |
"""
(D / "tokenizer_ab_report.md").write_text(md, encoding="utf-8")
print("finalized")
