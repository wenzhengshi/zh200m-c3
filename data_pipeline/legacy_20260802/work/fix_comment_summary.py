import json
from pathlib import Path

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
D = sorted((WS / "outputs/comment2019zh").glob("dedup_v1_*"))[-1]
print("dir:", D)
p = D / "summary.json"
s = json.loads(p.read_text(encoding="utf-8"))
s["dataset"] = "comment2019zh"
p.write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
log = (D / "run.log").read_text(encoding="utf-8")
(D / "run.log").write_text(log.replace("dataset: webText2019zh", "dataset: comment2019zh"), encoding="utf-8")
print("fixed")
