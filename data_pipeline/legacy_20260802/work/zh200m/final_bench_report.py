#!/usr/bin/env python3
"""Final cpu_benchmark_report from clean llama-bench rows + TTFT contexts + smoke measurements."""
import json
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8")
ttft = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out_dir = Path(sys.argv[3])

rows = []
cur = None
for line in raw.splitlines():
    m = re.match(r"### threads=(\d+) prompt=(\d+)", line)
    if m:
        cur = (int(m.group(1)), int(m.group(2)))
        continue
    if cur and line.startswith("|") and "model" not in line and "---" not in line:
        parts = [p.strip() for p in line.split("|")]
        try:
            test = parts[6]
            tps = float(parts[7].split()[0])
            kind = "prompt_processing" if test.startswith("pp") else "generation"
            rows.append({"threads": cur[0], "prompt_tokens": cur[1], "model": parts[1],
                         "size": parts[2], "params": parts[3], "test": test,
                         "tokens_per_second": tps, "kind": kind})
        except (ValueError, IndexError):
            pass

report = {
    "benchmark": rows,
    "ttft_by_context": ttft,
    "smoke_measurements": {
        "model": "model-q5_k_m.gguf (196M)",
        "load_time_ms": 1017.5,
        "peak_rss_mb": 200.6,
        "kv_cache_mb_analytic_f16": 10.0,
        "context": 2048,
        "threads": 2,
    },
    "note": "本机4核WSL烟雾数据；1/2/4线程为本机实测，6/8/12/16线程与目标硬件正式数据须执行 benchmark_cpu.sh 扩展矩阵。",
    "disclaimer": "未在目标CPU完成基准前，不承诺Token/s。",
}
(out_dir / "cpu_benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

md = ["# CPU 性能基准（本机烟雾数据，Q5_K_M 196M）", "",
      "## llama-bench（pp/tg, tokens/s）", "",
      "| threads | prompt tokens | test | tok/s |", "|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['threads']} | {r['prompt_tokens']} | {r['test']} | {r['tokens_per_second']:.2f} |")
md += ["", "## TTFT by context（llama-server, 2线程, 15-token prompt）", "",
       "| context | prompt_ms | predicted_per_token_ms | gen tok/s |", "|---|---|---|---|"]
for t in ttft:
    md.append(f"| {t['ctx']} | {t['prompt_ms']:.1f} | {t['predicted_per_token_ms']:.2f} | {t['predicted_per_second']:.2f} |")
md += ["", "## 补充测量", "",
       "- 模型加载时间: 1.02 s",
       "- 峰值 RSS: ~200.6 MB",
       "- KV Cache（解析值, f16, ctx2048）: 10 MB",
       "", report["note"], report["disclaimer"]]
(out_dir / "cpu_benchmark_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("rows:", len(rows), "ttft:", len(ttft))
