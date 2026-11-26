#!/usr/bin/env python3
"""Parse llama-bench raw output (threads/prompt matrix) into cpu_benchmark_report.json/.md."""
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8")
out_dir = Path(sys.argv[2])
rows = []
cur = None
for line in raw.splitlines():
    m = __import__("re").match(r"### threads=(\d+) prompt=(\d+)", line)
    if m:
        cur = (int(m.group(1)), int(m.group(2)))
        continue
    if cur and line.startswith("|") and "model" not in line and "---" not in line:
        parts = [p.strip() for p in line.split("|")]
        try:
            test = parts[6]
            tps = float(parts[7].split()[0])
            kind = "prompt_processing" if test.startswith("pp") else "generation"
            rows.append({"threads": cur[0], "prompt_tokens": cur[1],
                         "model": parts[1], "size": parts[2], "params": parts[3],
                         "test": test, "tokens_per_second": tps, "kind": kind})
        except (ValueError, IndexError):
            pass
report = {
    "benchmark": rows,
    "note": "本机为4核WSL；1/2/4线程为实测；6/8/12/16线程与不同硬件的正式数据须在目标CPU上执行 benchmark_cpu.sh 扩展矩阵。",
    "disclaimer": "未在目标CPU完成基准前，不承诺Token/s；当前数值仅为本机烟雾基准。",
}
(out_dir / "cpu_benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
md = ["# CPU 性能基准（本机烟雾数据，Q5_K_M 196M）", "",
      "| threads | prompt tokens | test | tok/s |", "|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['threads']} | {r['prompt_tokens']} | {r['test']} | {r['tokens_per_second']:.2f} |")
md.append("")
md.append(report["note"])
md.append(report["disclaimer"])
(out_dir / "cpu_benchmark_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("rows:", len(rows))
