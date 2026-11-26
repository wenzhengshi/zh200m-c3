import json
from pathlib import Path

D = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace/outputs/cross_source/zh200m_cpu_inference_v1_20260802_113604/bench")
extra = {
    "smoke_measurements": {
        "model": "model-q5_k_m.gguf (196M)",
        "load_time_ms": 1017.5,
        "peak_rss_mb": 200.6,
        "time_to_first_token_ms": 135.6,
        "generation_tokens_per_second": 60.7,
        "kv_cache_mb_analytic_f16": 10.0,
        "context": 2048,
        "threads": 2,
        "note": "本机烟雾测量；目标CPU正式基准需扩展6/8/12/16线程与KV实测。",
    }
}
r = json.loads((D / "cpu_benchmark_report.json").read_text(encoding="utf-8"))
r.update(extra)
(D / "cpu_benchmark_report.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
md = (D / "cpu_benchmark_report.md").read_text(encoding="utf-8")
md += "\n## 补充烟雾测量（Q5_K_M, ctx 2048, 2线程）\n\n- 模型加载时间: 1.02 s\n- 峰值 RSS: ~200.6 MB\n- 首 Token 延迟: ~135.6 ms\n- 生成: ~60.7 tok/s\n- KV Cache（解析值, f16）: 10 MB\n"
(D / "cpu_benchmark_report.md").write_text(md, encoding="utf-8")
print("benchmark reports updated")
