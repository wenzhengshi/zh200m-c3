#!/usr/bin/env python3
"""Post-training quality validation across original/GGUF/ONNX variants (run after training)."""
import argparse
import http.client
import json
import math
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="gguf paths")
    ap.add_argument("--val-text", required=True, help="validation text file (llama-perplexity input)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llamacpp-dir", required=True)
    ap.add_argument("--prompts", nargs="+", default=["新华社报道，", "今天天气不错，"])
    a = ap.parse_args()
    ll = Path(a.llamacpp_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in a.models:
        r = subprocess.run(
            [str(ll / "llama-perplexity"), "-m", m, "-f", a.val_text, "-c", "2048"],
            capture_output=True, text=True, timeout=3600,
        )
        # parse PPL from stderr
        ppl = None
        for line in (r.stdout + r.stderr).splitlines():
            if "PPL" in line and "=" in line:
                try:
                    ppl = float(line.split("=")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        # generation via llama-server (llama-cli hangs in this sandbox)
        import time
        port = 18083
        srv = subprocess.Popen(
            [str(ll / "llama-server"), "-m", m, "-t", "2", "-c", "512",
             "--port", str(port), "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        gens = {}
        try:
            for _ in range(60):
                time.sleep(1)
                try:
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                    c.request("GET", "/health")
                    if "ok" in c.getresponse().read().decode():
                        c.close()
                        break
                    c.close()
                except Exception:
                    pass
            for p in a.prompts:
                body = json.dumps({"prompt": p, "n_predict": 32, "temperature": 0,
                                   "cache_prompt": True, "stream": False})
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
                c.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
                gens[p] = json.loads(c.getresponse().read().decode()).get("content", "")
                c.close()
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=10)
            except Exception:
                srv.kill()
        results[m] = {"ppl": ppl, "generations": gens}
        print(m, "ppl:", ppl, flush=True)
    (out / "quality_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
