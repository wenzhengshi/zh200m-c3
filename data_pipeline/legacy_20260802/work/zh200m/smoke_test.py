#!/usr/bin/env python3
"""2-step CPU smoke test + checkpoint/resume + GGUF export + llama.cpp load/generate."""
import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from model import build_model


def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert cfg["parameter_count"] == 196153216
    assert cfg["vocab_size"] == 32000
    assert cfg["max_seq_len"] == 2048
    assert cfg["random_init"] is True
    assert cfg["init_checkpoint"] is None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    model = build_model(cfg)
    param_count = model.count_parameters()
    print("params:", param_count, flush=True)

    # synthetic or real data
    seq_len = cfg["max_seq_len"]
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir and (data_dir / "train.bin").exists():
        stream = np.memmap(data_dir / "train.bin", dtype=np.int32, mode="r")
        tokens = stream[:3 * seq_len + 1].astype(np.int64)
    else:
        tokens = torch.randint(1, cfg["vocab_size"], (3 * seq_len + 1,)).numpy()
    seqs = len(tokens) // seq_len
    print("smoke seqs:", seqs, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        inp = torch.from_numpy(tokens[step * seq_len:(step + 1) * seq_len]).unsqueeze(0)
        lab = torch.from_numpy(tokens[step * seq_len + 1:(step + 1) * seq_len + 1]).unsqueeze(0)
        logits = model(inp)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1))
        loss.backward()
        opt.step()
        print(f"smoke step {step} loss={loss.item():.4f}", flush=True)

    # save checkpoint
    ckpt_path = out / "smoke_checkpoint.pt"
    torch.save({"model": model.state_dict(), "global_step": 2, "trained_tokens": 4096}, ckpt_path)
    # resume check: reload and compare params
    model2 = build_model(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ck["model"])
    ok = all(
        torch.allclose(a, b)
        for a, b in zip(model.parameters(), model2.parameters())
    )
    print("resume param consistency:", ok, flush=True)
    assert ok and ck["global_step"] == 2

    # free memory before GGUF export (subprocess)
    del model, model2, opt
    gc.collect()

    # GGUF export via HF + official converter (hand-written writer had a decode issue)
    gguf_path = out / "model-f16.gguf"
    hf_dir = out / "hf_model"
    subprocess.run(
        [
            "python3", str(Path(__file__).parent / "hf_export.py"),
            "--checkpoint", str(ckpt_path),
            "--config", args.config,
            "--sp-model", args.sp_model,
            "--out-dir", str(hf_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python3", args.convert_script,
            "--outfile", str(gguf_path),
            "--outtype", "f16",
            str(hf_dir),
        ],
        check=True,
    )
    print("gguf size:", gguf_path.stat().st_size, flush=True)

    # llama.cpp CPU load + tokenizer encode + generate via llama-server (llama-cli hangs in this sandbox)
    ll = Path(args.llamacpp_dir)
    tok = ll / "llama-tokenize"
    prompt = "新华社报道，"
    enc = subprocess.run(
        [str(tok), "-m", str(gguf_path), "-p", prompt, "--no-bos"],
        capture_output=True, text=True, timeout=120,
    )
    print("tokenize rc:", enc.returncode, flush=True)
    ids = [int(x) for x in re.findall(r"\d+", enc.stdout) if 0 <= int(x) < cfg["vocab_size"]]
    print("tokenize ids sample:", ids[:10], "count:", len(ids), flush=True)
    assert len(ids) > 0, enc.stdout + enc.stderr

    import json as _json
    import http.client
    port = 18081
    srv = subprocess.Popen(
        [
            str(ll / "llama-server"), "-m", str(gguf_path),
            "-t", str(args.threads), "-c", "256",
            "--port", str(port), "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = None
        for _ in range(60):
            time.sleep(1)
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                c.request("GET", "/health")
                health = c.getresponse().read().decode()
                c.close()
                if "ok" in health:
                    break
            except Exception:
                pass
        print("server health:", health, flush=True)
        assert health and "ok" in health
        body = _json.dumps(
            {"prompt": prompt, "n_predict": 2, "temperature": 0, "cache_prompt": True,
             "stream": False, "return_tokens": True}
        )
        t0 = time.time()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        c.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
        resp = c.getresponse()
        data = _json.loads(resp.read().decode())
        c.close()
        elapsed = time.time() - t0
        timings = data.get("timings", {})
        first_token_ms = timings.get("prompt_ms") + timings.get("predicted_per_token_ms")
        gen_tok_s = timings.get("predicted_per_second")
        gen_tokens = data.get("tokens", [])
        gen_text = data.get("content", "")
        print("first-token latency ms:", first_token_ms, "gen tok/s:", gen_tok_s, flush=True)
        print("generated tokens ids:", gen_tokens, "text:", repr(gen_text), flush=True)
        assert resp.status == 200
        assert len(gen_tokens) >= 1
        assert all(0 <= t < cfg["vocab_size"] for t in gen_tokens), "token id out of range"
        # KV cache: second completion with cache_prompt should work (cache reuse)
        body2 = _json.dumps({"prompt": prompt + gen_text, "n_predict": 1, "temperature": 0,
                             "cache_prompt": True, "stream": False})
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        c.request("POST", "/completion", body=body2, headers={"Content-Type": "application/json"})
        data2 = _json.loads(c.getresponse().read().decode())
        c.close()
        kv_ok = bool(data2.get("content", "").strip()) and data2.get("timings", {}).get("prompt_ms", 0) < 10000
        print("kv cache second completion ok:", kv_ok, flush=True)
        assert kv_ok
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()

    report = {
        "params": param_count,
        "resume_consistent": bool(ok),
        "gguf_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
        "tokenize_ids_count": len(ids),
        "token_id_min": min(ids),
        "token_id_max": max(ids),
        "vocab_size": cfg["vocab_size"],
        "first_token_latency_ms": round(float(first_token_ms), 2) if first_token_ms else None,
        "generation_tokens_per_second": round(gen_tok_s, 3) if gen_tok_s else None,
        "kv_cache_ok": bool(kv_ok),
        "elapsed_smoke_seconds": round(elapsed, 2),
    }
    (out / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sp-model", required=True)
    ap.add_argument("--llamacpp-dir", required=True)
    ap.add_argument("--convert-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    run(ap.parse_args())
