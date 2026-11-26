#!/usr/bin/env python3
"""Load FP32/INT8 ONNX with onnxruntime and compare logits against the torch model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx-fp32", required=True)
    ap.add_argument("--onnx-int8", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    seq = 8
    ids = torch.randint(0, cfg["vocab_size"], (1, seq), dtype=torch.long)
    with torch.no_grad():
        ref = model(ids).numpy()

    results = {}
    for name, path in (("fp32", a.onnx_fp32), ("int8", a.onnx_int8)):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input_ids": ids.numpy()})[0]
        err = np.abs(out - ref)
        results[name] = {
            "shape": list(out.shape),
            "max_logit_error": float(err.max()),
            "mean_logit_error": float(err.mean()),
        }
        print(name, "max_err:", results[name]["max_logit_error"], "mean_err:", results[name]["mean_logit_error"])

    Path(a.out).write_text(json.dumps({"ref_shape": list(ref.shape), **results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
