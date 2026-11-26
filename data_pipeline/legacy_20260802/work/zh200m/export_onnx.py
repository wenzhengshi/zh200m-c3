#!/usr/bin/env python3
"""Export checkpoint to FP32 ONNX and INT8 (dynamic-quantized) ONNX."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import build_model


def export(checkpoint, config, out_fp32, out_int8):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    seq = 16
    dummy = torch.randint(0, cfg["vocab_size"], (1, seq), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_fp32),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
        opset_version=17,
    )
    print("fp32 onnx:", out_fp32)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(out_fp32),
        str(out_int8),
        weight_type=QuantType.QInt8,
    )
    print("int8 onnx:", out_int8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-fp32", required=True)
    ap.add_argument("--out-int8", required=True)
    a = ap.parse_args()
    export(a.checkpoint, a.config, Path(a.out_fp32), Path(a.out_int8))
