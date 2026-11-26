#!/usr/bin/env python3
"""Export the trained checkpoint to a LLaMA-arch F16 GGUF file."""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from model import build_model


def export(checkpoint, config, sp_model, out_path, model_name="zh200m"):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from gguf import GGUFWriter
    writer = GGUFWriter(str(out_path), "llama")
    writer.add_name(model_name)
    writer.add_context_length(cfg["max_seq_len"])
    writer.add_embedding_length(cfg["dim"])
    writer.add_block_count(cfg["n_layer"])
    writer.add_head_count(cfg["n_head"])
    writer.add_head_count_kv(cfg["n_kv_head"])
    writer.add_feed_forward_length(cfg["ffn_dim"])
    writer.add_rope_dimension_count(cfg["dim"] // cfg["n_head"])
    writer.add_rope_freq_base(cfg["rope_theta"])
    writer.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    writer.add_file_type(1)  # F16

    # tokenizer metadata from the SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    scores = [float(sp.get_score(i)) for i in range(sp.get_piece_size())]
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types([0] * sp.get_piece_size())
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    sd = model.state_dict()
    tensors = {}
    tensors["token_embd.weight"] = sd["embed_tokens.weight"]
    tensors["output_norm.weight"] = sd["norm.weight"]
    tensors["output.weight"] = sd["embed_tokens.weight"]  # tied
    for i in range(cfg["n_layer"]):
        tensors[f"blk.{i}.attn_norm.weight"] = sd[f"layers.{i}.attn_norm.weight"]
        tensors[f"blk.{i}.attn_q.weight"] = sd[f"layers.{i}.attn.q_proj.weight"]
        tensors[f"blk.{i}.attn_k.weight"] = sd[f"layers.{i}.attn.k_proj.weight"]
        tensors[f"blk.{i}.attn_v.weight"] = sd[f"layers.{i}.attn.v_proj.weight"]
        tensors[f"blk.{i}.attn_output.weight"] = sd[f"layers.{i}.attn.o_proj.weight"]
        tensors[f"blk.{i}.ffn_norm.weight"] = sd[f"layers.{i}.ffn_norm.weight"]
        tensors[f"blk.{i}.ffn_gate.weight"] = sd[f"layers.{i}.mlp.gate_proj.weight"]
        tensors[f"blk.{i}.ffn_up.weight"] = sd[f"layers.{i}.mlp.up_proj.weight"]
        tensors[f"blk.{i}.ffn_down.weight"] = sd[f"layers.{i}.mlp.down_proj.weight"]

    for name, t in tensors.items():
        data = t.detach().to(torch.float16).numpy()
        shape = list(data.shape)
        writer.add_tensor(name, data, raw_shape=shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("GGUF written:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sp-model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.checkpoint, a.config, a.sp_model, Path(a.out))
