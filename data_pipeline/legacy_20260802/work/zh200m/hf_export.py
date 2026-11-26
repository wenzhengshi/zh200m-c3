#!/usr/bin/env python3
"""Save a checkpoint in HF LLaMA format (config.json + model.safetensors + tokenizer.model)."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sp-model", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg["vocab_size"],
        "hidden_size": cfg["dim"],
        "intermediate_size": cfg["ffn_dim"],
        "num_hidden_layers": cfg["n_layer"],
        "num_attention_heads": cfg["n_head"],
        "num_key_value_heads": cfg["n_kv_head"],
        "head_dim": cfg["dim"] // cfg["n_head"],
        "max_position_embeddings": cfg["max_seq_len"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "torch_dtype": "float32",
    }

    sd = model.state_dict()
    rename = {}
    rename["embed_tokens.weight"] = "model.embed_tokens.weight"
    rename["norm.weight"] = "model.norm.weight"
    rename["lm_head.weight"] = "lm_head.weight"
    for i in range(cfg["n_layer"]):
        rename[f"layers.{i}.attn_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        rename[f"layers.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
        rename[f"layers.{i}.attn.q_proj.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        rename[f"layers.{i}.attn.k_proj.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        rename[f"layers.{i}.attn.v_proj.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        rename[f"layers.{i}.attn.o_proj.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        rename[f"layers.{i}.mlp.gate_proj.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        rename[f"layers.{i}.mlp.up_proj.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
        rename[f"layers.{i}.mlp.down_proj.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
    out = {}
    for k, v in sd.items():
        out[rename[k]] = v.detach().float().contiguous()
    # lm_head omitted: tie_word_embeddings=True; converter derives output.weight from embed_tokens

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(hf_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_file(out, str(out_dir / "model.safetensors"))
    shutil.copy(a.sp_model, out_dir / "tokenizer.model")
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "add_bos_token": False,
                "add_eos_token": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("HF model saved to", out_dir)


if __name__ == "__main__":
    main()
