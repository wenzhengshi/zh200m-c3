import torch
from transformers import LlamaConfig, LlamaForCausalLM

cfg = LlamaConfig(
    vocab_size=32000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    bos_token_id=1,
    eos_token_id=2,
)
m = LlamaForCausalLM(cfg)
m.save_pretrained("/tmp/toy_llama")
import shutil
shutil.copy(
    "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model",
    "/tmp/toy_llama/tokenizer.model",
)
import json
from pathlib import Path
Path("/tmp/toy_llama/tokenizer_config.json").write_text(
    json.dumps({"model_type": "llama", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}),
    encoding="utf-8",
)
print("toy saved")
