#!/usr/bin/env python3
"""Dump the existing zh_bpe_32k SentencePiece model config (read-only)."""
import json

import sentencepiece as spm

MODEL = "/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/tokenizer/zh_bpe_32k.model"
sp = spm.SentencePieceProcessor(model_file=MODEL)
proto = sp.serialized_model_proto()

# parse proto fields via the trainer/normalizer spec JSON embedded in the model
import re
text = proto.decode("utf-8", errors="replace")
print("--- raw proto head ---")
print(text[:2000])

print("\n--- python API ---")
print("vocab_size:", sp.get_piece_size())
print("unk_id:", sp.unk_id(), "unk_piece:", sp.id_to_piece(sp.unk_id()))
print("bos_id:", sp.bos_id(), "bos_piece:", sp.id_to_piece(sp.bos_id()) if sp.bos_id() >= 0 else None)
print("eos_id:", sp.eos_id(), "eos_piece:", sp.id_to_piece(sp.eos_id()) if sp.eos_id() >= 0 else None)
print("pad_id:", sp.pad_id(), "pad_piece:", sp.id_to_piece(sp.pad_id()) if sp.pad_id() >= 0 else None)
