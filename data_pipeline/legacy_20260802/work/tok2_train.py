#!/usr/bin/env python3
"""Train C1 (cc=1.0, bf=false) and C2 (cc=0.9995, bf=true) with replicated existing config."""
import json
from pathlib import Path

import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
SAMPLES = AB / "samples"
TRAIN = SAMPLES / "tokenizer_train.txt"
print("ab dir:", AB)
print("sentencepiece version:", spm.__version__)

COMMON = dict(
    model_type="bpe",
    vocab_size=32000,
    normalization_rule_name="identity",
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    split_digits=True,
    max_sentencepiece_length=16,
    shuffle_input_sentence=False,
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
    user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>", "<|n|>", "<|tab|>"],
    num_threads=8,
)


def run(label, coverage, byte_fallback):
    prefix = str(AB / label)
    log = AB / f"{label}_train.log"
    params = dict(COMMON)
    params.update(character_coverage=coverage, byte_fallback=byte_fallback)
    params.update(input=str(TRAIN), model_prefix=prefix)
    with log.open("w", encoding="utf-8") as lf:
        try:
            spm.SentencePieceTrainer.train(**params)
            status = "OK"
        except Exception as e:
            lf.write(f"\nEXCEPTION: {type(e).__name__}: {e}\n")
            status = f"FAILED: {type(e).__name__}"
    print(f"{label}: {status}", flush=True)
    return status


status = {}
status["c1"] = run("c1_bpe_32k", 1.0, False)
status["c2"] = run("c2_bpe_32k", 0.9995, True)

(AB / "train_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("done:", status)
