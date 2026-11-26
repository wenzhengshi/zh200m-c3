import json
from pathlib import Path

import sentencepiece as spm

WS = Path("/home/xmkjsoft/jicreate_chinese_lm_0.2b_runtime/data/codex_cleaning_workspace")
AB = sorted((WS / "outputs/cross_source").glob("tokenizer_ab_v2_*"))[-1]
print("ab dir:", AB)
for label in ("c1_bpe_32k", "c2_bpe_32k"):
    m = AB / f"{label}.model"
    sp = spm.SentencePieceProcessor(model_file=str(m))
    byte_pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if sp.id_to_piece(i).startswith("<0x")]
    single_char_pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if len(sp.id_to_piece(i)) == 1 and not sp.id_to_piece(i).startswith("<")]
    info = {
        "model": label,
        "vocab_size": sp.get_piece_size(),
        "unk_id": sp.unk_id(), "bos_id": sp.bos_id(), "eos_id": sp.eos_id(), "pad_id": sp.pad_id(),
        "byte_piece_count": len(byte_pieces),
        "single_char_piece_count": len(single_char_pieces),
        "sample_byte_pieces": byte_pieces[:5],
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    # smoke encode
    t = "新华社上海7月26日电 时隔多年，国产经典动漫形象“葫芦兄弟”再现屏幕。"
    ids = sp.encode(t, out_type=int)
    print("  smoke tokens:", len(ids), "unk:", sum(1 for x in ids if x == sp.unk_id()), "roundtrip:", sp.decode(ids) == t)
