from pathlib import Path

p = Path("/mnt/c/Users/xmkjsoft/Documents/Codex/2026-08-01/files-mentioned-by-the-user-c/work/bind_final.py")
src = p.read_text(encoding="utf-8")

start = src.index("# global deterministic doc-level shuffle")
end = src.index("train_stream.astype(np.int32).tofile(OUT / \"train.bin\")") + len("train_stream.astype(np.int32).tofile(OUT / \"train.bin\")")

new_block = '''# global deterministic doc-level shuffle
rg = random.Random(SEED + 1000)
rg.shuffle(sampled)

del train_doc_info
for s in TARGETS:
    mmaps[s]._mmap.close()
del mmaps

# read doc tokens from pool bins (cache idx)
pool_off = {}
for s in TARGETS:
    m = {}
    with (TMP / f"pool_{s}.idx").open("r", encoding="utf-8") as f:
        for line in f:
            uid, off, ln = line.rstrip("\\n").split("\\t")
            m[uid] = (int(off), int(ln))
    pool_off[s] = m

mmaps = {s: np.memmap(TMP / f"pool_{s}.bin", dtype=np.int32, mode="r") for s in TARGETS}
with open(OUT / "train.bin", "wb") as tf:
    for s, uid, ln in sampled:
        off, ln2 = pool_off[s][uid]
        tf.write(mmaps[s][off:off + ln2].tobytes())
total_written = (OUT / "train.bin").stat().st_size // 4
print("train stream tokens (before trim):", total_written, flush=True)

# trim to cap (CAP = 747110400 = 364800 * 2048, exact)
full_seqs = min(total_written, CAP) // SEQ_LEN
final_tokens = full_seqs * SEQ_LEN
with open(OUT / "train.bin", "r+b") as tf:
    tf.truncate(final_tokens * 4)
dropped = total_written - final_tokens
print("train seqs:", full_seqs, "tokens:", final_tokens, "dropped:", dropped, flush=True)

# walk the sampled order to build the manifest within final_tokens
src_actual_final = Counter()
consumed = 0
man_final = []
for s, uid, ln in sampled:
    if consumed + ln > final_tokens:
        break
    consumed += ln
    src_actual_final[s] += ln
    man_final.append({"document_id": uid, "source": s, "tokens": ln})
print("final per-source:", dict(src_actual_final), "total:", consumed, flush=True)
'''

src = src[:start] + new_block + src[end:]
p.write_text(src, encoding="utf-8")
print("patched")
