def params(d_model=896, n_layer=20, n_head=14, n_kv_head=2, ffn=None, tie=True, vocab=32000):
    head_dim = d_model // n_head
    q = d_model * d_model
    k = d_model * (n_kv_head * head_dim)
    v = d_model * (n_kv_head * head_dim)
    o = d_model * d_model
    attn = q + k + v + o
    f = ffn or 0
    mlp = d_model * f + f * d_model
    norms = 2 * d_model
    layer = attn + mlp + norms
    emb = vocab * d_model
    head = emb if not tie else 0
    final_norm = d_model
    return emb + head + layer * n_layer + final_norm, attn, mlp, norms

for ffn, tie in ((2432, True), (3648, True), (2848, False), (2432, False)):
    total, attn, mlp, norms = params(ffn=ffn, tie=tie)
    print(f"ffn={ffn} tie={tie}: total={total:,} (attn={attn:,} mlp={mlp:,} norms={norms:,}/layer)")
