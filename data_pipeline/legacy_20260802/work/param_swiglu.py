def llama_params(d=896, L=20, h=14, kv=2, f=2432, vocab=32000):
    hd = d // h
    attn = d * d + 2 * (d * kv * hd) + d * d
    mlp = 3 * d * f  # SwiGLU: gate/up/down
    norms = 2 * d
    layer = attn + mlp + norms
    emb = vocab * d
    return emb + layer * L + d

total = llama_params()
print("SwiGLU ffn=2432 tied:", total)
print("target 196153216:", total == 196153216)
