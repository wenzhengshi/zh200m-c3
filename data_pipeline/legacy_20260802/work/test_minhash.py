import numpy as np

MASK64 = (1 << 64) - 1
BASE = 131
K = 64


def rolling_shingles(text, n=4):
    codes = [ord(c) for c in text]
    if len(codes) < n:
        return []
    h = 0
    pw = 1
    for i in range(n):
        h = (h * BASE + codes[i]) & MASK64
        pw = (pw * BASE) & MASK64
    out = [h]
    for i in range(n, len(codes)):
        h = (h * BASE + codes[i] - codes[i - n] * pw) & MASK64
        out.append(h)
    return out


def bottom64(sh):
    if len(sh) <= K:
        return sorted(sh)
    arr = (np.array(sh, dtype=np.uint64) * 0x9E3779B97F4A7C15) & MASK64
    return sorted(np.partition(arr, K - 1)[:K].tolist())


def jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


t1 = "黑嘴树鸭（学名：）是鸭科树鸭属的鸟类，已知仅分布于加勒比地区。" * 20
t2 = "黑嘴树鸭（学名：）是鸭科树鸭属的鸟类，已知仅分布于加勒比地区。" * 20 + "追加内容。" * 5
t3 = "完全不同的文本内容，关于量子力学与相对论的讨论。" * 20

s1, s2, s3 = bottom64(rolling_shingles(t1)), bottom64(rolling_shingles(t2)), bottom64(rolling_shingles(t3))
print("sim(1,2):", round(jac(s1, s2), 3), "(expected > 0.85)")
print("sim(1,3):", round(jac(s1, s3), 3), "(expected ~0)")
print("shingle counts:", len(rolling_shingles(t1)), len(rolling_shingles(t2)), len(rolling_shingles(t3)))
print("sketch lens:", len(s1), len(s2), len(s3))
