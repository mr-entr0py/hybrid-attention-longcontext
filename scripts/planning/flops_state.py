# Analytic forward FLOPs/token and inference-state bytes per config (INF; arithmetic only).
V = 16384; L = 12; C = 64
cfg = {
 384: dict(dff=1024, Ha=6, Hl=3, gdn=595_334, fg=592_896, fgc=597_504, gdnE=594_050),
 512: dict(dff=1536, Ha=8, Hl=4, gdn=1_056_904, fg=1_053_696, fgc=1_059_840, gdnE=1_054_210),
}
def gdn_core(H, dk, dv):  # chunked delta-rule core FLOPs per token per layer (C=64), approx
    return H * (6 * dk * dv + 2 * C * (3 * dk + 2 * dv) + C * C / 3)
def state_bytes(d, k, T, linear="A"):
    c = cfg[d]; kv = k * T * 2 * d * 2
    if linear == "A":
        H, dk, dv = c["Hl"], 64, 128
    else:  # GDN-E
        H, dk, dv = 1, d // 2, d
    per = H * dk * dv * 4 + (H * dk * 2 + H * dv) * 4 * 4
    return kv + (L - k) * per, per
rows = []
for d in (384, 512):
    c = cfg[d]; mlp = 3 * d * c["dff"]; norms = 2 * d
    for name, k, lin, fullmix in [("k0", 0, "A", "fg"), ("k1", 1, "A", "fg"), ("k2", 2, "A", "fg"),
                                  ("k3", 3, "A", "fg"), ("k6", 6, "A", "fg"), ("k12", 12, "A", "fg"),
                                  ("GDN-E k0", 0, "E", "fg"), ("F_gc k12", 12, "A", "fgc")]:
        gparams = c["gdn"] if lin == "A" else c["gdnE"]
        nonemb = (L - k) * (gparams + mlp + norms) + k * (c[fullmix] + mlp + norms) + d
        dense = 2 * nonemb + 2 * d * V
        H, dk, dv = (c["Hl"], 64, 128) if lin == "A" else (1, d // 2, d)
        core_lin = (L - k) * gdn_core(H, dk, dv)
        out = {}
        for T in (4096, 32768):
            att = k * 2 * T * d
            out[T] = dense + core_lin + att
        sb = {T: state_bytes(d, k, T, lin)[0] for T in (1024, 2048, 4096, 8192, 16384, 32768)}
        rows.append((d, name, nonemb, out[4096], out[32768], sb, state_bytes(d, k, 1, lin)[1]))
for d, name, ne, f4, f32, sb, per in rows:
    print(f"d{d} {name:9s} nonemb={ne:>11,} fwd@4K={f4/1e6:7.1f}M fwd@32K={f32/1e6:7.1f}M  per-lin-layer={per:,}B  "
          + " ".join(f"{T//1024}K:{sb[T]/2**20:.2f}" for T in sb))
