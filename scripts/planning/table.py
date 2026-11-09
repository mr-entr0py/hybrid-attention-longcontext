V = 16384; L = 12; C = 64
cfg = {384: dict(dff=1024, Hl=3, gdn=595_334, fg=592_896, fgc=597_504, gdnE=594_050, A_tok=0.6e9, B_tok=60e6),
       512: dict(dff=1536, Hl=4, gdn=1_056_904, fg=1_053_696, fgc=1_059_840, gdnE=1_054_210, A_tok=0.8e9, B_tok=0.2e9)}
TEF = 65.536e6
def core(H, dk, dv): return H * (6 * dk * dv + 2 * C * (3 * dk + 2 * dv) + C * C / 3)
def pat(pos, ch="A", base="-"):
    s = [base] * 12
    for p in pos: s[p] = ch
    return "".join(s)
cent = {0: [], 1: [6], 2: [3, 9], 3: [2, 6, 10], 4: [1, 4, 7, 10], 6: [1, 3, 5, 7, 9, 11], 12: list(range(12))}
rows = []
for d in (512, 384):
    c = cfg[d]; mlp = 3 * d * c["dff"]; nm = 2 * d; emb = V * d
    confs = [(f"C{k}", k, "A", c["fg"], "A", pat(cent[k])) for k in (0, 1, 2, 3, 4, 6, 12)]
    confs += [("X-gc12 (F_gc)", 12, "A", c["fgc"], "A", pat(cent[12], "C")),
              ("X-E0 (GDN-E)", 0, "E", c["fg"], "E", "E" * 12)]
    for name, k, lin, fparams, _, p in confs:
        gp = c["gdn"] if lin == "A" else c["gdnE"]
        ne = (L - k) * (gp + mlp + nm) + k * (fparams + mlp + nm) + d
        H, dk, dv = (c["Hl"], 64, 128) if lin == "A" else (1, d // 2, d)
        dense = 2 * ne + 2 * d * V; lin_core = (L - k) * core(H, dk, dv)
        f = {T: dense + lin_core + k * 2 * T * d for T in (4096, 32768)}
        per_lin = H * dk * dv * 4 + (2 * H * dk + H * dv) * 16
        convA = 3 * d * 16 if "gc" in name else 0
        sb = {T: k * T * 4 * d + (L - k) * per_lin + k * convA for T in (1024, 4096, 16384, 32768)}
        train_pf = (3 * f[4096] * c["A_tok"] + 4 * f[32768] * (c["B_tok"] + TEF)) / 1e15
        rows.append(f"| D{d} | {name} | `{p}` | {ne:,} | {emb:,} | {ne+emb:,} | {f[4096]/1e6:.1f} | {f[32768]/1e6:.1f} | "
                    + " | ".join(f"{sb[T]/2**20:.2f}" for T in sb) + f" | {train_pf:.0f} |")
print("| Scale | Config | Pattern (layer 0→11) | Non-emb params | Emb (tied) | Total | Fwd MFLOP/tok @4K | @32K | State MiB @1K | @4K | @16K | @32K | Train PFLOP/run (nominal) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows: print(r)
