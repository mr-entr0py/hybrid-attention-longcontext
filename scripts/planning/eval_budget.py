# Token budget and prefill-time model for the frozen evaluation suites (INF; calibrated to Review B's
# contended D512 single-layer timings: flash 32K 12.02 ms, 4K 0.26 ms; GDN layer 32K 2.55 ms, 4K 0.69 ms;
# 4d^2 projection 32K 0.78 ms).
K = 1024
Ls = [1*K, 2*K, 4*K, 8*K, 16*K, 32*K]
def cells_confirmatory():
    c = []  # (task, level, T, contexts)
    for D in (3, 15):
        for T in Ls: c.append(("R1", D, T, 400 if T >= 16*K else 300))
    for T in Ls[:3]:
        for N in (8, 32): c.append(("Rload", N, T, 300))
    for N in (8, 32, 128, 512): c.append(("Rload", N, 8*K, 300))
    for T in (16*K, 32*K):
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 400))
    c.append(("Rload", 2048, 32*K, 200))
    for T in Ls[:3]:
        for h in (1, 2): c.append(("Mhop", h, T, 300))
    for T in Ls[3:]:
        for h in (1, 2, 3, 4): c.append(("Mhop", h, T, 300))
    for T in Ls[:3]:
        for r in (4, 2): c.append(("Afreq", r, T, 300))
    for T in Ls[3:]:
        for r in (4, 2, 1.5, 1.25): c.append(("Afreq", r, T, 300))
    return c
def cells_mechanism():
    c = []
    for filler in ("natural", "repetitive"): c.append(("Rdist", filler, 32*K, 300))
    for U in (2, 4):
        for T in (4*K, 16*K, 32*K): c.append(("R3", U, T, 200))
    for task in ("R1D3", "RloadN32", "Mhop2", "Afreq2"):
        for T in (4*K, 32*K): c.append(("C2", task, T, 200))
    for T in (8*K, 32*K): c.append(("Acount", 0, T, 200))
    for order in ("fwd", "rev"): c.append(("Morder", order, 16*K, 200))
    c.append(("Mhop", 5, 32*K, 200)); c.append(("Afreq", 1.1, 32*K, 200))
    for task in ("R1D3", "Mhop2", "Afreq2"):
        for T in (8*K, 32*K): c.append(("T4slice", task, T, 100))
    return c
def cells_dev_gate():  # generator seed 77; G-REF; scored on k in {0,12} (+hybrids only for inversion)
    c = []
    for T in (16*K, 32*K):
        for D in (3, 15): c.append(("R1", D, T, 200))
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 200))
    return c
def cells_dev_ckpt():  # intermediate checkpoints (TEF dose ladder, FEAS)
    c = []
    for T in (4*K, 16*K, 32*K):
        for task in ("R1D3", "RloadN32", "RloadN128", "Mhop2", "Afreq2"): c.append((task, 0, T, 100))
    return c
def tier0():
    # LM-1 400 docs x 32K; LM-2 2000 x 4K; LM-4 2 variants x 200 items x 5 gaps
    lm1 = 400 * 32 * K; lm2 = 2000 * 4 * K
    lm4 = 2 * 200 * sum(512 + g for g in (0, 512, 2*K, 8*K, 30*K))
    return {"LM1": lm1, "LM2": lm2, "LM4": lm4}
def toks(cells): return sum(T * n for _, _, T, n in cells)
# time model per 1 forward of length T at scale d with k attention layers (ms)
def t_fwd(T, k, d):
    s_att = 0.75 if d == 384 else 1.0
    s_dense = 0.53 if d == 384 else 1.0
    att_core = 12.02 * (T / (32*K))**2 if T >= 8*K else {1*K: 0.04, 2*K: 0.09, 4*K: 0.26}[T]
    att_layer = s_att * att_core + s_dense * (0.78 + 1.82) * T / (32*K) + 0.05
    gdn_layer = 0.45 + s_att * 2.08 * T / (32*K) + s_dense * 1.82 * T / (32*K)
    return k * att_layer + (12 - k) * gdn_layer
def hours(cells, k, d, contended=1.3, overhead=1.2):
    ms = sum(n * t_fwd(T, k, d) for _, _, T, n in cells)
    return ms / 1000 / 3600 * contended * overhead
conf, mech, devg, devc = cells_confirmatory(), cells_mechanism(), cells_dev_gate(), cells_dev_ckpt()
t0 = tier0()
print("tokens (M): confirmatory %.1f | mechanism %.1f | dev-gate %.1f | dev-ckpt %.1f | tier0 %.1f" % (
    toks(conf)/1e6, toks(mech)/1e6, toks(devg)/1e6, toks(devc)/1e6, sum(t0.values())/1e6))
t0cells = [("LM1", 0, 32*K, 400), ("LM2", 0, 4*K, 2000)] + [("LM4", g, 512*2 + g, 200*2) for g in (0, 512, 2*K, 8*K, 30*K)]
def lm4fix(cells):  # LM4 lengths are not in table; approximate with nearest length
    out = []
    for t, l, T, n in cells:
        if t == "LM4":
            TT = min(Ls, key=lambda x: abs(x - T)); out.append((t, l, TT, n))
        else: out.append((t, l, T, n))
    return out
t0cells = lm4fix(t0cells)
for d in (384, 512):
    for k in (0, 1, 3, 12):
        print(f"d{d} k{k:2d}: conf {hours(conf,k,d):.2f} h | mech {hours(mech,k,d):.2f} | devgate {hours(devg,k,d):.2f} | devckpt {hours(devc,k,d):.3f} | tier0 {hours(t0cells,k,d):.2f}")
