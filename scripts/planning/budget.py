# Stage-by-stage GPU-h (contended), same per-token rates as the debate record (B1 measured D512; D384 = x1.35 A / x1.33 B).
KS = (0, 1, 2, 3, 6, 12)
rateB512 = dict(zip(KS, (70.5e3, 64e3, 59e3, 51e3, 44e3, 31.9e3)))
P = {
 "D384": dict(A_rate=88.9e3 * 1.35, B_rate={k: v * 1.33 for k, v in rateB512.items()}, A_tok=0.6e9, B_tok=60e6, n0=5,
              feasB=(30e6, 90e6), feas_seeds=2, eval_A=9.8, evalB=3.0, lr_tok=0.15e9),
 "D512": dict(A_rate=88.9e3, B_rate=rateB512, A_tok=0.8e9, B_tok=0.2e9, n0=4,
              feasB=(0.1e9, 0.3e9), feas_seeds=1, eval_A=12.1, evalB=5.0, lr_tok=0.15e9),
}
TEF = 65.536e6; TEF4_extra = 3 * TEF
h = lambda tok, rate: tok / rate / 3600
def run(p, k, mult=1.0, Btok=None):
    Btok = p["B_tok"] if Btok is None else Btok
    return mult * (h(p["A_tok"], p["A_rate"]) + h(Btok, p["B_rate"][k]) + h(TEF, p["B_rate"][k]))
def feas(p, seeds, eval_h):
    A = 0.3e9; b1, b2 = p["feasB"]
    t = 0.0
    for k in (0, 12):
        t += seeds * (h(A, p["A_rate"]) + h(b1 + b2, p["B_rate"][k]) + 2 * h(TEF, p["B_rate"][k]))
    t += h(A, p["A_rate"]) + h(b1, p["B_rate"][12]) + h(TEF, p["B_rate"][12])  # SSMax variant, 1 seed, small branch
    return t + eval_h
out = {}
for name, p in P.items():
    rows = []
    rows.append(("S0 infrastructure + verification", 3.0))
    rows.append(("S1 Zoology MQAR reproduction", 3.0))
    f384 = feas(P["D384"], 2, 0.9)
    if name == "D384": rows.append(("S2 FEAS-384 (2 seeds + SSMax)", f384))
    else:
        rows.append(("S2 FEAS-384 (run first, failed)", f384))
        rows.append(("S2 FEAS-512 (seed 101 + SSMax)", feas(p, 1, 0.8)))
    rows.append(("S3 LR sweep k{0,3,12} x 4 LR x 0.15B", 12 * h(p["lr_tok"], p["A_rate"])))
    core = p["n0"] * sum(run(p, k) for k in (0, 1, 3, 12))
    rows.append((f"S4 core k{{0,1,3,12}} x {p['n0']} seeds", core))
    ctrl = 2 * run(p, 12, 1.06) + run(p, 3) + run(p, 12)
    rows.append(("S5a controls F_gc k12 x2 + NoPE k{3,12} x1", ctrl))
    tef4 = sum(h(TEF4_extra, p["B_rate"][k]) for k in (0, 1, 3, 12))
    twoB = sum(h(2 * p["B_tok"], p["B_rate"][k]) + h(TEF, p["B_rate"][k]) for k in (0, 3, 12))
    rows.append(("S5b branches TEF-4000 {0,1,3,12} + 2xB {0,3,12}", tef4 + twoB))
    rows.append(("S6 evaluation (frozen suites)", p["eval_A"]))
    rows.append(("S6 cost benchmarks (idle windows)", 3.0))
    tierA = sum(v for _, v in rows)
    trig = 2 * sum(run(p, k) for k in (0, 1, 3, 12))
    curve = 2 * (run(p, 2) + run(p, 6))
    place = 2 * p["n0"] * run(p, 1)  # k*=1 shown
    gdnE = 2 * (run(p, 0, 1.17) + run(p, 1, 1.17))
    tierB = curve + place + gdnE + p["evalB"]
    condB = 2 * (run(p, 1, 1.06) + run(p, 3, 1.06)) + run(p, 3) + run(p, 12) + (p["n0"] - 2) * run(p, 2)
    print(f"===== {name} path  (per-run h: " + ", ".join(f"k{k} {run(p,k):.2f}" for k in KS) + ")")
    for r, v in rows: print(f"  {r:52s} {v:6.1f}")
    print(f"  {'TIER A base':52s} {tierA:6.1f}   with 20% contingency {1.2*tierA:6.1f}")
    print(f"  {'Tier A conditional +2 anchor seeds (U-05)':52s} {trig:6.1f}")
    print(f"  {'TIER B (curve %.1f, placement k*=1 %.1f, GDN-E %.1f, eval %.1f)' % (curve, place, gdnE, p['evalB']):52s} {tierB:6.1f}")
    print(f"  {'Tier B conditional (Fgc k1,3 x2; NoPE s1; top-up)':52s} {condB:6.1f}")
    # minimum falsification experiment
    lr_red = 6 * h(p["lr_tok"], p["A_rate"])
    core3 = p["n0"] * sum(run(p, k) for k in (0, 3, 12))
    br = sum(h(TEF4_extra, p["B_rate"][k]) for k in (0, 3, 12)) + twoB
    fz = f384 if name == "D384" else f384 + feas(p, 1, 0.8)
    ev = 2.5 if name == "D384" else 3.5
    mn = 3.0 + 1.5 + fz + lr_red + core3 + br + ev
    print(f"  {'MIN H1-falsification experiment':52s} {mn:6.1f}   with 20% {1.2*mn:6.1f}")
