"""Agent-6 resource model (report-only). All inputs are cited in reports/compute/experiment_schedule.md.
Prints per-run, per-suite and per-stage GPU-hours for D384/D512 under exclusive and CARLA-shared conditions."""
import json, math, collections
K = 1024
KS = (0, 1, 2, 3, 6, 12)
# ---------------- training throughput (tok/s), exclusive, production data ----------------
# central = G0 per-window regression intercepts at 0% foreign SM (pooled beta=0.635), .audit/agent6-schedule/g0_fit.py
# optimistic = idle bench (Stage A) / R2(c) PG19 kernel-sum mix (Stage B); see report §2
def lin_k(v0, v12, k):            # per-token time linear in k (validated on the idle bench: C3 pred 10.20 vs 10.23 us/tok)
    t = 1 / v0 + (1 / v12 - 1 / v0) * k / 12
    return 1 / t
def stageA(scale, k, opt=False):
    if scale == "D384":
        pts = {0: 127.0e3, 3: 128.8e3, 12: 137.8e3} if opt else {0: 117.9e3, 3: 116.8e3, 12: 121.2e3}
    else:
        pts = {0: 97.5e3, 3: 97.9e3, 12: 99.1e3} if opt else {0: 92.8e3, 3: 90.1e3, 12: 88.1e3}
    if k in pts: return pts[k]
    lo, hi = (0, 3) if k < 3 else (3, 12)
    return pts[lo] + (pts[hi] - pts[lo]) * (k - lo) / (hi - lo)
def stageB(scale, k, opt=False):
    if scale == "D384":
        v0, v12 = (104.5e3, 54.9e3) if opt else (85.5e3, 50.9e3)
    else:
        v0, v12 = (77.8e3, 39.0e3) if opt else (75.2e3, 36.1e3)
    return lin_k(v0, v12, k)
SC = {"D384": dict(A=600047616, B=60e6, Bfeas=(30015488, 90046464), lr_tok=1145 * 131072, n0=5),
      "D512": dict(A=800063488, B=0.2e9, Bfeas=(0.1e9, 0.3e9), lr_tok=1145 * 131072, n0=4)}
TEF = 65536000
RUN_OVH = 1.05      # checkpoint/validation/startup overhead [INF]
h = lambda tok, rate: tok / rate / 3600
def run_h(scale, k, opt=False, mult=1.0, Btok=None, Atok=None):
    s = SC[scale]; Btok = s["B"] if Btok is None else Btok; Atok = s["A"] if Atok is None else Atok
    return RUN_OVH * mult * (h(Atok, stageA(scale, k, opt)) + h(Btok + TEF, stageB(scale, k, opt)))
def carla(f): return 1 / (1 - 0.635 * f)
# ---------------- evaluation model ----------------
# per-query calls = n_cand + 1 (gold continuation + prompt extend + n_cand-1 candidates) ; without candidates = 1
# per-call cost (ms): probe (CARLA-contended, B=1) C0 11.85, C3 11.87, C12 7.89; smoke (no CARLA) ~15 ms
CALL = {0: 11.9, 1: 11.9, 2: 11.9, 3: 11.9, 6: 10.0, 12: 7.8}
CALL_GRAPH = 2.0   # [INF] CUDA-graph replay of the 3-token extend; cost bench graph decode 0.9-1.8 ms/token
def bsz(T): return 8 if T <= 4 * K else (2 if T <= 16 * K else 1)
# idle eager / graph prefill B=1 (ms), cost bench session 1 (flash, cold L2); C0 from 'none' backend rows
PRE = {"D384": {"eager": {0: {1: 13.15, 4: 13.09, 16: 19.91, 32: 52.2}, 1: {1: 12.71, 4: 12.58, 16: 22.04, 32: 60.59},
                          2: {1: 12.18, 4: 12.02, 16: 24.31, 32: 69.14}, 3: {1: 11.6, 4: 11.6, 16: 26.54, 32: 77.65},
                          6: {1: 9.85, 4: 9.97, 16: 33.39, 32: 103.22}, 12: {1: 5.25, 4: 7.87, 16: 46.75, 32: 154.42}},
                "graph": {0: {1: 2.28, 4: 5.36, 16: 19.28, 32: 51.95}, 1: {1: 2.29, 4: 5.56, 16: 21.5, 32: 60.37},
                          2: {1: 2.32, 4: 5.75, 16: 23.89, 32: 68.82}, 3: {1: 2.32, 4: 5.96, 16: 26.14, 32: 77.37},
                          6: {1: 2.36, 4: 6.55, 16: 33.07, 32: 102.97}, 12: {1: 2.44, 4: 7.74, 16: 46.62, 32: 154.31}}},
       "D512": {"eager": {0: {1: 13.29, 4: 13.38, 16: 31.76, 32: 77.18}, 1: {1: 12.72, 4: 13.11, 16: 34.96, 32: 88.61},
                          2: {1: 12.29, 4: 12.3, 16: 37.64, 32: 100.12}, 3: {1: 11.91, 4: 12.04, 16: 40.74, 32: 111.42},
                          6: {1: 9.88, 4: 10.51, 16: 49.96, 32: 146.09}, 12: {1: 5.27, 4: 11.83, 16: 67.98, 32: 214.79}},
                "graph": {0: {1: 2.86, 4: 7.99, 16: 31.71, 32: 77.03}, 1: {1: 2.9, 4: 8.3, 16: 34.84, 32: 88.51},
                          2: {1: 2.93, 4: 8.62, 16: 38.1, 32: 99.95}, 3: {1: 2.98, 4: 8.9, 16: 40.93, 32: 111.27},
                          6: {1: 3.07, 4: 9.79, 16: 49.83, 32: 145.61}, 12: {1: 3.27, 4: 11.62, 16: 68.16, 32: 214.79}}}}
def interp(tab, Tk):
    xs = sorted(tab)
    if Tk in tab: return tab[Tk]
    for a, b in zip(xs, xs[1:]):
        if a < Tk < b: return tab[a] + (tab[b] - tab[a]) * (Tk - a) / (b - a)
    return tab[xs[-1]] * Tk / xs[-1]
def prefill_batch_ms(scale, k, T, B):
    Tk = T // K
    e = interp(PRE[scale]["eager"][k], Tk); g = interp(PRE[scale]["graph"][k], Tk)
    return max(e, B * g)
def cell_cost_s(scale, k, T, n_ctx, n_cand, cand=True, call_ms=None, Q=4, rescore=True):
    B = bsz(T); nb = math.ceil(n_ctx / B)
    c = (CALL[k] if call_ms is None else call_ms) * (1.1 if scale == "D512" else 1.0)
    calls = nb * Q * ((n_cand + 1) if cand else 1)
    t = nb * prefill_batch_ms(scale, k, T, B) + calls * c
    if rescore:  # 5% of contexts, Q single-pass forwards each at B=1
        t += 0.05 * n_ctx * Q * interp(PRE[scale]["eager"][k], T // K)
    return t / 1000.0, calls
def ncand(task, lvl):
    if task == "R1": return lvl + 1 + 4
    if task == "Rload": return min(lvl, 32) + 4
    if task in ("Mhop", "Morder"): return 8
    if task in ("Afreq",): return 2
    if task == "Rdist": return 20
    if task == "R3": return 4 * (lvl + 1) + 4
    if task == "Acount": return 10
    return 8
L6 = [1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K]
def CS():
    c = []
    for D in (3, 15):
        for T in L6: c.append(("R1", D, T, 300 if T >= 16 * K else 200))
    for T in L6[:3]:
        for N in (8, 32): c.append(("Rload", N, T, 200))
    for N in (8, 32, 128, 512): c.append(("Rload", N, 8 * K, 200))
    for T in (16 * K, 32 * K):
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 300))
    c.append(("Rload", 2048, 32 * K, 150))
    for T in L6[:3]:
        for hh in (1, 2): c.append(("Mhop", hh, T, 200))
    for T in L6[3:]:
        for hh in (1, 2, 3, 4): c.append(("Mhop", hh, T, 200))
    for T in L6[:3]:
        for r in (4, 2): c.append(("Afreq", r, T, 200))
    for T in L6[3:]:
        for r in (4, 2, 1.5, 1.25): c.append(("Afreq", r, T, 200))
    # C0 no-filler controls (one per level, all CS levels except N=2048); short contexts -> B=8, T~2K proxy
    for D in (3, 15): c.append(("R1", D, 2 * K, 200))
    for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, 2 * K, 200))
    for hh in (1, 2, 3, 4): c.append(("Mhop", hh, 2 * K, 200))
    for r in (4, 2, 1.5, 1.25): c.append(("Afreq", r, 2 * K, 200))
    return c
def DG(k):
    c = []
    for T in (16 * K, 32 * K):
        for D in (3, 15): c.append(("R1", D, T, 200))
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 200))
        if k == 12:
            for hh in (1, 2, 3, 4): c.append(("Mhop", hh, T, 200))
    for D in (3, 15): c.append(("R1", D, 2 * K, 200))
    for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, 2 * K, 200))
    return c
def DC():
    c = []
    for T in (4 * K, 16 * K, 32 * K):
        c += [("R1", 3, T, 100), ("Rload", 32, T, 100), ("Rload", 128, T, 100), ("Mhop", 2, T, 100), ("Afreq", 2, T, 100)]
    return c
def FD():
    c = []
    for T in (4 * K, 8 * K, 16 * K, 32 * K):
        for D in (3, 15): c.append(("R1", D, T, 100))
    for N in (8, 32): c.append(("Rload", N, 4 * K, 100))
    for N in (8, 32, 128, 512): c.append(("Rload", N, 8 * K, 100))
    for T in (16 * K, 32 * K):
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 100))
    for D in (3, 15): c.append(("R1", D, 2 * K, 100))
    for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, 2 * K, 100))
    for T in (8 * K, 32 * K):
        for hh in (1, 2, 3, 4): c.append(("Mhop", hh, T, 100))
        for r in (4, 2, 1.5, 1.25): c.append(("Afreq", r, T, 100))
    return c
def P4cells():
    c = []
    for D in (3, 15): c += [("R1", D, 4 * K, 100), ("R1", D, 32 * K, 100), ("R1", D, 2 * K, 100)]
    for N in (8, 32): c.append(("Rload", N, 4 * K, 100))
    for N in (8, 32, 128, 512, 1024): c += [("Rload", N, 32 * K, 100), ("Rload", N, 2 * K, 100)]
    return c
def MS():
    c = [("Rdist", 0, 32 * K, 200), ("Rdist", 0, 32 * K, 200)]
    for U in (2, 4):
        for T in (4 * K, 16 * K, 32 * K): c.append(("R3", U, T, 150))
    for T in (4 * K, 32 * K):
        c += [("R1", 3, T, 150), ("Rload", 32, T, 150), ("Mhop", 2, T, 150), ("Afreq", 2, T, 150)]
    for T in (8 * K, 32 * K): c.append(("Acount", 0, T, 150))
    c += [("Morder", 2, 16 * K, 150), ("Morder", 2, 16 * K, 150), ("Mhop", 5, 32 * K, 150), ("Afreq", 1.1, 32 * K, 150)]
    for T in (8 * K, 32 * K): c += [("R1", 3, T, 100), ("Mhop", 2, T, 100), ("Afreq", 2, T, 100)]
    return c
def RETR_PRIMARY():  # dose-ladder TEF 2000/4000: CS retrieval cells at >= 8K
    return [x for x in CS() if x[0] in ("R1", "Rload") and x[2] >= 8 * K and not (x[2] == 2 * K)]
def suite_h(scale, k, cells, cand=True, call_ms=None):
    t = 0; calls = 0
    for task, lvl, T, n in cells:
        s, cl = cell_cost_s(scale, k, T, n, ncand(task, lvl), cand=cand, call_ms=call_ms)
        t += s; calls += cl
    return t / 3600, calls
def tier0_h(scale, k):
    e32 = PRE[scale]["eager"][k][32]; e4 = PRE[scale]["eager"][k][4]
    lm1 = 279 * e32 * 1.3; lm2 = (2000 / 8) * max(e4, 8 * PRE[scale]["graph"][k][4]) * 1.3
    lm4 = 2 * 200 * sum(interp(PRE[scale]["eager"][k], max(1, (512 + g) // K)) for g in (0, 512, 2 * K, 8 * K, 30 * K)) * 1.3
    r0 = 100 * (e4 + e32) * 1.2
    return (lm1 + lm2 + lm4 + r0) / 1000 / 3600
if __name__ == "__main__":
    out = {}
    for scale in ("D384", "D512"):
        print(f"\n######## {scale}")
        print("per-run train GPU-h (A + B(default) + TEF1000), exclusive central / optimistic / CARLA f=0.5 central:")
        for k in KS:
            print(f"  k={k:2d}: {run_h(scale,k):.2f} / {run_h(scale,k,opt=True):.2f} / {run_h(scale,k)*carla(0.5):.2f}   "
                  f"[A {h(SC[scale]['A'],stageA(scale,k))*RUN_OVH:.2f} h @ {stageA(scale,k)/1e3:.1f}K; B+TEF {h(SC[scale]['B']+TEF,stageB(scale,k))*RUN_OVH:.2f} h @ {stageB(scale,k)/1e3:.1f}K]")
        print("per-checkpoint eval GPU-h (current scorer; candidates) CS / DG / DC / FD / MS / RETR / T0 ; CS calls")
        for k in (0, 1, 3, 12):
            cs, csc = suite_h(scale, k, CS()); dg, _ = suite_h(scale, k, DG(k)); dc, _ = suite_h(scale, k, DC())
            fd, fdc = suite_h(scale, k, FD()); ms, _ = suite_h(scale, k, MS()); rp, _ = suite_h(scale, k, RETR_PRIMARY())
            print(f"  k={k:2d}: CS {cs:.2f} | DG {dg:.2f} | DC {dc:.2f} | FD {fd:.2f} | MS {ms:.2f} | RETR {rp:.2f} | T0 {tier0_h(scale,k):.2f} | CS calls {csc} FD calls {fdc}")
        print("  same, dev suites without candidates (DG/DC/FD/RETR):")
        for k in (0, 1, 3, 12):
            dg, _ = suite_h(scale, k, DG(k), cand=False); dc, _ = suite_h(scale, k, DC(), cand=False)
            fd, _ = suite_h(scale, k, FD(), cand=False); rp, _ = suite_h(scale, k, RETR_PRIMARY(), cand=False)
            print(f"  k={k:2d}: DG {dg:.2f} | DC {dc:.2f} | FD {fd:.2f} | RETR {rp:.2f}")
        print("  CS/MS with CUDA-graph candidate calls (2 ms, projection):")
        for k in (0, 1, 3, 12):
            cs, _ = suite_h(scale, k, CS(), call_ms=CALL_GRAPH); ms, _ = suite_h(scale, k, MS(), call_ms=CALL_GRAPH)
            print(f"  k={k:2d}: CS {cs:.2f} | MS {ms:.2f}")
