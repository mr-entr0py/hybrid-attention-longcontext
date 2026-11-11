"""Agent-6 resource/schedule model v2 (report-only; CPU). Every input is cited in
reports/compute/experiment_schedule.md §2-§4. Prints per-run, per-checkpoint and per-stage GPU-hours for the
D384 and D512 paths, exclusive vs CARLA-shared, and writes schedule_model.json.

Inputs
- Training rates (tok/s, exclusive, production trainer, real StageStream data): per-window regression of
  G0 pass-1/2 throughput on foreign SM% (g0_fit.py), intercept at 0% foreign SM. k in {1,2,6} interpolated
  linearly in per-token time (checked on the idle bench: C3 predicted 10.20 vs measured 10.23 us/token).
- TEF rate = Stage-B rate (both are 1x32K checkpointed, attention load dominated by >=16K documents/items).
- CARLA factors: light 1.15 (G0 pass-1/2 contention, foreign SM 10-25%), heavy 1.45 (probe2 2026-11-10 20:50,
  same code path as the idle bench: 4K x0.71-0.73, 32K x0.54-0.74).
- Eval: exact scorer call counts from the frozen items (count_calls.py); per-call cost measured (probe2, eager,
  launch-bound, identical at T=4K B=8 and T=32K B=1, and identical to the 15:32 probe): C0 11.84 ms,
  C3 11.23 ms, C12 7.77 ms; prefill from cost-bench session 1 eager tables (idle).
"""
import json, math
K = 1024
OUT = "/home/sedrica-lynx/Downloads/outeropt/.audit/agent6-schedule/"
CC = json.load(open(OUT + "call_counts.json"))

# ----------------------------------------------------------------------------------------- training
RA = {"D384": {0: 117.9e3, 3: 116.8e3, 12: 121.2e3}, "D512": {0: 92.8e3, 3: 90.1e3, 12: 88.1e3}}
RB = {"D384": {0: 85.3e3, 12: 50.8e3}, "D512": {0: 75.0e3, 12: 36.1e3}}
def rate_A(sc, k):
    p = RA[sc]
    if k in p: return p[k]
    lo, hi = (0, 3) if k < 3 else (3, 12)
    t = 1 / p[lo] + (1 / p[hi] - 1 / p[lo]) * (k - lo) / (hi - lo)
    return 1 / t
def rate_B(sc, k):
    p = RB[sc]; t = 1 / p[0] + (1 / p[12] - 1 / p[0]) * k / 12
    return 1 / t
OVH = 1.05            # startup, checkpoint writes, validation, milestones [INF]
TOK = {"D384": dict(A=600047616, B=60e6, Bfeas=(30015488, 90046464), Afeas=300023808, n0=5),
       "D512": dict(A=800063488, B=0.2e9, Bfeas=(0.1e9, 0.3e9), Afeas=300023808, n0=4)}
TEF = 65536000
TEF4000_EXTRA = 3000 * 65536
LR_TOK = 1145 * 131072
def h(tok, r): return tok / r / 3600 * OVH
def run(sc, k, A=True, B=None, tef=True, mult=1.0):
    s = TOK[sc]; Bt = s["B"] if B is None else B
    a = h(s["A"], rate_A(sc, k)) if A is True else (h(A, rate_A(sc, k)) if A else 0.0)
    b = h(Bt, rate_B(sc, k)) if Bt else 0.0
    t = h(TEF, rate_B(sc, k)) if tef else 0.0
    return mult * (a + b + t), mult * a, mult * b, mult * t

# ----------------------------------------------------------------------------------------- evaluation
CALL_MS = {0: 11.84, 3: 11.23, 12: 7.77}
def call_ms(sc, k):
    v = CALL_MS[k] if k in CALL_MS else (12 - k) * CALL_MS[0] / 12 + k * CALL_MS[12] / 12
    return v * (1.05 if sc == "D512" else 1.0)   # [INF] D512 same launch count, slightly larger kernels
EAGER_B1 = {"D384": {0: (13.15, 13.09, 19.91, 52.20), 1: (12.71, 12.58, 22.04, 60.59), 2: (12.18, 12.02, 24.31, 69.14),
                     3: (11.60, 11.60, 26.54, 77.65), 6: (9.85, 9.97, 33.39, 103.22), 12: (5.25, 7.87, 46.75, 154.42)},
            "D512": {0: (13.29, 13.38, 31.76, 77.18), 1: (12.72, 13.11, 34.96, 88.61), 2: (12.29, 12.30, 37.64, 100.12),
                     3: (11.91, 12.04, 40.74, 111.42), 6: (9.88, 10.51, 49.96, 146.09), 12: (5.27, 11.83, 67.98, 214.79)}}
EAGER_B8 = {"D384": {0: (13.38, 44.84, 203.01, 408.91), 1: (12.86, 46.81, 223.05, 478.97), 2: (12.21, 48.86, 243.08, 549.27),
                     3: (12.04, 50.96, 263.37, 620.07), 6: (10.58, 56.96, 323.79, 831.17), 12: (11.49, 69.11, 444.87, 1150.0)},
            "D512": {0: (13.67, 70.34, 294.79, 629.0), 1: (13.95, 73.11, 321.55, 727.73), 2: (14.30, 75.88, 348.31, 827.06),
                     3: (14.37, 78.68, 374.94, 941.57), 6: (15.38, 87.16, 484.03, 1200.0), 12: (16.76, 104.08, 616.12, 1600.0)}}
XS = (1 * K, 4 * K, 16 * K, 32 * K)
def interp(vals, T):
    if T <= XS[0]: return vals[0] * max(T, 64) / XS[0] if T < 256 else vals[0]
    for i in range(3):
        if XS[i] <= T <= XS[i + 1]:
            return vals[i] + (vals[i + 1] - vals[i]) * (T - XS[i]) / (XS[i + 1] - XS[i])
    return vals[-1] * T / XS[-1]
def prefill_ms(sc, k, T, B):
    kk = k if k in EAGER_B1[sc] else min(EAGER_B1[sc], key=lambda x: abs(x - k))
    b1 = interp(EAGER_B1[sc][kk], T); b8 = interp(EAGER_B8[sc][kk], T)
    if B == 1: return b1
    if B == 8: return b8
    return max(b1, b8 * B / 8)
def suite_h(sc, k, suite, cand=True, cells=None, rescore=True):
    tot = 0.0
    for c in CC[suite]["cells"]:
        if cells is not None and not cells(c["cell"]): continue
        calls = c["calls_cand"] if cand else c["calls_nocand"]
        t = c["batches"] * prefill_ms(sc, k, c["T_max"], c["B"]) + calls * call_ms(sc, k)
        if rescore:   # 5% of contexts, one single-pass forward per query at B=1
            t += 0.05 * c["n_q"] * prefill_ms(sc, k, c["T_max"], 1)
        tot += t
    return tot / 1000 / 3600 * 1.03   # +3% per-cell I/O, item loading, meta writing [INF]
is_mhop = lambda n: n.startswith("M-hop")
RETR_LONG = lambda n: (n.startswith("R1") or n.startswith("R-load")) and (n.endswith("_16K") or n.endswith("_32K") or "nofill" in n)
def DG(sc, k, cand=True): return suite_h(sc, k, "dev/DG", cand, cells=(None if k == 12 else (lambda n: not is_mhop(n))))
def CS(sc, k, cand=True): return suite_h(sc, k, "test/CS", cand)
def RETR(sc, k, cand=True): return suite_h(sc, k, "test/CS", cand, cells=RETR_LONG)
def MS(sc, k, cand=True): return suite_h(sc, k, "test/MS", cand)
def DC(sc, k, cand=True): return suite_h(sc, k, "dev/DC", cand)
def FD(sc, k, cand=True): return suite_h(sc, k, "dev/FD", cand)
def T0(sc, k): return suite_h(sc, k, "test/Tier0", False, rescore=False)
FD_VERDICT = lambda n: n.startswith("R1") or n.startswith("R-load")
def P4(sc, k):   # fp32 compute + tiled fp32 attention, no candidates, verdict cells [INF: x3 GDN, x6 attention vs bf16]
    base = suite_h(sc, k, "dev/FD", False, cells=FD_VERDICT, rescore=False)
    return base * (3.0 * (12 - k) / 12 + 6.0 * k / 12)
def P1(sc, k): return suite_h(sc, k, "dev/FD", False, rescore=False) * 1.1

def per_ckpt_eval(sc, k, mode):
    """mode 'cand' = current scorer everywhere; 'cs_only' = candidates only on CS/MS; 'graph' = CUDA-graph calls (~1.5 ms)."""
    global CALL_MS
    saved = dict(CALL_MS)
    if mode == "graph":
        CALL_MS = {0: 1.5, 3: 1.6, 12: 1.9}
    cand_dev = mode != "cs_only"
    r = dict(CS=CS(sc, k), MS=MS(sc, k), RETR=RETR(sc, k, cand=True if mode != "cs_only" else False),
             DG=DG(sc, k, cand_dev), DC=DC(sc, k, cand_dev), FD=FD(sc, k, cand_dev), T0=T0(sc, k))
    CALL_MS = saved
    return r

# ----------------------------------------------------------------------------------------- stage budgets
def stage_budget(sc, mode="cand", carla=1.0, k_star=1):
    """Tier-A stage budget. carla multiplies training only (eval calls are launch-bound; prefill share small)."""
    n0 = TOK[sc]["n0"]; E = {k: per_ckpt_eval(sc, k, mode) for k in (0, 1, 2, 3, 6, 12)}
    S = {}
    # Phase II pre-FEAS (PI rulings D089; prereg gpu_h fields)
    S["PhaseII pre-FEAS (O4 Z/B/A, TC-PF, FEAS-SMOKE, R2e-X, VT-13r, VT-22b, VT-23b, R6, SUITE-PF)"] = (1.0 + 1.0 + 2.4 + 0.45 + 0.2 + 0.6 + 0.55 + 0.15 + 0.2 + 0.45 + 0.35, 0.0)
    # FEAS-384 (plan.yaml S2): 5 Stage A (300M), branches 30M/90M x {C0,C12} x {101,102} + SSMax 30M, 9 TEF
    def feas(scale, seeds, Bs, ssmax=True):
        tr = 0.0; ev = 0.0; ck = []
        A = TOK[scale]["Afeas"]
        for s in seeds:
            for k in (0, 12):
                tr += h(A, rate_A(scale, k))
                for Bt in Bs:
                    tr += h(Bt, rate_B(scale, k)) + h(TEF, rate_B(scale, k)); ck.append(k)
        if ssmax:
            tr += h(A, rate_A(scale, 12)) + h(Bs[0], rate_B(scale, 12)) + h(TEF, rate_B(scale, 12)); ck.append(12)
        Es = {k: per_ckpt_eval(scale, k, mode) for k in (0, 12)}
        for k in ck:   # FD P0 at TEF1000, DC at TEF250, P1 full FD, P4 verdict cells, LM-2 x2 (~0.01 h)
            ev += Es[k]["FD"] + Es[k]["DC"] + P1(scale, k) + P4(scale, k) + 0.02
        ev += 5 * 0.05   # RP-2 LM-3 reading on end-of-Stage-A checkpoints + R8 ICL helper [INF]
        return tr, ev
    tr, ev = feas("D384", (101, 102), TOK["D384"]["Bfeas"])
    S["S2 FEAS-384"] = (tr * carla, ev)
    if sc == "D512":
        tr, ev = feas("D512", (101,), TOK["D512"]["Bfeas"])
        S["S2 FEAS-512 (conditional on FEAS-384 failing)"] = (tr * carla, ev)
    S["S3 LR sweep 12 x 0.15B (+R9 k=1 run)"] = (sum(h(LR_TOK, rate_A(sc, k)) for k in (0, 3, 12)) * 4 * carla + h(LR_TOK, rate_A(sc, 1)) * carla, 0.15)
    S["G5 add-only Triton re-warm"] = (0.45, 0.0)
    # S4a + S4b core
    core_tr = sum(run(sc, k)[0] for k in (0, 1, 3, 12)) * n0
    core_ev = n0 * sum(E[k]["CS"] + 2 * E[k]["DC"] + E[k]["T0"] for k in (0, 1, 3, 12)) + n0 * (E[0]["DG"] + E[12]["DG"])
    S["S4 core k in {0,1,3,12} x n0 (incl. S4a G-FULL)"] = (core_tr * carla, core_ev)
    S["VT-13 standing replicate x2 (C12 s0, 1,000 steps audit mode)"] = (2 * h(1000 * 131072, rate_A(sc, 12) * 0.75), 0.0)
    # S5a controls
    c_tr = 2 * run(sc, 12, mult=1.06)[0] + run(sc, 3)[0] + run(sc, 12)[0]
    c_ev = 2 * (E[12]["CS"] + 2 * E[12]["DC"] + E[12]["T0"]) + (E[3]["CS"] + 2 * E[3]["DC"] + E[3]["T0"]) + (E[12]["CS"] + 2 * E[12]["DC"] + E[12]["T0"])
    S["S5a controls (X-gc12 x2, X-nope3, X-nope12)"] = (c_tr * carla, c_ev)
    # S5b seed-0 branches
    b_tr = sum(h(TEF4000_EXTRA, rate_B(sc, k)) for k in (0, 1, 3, 12)) + sum(run(sc, k, A=False, B=2 * TOK[sc]["B"])[0] for k in (0, 3, 12))
    # INTERFACES.md L127: dose guard = DC at tef2000 and tef4000 of the tef4000 branch (CS at tef4000 optional);
    # saturation guard = CS of the stageB2x branch (+ DC at TEF 250/500 and Tier 0 like any run)
    b_ev = sum(2 * E[k]["DC"] for k in (0, 1, 3, 12)) + sum(E[k]["CS"] + 2 * E[k]["DC"] + E[k]["T0"] for k in (0, 3, 12))
    S["S5b seed-0 TEF-4000 x4 + 2x Stage-B x3"] = (b_tr * carla, b_ev)
    # MS on C0/C12 seeds 0-1, C1, C3, X-gc12, X-nope3, X-nope12 (seed 0)
    S["S6 MS suite (9 checkpoints)"] = (0.0, 2 * E[0]["MS"] + 2 * E[12]["MS"] + E[1]["MS"] + E[3]["MS"] + 2 * E[12]["MS"] + E[3]["MS"])
    S["S6 cost benchmarks session 2 (idle window)"] = (3.0, 0.0)
    trig_tr = sum(run(sc, k)[0] for k in (0, 1, 3, 12)) * 2 * carla
    trig_ev = 2 * sum(E[k]["CS"] + 2 * E[k]["DC"] + E[k]["T0"] for k in (0, 1, 3, 12)) + 2 * (E[0]["DG"] + E[12]["DG"])
    tierB_tr = (2 * run(sc, 2)[0] + 2 * run(sc, 6)[0] + 2 * n0 * run(sc, k_star)[0] + 2 * run(sc, 0, mult=1.17)[0] + 2 * run(sc, k_star, mult=1.17)[0]) * carla
    tierB_ev = 2 * (E[2]["CS"] + 2 * E[2]["DC"] + E[2]["T0"]) + 2 * (E[6]["CS"] + 2 * E[6]["DC"] + E[6]["T0"]) + 2 * n0 * (E[k_star]["CS"] + 2 * E[k_star]["DC"] + E[k_star]["T0"]) + 4 * (E[0]["CS"] + 2 * E[0]["DC"] + E[0]["T0"])
    return S, (trig_tr, trig_ev), (tierB_tr, tierB_ev)

if __name__ == "__main__":
    res = {}
    for sc in ("D384", "D512"):
        print(f"\n################ {sc}")
        print("Per-run training GPU-h, exclusive central (A / B(default) / TEF1000 / total); x1.15 light CARLA; x1.45 heavy CARLA; protocol §13.2")
        prot = {"D384": {0: 1.76, 1: 1.80, 2: 1.83, 3: 1.90, 6: 1.98, 12: 2.21}, "D512": {0: 3.55, 1: 3.65, 2: 3.75, 3: 3.95, 6: 4.18, 12: 4.81}}
        for k in (0, 1, 2, 3, 6, 12):
            t, a, b, tf = run(sc, k)
            print(f"  k={k:2d}: A {a:.2f} @ {rate_A(sc,k)/1e3:.1f}K | B {b:.2f} | TEF {tf:.2f} @ {rate_B(sc,k)/1e3:.1f}K | total {t:.2f} | light {t*1.15:.2f} | heavy {t*1.45:.2f} | protocol {prot[sc][k]:.2f}")
        print("Per-checkpoint eval GPU-h: mode cand (current scorer) | cs_only (candidates only on CS/MS) | graph (CUDA-graph calls)")
        for k in (0, 1, 3, 12):
            for mode in ("cand", "cs_only", "graph"):
                e = per_ckpt_eval(sc, k, mode)
                print(f"  k={k:2d} {mode:7s}: " + " | ".join(f"{n} {v:.2f}" for n, v in e.items()))
        print("Protocol §9.9 per model (D384/D512): CS 0.12/0.14/0.16/0.29 h | 0.17/0.19/0.23/0.41 h")
        for mode in ("cand", "cs_only", "graph"):
            for cl, cf in (("exclusive", 1.0), ("CARLA-light", 1.15), ("CARLA-heavy", 1.45)):
                S, trig, tb = stage_budget(sc, mode, cf)
                tr = sum(v[0] for v in S.values()); ev = sum(v[1] for v in S.values())
                key = f"{sc}|{mode}|{cl}"
                res[key] = dict(stages={n: [round(a, 2), round(b, 2)] for n, (a, b) in S.items()}, train=tr, eval=ev,
                                trigger=[round(trig[0], 2), round(trig[1], 2)], tierB=[round(tb[0], 2), round(tb[1], 2)])
                if cl == "exclusive" or mode == "cand":
                    print(f"\n  == {key}: Tier-A GPU-h (incl. pre-FEAS Phase II) train {tr:.1f} + eval {ev:.1f} = {tr+ev:.1f};"
                          f" trigger +{trig[0]+trig[1]:.1f}; Tier B +{tb[0]+tb[1]:.1f}")
                    if cl == "exclusive" and mode == "cand":
                        for n, (a, b) in S.items():
                            print(f"     {n:75s} train {a:6.2f}  eval {b:6.2f}  total {a+b:6.2f}")
    json.dump(res, open(OUT + "schedule_model.json", "w"), indent=1)
