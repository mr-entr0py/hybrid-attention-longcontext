"""Checkpoint storage model (CPU). Sizes measured: D384 full ckpt 331,376,123 B and bf16 milestone 55,190,043-55,271,691 B
(experiments/s0/smoke/ckpt, experiments/s0/g0_rerun/.../milestones); D512 bf16 milestone 98,761,499-98,861,579 B;
D512 full = 12 B/param x 49.4M (fp32 weights + Adam m,v) [INF, same layout as D384: 331.4e6/27.6e6 = 12.0 B/param]."""
FULL = {"D384": 331.4e6, "D512": 12.0 * 49.40e6}
MIL = {"D384": 55.2e6, "D512": 98.8e6}
def unit(sc, n_full_kept, n_mil):
    return n_full_kept * FULL[sc] + n_mil * MIL[sc]
def tierA(sc, prune):
    n0 = 5 if sc == "D384" else 4
    f = 1 if prune else 3   # per phase: final.pt (+2 rolling step_*.pt kept by keep_last=2 if not pruned)
    main = unit(sc, 3 * f, 9)                 # A(final=S) + B(final) + TEF(final); 9 milestones
    anchor_extra = unit(sc, f, 2)             # TEF-4000 continuation: final + milestones 2000/4000
    b2x = unit(sc, 2 * f, 4)                  # 2x Stage B + its TEF
    feas_A = unit(sc if sc == "D384" else "D384", f, 5)   # FEAS-384 Stage A (300M): final + 4 milestones (scaled)
    feas_BT = unit("D384", 2 * f, 4)
    feas384 = 5 * feas_A + 9 * feas_BT
    feas512 = (3 * unit("D512", f, 5) + 5 * unit("D512", 2 * f, 4)) if sc == "D512" else 0
    lr = 12 * unit(sc, f, 1)
    core = 4 * n0 * main; ctrl = 4 * main; s5b = 4 * anchor_extra + 3 * b2x; trig = 8 * main
    tot = core + ctrl + s5b + feas384 + feas512 + lr
    return dict(core=core, controls=ctrl, s5b=s5b, feas=feas384 + feas512, lr=lr, total=tot, trigger=trig)
for sc in ("D384", "D512"):
    for prune in (False, True):
        r = tierA(sc, prune)
        print(sc, "pruned" if prune else "keep_last=2 retained", {k: round(v / 1e9, 1) for k, v in r.items()}, "GB")
