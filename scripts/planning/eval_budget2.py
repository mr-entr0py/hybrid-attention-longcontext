import importlib.util, sys
spec = importlib.util.spec_from_file_location("eb", "eval_budget.py"); eb = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; 
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(eb)
K = 1024; Ls = eb.Ls
def conf():
    c = []
    for D in (3, 15):
        for T in Ls: c.append(("R1", D, T, 300 if T >= 16*K else 200))
    for T in Ls[:3]:
        for N in (8, 32): c.append(("Rload", N, T, 200))
    for N in (8, 32, 128, 512): c.append(("Rload", N, 8*K, 200))
    for T in (16*K, 32*K):
        for N in (8, 32, 128, 512, 1024): c.append(("Rload", N, T, 300))
    c.append(("Rload", 2048, 32*K, 150))
    for T in Ls[:3]:
        for h in (1, 2): c.append(("Mhop", h, T, 200))
    for T in Ls[3:]:
        for h in (1, 2, 3, 4): c.append(("Mhop", h, T, 200))
    for T in Ls[:3]:
        for r in (4, 2): c.append(("Afreq", r, T, 200))
    for T in Ls[3:]:
        for r in (4, 2, 1.5, 1.25): c.append(("Afreq", r, T, 200))
    return c
def retr_primary():  # dose-ladder / branch check: retrieval ladder at 8K,16K,32K only
    return [x for x in conf() if x[0] in ("R1", "Rload") and x[2] >= 8*K]
def mech():
    c = []
    for f in ("natural", "repetitive"): c.append(("Rdist", f, 32*K, 200))
    for U in (2, 4):
        for T in (4*K, 16*K, 32*K): c.append(("R3", U, T, 150))
    for task in ("R1D3", "RloadN32", "Mhop2", "Afreq2"):
        for T in (4*K, 32*K): c.append(("C2", task, T, 150))
    for T in (8*K, 32*K): c.append(("Acount", 0, T, 150))
    for o in ("fwd", "rev"): c.append(("Morder", o, 16*K, 150))
    c.append(("Mhop", 5, 32*K, 150)); c.append(("Afreq", 1.1, 32*K, 150))
    for task in ("R1D3", "Mhop2", "Afreq2"):
        for T in (8*K, 32*K): c.append(("T4", task, T, 100))
    return c
devg, devc = eb.cells_dev_gate(), eb.cells_dev_ckpt()
t0 = [("LM1", 0, 32*K, 400), ("LM2", 0, 4*K, 2000), ("LM4", 0, 1*K, 400), ("LM4", 0, 1*K, 400),
      ("LM4", 0, 4*K, 400), ("LM4", 0, 8*K, 400), ("LM4", 0, 32*K, 400)]
H = eb.hours; toks = eb.toks
print("tokens(M) conf %.1f retr-primary %.1f mech %.1f devgate %.1f devckpt %.1f tier0 %.1f" % (
  toks(conf())/1e6, toks(retr_primary())/1e6, toks(mech())/1e6, toks(devg)/1e6, toks(devc)/1e6, toks(t0)/1e6))
for d, seeds in ((384, 5), (512, 4)):
    ks = (0, 1, 3, 12); tot = {}
    tot["core conf+tier0"] = sum(seeds * (H(conf(), k, d) + H(t0, k, d)) for k in ks)
    tot["core devgate k0,k12"] = seeds * (H(devg, 0, d) + H(devg, 12, d))
    tot["core devckpt 250/500/1000"] = sum(seeds * 3 * H(devc, k, d) for k in ks)
    tot["mech (k0,k12 s0-1; k1,k3 s0)"] = 2 * (H(mech(), 0, d) + H(mech(), 12, d)) + H(mech(), 1, d) + H(mech(), 3, d)
    tot["Fgc k12 x2 (conf+t0+devg)"] = 2 * (H(conf(), 12, d) + H(t0, 12, d) + H(devg, 12, d))
    tot["NoPE k3,k12 (conf+t0; devg k12)"] = H(conf(), 3, d) + H(t0, 3, d) + H(conf(), 12, d) + H(t0, 12, d) + H(devg, 12, d)
    tot["TEF dose 2000/4000 retr-primary + devckpt"] = sum(2 * H(retr_primary(), k, d) + 2 * H(devc, k, d) for k in ks)
    tot["2xB k0,3,12 (conf + devg k0,k12)"] = sum(H(conf(), k, d) for k in (0, 3, 12)) + H(devg, 0, d) + H(devg, 12, d)
    s = sum(tot.values())
    print(f"--- d{d} (n0={seeds}) Tier-A evaluation total {s:.1f} GPU-h")
    for k_, v in tot.items(): print(f"   {k_:45s} {v:5.2f}")
    print("   per-model conf: " + ", ".join(f"k{k} {H(conf(),k,d):.2f}" for k in ks))
