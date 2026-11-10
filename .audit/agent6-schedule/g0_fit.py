import json, numpy as np, collections
W = json.load(open(".audit/agent6-schedule/g0_windows.json"))
idle = {("A","D384","C0"):127.0,("A","D384","C3"):128.8,("A","D384","C12"):137.8,
        ("A","D512","C0"):97.5,("A","D512","C3"):97.9,("A","D512","C12"):99.1,
        ("B","D384","C0"):107.3,("B","D384","C3"):97.7,("B","D384","C12"):77.9,
        ("B","D512","C0"):78.9,("B","D512","C3"):71.6,("B","D512","C12"):56.0}
cells = collections.defaultdict(list)
for w in W:
    cells[(w["stage"], w["scale"], w["cfg"])].append(w)
print("Per-cell OLS: tok/s(K) = a + b*foreignSM%   (all windows; then excluding each run's first window)")
rows=[]
for key in sorted(cells):
    ws = cells[key]
    for excl in (False, True):
        sel = [w for w in ws if not (excl and w["step"] == 9)]
        x = np.array([w["foreign_sm"] for w in sel]); y = np.array([w["tok_s"]/1e3 for w in sel])
        if len(sel) >= 3 and x.std() > 0:
            b, a = np.polyfit(x, y, 1)
        else:
            a, b = float("nan"), float("nan")
        lo = y[x <= 5.0]
        print(f"{key} excl_first={excl!s:5} n={len(sel):2d} a(0%)={a:6.1f}K slope={b:6.3f}K/% "
              f"mean={y.mean():6.1f}K  meanFSM={x.mean():5.1f}%  windows<=5%FSM: n={len(lo)} mean={lo.mean() if len(lo) else float('nan'):6.1f}K  idle={idle.get(key)}")
        if excl: rows.append((key, a, b, y.mean(), x.mean(), len(sel)))
# pooled relative model: y_i = a_c * (1 - beta * x_i/100); estimate a_c per cell, shared beta, by grid search
print()
def fit(stage_filter, excl=True):
    sel = [w for w in W if w["stage"] in stage_filter and not (excl and w["step"] == 9)]
    keys = sorted({(w["stage"], w["scale"], w["cfg"]) for w in sel})
    best = None
    for beta in np.linspace(0, 2.0, 2001):
        sse = 0; acs = {}
        for k in keys:
            ws = [w for w in sel if (w["stage"], w["scale"], w["cfg"]) == k]
            z = np.array([1 - beta * w["foreign_sm"]/100 for w in ws]); y = np.array([w["tok_s"]/1e3 for w in ws])
            a = (z @ y) / (z @ z); acs[k] = a
            sse += ((y - a*z)**2).sum()
        if best is None or sse < best[0]: best = (sse, beta, acs)
    return best
for sf in (("A",), ("B",), ("A","B")):
    sse, beta, acs = fit(sf)
    print(f"pooled stages={sf}: beta={beta:.3f}  (throughput multiplier at f% foreign SM = 1 - beta*f/100)")
    for k, a in acs.items():
        print(f"   {k}: a(0%)={a:6.1f}K  idle bench={idle.get(k)}  ratio a/idle={a/idle[k] if k in idle else float('nan'):.3f}")
