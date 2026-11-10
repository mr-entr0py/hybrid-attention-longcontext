"""Per-window G0 throughput vs foreign-SM contention (report-only analysis; reads experiments/s0/g0*/)."""
import json, glob, os, statistics as st, collections, sys
out = []
for pas, root in (("p1", "experiments/s0/g0"), ("p2", "experiments/s0/g0_rerun")):
    for mf in sorted(glob.glob(f"{root}/runs_root/runs/*/metrics.jsonl")):
        rid = mf.split("/")[-2]
        stage = "B" if rid.startswith("S0G0B") else "A"
        parts = rid.split("-")
        scale, cfg = parts[1], parts[2]
        gf = f"{root}/gpu/{stage}_{scale}_{cfg}.jsonl"
        if not os.path.exists(gf):
            continue
        g = [json.loads(l) for l in open(gf)]
        recs = [json.loads(l) for l in open(mf)]
        start = [r for r in recs if r.get("kind") == "start"][0]["ts"]
        prev = start
        for r in recs:
            if r.get("kind") != "train":
                continue
            ts = r["ts"]
            win = [x for x in g if prev <= x["ts"] <= ts]
            fsm = st.mean([x["foreign_sm_pct"] for x in win]) if win else float("nan")
            own = st.mean([x["own_sm_pct"] for x in win]) if win else float("nan")
            ffb = st.mean([x["foreign_fb_mib"] for x in win]) if win else float("nan")
            # which foreign procs had sm>0
            act = collections.Counter()
            for x in win:
                for p in x["processes"]:
                    if p["pid"] != x["pid"] and (p.get("sm") or 0) > 0:
                        act[p["name"]] += p["sm"]
            out.append(dict(pass_=pas, stage=stage, scale=scale, cfg=cfg, step=r["step"], tok_s=r["tok_s"],
                            n=len(win), foreign_sm=fsm, own_sm=own, foreign_fb=ffb, active=dict(act)))
            prev = ts
for o in out:
    print(f"{o['pass_']} {o['stage']} {o['scale']} {o['cfg']:>3} step{o['step']:>3} tok/s {o['tok_s']/1e3:6.1f}K  n={o['n']:2d} foreignSM {o['foreign_sm']:5.1f}  ownSM {o['own_sm']:5.1f}  foreignFB {o['foreign_fb']:6.0f}  active={o['active']}")
json.dump(out, open(".audit/agent6-schedule/g0_windows.json", "w"), indent=1)
