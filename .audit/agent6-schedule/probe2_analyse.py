"""Attribute foreign SM (pmon side-car) to each probe2 measurement window; compare with idle bench."""
import json, collections, datetime, statistics as st
D = "/home/sedrica-lynx/Downloads/outeropt/.audit/agent6-schedule/"
recs = [json.loads(l) for l in open(D + "probe2.jsonl")]
own = recs[0]["pid"]
per_sec = collections.defaultdict(lambda: {"foreign": 0.0, "own": 0.0, "fb_foreign": 0.0, "procs": {}})
for line in open(D + "probe2_pmon.txt"):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split()
    date, tm, pid, typ, sm, fb, name = p[0], p[1], int(p[3]), p[4], p[5], p[10], p[-1]
    ts = datetime.datetime.strptime(date + tm, "%Y%m%d%H:%M:%S").timestamp()
    s = 0.0 if sm == "-" else float(sm)
    fbv = 0.0 if fb == "-" else float(fb)
    if pid == own:
        per_sec[ts]["own"] += s
    else:
        per_sec[ts]["foreign"] += s
        per_sec[ts]["fb_foreign"] += fbv
        if s > 0: per_sec[ts]["procs"][name] = per_sec[ts]["procs"].get(name, 0) + s
idle = {("D384","C0","4k"):127.0,("D384","C3","4k"):128.8,("D384","C12","4k"):137.8,
        ("D384","C0","32k"):107.3,("D384","C3","32k"):97.7,("D384","C12","32k"):77.9,
        ("D512","C0","32k"):78.9,("D512","C3","32k"):71.6,("D512","C12","32k"):56.0}
rows = []
for r in recs:
    if r.get("test") not in ("T1", "T2") or r.get("status") != "ok":
        continue
    a, b = r["wall_start"], r["wall_end"]
    secs = [v for t, v in per_sec.items() if a - 0.5 <= t <= b + 0.5]
    f = st.mean([v["foreign"] for v in secs]) if secs else float("nan")
    o = st.mean([v["own"] for v in secs]) if secs else float("nan")
    fb = st.mean([v["fb_foreign"] for v in secs]) if secs else float("nan")
    if r["test"] == "T1":
        key = (r["scale"], r["config"], r["shape"])
        ib = idle.get(key) if r["batch"] == "packed" else None
        ratio = (r["tok_per_s_median"] / 1e3 / ib) if ib else None
        print(f"T1 {r['scale']} {r['config']:>3} {r['shape']:>3} {r['batch']:>6}: {r['tok_per_s_median']/1e3:6.1f}K (best {r['tok_per_s_best']/1e3:6.1f}K)  "
              f"foreignSM {f:5.1f}%  ownSM {o:5.1f}%  foreignFB {fb:6.0f} MiB  n_sec={len(secs)}  idle={ib}  ratio={ratio if ratio is None else round(ratio,3)}")
        rows.append(dict(key=key, batch=r["batch"], tok=r["tok_per_s_median"], best=r["tok_per_s_best"], fsm=f, idle=ib))
    else:
        print(f"T2 {r['scale']} {r['config']:>3} T={r['T']:>5} B={r['B']}: prefill {r['prefill_ms']:7.2f} ms  cand {r['cand_call_ms']:6.2f}  gold {r['gold_call_ms']:6.2f}  extend6 {r['extend6_ms']:6.2f} ms  foreignSM {f:5.1f}%  n_sec={len(secs)}")
# single vs packed ratio (same config, adjacent windows)
print()
for sc, cf in (("D384","C0"),("D384","C3"),("D384","C12"),("D512","C0"),("D512","C12")):
    pk = [x for x in rows if x["key"] == (sc, cf, "32k") and x["batch"] == "packed"][0]
    sg = [x for x in rows if x["key"] == (sc, cf, "32k") and x["batch"] == "single"][0]
    print(f"{sc} {cf}: single/packed tok/s = {sg['tok']/pk['tok']:.3f}  (best {sg['best']/pk['best']:.3f}); foreignSM packed {pk['fsm']:.1f}% single {sg['fsm']:.1f}%")
