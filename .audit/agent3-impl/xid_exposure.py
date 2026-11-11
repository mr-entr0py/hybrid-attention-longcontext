"""Exposure of each job family to the Xid-8 faults: sum of start->end wall time per (queue, family).
Reads the S1/R4/R5 queue logs (start/end lines). Output: per-run durations, faults, totals."""
import re, sys, json
from datetime import datetime
logs = {"S1": "experiments/s1_zoology/logs/queue.log", "R4": "experiments/repro/R4/logs/queue.log",
        "R5": "experiments/repro/R5/logs/queue.log"}
pat_s = re.compile(r"^\[(?:\w+-queue)\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) start (\S+) attempt (\d+)")
pat_e = re.compile(r"^\[(?:\w+-queue)\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) end (\S+) attempt (\d+) rc=(\-?\d+)(?: flagged=(\w+))?")
rows = []
for q, p in logs.items():
    open_ = {}
    for ln in open(p):
        m = pat_s.match(ln)
        if m:
            open_[(m.group(2), m.group(3))] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            continue
        m = pat_e.match(ln)
        if m:
            k = (m.group(2), m.group(3))
            t1 = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            t0 = open_.pop(k, None)
            if t0 is None:
                continue
            name = m.group(2)
            fam = "gdn" if ("gdn" in name or "gated_delta" in name) else "attn"
            rows.append(dict(queue=q, run=name, attempt=int(m.group(3)), rc=int(m.group(4)),
                             flagged=m.group(5), start=str(t0), end=str(t1), secs=(t1 - t0).total_seconds(), family=fam))
tot = {}
for r in rows:
    key = (r["queue"], r["family"])
    t = tot.setdefault(key, dict(secs=0.0, n=0, crashes_rc1=0))
    t["secs"] += r["secs"]; t["n"] += 1; t["crashes_rc1"] += int(r["rc"] != 0)
for r in rows:
    print(json.dumps(r))
print("TOTALS")
for k, v in sorted(tot.items()):
    print(k, {**v, "hours": round(v["secs"] / 3600, 2)})
g = sum(v["secs"] for (q, f), v in tot.items() if f == "gdn") / 3600
a = sum(v["secs"] for (q, f), v in tot.items() if f == "attn") / 3600
print(f"GDN-family Zoology-harness wall hours: {g:.2f}; attention-family: {a:.2f}")
