"""Summarise every R5 attempt window: rc, infra events, exclusive start, lock wait, and foreign GPU processes
(seconds of overlap, peak MiB). Reads experiments/repro/R5/infra/*.infra.json and the attempt logs.

    python scripts/repro/r5_infra_summary.py   -> experiments/repro/R5/r5_infra_summary.json (+ stdout table)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R5 = REPO / "experiments" / "repro" / "R5"


def main() -> None:
    rows = []
    for p in sorted(R5.glob("infra/R5-*.infra.json"), key=lambda q: json.loads(q.read_text())["start"]):
        d = json.loads(p.read_text())
        tag = d["tag"]
        rid, k = tag[3:].rsplit("-a", 1)
        log = (R5 / "logs" / f"{rid}.attempt{k}.log").read_text(errors="replace")
        m = re.search(r"acquired after ([\d.]+) s", log)
        own = None
        mm = re.search(r"run_one_r5", " ".join(d.get("cmd", [])))
        foreign: dict[str, dict] = {}
        samples = d.get("samples", [])
        iv = d.get("gpu", {}).get("interval_s", 10.0)
        # own pid = the .venv python with the largest memory that is present in most samples
        counts: dict[str, int] = {}
        for x in samples:
            for a in x.get("compute_apps", []):
                pid, name = a.split(",")[0].strip(), a.split(",")[1].strip()
                if "outeropt/.venv/bin/python" in name:
                    counts[pid] = counts.get(pid, 0) + 1
        own = max(counts, key=counts.get) if counts else None
        for x in samples:
            for a in x.get("compute_apps", []):
                parts = [s.strip() for s in a.split(",")]
                pid, name, mib = parts[0], parts[1], float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
                if pid == own or "CARLA" in name or "gnome-remote-desktop" in name:
                    continue
                f = foreign.setdefault(pid, {"name": name, "samples": 0, "peak_mib": 0.0, "first_s": round(x["ts"] - d["start"])})
                f["samples"] += 1
                f["peak_mib"] = max(f["peak_mib"], mib)
        rows.append({"attempt": tag, "start": d["start_iso"], "end": d["end_iso"], "rc": d.get("returncode"),
                     "infra_events": [e["line"].split("kernel: ")[-1] for e in d.get("infra_events", [])],
                     "flagged": d.get("flagged"), "lock_wait_s": float(m.group(1)) if m else None,
                     "own_pid": own, "foreign_gpu_processes": {pid: {**f, "approx_overlap_s": f["samples"] * iv}
                                                               for pid, f in foreign.items()},
                     "max_temp_c": d.get("gpu", {}).get("max_temp_c"), "any_slowdown": d.get("gpu", {}).get("any_slowdown"),
                     "journal_ok": d.get("journal_ok"), "wall_s": round(d["end"] - d["start"], 1)})
    (R5 / "r5_infra_summary.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        fg = "; ".join(f"{pid} {f['name']} ~{f['approx_overlap_s']:.0f}s peak {f['peak_mib']:.0f}MiB"
                       for pid, f in r["foreign_gpu_processes"].items()) or "none"
        print(f"{r['attempt']} | {r['start'][11:]}-{r['end'][11:]} | rc={r['rc']} | events={len(r['infra_events'])} | "
              f"wait={r['lock_wait_s']} | foreign: {fg}")


if __name__ == "__main__":
    main()
