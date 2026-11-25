#!/usr/bin/env python
"""R2-LIVE v2 C2 compliance scan (experiments/prereg/R2_live.v2.yaml, C2_live_window). CPU only, read only.

Window: from the first O4-Z lock monitor file to now (``--final`` once SUITE-PF has ended; otherwise the output
is marked INTERIM). Checks the four registered requirements and prints counts, run ids and process names only
(never a loss or score). Limitation, recorded in the output: lock monitor samples exist only while a project lock
is held, so a project process started outside the lock while nobody held it would not appear in them.
Output: experiments/repro/R2-LIVE/c2_compliance.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALLOWED_FOREIGN = ("gnome-remote-desktop", "Xorg", "gnome-shell", "Xwayland")
TERMINAL = {"completed", "crashed", "failed", "infra_failed_attempt", "suspect_sw_failure", "paused", "diverged", "oom"}


def rows_of(p: Path):
    for ln in p.read_text(errors="replace").splitlines():
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="SUITE-PF has ended: the window is closed")
    a = ap.parse_args()
    mons = sorted((REPO / "runs" / "_logs" / "gpumon_lock").glob("*.jsonl"))
    starts = [m for m in mons if "O4-Z" in m.name or "O4_Z" in m.name or "O4Z" in m.name]
    if not starts:
        raise SystemExit("no O4-Z lock monitor file found: cannot fix the window start")
    first = next(rows_of(starts[0]))["ts"]
    n_files = n_samples = 0
    bad_lock, foreign, multi, recorded, carla = [], [], [], [], []
    labels = collections.Counter()
    for m in mons:
        seen = False
        for r in rows_of(m):
            if r.get("ts", 0) < first:
                continue
            seen = True
            n_samples += 1
            lab = r.get("holder_label") or m.stem
            proj = [p for p in r.get("procs", []) if p.get("project")]
            for p in proj:
                if p.get("lock_ok") is False:
                    bad_lock.append({"file": m.name, "ts": r["ts"], "pid": p.get("pid")})
            for p in r.get("procs", []):
                if not p.get("project") and not any(x in (p.get("name") or "") for x in ALLOWED_FOREIGN):
                    hit = {"file": m.name, "ts": r["ts"], "holder": lab, "name": Path(str(p.get("name"))).name}
                    # the registered exception is the IDLE CARLA-env client; idleness is not measured by these samples
                    (carla if "/carla_env/bin/python" in str(p.get("name")) else foreign).append(hit)
            if len(proj) > 1:
                multi.append({"file": m.name, "ts": r["ts"], "holder": lab, "pids": [p.get("pid") for p in proj]})
            if r.get("violations"):
                recorded.append({"file": m.name, "ts": r["ts"], "holder": lab, "violations": r["violations"]})
        if seen:
            n_files += 1
            labels[m.stem.split("_", 1)[1].rsplit("_", 1)[0]] += 1

    regs = [REPO / "experiments" / "registry.jsonl", *sorted((REPO / "experiments" / "repro").rglob("registry.jsonl")),
            *sorted((REPO / "experiments").glob("feas_smoke*/registry.jsonl"))]
    missing_infra, journal_bad, n_rows = [], [], 0
    for reg in regs:
        for r in rows_of(reg):
            st = r.get("status")
            if r.get("ts", 0) < first or not (st == "attempt_end" or st in TERMINAL):
                continue
            n_rows += 1
            where = {"registry": str(reg.relative_to(REPO)), "run_id": r.get("run_id"), "status": st, "ts": r.get("ts")}
            if "infra_class" not in r and "infra" not in r:
                missing_infra.append(where)
                continue
            blk = r.get("infra") if isinstance(r.get("infra"), dict) else {}
            if r.get("journal_ok", blk.get("journal_ok")) is not True:
                journal_bad.append(where)

    def cap(x):
        return {"n": len(x), "examples": x[:25]}

    out = {"item": "R2-LIVE C2", "prereg": "experiments/prereg/R2_live.v2.yaml",
           "status": "FINAL" if a.final else "INTERIM (window still open: SUITE-PF has not ended)",
           "generated": time.strftime("%FT%T%z"), "window_start_ts": first, "n_monitor_files": n_files,
           "n_samples": n_samples, "n_registries": len(regs), "n_terminal_and_attempt_end_rows": n_rows,
           "req1_project_lock_ok_false": cap(bad_lock), "req2_foreign_compute_samples": cap(foreign),
           "req2_foreign_by_name": dict(collections.Counter(x["name"] for x in foreign)),
           "carla_env_client_samples_idleness_not_checked": cap(carla),
           "more_than_one_project_process": cap(multi), "monitor_recorded_violations": cap(recorded),
           "req3_journal_ok_not_true": cap(journal_bad), "req4_rows_without_infra_fields": cap(missing_infra),
           "limitation": "monitor samples exist only while a project lock is held",
           "monitor_labels": dict(labels)}
    zero = not (bad_lock or foreign or recorded or journal_bad or missing_infra)
    out["zero_violations"] = zero
    d = REPO / "experiments" / "repro" / "R2-LIVE"
    d.mkdir(parents=True, exist_ok=True)
    (d / "c2_compliance.json").write_text(json.dumps(out, indent=1))
    print(out["status"], "| files", n_files, "samples", n_samples, "rows", n_rows)
    for k, v in out.items():
        if isinstance(v, dict) and "n" in v:
            print(f"  {k}: {v['n']}")
    print("  foreign by name:", out["req2_foreign_by_name"], "| zero_violations:", zero)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
