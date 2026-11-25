#!/usr/bin/env python3
"""Orchestrator for the O4 soak on sedrica-mb (runs detached, every 10 min).

1. Keeps the O4 queue alive: if `o4_run.py all` died before the verdict exists, restart it (it resumes from state.json).
2. Writes experiments/repro/O4/PROGRESS.md (parts, events, verdict, alerts).
3. Publishes ONLY small result files to branch `o4-results` via a separate git worktree (the live tree that the
   trainer reads is never switched or modified). Never touches main, never force-pushes.
Stops itself after the verdict is published or after MAX_H hours.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path.home() / "Downloads" / "outeropt"
OUT = REPO / "experiments" / "repro" / "O4"
Q = Path.home() / "outeropt-queue"
WT = Q / "wt"
BRANCH = "o4-results"
MAX_H = 24
LOG = Q / "orchestrator.log"
PY = str(REPO / ".venv" / "bin" / "python")


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')} {m}\n")


def sh(*a, cwd=REPO, check=False):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, check=check)


def queue_alive():
    try:
        os.kill(int((Q / "o4.pid").read_text()), 0)
        return True
    except Exception:
        return False


def state():
    p = OUT / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def progress(st):
    rows, alerts = [], []
    for k in ["Z", "B", "A1", "S1", "S2", "S3", "S4", "A3"]:
        v = st.get(k)
        if not v:
            rows.append(f"| {k} | pending | | | |")
            continue
        rows.append(f"| {k} | {v['reason']} | {v['n_events']} | {v['duration_s']}s | {'BLOCKED' if v['blocked'] else 'ok'} |")
        if v["n_events"]:
            alerts.append(f"{k}: {v['n_events']} kernel-log event(s) (Xid/RC watchdog/OOM) - see {k}/{k}.infra.json")
        if v.get("cuda_error_in_log"):
            alerts.append(f"{k}: CUDA error text in child log")
        if v["blocked"]:
            alerts.append(f"{k}: BLOCKED (foreign={v.get('foreign_disallowed')} journal_ok={v.get('journal_ok')}) - repeated later")
    verdict = (OUT / "o4_result.json").read_text() if (OUT / "o4_result.json").exists() else "{}"
    zres = (OUT / "o4z_result.json").read_text() if (OUT / "o4z_result.json").exists() else "{}"
    md = ["# O4 soak progress (auto-generated)", f"_updated {time.strftime('%F %T %Z')}_", "",
          "| part | end reason | kernel events | duration | window |", "|---|---|---|---|---|", *rows, "",
          "## Alerts", *(["- " + a for a in alerts] or ["- none"]), "", "## O4-Z result", "```json", zres, "```",
          "## O4-A verdict", "```json", verdict, "```", ""]
    (OUT / "PROGRESS.md").write_text("\n".join(md))
    return alerts


def stage_files():
    """Copy small result files into the worktree; strip raw GPU samples; keep log tails."""
    dst = WT / "experiments" / "repro" / "O4"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("state.json", "o4z_result.json", "o4_result.json", "PROGRESS.md"):
        if (OUT / name).exists():
            shutil.copy(OUT / name, dst / name)
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        (dst / d.name).mkdir(exist_ok=True)
        for f in d.glob("*.infra.json"):
            j = json.loads(f.read_text())
            j.pop("samples", None)
            (dst / d.name / f.name).write_text(json.dumps(j, indent=1, default=str))
        for f in list(d.glob("*.log")):
            tail = "\n".join(f.read_text(errors="replace").splitlines()[-300:])
            (dst / d.name / (f.name + ".tail.txt")).write_text(tail)
        for f in d.glob("runs_*/*.json"):  # O4-Z/B result JSONs (small)
            if f.stat().st_size < 2_000_000:
                (dst / d.name / f.parent.name).mkdir(exist_ok=True)
                shutil.copy(f, dst / d.name / f.parent.name / f.name)


def publish(msg):
    if not WT.exists():
        sh("git", "fetch", "-q", "origin")
        r = sh("git", "worktree", "add", "-q", "-B", BRANCH, str(WT), "HEAD")
        if r.returncode:
            log("worktree add failed: " + r.stderr[-300:])
            return False
    stage_files()
    sh("git", "add", "-A", "experiments/repro/O4", cwd=WT)
    if not sh("git", "status", "--porcelain", cwd=WT).stdout.strip():
        return True
    c = sh("git", "commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>", cwd=WT)
    if c.returncode:
        log("commit failed: " + c.stderr[-300:])
        return False
    p = sh("git", "push", "-q", "origin", BRANCH, cwd=WT)  # never --force, never main
    log("push rc=%d %s" % (p.returncode, p.stderr[-200:].strip()))
    return p.returncode == 0


def main():
    t0, last_sig, fails = time.time(), None, 0
    log("orchestrator start")
    while time.time() - t0 < MAX_H * 3600:
        st = state()
        done = (OUT / "o4_result.json").exists() and json.loads((OUT / "o4_result.json").read_text()).get("verdict") in ("PASS", "FAIL")
        if not queue_alive() and not done:
            log("queue dead -> restarting")
            with open(Q / "o4_all.log", "a") as lf:
                p = subprocess.Popen([PY, "scripts/repro/o4_run.py", "all"], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            (Q / "o4.pid").write_text(str(p.pid))
        alerts = progress(st)
        sig = json.dumps([{k: (v["reason"], v["n_events"], v["blocked"]) for k, v in st.items()}, done])
        if sig != last_sig and fails < 5:
            ok = publish("O4 soak progress: " + ", ".join(f"{k}={v['reason']}" for k, v in st.items()) if st else "O4 soak: queued")
            fails = 0 if ok else fails + 1
            last_sig = sig if ok else last_sig
        if done and last_sig:
            log("verdict published; orchestrator done")
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
