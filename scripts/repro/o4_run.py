#!/usr/bin/env python
"""O4 environment-soak harness (prereg experiments/prereg/O4_soak.v2.yaml, PI final rulings 2026-11-10).

    python scripts/repro/o4_run.py status                 # what is done / pending
    python scripts/repro/o4_run.py all [--allow-dirty]    # wait for an idle GPU, then Z -> B -> A1 -> A2 S1..S4 -> A3 -> evaluate
    python scripts/repro/o4_run.py part Z|B|A1|S1|S2|S3|S4|A3   # one part (same waiting rules)
    python scripts/repro/o4_run.py evaluate               # recompute o4z_result.json / o4_result.json from stored part results

Design rules (all from the prereg):
* Every part runs inside ONE project GPU lock and ONE InfraTagger window (kernel-log Xid / RC watchdog / OOM scan,
  nvidia-smi every 5 s). Windows must be exclusive: allowed foreign GPU users are the display/desktop clients and the
  CARLA-env python client only. The CARLA simulator itself (CarlaUE4-*) is NOT allowed.
* Before a part starts the harness WAITS until the GPU has been idle (no disallowed foreign compute process and
  utilisation < 10 %) for a settle time. If a disallowed process appears mid-part, the part is stopped (so we never
  fight another job for the GPU), recorded as BLOCKED, and repeated later. BLOCKED is never PASS.
* O4-Z / O4-B are report-only; O4-A (A1 probes, A2 soak S1-S4, A3 probes) is verdict-bearing (P1-P4).
* Nothing here edits code to hit a number; this file only orchestrates scripts/train.py and the R5 Zoology cell.
* Choices the prereg does not fix (documented in the result files): LR 1.4e-3 for every O4 run (middle of the
  lr_sweep grid), schedule.steps per segment large enough that the 30-min time box always ends the run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridlc.training.compat import git_dirty_paths  # noqa: E402
from hybridlc.utils.gpu_lock import compute_apps, gpu_lock, is_project_process  # noqa: E402
from hybridlc.utils.infra import InfraTagger, _allowed_foreign, kernel_events  # noqa: E402

OUT = REPO / "experiments" / "repro" / "O4"
STATE = OUT / "state.json"
CELL = "gdn_zoology-d64-lr1.0e-03-s125"
ZCKPT = REPO / "data" / "zoology" / "ckpt" / f"{CELL}.pt"
LR = "1.4e-3"
BOX_A2_S = 1800
ORDER = ["Z", "B", "A1", "S1", "S2", "S3", "S4", "A3"]
SEGMENTS = {  # name: (model, training, seed index, schedule.steps)
    "S1": ("C0", "stage_A", 306, 4000),   # Stage-A shapes, 2 x 4K, no activation checkpointing
    "S2": ("C0", "stage_B", 307, 3000),   # Stage-B shapes, 1 x 32K, activation checkpointing
    "S3": ("C3", "tef", 308, 6000),       # TEF shapes, 1 x 32K, buckets 512..32K, share 0.9, answer_norm block
    "S4": ("C12", "stage_B", 309, 3000),  # Stage-B shapes
}
CUDA_ERR = re.compile(r"launch timeout|illegal memory access|unspecified launch failure|uncorrectable ECC|"
                      r"cudaErrorLaunchTimeout|CUDA error(?!: out of memory)", re.I)
OOM = re.compile(r"out of memory|oom-kill|Out of memory", re.I)
ALLOW_DIRTY = False


def log(msg: str) -> None:
    print(f"[o4 {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    os.replace(tmp, STATE)


def foreign_disallowed() -> list[str]:
    bad = []
    for a in compute_apps():
        if "error" in a or is_project_process(a["pid"], a["name"]) or _allowed_foreign(a["name"]):
            continue
        bad.append(a["name"].split("/")[-1])
    return bad


def gpu_util() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.split()[0])
    except Exception:
        return 100


def wait_idle(settle_s: float = 300.0, poll_s: float = 30.0) -> None:
    """Block until the GPU has been idle (nothing disallowed, util < 10 %) for settle_s straight seconds."""
    since, last_msg = None, 0.0
    while True:
        bad, util = foreign_disallowed(), gpu_util()
        if not bad and util < 10:
            since = since or time.time()
            if time.time() - since >= settle_s:
                return
        else:
            since = None
            if time.time() - last_msg > 900:
                log(f"waiting for idle GPU: util={util}% foreign={bad}")
                last_msg = time.time()
        time.sleep(poll_s)


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty_files": len(git_dirty_paths())}  # code paths only, same rule as scripts/train.py


def kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def base_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update({"TRITON_CACHE_DIR": str(REPO / ".cache" / "triton-diag"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "0", "PYTHONFAULTHANDLER": "1", "FLA_CACHE_RESULTS": "1", "FLA_FLASH_QLA": "0"})
    env.update(extra or {})
    return env


def run_window(name: str, launch, box_s: float, stop_on_event: bool = False, max_children: int = 10**6) -> dict:
    """Run launch(i, dir)->Popen|None children inside one lock + infra window. Returns the part result."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    wait_idle()
    res: dict = {"part": name, "box_s": box_s, "children": []}
    with gpu_lock(f"O4:{name}", wait_exclusive=True, exclusive_settle_s=30):
        tagger = InfraTagger(d / f"{name}.infra.json", tag=f"O4-{name}").start()
        t0, reason, blocked, i, fast_fail = time.time(), "completed", [], 0, 0
        while i < max_children and reason == "completed":
            proc = launch(i, d)
            if proc is None:
                break
            t_child, last_chk = time.time(), time.time()
            while True:
                rc = proc.poll()
                now = time.time()
                if rc is not None:
                    res["children"].append({"i": i, "rc": rc, "s": round(now - t_child, 1)})
                    fast_fail = fast_fail + 1 if (rc != 0 and now - t_child < 30) else 0
                    if fast_fail >= 2:
                        reason = "crashloop"
                    break
                if now - t0 >= box_s:
                    kill(proc)
                    res["children"].append({"i": i, "rc": "SIGTERM(timebox)", "s": round(time.time() - t_child, 1)})
                    reason = "timebox"
                    break
                bad = foreign_disallowed()
                if bad:
                    kill(proc)
                    res["children"].append({"i": i, "rc": "SIGTERM(foreign)", "s": round(time.time() - t_child, 1)})
                    reason, blocked = "blocked_foreign", bad
                    break
                if stop_on_event and now - last_chk > 30:
                    last_chk = now
                    if kernel_events(tagger.t_start, now)["events"]:
                        kill(proc)
                        res["children"].append({"i": i, "rc": "SIGTERM(event)", "s": round(time.time() - t_child, 1)})
                        reason = "event_stop"
                        break
                time.sleep(5)
            i += 1
            if reason == "completed" and time.time() - t0 >= box_s:
                reason = "timebox"
        infra = tagger.finish()
    logs = " ".join(p.read_text(errors="replace")[-200000:] for p in d.glob("*.log"))
    res.update({"reason": reason, "blocked_by": blocked, "duration_s": round(infra["end"] - infra["start"], 1),
                "start_iso": infra["start_iso"], "end_iso": infra["end_iso"], "n_events": len(infra["infra_events"]),
                "events": infra["infra_events"][:20], "journal_ok": infra["journal_ok"],
                "foreign_disallowed": infra["infra"]["foreign_disallowed"],
                "unlocked_project_processes": infra["infra"]["unlocked_project_processes"],
                "cuda_error_in_log": bool(CUDA_ERR.search(logs)), "oom_in_log": bool(OOM.search(logs)),
                "git": git_state()})
    res["blocked"] = bool(reason == "blocked_foreign" or res["foreign_disallowed"] or not res["journal_ok"])
    return res


def spawn(cmd: list[str], logfile: Path, env: dict) -> subprocess.Popen:
    f = open(logfile, "ab")
    log("launch: " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def aside_ckpt(d: Path) -> None:
    if ZCKPT.exists():
        (d / "aside").mkdir(exist_ok=True)
        shutil.move(str(ZCKPT), str(d / "aside" / f"{CELL}.{int(time.time())}.pt"))


def launch_zoology(extra_env: dict):
    def go(i: int, d: Path):
        aside_ckpt(d)  # never reuse a checkpoint: every iteration starts at epoch 0
        env = base_env({"O4_DIR": str(d), "O4_FAULT_LOG": str(d / f"fault_{i}.log"), **extra_env})
        return spawn([sys.executable, "scripts/s1_zoology/run_one_o4.py", "--run-id", CELL, "--out-dir", str(d / f"runs_{i}")],
                     d / f"child_{i}.log", env)
    return go


def train_cmd(model: str, training: str, seed: int, steps: int, d: Path, tag: str, accum: int | None = None) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--model", f"configs/model/D384/{model}.yaml",
           "--training", f"configs/training/{training}.yaml", "--exp-stage", "O4", "--seed", str(seed), "--lr", LR,
           "-o", f"schedule.steps={steps}", "--tag", tag, "--runs-root", str(d / "runs"), "--ckpt-root", str(d / "ckpt"),
           "--registry", str(d / "registry.jsonl")]
    if accum:
        cmd += ["-o", f"data.accum={accum}"]
    return cmd + (["--allow-dirty"] if ALLOW_DIRTY else [])


def launch_probe(i: int, d: Path):
    if i >= 2:
        return None
    return spawn(train_cmd("C3", "vt13_audit", 305, 50, d, f"p{i}", accum=2), d / f"child_p{i}.log", base_env())


def launch_segment(name: str):
    model, training, seed, steps = SEGMENTS[name]

    def go(i: int, d: Path):
        if i >= 1:
            return None
        return spawn(train_cmd(model, training, seed, steps, d, name.lower()), d / f"child_{name}.log", base_env())
    return go


def run_part(name: str) -> dict:
    if name == "Z":
        r = run_window("Z", launch_zoology({}), 3600)
    elif name == "B":
        r = run_window("B", launch_zoology({"CUDA_LAUNCH_BLOCKING": "1"}), 3600, stop_on_event=True)
    elif name in ("A1", "A3"):
        r = run_window(name, launch_probe, 1800, max_children=2)
        r["probe"] = probe_compare(OUT / name)
    else:
        r = run_window(name, launch_segment(name), BOX_A2_S, max_children=1)
    return r


def _trace(d: Path, tag: str) -> tuple[list, str | None]:
    tr = sorted(d.glob(f"runs/**/*{tag}*/trace.jsonl")) or sorted(d.glob(f"runs/**/trace.jsonl"))
    tr = [p for p in tr if f"{tag}" in str(p)]
    if not tr:
        return [], None
    rows = {}
    for line in tr[-1].read_text().splitlines():
        try:
            j = json.loads(line)
            rows[j["step"]] = (j["step"], j["loss_hex"], j["gnorm_hex"], j["batch_hash"])
        except Exception:
            continue
    psha = None
    for line in (tr[-1].parent / "metrics.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("kind") == "end":
            psha = j.get("param_sha256")
    return [rows[k] for k in sorted(rows)], psha


def probe_compare(d: Path) -> dict:
    (t0, s0), (t1, s1) = _trace(d, "p0"), _trace(d, "p1")
    return {"n_steps": [len(t0), len(t1)], "trace_identical": bool(t0) and t0 == t1, "param_sha256": [s0, s1],
            "sha_identical": bool(s0) and s0 == s1}


def evaluate() -> dict:
    st = load_state()
    z = st.get("Z")
    if z:
        e = z["n_events"]
        meaning = ("GDN-workload hang persists on the exclusive GPU (E_Z >= 1): PI decision row needed; see prereg O4-Z.meaning"
                   if e else "no event in the O4-Z window (E_Z = 0): evidence the R5-rate trigger was contention-related or receded")
        (OUT / "o4z_result.json").write_text(json.dumps({"E_Z": e, "exposure_h": round(z["duration_s"] / 3600, 3), "events": z["events"],
                                                          "valid_window": not z["blocked"], "meaning": meaning}, indent=1))
    A = {k: st.get(k) for k in ["A1", "S1", "S2", "S3", "S4", "A3"]}
    if any(v is None for v in A.values()):
        out = {"verdict": "INCOMPLETE", "missing": [k for k, v in A.items() if v is None]}
    else:
        P1 = all(v["n_events"] == 0 and v["journal_ok"] for v in A.values())
        P2 = all(A[s]["reason"] == "timebox" and A[s]["duration_s"] >= BOX_A2_S - 60 and not A[s]["cuda_error_in_log"]
                 for s in SEGMENTS)
        pa1, pa3 = A["A1"]["probe"], A["A3"]["probe"]
        P3 = (pa1["trace_identical"] and pa1["sha_identical"] and pa3["trace_identical"] and pa3["sha_identical"]
              and pa1["param_sha256"][0] == pa3["param_sha256"][0])
        P4 = not any(v["oom_in_log"] or any(OOM.search(str(ev)) for ev in v["events"]) for v in A.values())
        blocked = any(v["blocked"] for v in A.values())
        verdict = "BLOCKED" if blocked else ("PASS" if (P1 and P2 and P3 and P4) else "FAIL")
        out = {"verdict": verdict, "P1_no_kernel_events": P1, "P2_full_time_boxes": P2, "P3_bitwise_determinism": P3,
               "P4_no_oom": P4, "probes": {"A1": pa1, "A3": pa3},
               "segments": {s: {k: A[s][k] for k in ("reason", "duration_s", "n_events", "cuda_error_in_log")} for s in SEGMENTS}}
    (OUT / "o4_result.json").write_text(json.dumps(out, indent=1))
    log(f"evaluate: {out['verdict']}")
    return out


def do_part(name: str, st: dict, retries: int = 100) -> None:
    for attempt in range(retries):
        res = run_part(name)
        res["attempt"] = attempt
        st[name] = res
        save_state(st)
        log(f"part {name}: reason={res['reason']} events={res['n_events']} blocked={res['blocked']} dur={res['duration_s']}s")
        if not res["blocked"]:
            return
        time.sleep(60)
    log(f"part {name}: still blocked after {retries} attempts")


def main() -> int:
    global ALLOW_DIRTY
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "all", "part", "evaluate"])
    ap.add_argument("name", nargs="?", choices=ORDER)
    ap.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree (recorded; prereg wants a clean committed tree)")
    a = ap.parse_args()
    ALLOW_DIRTY = a.allow_dirty
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    if a.cmd == "status":
        for k in ORDER:
            v = st.get(k)
            print(k, "-" if not v else f"{v['reason']} events={v['n_events']} blocked={v['blocked']} {v['duration_s']}s")
        return 0
    if a.cmd == "evaluate":
        evaluate()
        return 0
    g = git_state()
    if g["dirty_files"] and not ALLOW_DIRTY:
        log(f"REFUSED: code paths have {g['dirty_files']} uncommitted changes (prereg precondition: committed clean tree). "
            "Commit, or rerun with --allow-dirty (recorded as a deviation).")
        return 3
    todo = ORDER if a.cmd == "all" else [a.name]
    for name in todo:
        if a.cmd == "all" and st.get(name) and not st[name]["blocked"]:
            continue
        do_part(name, st)
    if a.cmd == "all":
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
