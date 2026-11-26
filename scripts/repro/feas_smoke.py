#!/usr/bin/env python
"""FEAS-SMOKE (experiments/prereg/FEAS_smoke.yaml v1): integration smoke of the FEAS chain. No evidential claim.

    python scripts/repro/feas_smoke.py --phase yaml,fakes                 # CPU only (CUDA hidden)
    python scripts/repro/gpu_lock.py run --label FEAS-SMOKE --wait-exclusive -- \\
        python scripts/repro/feas_smoke.py --phase gpu                    # the GPU chain, <= 0.2 GPU-h
    python scripts/repro/feas_smoke.py --phase verdict

Mechanism note (reported in smoke_result.json): a child run resolves its parent from the parent's YAML, which
CLI ``-o`` overrides do not reach. So the shortening is written into the ``overrides:`` block of generated copies
of the real FEAS YAMLs, and ``parent:`` points at the copy. The driver asserts that nothing else differs.
Outputs: experiments/feas_smoke/{yaml,runs,ckpt,registry.jsonl,results,logs,feas_decisions.json,timings.json,
smoke_result.json}. The study registry, runs and results are never written by the chain.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
OUT = REPO / "experiments" / "feas_smoke"
FEAS_YAML = REPO / "configs" / "experiments" / "feas"
PY = sys.executable
ARCHIVE = REPO / "runs" / "_artifacts" / "triton_cache_prefeas.tar"
CACHE_ROOT = REPO / ".cache" / "triton-feas-smoke"
XID8 = ["NVRM: Xid (PCI:0000:02:00): 8, pid=4242, name=python, channel 0x00000003",
        "NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7"]
XID31 = ["NVRM: Xid (PCI:0000:02:00): 31, pid=4242, name=python, channel 0x0d000025, MMU Fault"]
# run names in launch order; TEF_FAULT gets the infra injection
CHAIN_AB = ["C0_s101_A", "C12_s101_A", "C12ssmax_s101_A", "C0_s101_B030", "C0_s101_B090",
            "C12_s101_B030", "C12_s101_B090", "C12ssmax_s101_B030"]
TEF_FAULT = "C0_s101_B030_T"
CHAIN_T = ["C0_s101_B090_T", "C12_s101_B030_T", "C12_s101_B090_T", "C12ssmax_s101_B030_T"]
SCORE_RE = re.compile(r"(?i)(?<![a-z_])(em|em_tf|em_free|logp|mean_logp|flips?|flip_rate|score|rank|acc|accuracy)"
                      r"\b[\"']?\s*[=:]\s*[\[\-+]?\d")


def log(m: str) -> None:
    print(f"[feas-smoke {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fname(n: str) -> str:
    return f"S2_D384_{n}.yaml"


def short_overrides(n: str) -> dict:
    ssmax = "ssmax" in n
    if n.endswith("_T"):
        s, ck, ms, val = 20, 10, [10, 20], 10
    elif "_B0" in n:
        s = 5 if ssmax else 10
        ck, ms, val = 5, [s], s
    else:
        s = 10 if ssmax else 30
        ck, ms, val = 10, [10] if ssmax else [15, 30], s
    out = {"schedule.steps": s, "checkpoint.every": ck, "checkpoint.milestone_steps": ms,
           "checkpoint.milestone_tokens": [], "validation.every": val}
    if n.endswith("_B090"):
        # both branches are shortened to the same step count, so they would share one run id. The G-div fallback LR is
        # part of the run identity and is used only if G-div restarts a run (none of these runs may restart: every run
        # must exit as specified), so a 1e-7 offset separates the two lineages without changing what is trained
        out["gdiv.fallback_lr"] = 0.0014001
    return out


def make_yamls() -> dict:
    d = OUT / "yaml"
    d.mkdir(parents=True, exist_ok=True)
    report = {}
    for n in CHAIN_AB + [TEF_FAULT] + CHAIN_T:
        real = yaml.safe_load((FEAS_YAML / fname(n)).read_text())
        new = json.loads(json.dumps(real))
        new["overrides"] = {**(real.get("overrides") or {}), **short_overrides(n)}
        if real.get("parent"):
            new["parent"] = str(d / Path(real["parent"]).name)
        (d / fname(n)).write_text(yaml.safe_dump(new, sort_keys=False))
        diff = {k for k in set(real) | set(new) if real.get(k) != new.get(k)}
        assert diff <= {"overrides", "parent"}, (n, diff)
        added = {k: v for k, v in new["overrides"].items() if (real.get("overrides") or {}).get(k) != v}
        assert set(added) == set(short_overrides(n)), (n, added)
        report[n] = added
    from hybridlc.training.config import PathsConfig, load_run_config
    paths = PathsConfig(**roots_dict(OUT))
    ids = {}
    for n in CHAIN_AB + [TEF_FAULT] + CHAIN_T:
        c = load_run_config(d / fname(n), paths=paths)
        assert c.exp_stage == "S2" and c.schedule.steps == short_overrides(n)["schedule.steps"], n
        ids[n] = {"run_id": c.run_id, "parent_run_id": c.parent_run_id, "steps": c.schedule.steps}
    byid = {v["run_id"] for v in ids.values()}
    assert len(byid) == len(ids), f"duplicate run ids among the shortened runs: {len(byid)} ids for {len(ids)} runs"
    assert all(v["parent_run_id"] in byid for v in ids.values() if v["parent_run_id"]), "parent chain broken"
    (OUT / "yaml_check.json").write_text(json.dumps({"added_overrides": report, "runs": ids}, indent=1))
    log(f"yaml: {len(ids)} shortened copies; only overrides/parent differ; parent chain resolves")
    return ids


def roots_dict(root: Path) -> dict:
    return {"runs_root": str(root / "runs"), "ckpt_root": str(root / "ckpt"), "registry": str(root / "registry.jsonl")}


def runner(root: Path, env: dict, *args: str, logname: str, timeout: int = 1500) -> int:
    cmd = [PY, "scripts/runner.py", "--registry", str(root / "registry.jsonl"), "--runs-root", str(root / "runs"),
           "--ckpt-root", str(root / "ckpt"), "--min-free-gb", "20", "--no-monitor", *args]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / logname, "ab") as f:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def reg_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def statuses(rows: list[dict], rid: str | None = None) -> list[str]:
    return [r["status"] for r in rows if (rid is None or r.get("run_id") == rid)
            and r.get("status") not in ("attempt_end", "monitor", "note", "throughput_alert")]


def base_env(**extra: str) -> dict:
    e = dict(os.environ)
    for k in ("HYBRIDLC_FAULT", "HYBRIDLC_JOURNAL_SOURCE"):
        e.pop(k, None)
    e.update(extra)
    return e


# ------------------------------------------------------------------ CPU injections on training fakes
def phase_fakes() -> dict:
    from test_vt13_training import write_tiny

    res = {}
    base = OUT / "fakes"
    if base.exists():
        shutil.rmtree(base)
    cpu = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONHASHSEED="0", HYBRIDLC_GPUMON="0")
    # suspect_sw: Xid 31 at the same optimizer step twice -> suspect_sw_failure, then paused
    root = base / "suspect_sw"
    root.mkdir(parents=True)
    jf = root / "journal.txt"
    jf.write_text("")
    run = write_tiny(root, steps=20, every=10)
    env = base_env(**cpu, HYBRIDLC_JOURNAL_SOURCE=str(jf),
                   HYBRIDLC_FAULT=json.dumps({"kill_at": [15, 1], "inject_journal": XID31}))
    runner(root, env, "enqueue", str(run), "--allow-dirty", logname="fakes_suspect.log")
    runner(root, env, "run", "--max-jobs", "1", logname="fakes_suspect.log")
    first = statuses(reg_rows(root / "registry.jsonl"))
    for m in (root / "runs").glob("*/fault_kill_at.fired"):
        m.unlink()  # re-arm so the second attempt faults at the same optimizer step
    runner(root, env, "run", "--max-jobs", "1", logname="fakes_suspect.log")
    st = statuses(reg_rows(root / "registry.jsonl"))
    res["suspect_sw"] = {"after_first": first, "after_second": st,
                         "pass": "suspect_sw_failure" in first and st.count("suspect_sw_failure") == 2
                         and st[-1] == "paused" and "completed" not in st}
    # near miss: +1.2 nats for 60 steps -> a gdiv_near_miss note and no restart
    root = base / "near_miss"
    root.mkdir(parents=True)
    run = write_tiny(root, steps=200, every=50, lr=1e-7)  # flat loss, so only the planted excursion moves it
    env = base_env(**cpu, HYBRIDLC_FAULT=json.dumps({"loss_offset": [80, 60, 1.2]}))
    runner(root, env, "enqueue", str(run), "--allow-dirty", logname="fakes_nearmiss.log")
    runner(root, env, "run", logname="fakes_nearmiss.log")
    rows = reg_rows(root / "registry.jsonl")
    nm = [r for r in rows if "gdiv_near_miss" in json.dumps(r)]
    st = statuses(rows)
    res["near_miss"] = {"statuses": st, "n_near_miss_rows": len(nm),
                        "pass": len(nm) >= 1 and st.count("running") == 1 and st[-1] == "completed"}
    (OUT / "fakes_result.json").write_text(json.dumps(res, indent=1))
    log("fakes: " + ", ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in res.items()))
    return res


# ------------------------------------------------------------------ GPU chain
def item_files(cells: list[str]) -> str:
    """evaluate.py --only matches ITEM file names; map result-cell names to item files (all must resolve)."""
    from hybridlc.integration.evaluate import result_name

    by_result = {result_name(f.stem): f.name for f in (REPO / "experiments" / "items" / "dev" / "FD").glob("*.npz")}
    missing = [c for c in cells if c not in by_result]
    assert not missing, f"no FD item file for result cells {missing}"
    return ",".join(by_result[c] for c in cells)


def fresh_cache() -> Path:
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True)
    with tarfile.open(ARCHIVE) as tf:
        tf.extractall(CACHE_ROOT, filter="data")
    hits = [p for p in [CACHE_ROOT, *CACHE_ROOT.rglob("triton")] if p.is_dir()]
    from hybridlc.training.compat import triton_cache_hash
    want = [r for r in reg_rows(REPO / "experiments" / "registry.jsonl") if r.get("run_id") == "pre-FEAS-triton-cache"][-1]
    for p in sorted(hits, key=lambda q: len(q.parts)):
        if triton_cache_hash(str(p)) == want["triton_cache_hash"]:
            return p
    raise SystemExit("no directory of the fresh extraction matches the recorded pre-FEAS cache hash")


def files_in(d: Path) -> set[str]:
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


EVAL_CAP_D099 = "0.85"  # decision D099: only for P0 + P2/P3 re-score on k=12 checkpoints, only on an exclusive GPU
DISPLAY_CLIENTS = ("gnome-remote-desktop", "Xorg", "gnome-shell", "Xwayland")


def assert_gpu_exclusive() -> None:
    """D099 condition: no other compute process on the GPU immediately before a raised-cap step."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    others = [ln for ln in out.splitlines() if ln.strip() and not any(c in ln for c in DISPLAY_CLIENTS)
              and ln.split(",")[0].strip() != str(os.getpid())]
    if others:
        raise SystemExit(f"D099: GPU not exclusive, the 0.85 cap is not allowed now: {others}")


def evaluate(ckpt: Path, suite_args: list[str], env: dict, logname: str, results: Path) -> tuple[int, str, float]:
    cmd = [PY, "scripts/evaluate.py", "--ckpt", str(ckpt), "--results-root", str(results),
           "--runs-root", str(OUT / "runs"), "--registry", str(OUT / "registry.jsonl"), *suite_args]
    t = time.time()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    (OUT / "logs" / logname).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout + p.stderr, time.time() - t


def phase_gpu(ids: dict) -> dict:
    from hybridlc.analysis import feas as FE

    t0 = time.time()
    timings: dict = {"runs": {}, "eval": {}}
    res: dict = {}
    for stale in ("gpu_result.json", "smoke_result.json", "feas_decisions.json", "timings.json"):
        (OUT / stale).unlink(missing_ok=True)  # a crashed attempt must never inherit an earlier attempt's results
    for sub in ("runs", "ckpt", "results", "results_fixture", "logs/gpu"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / "registry.jsonl").unlink(missing_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    cache = fresh_cache()
    before = files_in(cache)
    env = base_env(TRITON_CACHE_DIR=str(cache))
    y = OUT / "yaml"

    def chain(names: list[str], e: dict, tag: str) -> None:
        for n in names:
            t = time.time()
            runner(OUT, e, "enqueue", str(y / fname(n)), logname=f"runner_{tag}.log")
            runner(OUT, e, "run", logname=f"runner_{tag}.log")
            rid = ids[n]["run_id"]
            st = statuses(reg_rows(OUT / "registry.jsonl"), rid)
            timings["runs"][n] = round(time.time() - t, 1)
            log(f"train {n}: {st[-1] if st else 'no rows'} in {timings['runs'][n]} s")
            if not st or st[-1] != "completed":
                raise SystemExit(f"FEAS-SMOKE: run {n} ({rid}) did not complete: {st}")
            if time.time() - t0 > 0.3 * 3600:
                raise SystemExit("FEAS-SMOKE: attempt passed 0.3 GPU-h; stopped as registered")

    chain(CHAIN_AB, env, "ab")
    jf = OUT / "journal_injected.txt"
    jf.write_text("")
    fenv = base_env(TRITON_CACHE_DIR=str(cache), HYBRIDLC_JOURNAL_SOURCE=str(jf),
                    HYBRIDLC_FAULT=json.dumps({"kill_at": [12, 1], "inject_journal": XID8}))
    chain([TEF_FAULT], fenv, "tef_infra")
    chain(CHAIN_T, env, "tef")

    rows = reg_rows(OUT / "registry.jsonl")
    frid = ids[TEF_FAULT]["run_id"]
    fst = statuses(rows, frid)
    res["infra_injection"] = {"statuses": fst, "pass": "infra_failed_attempt" in fst and fst[-1] == "completed"
                              and fst.count("running") == 2}
    res["clean_runs"] = {"pass": all(statuses(rows, v["run_id"]) == ["queued", "running", "completed"]
                                     for n, v in ids.items() if n != TEF_FAULT)}
    queued = {r["run_id"]: r for r in rows if r.get("status") == "queued"}
    res["registry_chain"] = {"pass": all(queued[v["run_id"]].get("parent_run_id") == v["parent_run_id"] for v in ids.values())}
    ck_ok, infra_ok, missing = True, True, []
    for n, v in ids.items():
        c = OUT / "ckpt" / v["run_id"]
        if not (c / "final.pt").is_file() or not list((c / "milestones").glob("*_bf16.pt")):
            ck_ok = False
            missing.append(n)
        ends = [r for r in rows if r.get("run_id") == v["run_id"] and r.get("status") == "attempt_end"]
        summ = OUT / "runs" / v["run_id"] / "summary.json"
        if not ends or not all("infra_class" in r for r in ends) or not summ.is_file() or "infra" not in summ.read_text():
            infra_ok = False
    res["checkpoints"] = {"pass": ck_ok, "missing": missing}
    res["infra_blocks"] = {"pass": infra_ok}

    # evaluation: P0 (+P2/P3 re-score) and P1 on all FD cells, P4 on the ladder verdict cells
    results = OUT / "results"
    p4_cells = item_files(FE.required_cells("P4", "D384"))
    tefs = [TEF_FAULT] + CHAIN_T
    rcs, leaks, meta_bad = {}, [], []
    for n in tefs:
        rid = ids[n]["run_id"]
        # the full checkpoint: P0/P2 round it to bf16 on load (identical to the milestone); P1/P3/P4 need its fp32 weights
        ck = OUT / "ckpt" / rid / "final.pt"
        for pid, extra in (("P0", ["--rescore", "--rescore-ref", "both"]), ("P1", ["--pass", "P1"]),
                           ("P4", ["--pass", "P4", "--only", p4_cells])):
            # evaluate.py appends -P1 / -P4 and -n2 to the base label itself
            if pid == "P0" and n.startswith("C12"):
                assert_gpu_exclusive()
                extra = [*extra, "--mem-fraction", EVAL_CAP_D099]
                log(f"eval {n} {pid}: mem fraction {EVAL_CAP_D099} (D099; GPU exclusive)")
            rc, text, wall = evaluate(ck, ["--suite", "FD", "--label", "tef20", "--n-contexts", "2", *extra], env,
                                      f"eval_{n}_{pid}.log", results)
            rcs[f"{n}/{pid}"] = rc
            timings["eval"][f"{n}/{pid}"] = round(wall, 1)
            leaks += [f"{n}/{pid}: {m.group(0)[:40]}" for m in SCORE_RE.finditer(text)]
            log(f"eval {n} {pid}: rc={rc} in {wall:.0f} s")
            if rc != 0:
                log(f"eval {n} {pid} FAILED (recorded; eval_exit will FAIL); see logs/eval_{n}_{pid}.log")
            if time.time() - t0 > 0.3 * 3600:
                raise SystemExit("FEAS-SMOKE: attempt passed 0.3 GPU-h; stopped as registered")
        n_p4 = len(list((results / rid / "tef20-P4-n2" / "FD").glob("*.jsonl")))
        if n_p4 < len(FE.required_cells("P4", "D384")):
            rcs[f"{n}/P4-cells"] = 1
            log(f"eval {n} P4 scored {n_p4} cells, expected {len(FE.required_cells('P4', 'D384'))}")
        metas = list((results / rid).rglob("_meta.json"))
        for lab in ("tef20-P1-n2", "tef20-P4-n2"):  # fp32-weight passes must really have scored fp32 weights
            mp = results / rid / lab / "FD" / "_meta.json"
            blob = " ".join(m.read_text() for m in (results / rid / lab).rglob("_meta.json"))
            if "bf16-rounded" in blob or "float32" not in blob:
                meta_bad.append(f"{rid}/{lab}: eval weights are not fp32")
        for mp in metas:
            blob = mp.read_text()
            if not all(k in blob for k in ("pass", "dtype", "autocast", "tf32", "attention")):
                meta_bad.append(str(mp.relative_to(results)))
    res["eval_exit"] = {"pass": all(v == 0 for v in rcs.values()), "n": len(rcs),
                        "failed": sorted(k for k, v in rcs.items() if v != 0)}
    res["eval_meta"] = {"pass": not meta_bad, "bad": meta_bad[:10]}
    ev = results / "_eval_log.jsonl"
    leaks += [f"_eval_log: {m.group(0)[:40]}" for m in SCORE_RE.finditer(ev.read_text() if ev.is_file() else "")]

    # blinding control: a non-S2 checkpoint must print scores
    ctl = sorted((REPO / "experiments" / "s0" / "smoke" / "ckpt").glob("S0SMT-*/milestones/*_bf16.pt"))[-1]
    cmd_res = OUT / "results_control"
    shutil.rmtree(cmd_res, ignore_errors=True)
    rc, text, _ = evaluate(ctl, ["--suite", "FD", "--label", "ctl", "--n-contexts", "2", "--only",
                                 item_files(FE.required_cells("P4", "D384")[:1]), "--runs-root",
                                 str(REPO / "experiments" / "s0" / "smoke" / "runs")], env, "eval_control.log", cmd_res)
    res["blinding"] = {"s2_leaks": leaks[:10], "n_s2_leaks": len(leaks), "control_exit": rc,
                       "control_prints_scores": bool(SCORE_RE.search(text)),
                       "pass": not leaks and rc == 0 and bool(SCORE_RE.search(text))}
    # W3: nothing new compiled in the fresh extraction
    new = sorted(files_in(cache) - before)
    res["w3_new_triton_files"] = {"pass": not new, "n": len(new), "examples": new[:10]}
    timings["gpu_wall_s"] = round(time.time() - t0, 1)
    (OUT / "timings.json").write_text(json.dumps(timings, indent=1))
    res.update(decision_code(ids, results))
    (OUT / "gpu_result.json").write_text(json.dumps(res, indent=1))
    log("gpu: " + ", ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in res.items()))
    return res


# ------------------------------------------------------------------ decision code: refuse, then full trace
def decision_code(ids: dict, results: Path) -> dict:
    from hybridlc.analysis import feas as FE

    reg = OUT / "registry.jsonl"
    design = FE.design_from_registry(FE.read_registry(reg))
    fix = OUT / "results_fixture"
    shutil.rmtree(fix, ignore_errors=True)
    # the -n2 labels are never read by the analysis; copy them to the plain labels in the fixture tree
    for d in design:
        for lab in ("tef20", "tef20-P1", "tef20-P4"):
            src = results / d.run_id / f"{lab}-n2"
            if src.is_dir():
                shutil.copytree(src, fix / d.run_id / lab)

    def lm2_of(rid: str) -> dict | None:
        return FE.lm2_from_metrics(OUT / "runs" / rid / "metrics.jsonl", 20)

    lm2 = {d.run_id: v for d in design if (v := lm2_of(d.run_id)) and v.get("before") is not None}
    (OUT / "lm2_seed101.json").write_text(json.dumps(lm2))

    def decide(tree: Path, design_file: Path, lm2_file: Path, out: Path, name: str) -> tuple[int, str]:
        cmd = [PY, "scripts/feas_decide.py", "--results", str(tree), "--registry", str(reg), "--design", str(design_file),
               "--lm2", str(lm2_file), "--ckpt", "tef20", "--out", str(out)]
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600,
                           env=base_env(CUDA_VISIBLE_DEVICES=""))
        (OUT / "logs" / name).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        return p.returncode, p.stdout + p.stderr

    rows101 = [dataclasses.asdict(d) for d in design]
    if not rows101:
        return {"decide_refuses_incomplete": {"pass": False, "why": "no design cells"},
                "decide_full_trace_on_fixture": {"pass": False, "why": "no design cells"}}
    (OUT / "design_seed101.json").write_text(json.dumps(rows101))
    rc, text = decide(fix, OUT / "design_seed101.json", OUT / "lm2_seed101.json", OUT / "must_not_exist.json",
                      "decide_incomplete.log")
    a = {"exit": rc, "names_seed_102": "seed=102" in text, "pass": rc == 2 and "seed=102" in text
         and not (OUT / "must_not_exist.json").exists()}
    # (b) fixture copy of the seed-101 rows, labelled synthetic, as seed 102
    syn = lambda r: f"{r}-SYNTHETIC102"
    rows102 = []
    for d in rows101:
        if d["arm"] != "plain":
            continue
        r = {**d, "run_id": syn(d["run_id"]), "seed": 102, "lineage": [syn(x) for x in d["lineage"]]}
        rows102.append(r)
        shutil.copytree(fix / d["run_id"], fix / r["run_id"])
        if d["run_id"] in lm2:
            lm2[r["run_id"]] = lm2[d["run_id"]]
    (OUT / "design_fixture.json").write_text(json.dumps(rows101 + rows102))
    (OUT / "lm2_fixture.json").write_text(json.dumps(lm2))
    rc, text = decide(fix, OUT / "design_fixture.json", OUT / "lm2_fixture.json", OUT / "feas_decisions.json",
                      "decide_fixture.log")
    blob = (OUT / "feas_decisions.json").read_text() if (OUT / "feas_decisions.json").is_file() else ""
    need = ["Q1", "Q2", "Q3", "Q4", "U-15", "I-08", "F-nearmiss", "F-infra", "F-precision", "TEF"]
    absent = [k for k in need if k.lower() not in blob.lower()]
    b = {"exit": rc, "trace_keys_absent": absent, "pass": rc == 0 and bool(blob) and not absent,
         "stderr_head": text[-1500:] if rc else ""}
    return {"decide_refuses_incomplete": a, "decide_full_trace_on_fixture": b}


def phase_diag(ids: dict) -> dict:
    """DIAGNOSTIC ONLY (never part of a verdict): P4 and the blinding control on the last attempt's checkpoints."""
    from hybridlc.analysis import feas as FE

    env = base_env(TRITON_CACHE_DIR=str(fresh_cache()))
    res_dir = OUT / "results_diag"
    shutil.rmtree(res_dir, ignore_errors=True)
    out = {}
    cells = FE.required_cells("P4", "D384")
    for n in [TEF_FAULT] + CHAIN_T:
        rid = ids[n]["run_id"]
        # the full checkpoint: P0/P2 round it to bf16 on load (identical to the milestone); P1/P3/P4 need its fp32 weights
        ck = OUT / "ckpt" / rid / "final.pt"
        rc, text, wall = evaluate(ck, ["--suite", "FD", "--label", "tef20", "--n-contexts", "2", "--pass", "P4",
                                       "--only", item_files(cells)], env, f"diag_{n}_P4.log", res_dir)
        got = len(list((res_dir / rid / "tef20-P4-n2" / "FD").glob("*.jsonl")))
        tail = [x for x in text.splitlines() if x.startswith('{"n_cells"')]
        out[n] = {"rc": rc, "cells": got, "expected": len(cells), "wall_s": round(wall, 1),
                  "leaks": len(SCORE_RE.findall(text)), "summary": tail[-1] if tail else None}
        log(f"diag P4 {n}: rc={rc} cells={got}/{len(cells)} {wall:.0f}s leaks={out[n]['leaks']}")
    ctl = sorted((REPO / "experiments" / "s0" / "smoke" / "ckpt").glob("S0SMT-*/milestones/*_bf16.pt"))[-1]
    rc, text, _ = evaluate(ctl, ["--suite", "FD", "--label", "ctl", "--n-contexts", "2", "--only",
                                 item_files(cells[:1]), "--runs-root", str(REPO / "experiments" / "s0" / "smoke" / "runs")],
                           env, "diag_control.log", OUT / "results_control_diag")
    out["control"] = {"rc": rc, "prints_scores": bool(SCORE_RE.search(text)),
                      "tail": [x[:200] for x in text.splitlines() if x.strip()][-6:]}
    log(f"diag control: rc={rc} prints_scores={out['control']['prints_scores']}")
    (OUT / "diag_result.json").write_text(json.dumps(out, indent=1))
    return out


def phase_verdict() -> dict:
    res = {}
    for f in ("fakes_result.json", "gpu_result.json"):
        res.update(json.loads((OUT / f).read_text()))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out = {"id": "FEAS-SMOKE", "prereg": "experiments/prereg/FEAS_smoke.yaml", "commit": head,
           "ts": time.strftime("%FT%T%z"), "items": res, "verdict": "PASS" if all(v["pass"] for v in res.values()) else "FAIL",
           "mechanism_note": "shortening applied through the overrides block of generated YAML copies, not CLI -o (see driver "
                             "docstring); cadence overrides include checkpoint.every and validation.every"}
    failing = sorted(k for k, v in res.items() if not v["pass"])
    out["failing_items"] = failing
    out["exceptions"] = {"D099": "0.85 evaluation memory cap for P0+re-score on k=12 checkpoints",
                         "D100": "w3_new_triton_files is a named residual (RR-W3-fp32); the verdict above is not changed by it"}
    out["feas_evaluation_permitted_under_D100"] = failing in ([], ["w3_new_triton_files"])
    (OUT / "smoke_result.json").write_text(json.dumps(out, indent=1))
    log(f"verdict {out['verdict']}: failing = {[k for k, v in res.items() if not v['pass']]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="yaml,fakes")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    phases = a.phase.split(",")
    ids = make_yamls() if {"yaml", "gpu", "diag"} & set(phases) else {}
    if "fakes" in phases:
        phase_fakes()
    if "gpu" in phases:
        phase_gpu(ids)
    if "diag" in phases:
        phase_diag(ids)
    if "decide" in phases:
        ids = json.loads((OUT / "yaml_check.json").read_text())["runs"]
        print(json.dumps(decision_code(ids, OUT / "results"), indent=1))
    if "verdict" in phases:
        return 0 if phase_verdict()["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
