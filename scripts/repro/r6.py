#!/usr/bin/env python
"""R6 v2: known-answer associative recall in our pipeline (prereg experiments/prereg/R6.v2.yaml, plan 2.6).

Launch (one GPU lock for the whole run; the train/eval children inherit the token):
    python scripts/repro/gpu_lock.py run --label R6 --wait-exclusive -- python scripts/repro/r6.py run [--attempt N]
Other subcommands:  items (CPU: generate the 8 R6 item files + K1 oracle check),  chain --ckpt P ... (R6-R5/R9 chain check).

Design: AA (2 layers) trains once to 2,289 steps (150M tokens) on 100% synthetic TEF at T = 1,024, LR 1e-3 constant after a
100-step warm-up. At step 763 / 1,526 the bf16 milestone is scored through scripts/evaluate.py (production path); the
trainer is SIGSTOPped during the evaluation and resumed (or killed on PASS). Score = mean over the two primary cells
(R1 D=3 @1K, R-load N=8 @1K) of EM_tf. PASS >= 0.90 at the first look; FAIL iff < 0.50 at 150M or the format-mismatch
signature (retrieval answer CE <= 0.50 over the 50 steps before a look AND score < 0.05) at any look; else INCONCLUSIVE.
The '--' (GDN) model then trains to the same step (descriptive only), and both checkpoints get the chain-consistency check.
An attempt whose window holds an infra event (Xid / RC watchdog / OOM line in the kernel journal) is discarded and repeated
from step 0 (checkpoints moved aside), at most 3 repeats. This script never tunes anything to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "experiments" / "repro" / "R6"
PY = sys.executable
SEED = 300
LOOKS = (763, 1526, 2289)
PRIMARY = ("R1_D3_1024", "R-load_N8_1024")
INFRA_PATTERN = "NVRM: Xid|RC watchdog|Out of memory|oom-kill"
MODELS = {"AA": "configs/model/D384/R6-AA.yaml", "GDN": "configs/model/D384/R6-GDN.yaml"}


def log(m: str) -> None:
    print(f"[r6 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def env() -> dict:
    e = dict(os.environ)
    e.update({"TRITON_CACHE_DIR": str(REPO / ".cache" / "triton-diag"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"})
    return e


def kernel_events(since: float) -> list[str]:
    r = subprocess.run(["journalctl", "-k", "--since", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)),
                        "--no-pager", "-g", INFRA_PATTERN], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip() and not l.startswith("-- ")]


def dirs(model: str, attempt: int) -> dict:
    root = OUT / f"attempt{attempt}" / model
    return dict(root=root, runs=root / "runs", ckpt=root / "ckpt", registry=root / "registry.jsonl", eval=OUT / f"attempt{attempt}" / "eval")


def train_cmd(model: str, d: dict) -> list[str]:
    return [PY, "scripts/train.py", "--model", MODELS[model], "--training", "configs/training/r6.yaml", "--exp-stage", "R6",
            "--seed", str(SEED), "--runs-root", str(d["runs"]), "--ckpt-root", str(d["ckpt"]), "--registry", str(d["registry"])]


def run_id(model: str, d: dict) -> str:
    return subprocess.run(train_cmd(model, d) + ["--print-run-id"], cwd=REPO, env={**env(), "CUDA_VISIBLE_DEVICES": ""},
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]


def milestone(d: dict, rid: str, step: int) -> Path | None:
    hits = sorted((d["ckpt"] / rid).glob(f"**/step_{step:07d}*bf16.pt"))
    return hits[0] if hits else None


def evaluate(ck: Path, d: dict, rid: str, step: int) -> int:
    cmd = [PY, "scripts/evaluate.py", "--ckpt", str(ck), "--suite", "R6", "--label", f"step{step}", "--results-root", str(d["eval"]),
           "--runs-root", str(d["runs"]), "--no-candidates", "--no-rescore"]
    with open(d["root"] / "evaluate.log", "ab") as f:
        return subprocess.run(cmd, cwd=REPO, env=env(), stdout=f, stderr=subprocess.STDOUT).returncode


def cell_files() -> dict[str, str]:
    from hybridlc.eval.suites import suite_cells
    from hybridlc.integration.evaluate import result_name
    return {c.name: result_name(Path(c.relpath).stem) for c in suite_cells("R6")}


def cell_em(d: dict, rid: str, step: int) -> dict[str, float]:
    out = {}
    for _, rn in cell_files().items():
        f = d["eval"] / rid / f"step{step}" / "R6" / f"{rn}.jsonl"
        rows = [json.loads(l) for l in open(f)]
        out[rn] = sum(bool(r["EM_tf"]) for r in rows) / len(rows)
    return out


def family_ce(d: dict, rid: str, look: int) -> float | None:
    recs = [json.loads(l) for l in open(d["runs"] / rid / "family_ce.jsonl")]
    recs = [r for r in recs if r["step"] <= look][-50:]
    return sum(r["answer_ce_retrieval"] for r in recs) / len(recs) if recs else None


def look_record(d: dict, rid: str, step: int) -> dict:
    em = cell_em(d, rid, step)
    score = sum(em[c] for c in PRIMARY) / 2
    ce = family_ce(d, rid, step)
    return dict(step=step, tokens=step * 65536, score=score, cell_em=em, answer_ce_retrieval_last50=ce,
                signature=bool(ce is not None and ce <= 0.50 and score < 0.05))


def decide(looks: list[dict]) -> dict:
    """R6-R2/R3/R4 on the AA looks (plan 2.6 as fixed by v2). The run continues past a look only if score < 0.90."""
    for lk in looks:
        if lk["signature"]:
            return dict(b_verdict="FAIL", reason=f"format-mismatch signature at step {lk['step']}")
        if lk["score"] >= 0.90:
            return dict(b_verdict="PASS", reason=f"score {lk['score']:.3f} >= 0.90 at step {lk['step']}", verdict_step=lk["step"])
    last = looks[-1]
    if last["step"] != LOOKS[-1]:
        return dict(b_verdict="PENDING", reason="next look not yet evaluated", verdict_step=None)
    if last["score"] < 0.50:
        return dict(b_verdict="FAIL", reason=f"score {last['score']:.3f} < 0.50 at 150M", verdict_step=last["step"])
    return dict(b_verdict="INCONCLUSIVE", reason=f"0.50 <= score {last['score']:.3f} < 0.90 at 150M", verdict_step=last["step"])


def train_with_looks(model: str, attempt: int, stop_at: int | None, decide_looks: bool) -> dict:
    """Train ``model``; evaluate each look while the trainer is stopped. AA stops at the first PASS; GDN at ``stop_at``."""
    d = dirs(model, attempt)
    for k in ("runs", "ckpt", "eval"):
        d[k].mkdir(parents=True, exist_ok=True)
    rid = run_id(model, d)
    t0 = time.time()
    log(f"{model}: run_id {rid}; launching")
    logf = open(d["root"] / "train.log", "ab")
    p = subprocess.Popen(train_cmd(model, d), cwd=REPO, env=env(), stdout=logf, stderr=subprocess.STDOUT)
    looks: list[dict] = []
    todo = [s for s in LOOKS if stop_at is None or s <= stop_at]
    while todo and p.poll() is None or (todo and p.poll() == 0):
        step = todo[0]
        ck = milestone(d, rid, step)
        if ck is None:
            if p.poll() is not None:
                break
            time.sleep(10)
            continue
        time.sleep(5)  # let the milestone write finish
        if p.poll() is None:
            p.send_signal(signal.SIGSTOP)
        log(f"{model}: look step {step}: evaluating {ck.name}")
        rc = evaluate(ck, d, rid, step)
        if rc != 0:
            if p.poll() is None:
                p.kill()
            return dict(model=model, rid=rid, status="eval_failed", rc=rc, looks=looks, events=kernel_events(t0))
        todo.pop(0)
        if decide_looks:
            looks.append(look_record(d, rid, step))
            log(f"{model}: step {step} score {looks[-1]['score']:.3f} ce {looks[-1]['answer_ce_retrieval_last50']}")
            stop = decide(looks)["b_verdict"] != "PENDING"
        else:
            looks.append(dict(step=step, cell_em=cell_em(d, rid, step)))
            stop = stop_at is not None and step >= stop_at
        if stop:
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
            p.wait()
            break
        if p.poll() is None:
            p.send_signal(signal.SIGCONT)
    if p.poll() is None:
        p.wait()
    events = kernel_events(t0)
    return dict(model=model, rid=rid, status="ok", looks=looks, events=events, train_rc=p.returncode, dirs={k: str(v) for k, v in d.items()})


def chain_cmd(ck: Path, d: dict, label: str, out: Path) -> int:
    cmd = [PY, "scripts/repro/r6.py", "chain", "--ckpt", str(ck), "--results-root", str(d["eval"]), "--runs-root", str(d["runs"]),
           "--label", label, "--out", str(out)]
    return subprocess.run(cmd, cwd=REPO, env=env()).returncode


def cmd_chain(a: argparse.Namespace) -> int:
    """R6-R5 / R6-R9: EM_tf flips of the stored cache-path rows vs a single-pass forward over ALL queries of the 8 cells."""
    from hybridlc.eval.scorer import rescore_check
    from hybridlc.integration.evaluate import load_checkpoint
    from hybridlc.utils.gpu_lock import guard
    import torch

    guard("r6-chain")
    ck = load_checkpoint(a.ckpt, "cuda", runs_root=a.runs_root, canonical_bf16=True)
    ref32 = load_checkpoint(a.ckpt, "cuda", runs_root=a.runs_root, canonical_bf16=False).model
    cells = {}
    from hybridlc.eval.suites import suite_cells
    from hybridlc.integration.evaluate import result_name
    for c in suite_cells("R6"):
        stem = Path(c.relpath).stem
        item = REPO / "experiments" / "items" / "dev" / c.relpath
        rows = Path(a.results_root) / ck.run_id / a.label / "R6" / f"{result_name(stem)}.jsonl"
        r16 = rescore_check(ck.model, item, rows, frac=1.0, ref="bf16")
        r32 = rescore_check(ck.model, item, rows, frac=1.0, ref="fp32", ref_model=ref32)
        stored = [json.loads(l) for l in open(rows)]
        em_p0 = sum(bool(r["EM_tf"]) for r in stored) / len(stored)
        em_16 = sum(r["EM_tf"] for r in r16["rows"]) / len(r16["rows"])
        em_32 = sum(r["EM_tf"] for r in r32["rows"]) / len(r32["rows"])
        cells[result_name(stem)] = dict(n=r16["n"], flips_bf16=r16["flips"], flips_fp32=r32["flips"], em_p0=em_p0, em_sp_bf16=em_16,
                                        em_sp_fp32=em_32, max_abs_dlogp_bf16=r16["max_abs_dlogp"])
        torch.cuda.empty_cache()
    n = sum(c["n"] for c in cells.values())
    f16 = sum(c["flips_bf16"] for c in cells.values())
    f32 = sum(c["flips_fp32"] for c in cells.values())
    # informative cell (FEAS-PREC v2): max(EM_a, EM_b) > 0.05 and min(EM_a, EM_b) < 0.95, pair = (P0, single-pass bf16)
    inf = {k: v for k, v in cells.items() if max(v["em_p0"], v["em_sp_bf16"]) > 0.05 and min(v["em_p0"], v["em_sp_bf16"]) < 0.95}
    ni = sum(v["n"] for v in inf.values())
    res = dict(ckpt=str(a.ckpt), run_id=ck.run_id, step=ck.step, label=a.label, n_queries=n, flips_bf16=f16, flip_rate_bf16=f16 / n,
               flips_fp32=f32, flip_rate_fp32_report_only=f32 / n, informative_cells=sorted(inf),
               restricted_flip_rate_bf16=(sum(v["flips_bf16"] for v in inf.values()) / ni) if ni else None, cells=cells)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    log(f"chain {a.label}: pooled bf16 flip {f16}/{n} = {f16 / n:.4%}; fp32 {f32 / n:.4%}; restricted {res['restricted_flip_rate_bf16']}")
    return 0


def cmd_items(_: argparse.Namespace) -> int:
    rc = subprocess.run([PY, "scripts/make_items.py", "--suite", "R6", "--jobs", "4"], cwd=REPO,
                        env={**env(), "CUDA_VISIBLE_DEVICES": ""}).returncode
    if rc:
        return rc
    # precondition: the R6 items pass the VT-24r K1 oracle (rule G-ORACLE) before the first R6 evaluation
    from hybridlc.eval.items import load_items
    from hybridlc.eval.oracle_check import item_ok
    from hybridlc.eval.suites import suite_cells
    bad = {}
    for c in suite_cells("R6"):
        items, _ = load_items(REPO / "experiments" / "items" / "dev" / c.relpath)
        n = sum(not item_ok(it) for it in items)
        if n:
            bad[c.relpath] = n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k1_oracle_r6_items.json").write_text(json.dumps(dict(violating_items=bad, ok=not bad), indent=1))
    log(f"K1 oracle on R6 items: {'OK' if not bad else bad}")
    return 1 if bad else 0


def cmd_run(a: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "experiments" / "prereg" / "R6.v2.yaml", OUT / "prereg.yaml")
    attempt, history = a.attempt, []
    while True:
        res = train_with_looks("AA", attempt, None, True)
        if res["events"] or res["status"] != "ok":
            history.append(dict(attempt=attempt, discarded=True, status=res["status"], events=res["events"]))
            log(f"attempt {attempt} discarded: {res['status']} events={len(res['events'])}")
            if len(history) > 3:
                (OUT / "r6_results.json").write_text(json.dumps(dict(overall="BLOCKED", history=history), indent=1))
                return 2
            aside = OUT / f"attempt{attempt}_discarded"
            shutil.move(str(OUT / f"attempt{attempt}"), str(aside))
            attempt += 1
            continue
        break
    b = decide(res["looks"])
    d = dirs("AA", attempt)
    vstep = b.get("verdict_step") or res["looks"][-1]["step"]
    out = dict(attempt=attempt, history=history, aa=res, b=b, verdict_step=vstep)
    ck = milestone(d, res["rid"], vstep) or next(iter(sorted((d["ckpt"] / res["rid"]).glob("**/*final_bf16.pt"))), None)
    ok_a = None
    if ck is not None:
        rc = chain_cmd(ck, d, f"step{vstep}", OUT / f"chain_AA_step{vstep}.json")
        if rc == 0:
            ch = json.load(open(OUT / f"chain_AA_step{vstep}.json"))
            ok_a = ch["flip_rate_bf16"] < 0.01
            out["a_chain_AA"] = dict(flip_rate_bf16=ch["flip_rate_bf16"], passes=ok_a)
    if ok_a is False or b["b_verdict"] == "FAIL":
        out["overall"] = "FAIL"
    elif ok_a and b["b_verdict"] == "PASS":
        out["overall"] = "PASS"
    else:
        out["overall"] = "INCONCLUSIVE"
    (OUT / "r6_results.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    log(f"AA done: overall {out['overall']} ({b['reason']})")
    # R6-R7 / R6-R9: '--' to the same token count, descriptive only
    g = train_with_looks("GDN", attempt, vstep, False)
    out["gdn"] = g
    dg = dirs("GDN", attempt)
    gck = milestone(dg, g["rid"], vstep) or next(iter(sorted((dg["ckpt"] / g["rid"]).glob("**/*final_bf16.pt"))), None)
    if gck is not None:
        if chain_cmd(gck, dg, f"step{vstep}", OUT / f"chain_GDN_step{vstep}.json") == 0:
            ch = json.load(open(OUT / f"chain_GDN_step{vstep}.json"))
            out["gdn_chain"] = dict(pooled=ch["flip_rate_bf16"], restricted=ch["restricted_flip_rate_bf16"], fp32=ch["flip_rate_fp32_report_only"],
                                    reopen_RR_G0_1_2=bool(ch["flip_rate_bf16"] >= 0.01 or (ch["restricted_flip_rate_bf16"] or 0) >= 0.01))
    (OUT / "r6_results.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("items").set_defaults(f=cmd_items)
    r = sp.add_parser("run")
    r.add_argument("--attempt", type=int, default=1)
    r.set_defaults(f=cmd_run)
    c = sp.add_parser("chain")
    for k in ("--ckpt", "--results-root", "--runs-root", "--label", "--out"):
        c.add_argument(k, required=True)
    c.set_defaults(f=cmd_chain)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
