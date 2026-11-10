#!/usr/bin/env python
"""VT-24r v2 checks K0-K4 and the summary (experiments/prereg/VT-24r.v2.yaml). CPU only.

    python scripts/repro/vt24r_check.py k0 [--items experiments/items_archive/pre-vt24r_97ef269]
    python scripts/repro/vt24r_check.py k1k2 [--workers 12]
    python scripts/repro/vt24r_check.py k3
    python scripts/repro/vt24r_check.py k5b
    python scripts/repro/vt24r_check.py summary

Outputs: experiments/repro/VT-24r/{k0_agreement.json, k1_scores/, k1k2_files.jsonl, k1k2.json,
k3_stability.json, k5b_after.json, k4_pytest.log (written by the caller), vt24r_summary.json}.

K0 runs the INDEPENDENT checker (hybridlc.eval.oracle_check, numpy window search) on the pre-fix item
files of VT-24's scope and compares its pred_ok misses with the 7 (file, context, query) triples VT-24
missed (experiments/repro/R3/vt24/run_20261110T055803/diagnosis_misses.json, the same 7 in the 12:52
rerun). K1/K2 score the regenerated items with devtools.InductionLM + scorer.score_file (the oracle
implementation independent of the repair's checker).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "experiments" / "repro" / "VT-24r"
RETRIEVAL = ("R1", "R-load", "R3", "R-dist")
K0_DIRS = ("test/CS", "test/MS", "dev/DG", "dev/FD")
K1_DIRS = ("test/CS", "test/MS", "dev/DG", "dev/FD", "dev/DC", "dev/MS")
VT24_MISSES = REPO / "experiments/repro/R3/vt24/run_20261110T055803/diagnosis_misses.json"
VT24_RERUN = REPO / "experiments/repro/R3/vt24/run_20261110T125240/vt24_files.jsonl"
LP_GOLD, LP_TOL = 4 * math.log(0.9), 1e-4


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def _meta() -> dict:
    from hybridlc.eval.items import eval_code_sha256

    return {"git_head": _git("rev-parse", "HEAD"), "git_dirty_files": len(_git("status", "--porcelain").splitlines()),
            "eval_code_sha256": eval_code_sha256(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ------------------------------------------------------------------------------------------ K0
def _k0_file(path: str) -> dict:
    from hybridlc.eval.items import load_items
    from hybridlc.eval.oracle_check import check_item

    items, cell = load_items(path)
    pred_miss, loc_miss, coinc_q, coinc_d, defect = [], [], 0, 0, 0
    n = 0
    for it in items:
        for r in check_item(it):
            n += 1
            key = (int(it["meta"]["context_id"]), int(r["query_idx"]))
            if not r["pred_ok"]:
                pred_miss.append(key)
            if not r["loc_ok"]:
                loc_miss.append(key)
            cd = sum(1 for d in r["digits"] if d["pred_ok"] and not d["loc_ok"])
            coinc_d += cd
            coinc_q += bool(r["pred_ok"] and not r["loc_ok"])
            defect += sum(1 for d in r["digits"] if d["loc_ok"] and not d["pred_ok"])
    return {"path": path, "task": cell.get("task"), "T": cell.get("T"), "n_queries": n, "pred_miss": pred_miss,
            "loc_miss": loc_miss, "coincidental_queries": coinc_q, "coincidental_digits": coinc_d,
            "checker_defect_digits": defect}


def cmd_k0(a: argparse.Namespace) -> dict:
    from hybridlc.eval.items import load_items

    root = Path(a.items)
    files = []
    for d in K0_DIRS:
        for p in sorted((root / d).glob("*.npz")):
            with zipfile.ZipFile(p) as zf:
                cell = json.loads(zf.read("meta.json"))["cell"]
            if cell.get("task") in RETRIEVAL and cell.get("T") in (16384, 32768):
                files.append(str(p))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_k0_file, files))
    # VT-24's 7 misses, keyed by (dir/stem, context, query)
    vt24 = set()
    for m in json.loads(VT24_MISSES.read_text()):
        rel = m["file"].split("experiments/items/")[1]
        vt24.add((rel, int(m["context_id"]), int(m["query_idx"])))
    rerun = set()
    for ln in VT24_RERUN.read_text().splitlines():
        r = json.loads(ln)
        if r.get("role") != "retrieval":
            continue
        rel = r["path"].split("experiments/items/")[1]
        for mm in r.get("misses_first20", []):
            rerun.add((rel, int(mm["context_id"]), int(mm["query_idx"])))
    checker = set()
    for r in rows:
        rel = Path(r["path"]).relative_to(root).as_posix()
        for c, q in r["pred_miss"]:
            checker.add((rel, c, q))
    defect = sum(r["checker_defect_digits"] for r in rows)
    res = {
        "check": "K0_checker_agreement", "items_root": str(root), "n_files": len(rows),
        "n_queries": sum(r["n_queries"] for r in rows),
        "vt24_misses": sorted(map(list, vt24)), "vt24_rerun_misses_equal": rerun == vt24,
        "checker_pred_miss": sorted(map(list, checker)),
        "checker_loc_miss_queries": sum(len(r["loc_miss"]) for r in rows),
        "coincidental_match_queries": sum(r["coincidental_queries"] for r in rows),
        "coincidental_match_digits": sum(r["coincidental_digits"] for r in rows),
        "checker_defect_digits": defect,
        "per_file": [{k: v for k, v in r.items() if k not in ("pred_miss", "loc_miss")} |
                     {"n_pred_miss": len(r["pred_miss"]), "n_loc_miss": len(r["loc_miss"])} for r in rows],
    }
    res["pass"] = bool(checker == vt24 and defect == 0 and len(rows) == 52)
    res["criteria"] = ("set of pred_ok=False queries == VT-24's 7 triples; no loc_ok=True with pred_ok=False; "
                       "52 VT-24 retrieval files covered")
    return res


# ------------------------------------------------------------------------------------------ K1 / K2
def _enumerate_k1() -> list[dict]:
    out = []
    for d in K1_DIRS:
        for p in sorted((REPO / "experiments" / "items" / d).glob("*.npz")):
            with zipfile.ZipFile(p) as zf:
                cell = json.loads(zf.read("meta.json"))["cell"]
            task = cell.get("task")
            role = "retrieval" if task in RETRIEVAL else ("mhop" if task == "M-hop" else None)
            if role:
                out.append({"path": str(p.relative_to(REPO)), "dir": d, "task": task, "T": cell.get("T"),
                            "variant": cell.get("variant"), "role": role})
    return out


def _k1_score(rec: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from hybridlc.eval.devtools import InductionLM
    from hybridlc.eval.scorer import score_file

    t0 = time.time()
    p = REPO / rec["path"]
    out = OUT / "k1_scores" / rec["dir"] / f"{p.stem}.jsonl"
    if out.exists():
        out.unlink()
    res = score_file(InductionLM(16384, order=6, p_hit=0.9), p, out, score_candidates=False, autocast=False,
                     extra_meta={"vt": "VT-24r K1/K2", "oracle": "InductionLM(16384, order=6, p_hit=0.9)"})
    rows = [json.loads(x) for x in Path(res).read_text().splitlines()]
    em = [bool(r["EM_tf"]) for r in rows]
    lp = [float(r["logp_gold"]) for r in rows]
    misses = [{"context_id": r["context_id"], "query_idx": r["query_idx"], "EM_tf": r["EM_tf"], "logp_gold": r["logp_gold"]}
              for r in rows if (not r["EM_tf"]) or abs(float(r["logp_gold"]) - LP_GOLD) > LP_TOL]
    row = dict(rec, n_queries=len(rows), em_mean=sum(em) / max(len(em), 1), n_misses=len(misses),
               misses_first20=misses[:20], secs=round(time.time() - t0, 1))
    row["pass"] = bool(rows and not misses) if rec["role"] == "retrieval" else bool(row["em_mean"] < 0.5)
    return row


def cmd_k1k2(a: argparse.Namespace) -> dict:
    files = _enumerate_k1()
    rows = []
    with open(OUT / "k1k2_files.jsonl", "w") as f, ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_k1_score, r): r for r in sorted(files, key=lambda r: -(r["T"] or 0))}
        for fu in as_completed(futs):
            try:
                row = fu.result()
            except Exception as e:  # noqa: BLE001 - recorded, counts as a failure
                row = dict(futs[fu], error=repr(e), **{"pass": False})
            rows.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
    retr = [r for r in rows if r["role"] == "retrieval"]
    mh = [r for r in rows if r["role"] == "mhop"]
    res = {"check": "K1_oracle + K2_negative_control", "n_retrieval_files": len(retr), "n_mhop_files": len(mh),
           "n_retrieval_queries": sum(r.get("n_queries", 0) for r in retr),
           "retrieval_queries_failing": sum(r.get("n_misses", 0) for r in retr),
           "retrieval_files_failing": sorted(r["path"] for r in retr if not r["pass"]),
           "mhop_files_failing": sorted(r["path"] for r in mh if not r["pass"]),
           "mhop_em_max": max((r.get("em_mean", 1.0) for r in mh), default=None),
           "lengths": sorted({str(r["T"]) for r in retr}), "dirs": list(K1_DIRS),
           "errors": [r["path"] for r in rows if "error" in r]}
    res["K1_pass"] = bool(retr and all(r["pass"] for r in retr))
    res["K2_pass"] = bool(mh and all(r["pass"] for r in mh))
    return res


# ------------------------------------------------------------------------------------------ K3
def cmd_k3(a: argparse.Namespace) -> dict:
    from hybridlc.eval.items import item_content_sha256, load_items

    base = json.loads((OUT / "k3_baseline.json").read_text())
    repairs = json.loads((REPO / "experiments/items/vt24r_repairs.json").read_text())
    rep_by_file: dict[str, list] = {}
    for r in repairs["repairs"]:
        rep_by_file.setdefault(r["file"], []).append(r)
    cells, problems = [], []
    for rel, info in sorted(base["files"].items()):
        p = REPO / "experiments" / "items" / rel
        if not p.is_file():
            problems.append(f"{rel}: missing after regeneration")
            continue
        reps = rep_by_file.get(rel, [])
        new_c = item_content_sha256(p)
        row = {"file": rel, "n_repairs": len(reps), "content_identical": new_c == info["content_sha256"]}
        if not reps:
            if new_c != info["content_sha256"]:
                problems.append(f"{rel}: zero repairs but content differs")
        else:
            row.update(_k3_repaired_diff(p, REPO / "experiments/items_archive/pre-vt24r_97ef269" / rel, reps))
            if not row["only_repaired_fields_differ"]:
                problems.append(f"{rel}: differences outside repaired facts: {row['unexpected'][:5]}")
        cells.append(row)
    return {"check": "K3_content_stability", "n_cells": len(cells), "n_repaired_cells": sum(c["n_repairs"] > 0 for c in cells),
            "n_repaired_facts": sum(c["n_repairs"] for c in cells), "cells": cells, "problems": problems,
            "pass": not problems}


def _k3_repaired_diff(new_p: Path, old_p: Path, reps: list[dict]) -> dict:
    """Compare a repaired cell with its archived pre-fix file item by item: everything except the
    repaired facts' 4 value-digit tokens and their derived fields must be identical."""
    import numpy as np

    from hybridlc.eval.items import load_items

    new, cn = load_items(new_p)
    old, co = load_items(old_p)
    unexpected = []
    if len(new) != len(old):
        return {"only_repaired_fields_differ": False, "unexpected": ["n_items"]}
    by_ctx: dict[int, list[dict]] = {}
    for r in reps:
        by_ctx.setdefault(int(r["context"]), []).append(r)
    for i, (a, b) in enumerate(zip(new, old)):
        rs = by_ctx.get(int(b["meta"]["context_id"]), [])
        ca, cb = np.asarray(a["context_ids"]), np.asarray(b["context_ids"])
        if len(ca) != len(cb):
            unexpected.append(f"ctx{i}:len")
            continue
        allowed = np.zeros(len(ca), bool)
        olds, news = set(), set()
        for r in rs:
            allowed[r["digit_start"]:r["digit_start"] + 4] = True
            olds.add(int(r["old_value"]))
            news.add(int(r["new_value"]))
        diff = np.flatnonzero(ca != cb)
        if np.any(~allowed[diff]):
            unexpected.append(f"ctx{i}:context@{diff[~allowed[diff]][:3].tolist()}")
        # arrays: only fact_value entries of repaired facts
        for nm in set(a["arrays"]) | set(b["arrays"]):
            x, y = np.asarray(a["arrays"].get(nm)), np.asarray(b["arrays"].get(nm))
            if x.shape != y.shape:
                unexpected.append(f"ctx{i}:{nm}:shape")
            elif nm == "fact_value":
                d = np.flatnonzero(x != y)
                if any(int(y[j]) not in olds or int(x[j]) not in news for j in d):
                    unexpected.append(f"ctx{i}:fact_value")
            elif not np.array_equal(x, y):
                unexpected.append(f"ctx{i}:{nm}")
        for qa, qb in zip(a["queries"], b["queries"]):
            if not np.array_equal(qa["prompt_ids"], qb["prompt_ids"]):
                unexpected.append(f"ctx{i}:prompt")
            ma = {k: v for k, v in qa["meta"].items() if k not in ("value", "cand_values")}
            mb = {k: v for k, v in qb["meta"].items() if k not in ("value", "cand_values")}
            if ma != mb:
                unexpected.append(f"ctx{i}:qmeta")
            if not rs and (qa["meta"] != qb["meta"] or qa["candidates"] != qb["candidates"]
                           or not np.array_equal(qa["answer_ids"], qb["answer_ids"])):
                unexpected.append(f"ctx{i}:query-unrepaired-item")
        ia = {k: v for k, v in a["meta"].items() if k != "vt24r_repair"}
        if ia != b["meta"]:
            unexpected.append(f"ctx{i}:item-meta")
    return {"only_repaired_fields_differ": not unexpected, "unexpected": unexpected[:20]}


# ------------------------------------------------------------------------------------------ K5b
def cmd_k5b(a: argparse.Namespace) -> dict:
    base = json.loads((OUT / "k5b_baseline.json").read_text())
    r = subprocess.run([sys.executable, str(REPO / "scripts/repro/vt24r_baselines.py"), "k5b", "--out",
                        str(OUT / "k5b_after.json")], capture_output=True, text=True)
    if r.returncode != 0:
        return {"check": "K5b", "pass": False, "error": r.stderr[-2000:]}
    after = json.loads((OUT / "k5b_after.json").read_text())
    seq_eq = [x == y for x, y in zip(base["sequences"], after["sequences"])]
    res = {"check": "K5b_full_block_hash", "aggregate_baseline": base["aggregate_sha256"],
           "aggregate_after": after["aggregate_sha256"], "block_plan_equal": base["block_plan_sha256"] == after["block_plan_sha256"],
           "n_sequences": len(seq_eq), "n_sequences_equal": sum(seq_eq),
           "vt17_equal": base["vt17_batch_hashes"] == after["vt17_batch_hashes"],
           "baseline_pre_fix_tree": base["provenance"]["pre_fix_tree"]}
    res["pass"] = bool(res["aggregate_baseline"] == res["aggregate_after"] and all(seq_eq) and len(seq_eq) == 2000
                       and res["block_plan_equal"] and res["vt17_equal"])
    return res


# ------------------------------------------------------------------------------------------ summary
def cmd_summary(a: argparse.Namespace) -> dict:
    def load(n):
        p = OUT / n
        return json.loads(p.read_text()) if p.exists() else None

    k0, k12, k3, k5b = load("k0_agreement.json"), load("k1k2.json"), load("k3_stability.json"), load("k5b_check.json")
    k4 = load("k4_result.json")
    k56 = load("k5a_k6_result.json")
    checks = {
        "K0": bool(k0 and k0["pass"]), "K1": bool(k12 and k12["K1_pass"]), "K2": bool(k12 and k12["K2_pass"]),
        "K3": bool(k3 and k3["pass"]), "K4": bool(k4 and k4["pass"]),
        "K5a": bool(k56 and k56.get("K5a")), "K5b": bool(k5b and k5b["pass"]), "K6": bool(k56 and k56.get("K6")),
    }
    rep = json.loads((REPO / "experiments/items/vt24r_repairs.json").read_text())
    return {"item": "VT-24r", "version": 2, "prereg": "experiments/prereg/VT-24r.v2.yaml",
            "prereg_sha256": hashlib.sha256((REPO / "experiments/prereg/VT-24r.v2.yaml").read_bytes()).hexdigest(),
            "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL",
            "repairs": {"n_records": len(rep["repairs"]), "n_distinct_facts": rep.get("n_distinct_facts"), "by_cell": rep.get("by_cell"), "by_template": rep.get("by_template"),
                        "triggered_by_filler": rep.get("triggered_by_filler")},
            "k0_coincidental_match_queries": k0 and k0["coincidental_match_queries"],
            "meta": _meta()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("k0", "k1k2", "k3", "k5b", "summary"):
        p = sub.add_parser(nm)
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--items", default=str(REPO / "experiments/items_archive/pre-vt24r_97ef269"))
    a = ap.parse_args(argv)
    fn = {"k0": cmd_k0, "k1k2": cmd_k1k2, "k3": cmd_k3, "k5b": cmd_k5b, "summary": cmd_summary}[a.cmd]
    res = fn(a)
    res.setdefault("meta", _meta())
    name = {"k0": "k0_agreement.json", "k1k2": "k1k2.json", "k3": "k3_stability.json", "k5b": "k5b_check.json",
            "summary": "vt24r_summary.json"}[a.cmd]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_file", "cells")}, default=str)[:3000])
    ok = res.get("pass", res.get("verdict") == "PASS" if "verdict" in res else res.get("K1_pass") and res.get("K2_pass"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
