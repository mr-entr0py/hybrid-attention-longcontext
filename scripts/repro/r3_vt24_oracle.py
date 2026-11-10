"""VT-24 (R3e) runner: InductionLM(order 6, p_hit 0.9) oracle answerability of the 16K/32K items.

    python scripts/repro/r3_vt24_oracle.py [--workers 12] [--limit-files N]

Scores every R1 / R-load / R3 / R-dist item file at 16K and 32K in experiments/items/{test/CS,
test/MS, dev/DG, dev/FD} (plus every M-hop file there as the negative control) through
``hybridlc.eval.scorer.score_file`` on the CPU, then applies the frozen criteria
(experiments/repro/R3/prereg.yaml, VT-24). A supplementary metadata oracle (report-only) checks
the item layout and that each gold value sits inside its recorded needle span.

Outputs (never overwritten; reruns go to a new run directory):
  experiments/repro/R3/vt24/run_<ts>/scores/<ns>/<suite>/<stem>.jsonl   (scorer rows + _meta.json)
  experiments/repro/R3/vt24/run_<ts>/vt24_files.jsonl                   (one row per file)
  experiments/repro/R3/vt24/run_<ts>/vt24_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_common as RC  # noqa: E402

DIRS = ["test/CS", "test/MS", "dev/DG", "dev/FD"]
RETRIEVAL = {"R1", "R-load", "R3", "R-dist"}
LENGTHS = {16384, 32768}
LP_GOLD = 4 * math.log(0.9)
LP_TOL = 1e-4
ORDER, P_HIT = 6, 0.9


def enumerate_files() -> list[dict]:
    from hybridlc.eval.items import load_items

    out = []
    for d in DIRS:
        for p in sorted((RC.REPO / "experiments" / "items" / d).glob("*.npz")):
            _, cell = load_items(p)
            T = cell.get("T")
            task = cell.get("task")
            if T not in LENGTHS:
                continue
            if task in RETRIEVAL:
                role = "retrieval"
            elif task == "M-hop":
                role = "mhop_negative_control"
            else:
                continue
            out.append({"path": str(p.relative_to(RC.REPO)), "dir": d, "task": task, "T": T,
                        "variant": cell.get("variant"), "level": cell.get("level"), "role": role,
                        "n_contexts": cell.get("n")})
    return out


def metadata_oracle(path: Path) -> dict:
    """Report-only layout/answer-presence checks using the item metadata."""
    import numpy as np

    from hybridlc.eval.items import load_items

    items, cell = load_items(path)
    T = int(cell["T"])
    n_q = bad_len = bad_presence = bad_key = 0
    examples = []
    for ci, it in enumerate(items):
        ctx = np.asarray(it["context_ids"])
        for q in it["queries"]:
            n_q += 1
            pr, an = np.asarray(q["prompt_ids"]), np.asarray(q["answer_ids"])
            if len(ctx) + len(pr) + len(an) != T:
                bad_len += 1
            m = q["meta"]
            starts, ends = m.get("needle_start", []), m.get("needle_end", [])
            if not starts:
                continue
            # R3: the gold is the latest update = needle with the largest start; others: first needle
            j = int(np.argmax(starts)) if cell.get("task") == "R3" else 0
            span = ctx[int(starts[j]):int(ends[j])]
            if not _contains(span, an):
                bad_presence += 1
                if len(examples) < 5:
                    examples.append({"context": ci, "query": int(m.get("query_idx", -1)), "issue": "answer not in needle"})
            key_toks = pr[3:5] if len(pr) >= 5 else pr  # "The code for {adj} {noun} is" -> key = 2 tokens (T1)
            if cell.get("variant") != "T4" and not _contains(span, key_toks):
                bad_key += 1
    return {"meta_n_queries": n_q, "meta_bad_length": bad_len, "meta_answer_not_in_needle": bad_presence,
            "meta_key_not_in_needle": bad_key, "meta_examples": examples}


def _contains(hay, needle) -> bool:
    import numpy as np

    hay, needle = np.asarray(hay), np.asarray(needle)
    n = len(needle)
    if n == 0 or len(hay) < n:
        return False
    w = np.lib.stride_tricks.sliding_window_view(hay, n)
    return bool((w == needle).all(1).any())


def score_one(rec: dict, out_root: str) -> dict:
    import torch

    torch.set_num_threads(1)
    from hybridlc.eval.devtools import InductionLM
    from hybridlc.eval.scorer import score_file

    t0 = time.time()
    p = RC.REPO / rec["path"]
    lm = InductionLM(16384, order=ORDER, p_hit=P_HIT)
    out = Path(out_root) / "scores" / rec["dir"] / f"{p.stem}.jsonl"
    res_path = score_file(lm, p, out, score_candidates=False, autocast=False,
                          extra_meta={"vt": "VT-24", "oracle": f"InductionLM(order={ORDER}, p_hit={P_HIT})"})
    rows = [json.loads(x) for x in Path(res_path).read_text().splitlines()]
    em = [bool(r["EM_tf"]) for r in rows]
    lp = [float(r["logp_gold"]) for r in rows]
    misses = [{"context_id": r["context_id"], "query_idx": r["query_idx"], "EM_tf": r["EM_tf"],
               "logp_gold": r["logp_gold"], "argmax_ids": r.get("argmax_ids"), "gold_ids": r.get("gold_ids"),
               "depth": r.get("depth"), "distance": r.get("distance")}
              for r in rows if (not r["EM_tf"]) or abs(float(r["logp_gold"]) - LP_GOLD) > LP_TOL]
    row = dict(rec, scores=str(Path(res_path).relative_to(RC.REPO)), n_queries=len(rows),
               em_mean=sum(em) / max(len(em), 1), n_em_fail=len(em) - sum(em),
               n_logp_off=sum(abs(v - LP_GOLD) > LP_TOL for v in lp),
               logp_min=min(lp) if lp else None, logp_max=max(lp) if lp else None,
               misses_first20=misses[:20], n_misses=len(misses), secs=time.time() - t0)
    if rec["role"] == "retrieval":
        row["pass"] = bool(len(rows) > 0 and row["n_em_fail"] == 0 and row["n_logp_off"] == 0)
        row.update(metadata_oracle(p))
    else:
        row["pass"] = bool(row["em_mean"] < 0.5)
    return row


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-files", type=int, default=0)
    a = ap.parse_args(argv)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_root = RC.OUT_ROOT / "vt24" / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=False)
    meta = RC.run_meta("VT-24", {"oracle": f"InductionLM(16384, order={ORDER}, p_hit={P_HIT})",
                                 "criteria": {"retrieval": "EM_tf=1 and |logp_gold-4ln0.9|<=1e-4 on every query",
                                              "mhop": "per-file mean EM_tf < 0.5"},
                                 "score_candidates": False, "device": "cpu", "workers": a.workers})
    files = enumerate_files()
    if a.limit_files:
        files = files[:a.limit_files]
    meta["files"] = files
    (out_root / "vt24_files_planned.json").write_text(json.dumps(files, indent=1))
    rows = []
    with open(out_root / "vt24_files.jsonl", "x") as f, ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(score_one, r, str(out_root)): r for r in files}
        for fu in as_completed(futs):
            try:
                row = fu.result()
            except Exception as e:  # recorded, counts as failure
                row = dict(futs[fu], error=repr(e), **{"pass": False})
            rows.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            print(f"[{len(rows)}/{len(files)}] {row['path']} role={row['role']} em={row.get('em_mean')} "
                  f"pass={row['pass']} ({row.get('secs', 0):.0f}s)", flush=True)
    retr = [r for r in rows if r["role"] == "retrieval"]
    neg = [r for r in rows if r["role"] != "retrieval"]
    summary = {"n_files": len(rows), "n_retrieval_files": len(retr), "n_mhop_files": len(neg),
               "n_retrieval_queries": sum(r.get("n_queries", 0) for r in retr),
               "retrieval_queries_failing": sum(r.get("n_misses", 0) for r in retr),
               "retrieval_files_failing": [r["path"] for r in retr if not r["pass"]],
               "mhop_files_failing": [r["path"] for r in neg if not r["pass"]],
               "mhop_em_max": max((r.get("em_mean", 1) for r in neg), default=None),
               "meta_oracle": {k: sum(r.get(k, 0) for r in retr) for k in
                               ("meta_n_queries", "meta_bad_length", "meta_answer_not_in_needle", "meta_key_not_in_needle")},
               "errors": [r["path"] for r in rows if "error" in r]}
    summary["verdict"] = "PASS" if (rows and all(r["pass"] for r in rows) and not summary["errors"]) else "FAIL"
    RC.finish_meta(meta)
    meta["summary"] = summary
    (out_root / "vt24_summary.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    s = main()
    sys.exit(0 if s["verdict"] == "PASS" else 1)
