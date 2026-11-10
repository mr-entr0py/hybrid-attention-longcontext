#!/usr/bin/env python
"""VT-24r v2 baselines, computed on the PRE-FIX tree (experiments/prereg/VT-24r.v2.yaml,
``baselines_before_the_fix``), and recomputed after the fix for K5b / K5a.

Subcommands (CPU only; no CUDA context)::

    python scripts/repro/vt24r_baselines.py k5b --out experiments/repro/VT-24r/k5b_baseline.json
    python scripts/repro/vt24r_baselines.py k3  --out experiments/repro/VT-24r/k3_baseline.json
    python scripts/repro/vt24r_baselines.py k5a --out experiments/repro/VT-24r/k5a_baseline.json [--find]

* ``k5b``: TEFStream(seed_index=101) block 0 at the protocol defaults (1,000 steps x 2 micro-batches
  of 32K): the block-plan hash, the SHA-256 of every sequence's input_ids, labels, loss_weights and
  answer_mask, one aggregate SHA-256; plus VT-17's batch hashes for seed indices 0 and 1, steps 0-2.
* ``k3``: per item file of experiments/items/{test,dev}: file SHA-256, content SHA-256
  (items.item_content_sha256) and the SHA-256 of every zip member (arrays and meta.json).
* ``k5a``: the planted-collision items of the K5a leak unit test, generated through ``generate()``
  exactly as the TEF calls it, in namespace 'train' and in namespace 'dev'. ``--find`` searches the
  attempt index for a level/rng whose item violates rule G-ORACLE (pre-fix), then records each
  item's serialised SHA-256 and the checker's per-query record.

Every output records the provenance of the code it ran on: git HEAD, and for each file under
src/hybridlc/eval/ and src/hybridlc/data/tef.py its git blob hash in the working tree and at HEAD
(``pre_fix_tree = true`` iff every pre-existing generator file equals HEAD).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

TRACKED = sorted([*(p.relative_to(REPO).as_posix() for p in (REPO / "src/hybridlc/eval").glob("*.py")),
                  "src/hybridlc/data/tef.py"])


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    files = {}
    for rel in TRACKED:
        wt = _git("hash-object", rel)
        head = _git("rev-parse", f"HEAD:{rel}") if _git("ls-files", rel) else None
        files[rel] = {"worktree_blob": wt, "head_blob": head, "equal_head": head == wt if head else None}
    gen_files = [r for r, v in files.items() if v["head_blob"] is not None]
    return {"git_head": _git("rev-parse", "HEAD"), "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": files, "pre_fix_tree": all(files[r]["equal_head"] for r in gen_files),
            "new_untracked_eval_files": [r for r, v in files.items() if v["head_blob"] is None]}


# ------------------------------------------------------------------------------------------ K5b
def _sha(a: np.ndarray, dt: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype(dt).tobytes()).hexdigest()


def _k5b_step(step: int) -> list[dict]:
    from hybridlc.data.tef import TEFStream

    s = _STREAM.get("s")
    if s is None:
        s = _STREAM["s"] = TEFStream(101, 1000)
    out = []
    for m in range(s.accum):
        b = s.get_numpy(step, m)
        out.append({"step": step, "micro": m, "input_ids": _sha(b["input_ids"][0], "<i4"),
                    "labels": _sha(b["labels"][0], "<i4"), "loss_weights": _sha(b["loss_weights"][0], "<f4"),
                    "answer_mask": _sha(b["answer_mask"][0], "|u1"), "batch_hash": b["batch_hash"]})
    return out


_STREAM: dict = {}


def plan_hash(stream, block: int) -> str:
    h = hashlib.sha256()
    for seq in stream.block_plan(block):
        for sg in seq:
            sp = sg.spec
            h.update(json.dumps([sg.kind, sg.length, sg.replay_kind, sg.replay_uid,
                                 None if sp is None else [sp.uid, sp.length, sp.family, sp.subtype, sp.repetitive]]
                                ).encode())
        h.update(b"|")
    return h.hexdigest()


def cmd_k5b(a: argparse.Namespace) -> dict:
    from hybridlc.data.tef import TEFStream

    t0 = time.time()
    s = TEFStream(101, 1000)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = [r for chunk in ex.map(_k5b_step, range(1000), chunksize=10) for r in chunk]
    agg = hashlib.sha256()
    for r in rows:
        agg.update("|".join([r["input_ids"], r["labels"], r["loss_weights"], r["answer_mask"]]).encode())
    vt17 = {}
    for seed in (0, 1):
        st = TEFStream(seed, 1000)
        vt17[str(seed)] = {f"{step}:{m}": st.get_numpy(step, m)["batch_hash"] for step in range(3) for m in range(2)}
    return {"item": "VT-24r K5b", "seed_index": 101, "block": 0, "n_sequences": len(rows),
            "block_plan_sha256": plan_hash(s, 0), "aggregate_sha256": agg.hexdigest(), "sequences": rows,
            "vt17_batch_hashes": vt17, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------ K3
def member_hashes(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in sorted(zf.namelist())}


def cmd_k3(a: argparse.Namespace) -> dict:
    from hybridlc.eval.items import item_content_sha256, sha256_file

    root = Path(a.items)
    files = {}
    for ns in ("test", "dev"):
        for p in sorted((root / ns).rglob("*.npz")):
            rel = p.relative_to(root).as_posix()
            files[rel] = {"sha256": sha256_file(p), "content_sha256": item_content_sha256(p),
                          "members": member_hashes(p)}
    manifests = {ns: hashlib.sha256((root / ns / "MANIFEST.json").read_bytes()).hexdigest() for ns in ("test", "dev")}
    return {"item": "VT-24r K3 baseline", "items_root": str(root), "n_files": len(files),
            "manifest_sha256": manifests, "files": files}


# ------------------------------------------------------------------------------------------ K5a
K5A_SPECS = {
    # namespace 'train': the TEF call path (generate(task, level, length, 1, rng, 'train', tok, filler));
    # T3's 6-token window before digit 4 ("the number" + " " + d1 d2 d3) does not contain the key, so a
    # later fact with the same 3-digit prefix collides.
    "train": dict(task="R-load", level={"N": 256, "template": "T3", "filler": "natural"}, T=8192),
    # namespace 'dev': T4's window before digit 3 ("'s code:" + " " + d1 d2) does not contain the key.
    "dev": dict(task="R-load", level={"N": 64, "template": "T4"}, T=4096),
}


def item_sha(item: dict) -> str:
    from hybridlc.eval.items import items_to_members

    mem = items_to_members([item], {"k5a": True})
    h = hashlib.sha256()
    for n in sorted(mem):
        h.update(n.encode() + b"\0" + hashlib.sha256(mem[n]).digest())
    return h.hexdigest()


def k5a_rng(ns: str, attempt: int):
    from hybridlc.eval.rng import child_rng, eval_rng, tef_rng

    if ns == "train":
        return child_rng(tef_rng(9101, "item", 0, 0), "attempt", attempt)
    return eval_rng("dev", "VT-24r-K5a", attempt)


def k5a_item(ns: str, attempt: int, tok, filler):
    from hybridlc.eval.generators import generate

    sp = K5A_SPECS[ns]
    return generate(sp["task"], dict(sp["level"]), sp["T"], 1, k5a_rng(ns, attempt), ns, tok, filler)[0]


def k5a_env():
    from hybridlc.eval.devtools import resolve_test_tokenizer
    from hybridlc.eval.filler import StoreFiller
    from hybridlc.eval.tokens import token_space

    tok, kind = resolve_test_tokenizer()
    ts = token_space(tok)
    return tok, kind, ts, {ns: StoreFiller(ts, ns) for ns in ("train", "dev")}


def cmd_k5a(a: argparse.Namespace) -> dict:
    from hybridlc.eval.oracle_check import check_item

    tok, kind, ts, fillers = k5a_env()
    out = {"item": "VT-24r K5a baseline", "tokenizer": kind, "tokenizer_fingerprint": ts.fingerprint,
           "specs": K5A_SPECS, "cases": {}}
    prev = json.loads(Path(a.out).read_text()) if (not a.find and Path(a.out).exists()) else None
    for ns in ("train", "dev"):
        attempts = range(200) if a.find else [prev["cases"][ns]["attempt"]]
        for att in attempts:
            it = k5a_item(ns, att, tok, fillers[ns])
            rec = check_item(it)
            bad = [r for r in rec if not r["loc_ok"]]
            if bad or not a.find:
                out["cases"][ns] = {"attempt": att, "item_sha256": item_sha(it), "n_loc_miss_queries": len(bad),
                                    "loc_miss_query_idx": [r["query_idx"] for r in bad],
                                    "context_sha256": hashlib.sha256(np.asarray(it["context_ids"], "<i4").tobytes()).hexdigest()}
                break
        else:
            raise SystemExit(f"k5a: no planted collision found for {ns} in 200 attempts")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("k5b", "k3", "k5a"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--items", default=str(REPO / "experiments" / "items"))
        p.add_argument("--find", action="store_true")
    a = ap.parse_args(argv)
    prov = provenance()
    res = {"k5b": cmd_k5b, "k3": cmd_k3, "k5a": cmd_k5a}[a.cmd](a)
    res["provenance"] = prov
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("sequences", "files", "provenance")}, default=str)[:2000])
    print(f"pre_fix_tree={prov['pre_fix_tree']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
