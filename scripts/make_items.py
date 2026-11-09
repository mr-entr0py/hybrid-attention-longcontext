#!/usr/bin/env python
"""Generate the frozen evaluation item files (protocol §9.3-9.6, §9.8) and their MANIFEST.

Usage (after ``data/tokenizer/`` and ``data/tokens/`` exist, i.e. after
``scripts/data_tokenizer.py`` and ``scripts/data_pretokenize.py``)::

    nohup python scripts/make_items.py --suite all --jobs 8 > experiments/items/make_items.log 2>&1 &

Options:
    --suite CS MS DG DC FD Tier0 | all
    --out experiments/items        output root (files go to <out>/<namespace>/<suite>/<cell>.npz)
    --tokens data/tokens           token stores (filler, LM-1/2/4 sources, LM-3 sample)
    --dev-filler                   use the C4 dev text filler instead of data/tokens (testing only)
    --tokenizer DIR                tokenizer directory (default data/tokenizer)
    --only SUBSTR                  only cells whose relpath contains SUBSTR
    --n-max N                      cap contexts per cell (smoke tests only; never for frozen files)
    --u15                          600 contexts in the 16K/32K confirmatory retrieval cells (U-15)
    --lm3-sample-tokens 100000000  size of the training sample for the LM-3 4-gram exclusion

Existing files are never overwritten (they are verified to load and skipped). The manifest
(``<out>/<namespace>/MANIFEST.json``) is rewritten at the end from the files on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

_STATE: dict = {}


def _init(args: dict) -> None:
    from hybridlc.eval.tokens import token_space

    if args["tokenizer"] == "dev":
        from hybridlc.eval.devtools import dev_tokenizer

        tok = dev_tokenizer(args["dev_tok_dir"])
    else:
        from hybridlc.data.tokenizer import load_tokenizer

        tok = load_tokenizer(Path(args["tokenizer"]))
    _STATE.update(tok=tok, ts=token_space(tok), args=args, fillers={})


def _filler(ns: str):
    f = _STATE["fillers"].get(ns)
    if f is None:
        from hybridlc.eval.filler import StoreFiller, dev_text_filler

        a = _STATE["args"]
        f = dev_text_filler(_STATE["ts"], ns) if a["dev_filler"] else StoreFiller(_STATE["ts"], ns, a["tokens"])
        _STATE["fillers"][ns] = f
    return f


def make_cell(cell_dict: dict) -> dict:
    """Generate and save one cell (runs in a worker)."""
    from hybridlc.eval import tier0
    from hybridlc.eval.generators import generate
    from hybridlc.eval.items import eval_code_sha256, save_items
    from hybridlc.eval.rng import eval_rng
    from hybridlc.eval.suites import Cell

    a = _STATE["args"]
    c = Cell(**{**cell_dict, "level": tuple(tuple(x) for x in cell_dict["level"])})
    out = Path(a["out"]) / c.namespace / c.relpath
    if out.exists():
        from hybridlc.eval.items import load_items

        load_items(out)
        return dict(path=str(out), skipped=True)
    t0 = time.time()
    tok, ts = _STATE["tok"], _STATE["ts"]
    n = min(c.n, a["n_max"]) if a["n_max"] else c.n
    rng = eval_rng(c.namespace, *c.rng_key())
    lv = c.gen_level()
    if c.task == "LM-1":
        banned_p = Path(a["out"]) / c.namespace / "Tier0" / "lm3_banned_4grams.npy"
        banned = np.load(banned_p) if banned_p.is_file() else None
        items = tier0.gen_lm1(lv, c.T, n, rng, c.namespace, tok, _filler(c.namespace), banned=banned)
    elif c.task == "LM-2":
        items = tier0.gen_lm2(lv, c.T, n, rng, c.namespace, tok, None, root=a["tokens"])
    elif c.task == "LM-4":
        items = tier0.gen_lm4(lv, c.T, n, rng, c.namespace, tok, _filler(c.namespace))
    else:
        items = generate(c.task, lv, c.T, n, rng, c.namespace, tok, _filler(c.namespace))
    meta = dict(namespace=c.namespace, suite=c.suite, task=c.task, level=c.level_dict, T=c.T, n=len(items),
                n_planned=c.n, variant=c.variant, name=c.name, purpose=c.purpose, rng_key=list(c.rng_key()),
                tokenizer_sha256=ts.fingerprint, eval_code_sha256=eval_code_sha256(),
                filler="dev_text" if a["dev_filler"] else "data/tokens")
    sha = save_items(out, items, meta)
    return dict(path=str(out), sha256=sha, n=len(items), seconds=round(time.time() - t0, 2), skipped=False)


def build_lm3_banned(args: dict, out_path: Path) -> None:
    """Top-1% 4-gram types of a training sample (85% FineWeb-Edu train / 15% PG19 train, Stage-A mix)."""
    from hybridlc.data.loader import get_store
    from hybridlc.eval.rng import eval_rng
    from hybridlc.eval.tier0 import build_lm3_banned as _banned

    n_tot = int(args["lm3_sample_tokens"])
    rng = eval_rng("test", "LM-3", "train-sample")
    parts = []
    for corpus, frac in (("fwe", 0.85), ("pg19", 0.15)):
        st = get_store(corpus, "train", args["tokens"])
        want = int(n_tot * frac)
        start = int(rng.integers(0, max(1, st.n_tokens - want)))
        parts.append(np.asarray(st.tokens[start:start + want]))
    banned = _banned(np.concatenate(parts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, banned)
    print(f"LM-3: {len(banned)} banned 4-gram types from {n_tot} tokens -> {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", default=["all"])
    ap.add_argument("--out", default=str(REPO / "experiments" / "items"))
    ap.add_argument("--tokens", default=str(REPO / "data" / "tokens"))
    ap.add_argument("--tokenizer", default=str(REPO / "data" / "tokenizer"), help="directory, or 'dev'")
    ap.add_argument("--dev-tok-dir", default=str(REPO / "data" / "dev_tokenizer"))
    ap.add_argument("--dev-filler", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-max", type=int, default=0)
    ap.add_argument("--u15", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--lm3-sample-tokens", type=int, default=100_000_000)
    ap.add_argument("--no-lm3-banned", action="store_true", help="skip the LM-3 exclusion set (testing)")
    args = ap.parse_args(argv)

    from hybridlc.eval.suites import SUITES, suite_cells

    names = list(SUITES) if args.suite == ["all"] else args.suite
    cells = [c for nm in names for c in suite_cells(nm, **({"u15": args.u15} if nm == "CS" else {}))]
    if args.only:
        cells = [c for c in cells if args.only in c.relpath]
    a = dict(vars(args))
    print(f"{len(cells)} cells; out={args.out}", flush=True)
    _init(a)
    # LM-3 exclusion set must exist before LM-1 items are made
    if any(c.task == "LM-1" for c in cells) and not args.no_lm3_banned:
        bp = Path(args.out) / "test" / "Tier0" / "lm3_banned_4grams.npy"
        if not bp.is_file():
            build_lm3_banned(a, bp)
    dicts = [dict(namespace=c.namespace, suite=c.suite, task=c.task, level=[list(x) for x in c.level], T=c.T, n=c.n,
                  variant=c.variant, purpose=c.purpose) for c in cells]
    results, failed = [], []
    if args.jobs <= 1:
        for d in dicts:
            try:
                results.append(make_cell(d))
                print(json.dumps(results[-1]), flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append((d, repr(e)))
                print("FAILED", d, repr(e), flush=True)
    else:
        # longest cells first for load balance
        order = sorted(range(len(dicts)), key=lambda i: -(dicts[i]["T"] or 0) * dicts[i]["n"])
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init, initargs=(a,)) as ex:
            futs = {ex.submit(make_cell, dicts[i]): i for i in order}
            for f in as_completed(futs):
                d = dicts[futs[f]]
                try:
                    results.append(f.result())
                    print(json.dumps(results[-1]), flush=True)
                except Exception as e:  # noqa: BLE001
                    failed.append((d, repr(e)))
                    print("FAILED", d, repr(e), flush=True)
    from hybridlc.eval.items import eval_code_sha256, write_manifest
    from hybridlc.eval.words import describe

    for ns in sorted({c.namespace for c in cells}):
        root = Path(args.out) / ns
        if root.is_dir():
            m = write_manifest(root, extra=dict(tokenizer_sha256=_STATE["ts"].fingerprint,
                                                eval_code_sha256=eval_code_sha256(),
                                                word_lists=describe(_STATE["tok"]),
                                                n_max_cap=args.n_max or None,
                                                filler="dev_text" if args.dev_filler else "data/tokens"))
            print(f"manifest: {m}", flush=True)
    print(f"done: {len(results)} ok, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
