#!/usr/bin/env python
"""Train the 16,384 byte-level BPE tokenizer (protocol §6.2).

Training text: 1 GB (10^9 UTF-8 bytes), 80% FineWeb-Edu shard 001 and 20% PG19
train, sampling seed 0. Documents are drawn in a seeded random order
(``np.random.default_rng(0)``; one generator, FWE first then PG19) and taken
whole until the byte budget is reached; the last document is truncated (at a
UTF-8 character boundary) so the budget is met exactly.

After training, bytes/token is measured on FineWeb-Edu shard-000 *dev* documents
and PG19 validation books (both never used for tokenizer training) and written to
``data/tokenizer/training_report.json``.

Usage: nohup python scripts/data_tokenizer.py > data/logs/tokenizer.log 2>&1 &
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from hybridlc.data import sources as S
from hybridlc.data.tokenizer import load_raw_tokenizer, tokenizer_sha256, train_tokenizer

TOTAL_BYTES = 10**9
FWE_FRACTION = 0.8
SAMPLING_SEED = 0


def _truncate_utf8(text: str, nbytes: int) -> str:
    return text.encode("utf-8")[:nbytes].decode("utf-8", errors="ignore")


def sample_fwe(rng: np.random.Generator, budget: int) -> tuple[list[str], dict]:
    path = S.raw_path(S.FWE_REPO, S.FWE_TOKENIZER_FILE)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    order = rng.permutation(len(texts))
    out, used, h = [], 0, hashlib.sha256()
    for i in order:
        t = texts[i]
        b = len(t.encode("utf-8"))
        if used + b >= budget:
            t = _truncate_utf8(t, budget - used)
            b = len(t.encode("utf-8"))
        out.append(t); used += b; h.update(int(i).to_bytes(8, "little"))
        if used >= budget - 4:  # within one (multi-byte) character of the budget
            break
    return out, dict(file=S.FWE_TOKENIZER_FILE, n_docs=len(out), bytes=used, n_available=len(texts),
                     row_order_sha256=h.hexdigest())


def sample_pg19(rng: np.random.Generator, budget: int) -> tuple[list[str], dict]:
    files = sorted(S.raw_path(S.PG19_REPO, "data").glob("train-*.parquet"))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    offs = np.concatenate([[0], np.cumsum(counts)])
    order = rng.permutation(int(offs[-1]))
    # Books average ~0.4 MB, so the first 2,000 candidates are ample for 200 MB.
    cand = order[:2000]
    need: dict[int, list[int]] = {}
    for g in cand:
        fi = int(np.searchsorted(offs, g, side="right") - 1)
        need.setdefault(fi, []).append(int(g - offs[fi]))
    texts_by_g: dict[int, str] = {}
    for fi, rows in need.items():
        tbl = pq.read_table(files[fi], columns=["text"]).take(rows)
        for r, t in zip(rows, tbl.column("text").to_pylist()):
            texts_by_g[int(offs[fi] + r)] = t
    out, used, h = [], 0, hashlib.sha256()
    for g in cand:
        t = texts_by_g[int(g)]
        b = len(t.encode("utf-8"))
        if used + b >= budget:
            t = _truncate_utf8(t, budget - used)
            b = len(t.encode("utf-8"))
        out.append(t); used += b; h.update(int(g).to_bytes(8, "little"))
        if used >= budget - 4:
            break
    assert used >= budget - 4, "not enough PG19 candidates"
    return out, dict(files="train-*", n_docs=len(out), bytes=used, n_available=int(offs[-1]),
                     row_order_sha256=h.hexdigest())


def compression_eval() -> dict:
    tok = load_raw_tokenizer(plain_specials=True)
    res = {}
    # FWE shard 000 dev half: first 3,000 odd-parity documents in file order.
    pf = pq.ParquetFile(S.raw_path(S.FWE_REPO, S.FWE_HELDOUT_FILE))
    dev = []
    for batch in pf.iter_batches(batch_size=1000, columns=["text", "id"]):
        for t, i in zip(batch.column("text").to_pylist(), batch.column("id").to_pylist()):
            if S.fwe_heldout_split(i) == "dev":
                dev.append(t)
        if len(dev) >= 3000:
            break
    dev = dev[:3000]
    val_path = next(S.raw_path(S.PG19_REPO, "data").glob("validation-*.parquet"))
    val = pq.read_table(val_path, columns=["text"]).column("text").to_pylist()
    for name, texts in [("fwe_dev_3000", dev), ("pg19_validation_50", val)]:
        nb = sum(len(t.encode("utf-8")) for t in texts)
        nt = sum(len(e.ids) for e in tok.encode_batch_fast(texts, add_special_tokens=False))
        res[name] = dict(bytes=nb, tokens=nt, bytes_per_token=nb / nt)
    # Digit isolation and whitespace convention (§6.2 answer encoding).
    probe = tok.encode(" 4821", add_special_tokens=False).tokens
    res["probe_' 4821'"] = probe
    return res


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SAMPLING_SEED)  # local generator, not the global RNG
    fwe_budget = int(TOTAL_BYTES * FWE_FRACTION)
    fwe, fwe_info = sample_fwe(rng, fwe_budget)
    pg, pg_info = sample_pg19(rng, TOTAL_BYTES - fwe_budget)
    print(f"sampled FWE {fwe_info['bytes']/1e6:.1f} MB ({fwe_info['n_docs']} docs), "
          f"PG19 {pg_info['bytes']/1e6:.1f} MB ({pg_info['n_docs']} books) in {time.time()-t0:.0f}s", flush=True)
    texts = fwe + pg
    t1 = time.time()
    train_tokenizer(texts, length=len(texts))
    train_s = time.time() - t1
    del texts, fwe, pg
    print(f"trained in {train_s:.0f}s; sha256 {tokenizer_sha256()}", flush=True)
    comp = compression_eval()
    report = dict(total_bytes=TOTAL_BYTES, fwe_fraction=FWE_FRACTION, sampling_seed=SAMPLING_SEED,
                  fwe=fwe_info, pg19=pg_info, train_seconds=train_s, tokenizer_sha256=tokenizer_sha256(),
                  compression=comp)
    (S.TOKENIZER_DIR / "training_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
