#!/usr/bin/env python
"""Pre-tokenise the pinned corpora into flat uint16 token files (protocol §6.1, §6.4).

Output per (corpus, split), in ``data/tokens/<corpus>/``:
  <split>.bin        flat uint16 tokens; every document is ``[0] + encode(text)``
  <split>.idx.npy    int64 document start offsets (len n_docs + 1; last = total)
  <split>.docs.npz   per-document id_hash / text_hash (first 16 bytes of SHA-256),
                     src_file (index into meta['source_files']), src_row
  <split>.meta.json  counts, length distribution, source file SHA-256s

Documents with empty text are skipped (counted in meta). The literal string
``<|endoftext|>`` inside a document is tokenised as plain text, so id 0 occurs
only at document starts (asserted).

Usage: nohup python scripts/data_pretokenize.py [--workers 22] > data/logs/pretok.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hybridlc.data import sources as S

TMP = S.TOKENS_DIR / "_tmp"
_TOK = None  # per-worker tokenizer


def _init_worker() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TOK
    from hybridlc.data.tokenizer import load_raw_tokenizer
    _TOK = load_raw_tokenizer(plain_specials=True)


def _h16(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()[:16]


def build_tasks() -> list[dict]:
    """One task per parquet row group, in canonical (file, row group) order."""
    import pyarrow.parquet as pq

    tasks: list[dict] = []
    files: list[tuple[str, str, str]] = []  # (corpus, repo, filename)
    files += [("fwe", S.FWE_REPO, f) for f in S.FWE_FILES]
    pg_dir = S.raw_path(S.PG19_REPO, "data")
    for p in sorted(pg_dir.glob("*.parquet")):
        files.append(("pg19", S.PG19_REPO, f"data/{p.name}"))
    for fidx, (corpus, repo, fname) in enumerate(files):
        path = S.raw_path(repo, fname)
        md = pq.ParquetFile(path).metadata
        row0 = 0
        for rg in range(md.num_row_groups):
            n = md.row_group(rg).num_rows
            if corpus == "fwe":
                split = "heldout" if fname == S.FWE_HELDOUT_FILE else "train"
            else:
                split = S.pg19_split_of_file(fname)
            tasks.append(dict(tid=len(tasks), corpus=corpus, split=split, file=fname, fidx=fidx,
                              path=str(path), rg=rg, row0=row0, n=n))
            row0 += n
    return tasks


def run_task(t: dict) -> dict:
    """Tokenise one row group; write temp outputs; return their description."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(t["path"])
    id_col = "id" if t["corpus"] == "fwe" else "url"
    out: dict[str, dict] = {}
    bufs: dict[str, dict[str, list]] = {}
    row = t["row0"]
    n_empty = 0
    for batch in pf.iter_batches(batch_size=1000 if t["corpus"] == "fwe" else 25,
                                 row_groups=[t["rg"]], columns=["text", id_col]):
        texts = batch.column("text").to_pylist()
        ids = batch.column(id_col).to_pylist()
        keep = [i for i, x in enumerate(texts) if x]
        n_empty += len(texts) - len(keep)
        encs = _TOK.encode_batch_fast([texts[i] for i in keep], add_special_tokens=False)
        for i, enc in zip(keep, encs):
            if t["split"] == "heldout":
                split = S.fwe_heldout_split(ids[i])
            else:
                split = t["split"]
            b = bufs.setdefault(split, dict(tok=[], idh=[], txh=[], row=[]))
            arr = np.empty(len(enc.ids) + 1, dtype=np.uint16)
            arr[0] = 0
            arr[1:] = enc.ids
            b["tok"].append(arr)
            b["idh"].append(_h16(ids[i]))
            b["txh"].append(_h16(texts[i]))
            b["row"].append(row + i)
        row += len(texts)
    for split, b in bufs.items():
        lens = np.fromiter((len(a) for a in b["tok"]), dtype=np.int64, count=len(b["tok"]))
        tok = np.concatenate(b["tok"]) if b["tok"] else np.zeros(0, np.uint16)
        # id 0 must occur exactly at document starts and ids 1..7 never (plain specials).
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        assert int((tok == 0).sum()) == len(lens) and np.all(tok[starts] == 0), "stray EOT"
        assert not np.any((tok >= 1) & (tok <= 7)), "special token inside document"
        stem = TMP / f"{t['tid']:06d}_{t['corpus']}_{split}"
        tok.tofile(str(stem) + ".bin")
        np.savez(str(stem) + ".npz", lens=lens,
                 id_hash=np.frombuffer(b"".join(b["idh"]), np.uint8).reshape(-1, 16),
                 text_hash=np.frombuffer(b"".join(b["txh"]), np.uint8).reshape(-1, 16),
                 src_row=np.asarray(b["row"], np.int32))
        out[split] = dict(stem=str(stem), n_docs=len(lens), n_tok=int(lens.sum()))
    return dict(tid=t["tid"], corpus=t["corpus"], fidx=t["fidx"], file=t["file"], out=out, n_empty=n_empty)


def length_stats(lens: np.ndarray) -> dict:
    """Document-length distribution in our tokens (lengths include the leading EOT)."""
    tot = int(lens.sum())
    d = dict(n_docs=int(len(lens)), n_tokens=tot)
    if len(lens) == 0:
        return d
    d.update(mean=float(lens.mean()), median=float(np.median(lens)), p99=float(np.percentile(lens, 99)),
             max=int(lens.max()), min=int(lens.min()))
    for thr in (2048, 4096, 8192, 16384, 32768, 65536):
        m = lens >= thr
        d[f"frac_docs_ge_{thr}"] = float(m.mean())
        d[f"frac_tokens_in_docs_ge_{thr}"] = float(lens[m].sum() / tot)
    return d


def _source_sha256s() -> dict[str, str]:
    """file -> SHA-256 from the download report (verified against HF LFS hashes)."""
    rep = json.loads((S.RAW_DIR / "download_report.json").read_text())
    return {r["file"]: r["sha256"] for r in rep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--only", default=None, help="restrict to one corpus (fwe|pg19)")
    ap.add_argument("--max-tasks", type=int, default=None, help="debug: first N tasks per file")
    ap.add_argument("--out", default=str(S.TOKENS_DIR))
    args = ap.parse_args()
    from hybridlc.data.tokenizer import tokenizer_sha256

    out_root = Path(args.out)
    tasks = build_tasks()
    if args.only:
        tasks = [t for t in tasks if t["corpus"] == args.only]
    if args.max_tasks:
        tasks = [t for t in tasks if t["rg"] < args.max_tasks]
    TMP.mkdir(parents=True, exist_ok=True)
    # Dispatch largest tasks first for load balance; assemble in canonical order afterwards.
    order = sorted(tasks, key=lambda t: (t["corpus"] != "pg19", -t["n"]))
    t0 = time.time()
    results: dict[int, dict] = {}
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(run_task, order, chunksize=1)):
            results[r["tid"]] = r
            if k % 100 == 0 or k == len(order) - 1:
                ntok = sum(o["n_tok"] for rr in results.values() for o in rr["out"].values())
                el = time.time() - t0
                print(f"[{el:7.0f}s] {k+1}/{len(order)} tasks, {ntok/1e9:.3f}B tokens, {ntok/el/1e6:.2f}M tok/s",
                      flush=True)
    shas = _source_sha256s()
    tok_sha = tokenizer_sha256()
    # Assemble each (corpus, split) in canonical task order.
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for t in tasks:
        r = results[t["tid"]]
        for split, o in r["out"].items():
            groups.setdefault((t["corpus"], split), []).append((t, o))
    summary = {}
    for (corpus, split), items in sorted(groups.items()):
        d = out_root / corpus
        d.mkdir(parents=True, exist_ok=True)
        files = sorted({t["file"] for t, _ in items}, key=lambda f: [t["fidx"] for t in tasks if t["file"] == f][0])
        fpos = {f: i for i, f in enumerate(files)}
        lens, idh, txh, srow, sfile = [], [], [], [], []
        with open(d / f"{split}.bin", "wb") as fh:
            for t, o in items:
                with open(o["stem"] + ".bin", "rb") as src:
                    shutil.copyfileobj(src, fh, 1 << 24)
                z = np.load(o["stem"] + ".npz")
                lens.append(z["lens"]); idh.append(z["id_hash"]); txh.append(z["text_hash"])
                srow.append(z["src_row"]); sfile.append(np.full(len(z["lens"]), fpos[t["file"]], np.int16))
        lens_a = np.concatenate(lens)
        idx = np.concatenate([[0], np.cumsum(lens_a)]).astype(np.int64)
        assert idx[-1] * 2 == (d / f"{split}.bin").stat().st_size
        np.save(d / f"{split}.idx.npy", idx)
        np.savez(d / f"{split}.docs.npz", id_hash=np.concatenate(idh), text_hash=np.concatenate(txh),
                 src_row=np.concatenate(srow), src_file=np.concatenate(sfile))
        n_empty = sum(results[t["tid"]]["n_empty"] for t in tasks
                      if t["corpus"] == corpus and t["file"] in fpos)
        meta = dict(corpus=corpus, split=split, dtype="uint16", eot_id=0,
                    tokenizer_sha256=tok_sha,
                    source_files=[dict(file=f, sha256=shas.get(f)) for f in files],
                    n_empty_docs_skipped_in_source_files=n_empty,
                    bin_sha256=None,
                    length_stats=length_stats(lens_a))
        (d / f"{split}.meta.json").write_text(json.dumps(meta, indent=1))
        summary[f"{corpus}/{split}"] = meta["length_stats"]
        print(f"{corpus}/{split}: {meta['length_stats']['n_docs']} docs, {idx[-1]/1e6:.1f}M tokens", flush=True)
    shutil.rmtree(TMP)
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
