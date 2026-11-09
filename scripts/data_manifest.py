#!/usr/bin/env python
"""Write ``data/MANIFEST.json`` (protocol §6.1) and ``data/stats.json``.

MANIFEST.json records:
  * the HF commit pins and, for every raw parquet file, its size, the LFS SHA-256
    reported by the HF API at the pinned revision and the locally computed SHA-256
    (from ``data/raw/download_report.json``; ``--rehash`` recomputes them);
  * the split assignment rules and which raw files feed which tokenised split;
  * the tokenizer SHA-256 and training-sample description;
  * every tokenised store (``data/tokens/<corpus>/<split>.{bin,idx.npy}``) with its
    SHA-256, document and token counts.

stats.json records the token counts per corpus/split, the document-length
distribution (fraction of tokens in documents >= 2K/4K/8K/16K/32K/64K tokens), the
effective-context histogram of the first Stage-A and Stage-B windows (§6.3
logging) and the epoch budget implied by the protocol token budgets.

Usage: python scripts/data_manifest.py [--rehash]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hybridlc.data import sources as S


def sha256_path(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def split_rules() -> dict:
    """Human- and machine-readable split assignment (protocol §6.1)."""
    pg = sorted(p.name for p in S.raw_path(S.PG19_REPO, "data").glob("*.parquet"))
    return {
        "fwe": {
            "train": dict(rule="all documents of shards 001-006", files=S.FWE_TRAIN_FILES,
                          use="training (Stage A/B FineWeb-Edu share, LR sweep)"),
            "test": dict(rule="sha256(doc_id) even (parity of the full digest = last hex nibble), shard 000",
                         files=[S.FWE_HELDOUT_FILE], use="test filler, LM-1, LM-2"),
            "dev": dict(rule="sha256(doc_id) odd, shard 000", files=[S.FWE_HELDOUT_FILE], use="dev filler"),
            "doc_id_field": "id",
        },
        "pg19": {
            "train": dict(rule="all train-000NN-of-00023 files", files=[f"data/{f}" for f in pg if f.startswith("train-")],
                          use="training (Stage A 4K windows, Stage B 32K book-bounded windows)"),
            "validation": dict(rule="validation file (50 books)", files=[f"data/{f}" for f in pg if f.startswith("validation-")],
                               use="dev filler"),
            "test": dict(rule="test file (100 books)", files=[f"data/{f}" for f in pg if f.startswith("test-")],
                         use="test filler and LM-1"),
            "doc_id_field": "url",
        },
        "notes": [
            "Training PG19 books never appear in evaluation filler (VT-11 asserts empty doc-id and text hash intersections).",
            "Held-out exact-duplicate filter (scripts/data_pretokenize.py::filter_heldout; flagged deviation/clarification): "
            "after the parity/file assignment, a held-out document is dropped if the SHA-256 of its text occurs in a "
            "training split, in a higher-priority held-out split (fwe/test, pg19/test, fwe/dev, pg19/validation), or "
            "earlier in its own split. FineWeb-Edu dedups only within a crawl dump, so ~3.5% of shard-000 texts recur "
            "in shards 001-006. Training stores are unmodified. Counts: tokens/<corpus>/<split>.meta.json['heldout_filter'].",
            "Every tokenised document is [<|endoftext|>=0] + encode(text); empty texts are skipped.",
            "Tokenizer training text came only from FWE shard 001 and PG19 train (never held-out data).",
        ],
    }


def stores_info() -> dict:
    out = {}
    for sp in S.SPLITS:
        base = S.TOKENS_DIR / sp.corpus / sp.split
        meta_p = Path(str(base) + ".meta.json")
        if not meta_p.exists():
            out[f"{sp.corpus}/{sp.split}"] = dict(missing=True, role=sp.role)
            continue
        meta = json.loads(meta_p.read_text())
        docs_p = Path(str(base) + ".docs.npz")
        out[f"{sp.corpus}/{sp.split}"] = dict(
            role=sp.role,
            bin=f"tokens/{sp.corpus}/{sp.split}.bin", bin_sha256=meta.get("bin_sha256"),
            idx=f"tokens/{sp.corpus}/{sp.split}.idx.npy", idx_sha256=meta.get("idx_sha256"),
            docs_npz_sha256=sha256_path(docs_p) if docs_p.exists() else None,
            n_docs=meta["length_stats"]["n_docs"], n_tokens=meta["length_stats"]["n_tokens"],
            tokenizer_sha256=meta.get("tokenizer_sha256"),
            source_files=meta.get("source_files"),
            n_empty_docs_skipped_in_source_files=meta.get("n_empty_docs_skipped_in_source_files"),
            heldout_filter=meta.get("heldout_filter"),
        )
    return out


def stage_context_stats() -> dict:
    """Effective-context histogram (tokens of same-document context available at each
    position) over the first mixing window of Stage A and Stage B at data_seed 0."""
    from hybridlc.data.loader import StageStream, effective_context_histogram

    res = {}
    for stage in ("A", "B"):
        st = StageStream(stage, 0)
        hist = None
        for step in range(st.window_steps):
            for m in range(st.accum):
                h, edges = effective_context_histogram(st.get_numpy(step, m)["cu_seqlens"])
                hist = h if hist is None else hist + h
        tot = int(hist.sum())
        res[stage] = dict(window_steps=st.window_steps, tokens=tot,
                          bin_lower_edges=[int(e) for e in edges[:-1]],
                          counts=[int(x) for x in hist],
                          frac_tokens_with_context_ge={str(t): float(hist[edges[:-1] >= t].sum() / tot)
                                                       for t in (1024, 2048, 4096, 8192, 16384)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehash", action="store_true", help="recompute raw-file SHA-256s instead of trusting the report")
    ap.add_argument("--no-context", action="store_true", help="skip the effective-context histograms")
    args = ap.parse_args()

    report = json.loads((S.RAW_DIR / "download_report.json").read_text())
    raw = []
    for r in report:
        p = S.raw_path(r["repo"], r["file"])
        sha = sha256_path(p) if args.rehash else r["sha256"]
        raw.append(dict(repo=r["repo"], revision=r["revision"], file=r["file"], local=str(p.relative_to(S.DATA_DIR)),
                        size=r["size"], lfs_sha256=r["lfs_sha256"], sha256=sha,
                        verified=bool(sha == r["lfs_sha256"] and p.stat().st_size == r["size"])))
    assert all(x["verified"] for x in raw), "raw file verification failed"

    tok_report_p = S.TOKENIZER_DIR / "training_report.json"
    tok_report = json.loads(tok_report_p.read_text()) if tok_report_p.exists() else None
    from hybridlc.data.tokenizer import tokenizer_sha256

    stores = stores_info()
    manifest = dict(
        protocol="reports/experimental_protocol.md v1.0 §6.1-6.2",
        pins={S.FWE_REPO: dict(revision=S.FWE_REVISION, config="sample-10BT"),
              S.PG19_REPO: dict(revision=S.PG19_REVISION)},
        raw_files=raw,
        split_assignment=split_rules(),
        tokenizer=dict(dir="tokenizer/", tokenizer_json_sha256=tokenizer_sha256(),
                       training=None if tok_report is None else {k: tok_report[k] for k in
                                                                 ("total_bytes", "fwe_fraction", "sampling_seed", "fwe", "pg19")}),
        tokens=stores,
    )
    S.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {S.MANIFEST_PATH}")

    # ------------------------------------------------------------------ stats
    stats: dict = dict(per_split={})
    for sp in S.SPLITS:
        mp = S.TOKENS_DIR / sp.corpus / f"{sp.split}.meta.json"
        if mp.exists():
            stats["per_split"][f"{sp.corpus}/{sp.split}"] = json.loads(mp.read_text())["length_stats"]
    if tok_report:
        stats["tokenizer_compression"] = tok_report.get("compression")
    # Budget check: protocol §6.3 token budgets vs available training tokens.
    if "fwe/train" in stats["per_split"] and "pg19/train" in stats["per_split"]:
        fwe_tr = stats["per_split"]["fwe/train"]["n_tokens"]
        pg_tr = stats["per_split"]["pg19/train"]["n_tokens"]
        budget = {}
        for scale, a_steps in (("D384", 4578), ("D512", 6104)):
            a_tok = a_steps * 131072
            b_tok = 200_000_000  # per-run cost assumption of §6.3 (FEAS-3 may change it)
            budget[scale] = dict(stageA_tokens=a_tok, stageA_fwe_epochs=0.85 * a_tok / fwe_tr,
                                 stageA_pg19_epochs=0.15 * a_tok / pg_tr,
                                 stageB_tokens_assumed=b_tok, stageB_fwe_epochs=0.4 * b_tok / fwe_tr,
                                 stageB_pg19_epochs=0.6 * b_tok / pg_tr)
        stats["epochs_implied_by_budget"] = budget
    if not args.no_context:
        stats["effective_context_first_window_seed0"] = stage_context_stats()
    (S.DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
