"""VT-11: data pipeline (protocol §6.1-6.4, §16).

Synthetic-corpus tests run anywhere; tests marked ``realdata`` need ``data/tokens``
(built by scripts/data_pretokenize.py) and are skipped when it is absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hybridlc.data import rng as R
from hybridlc.data import sources as S
from hybridlc.data.loader import IGNORE_INDEX, StageStream, effective_context_histogram, get_store
from hybridlc.data.storeio import synthetic_corpus

REAL = (S.TOKENS_DIR / "fwe" / "train.bin").exists() and (S.TOKENS_DIR / "pg19" / "train.bin").exists()
realdata = pytest.mark.skipif(not REAL, reason="data/tokens not built")
HAVE_TOK = (S.TOKENIZER_DIR / "tokenizer.json").exists()
needs_tokenizer = pytest.mark.skipif(not HAVE_TOK, reason="data/tokenizer not trained")


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    r = tmp_path_factory.mktemp("tokens")
    synthetic_corpus(r)
    return r


def small_A(root, seed=0):
    # Stage-A shape scaled down for speed: 8 seqs/step, 160-seq windows (24 PG19 = 15%).
    return StageStream("A", seed, seq_len=4096, micro_batch=2, accum=4,
                       mix=dict(pg19_per_window=24, window_sequences=160), root=root)


# ------------------------------------------------------------------ statelessness / determinism
def test_pure_function_of_seed_step_micro(root):
    a, b = small_A(root), small_A(root)
    keys = [(s, m) for s in range(25) for m in range(4)]
    ha = {k: a.get_numpy(*k)["batch_hash"] for k in keys}
    order = np.random.default_rng(1).permutation(len(keys))
    for i in order:  # different call order, fresh instance
        assert b.get_numpy(*keys[i])["batch_hash"] == ha[keys[i]]
    assert a.get_numpy(3, 1)["batch_hash"] == ha[(3, 1)]  # repeated call
    assert len(set(ha.values())) == len(ha)
    c = small_A(root, seed=1)
    assert c.get_numpy(0, 0)["batch_hash"] != ha[(0, 0)]


def test_hash_identical_across_processes(root):
    """Same (seed, step, micro) -> same batch hash in a fresh interpreter (resume / every k)."""
    code = ("import sys; from hybridlc.data.loader import StageStream;"
            f"s=StageStream('A',0,seq_len=4096,micro_batch=2,accum=4,"
            f"mix=dict(pg19_per_window=24,window_sequences=160),root={str(root)!r});"
            "print(s.get_numpy(17,3)['batch_hash'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == small_A(root).get_numpy(17, 3)["batch_hash"]


def test_torch_get_contract(root):
    import torch

    b = small_A(root).get(0, 0)
    assert b["input_ids"].shape == (1, 8192) and b["input_ids"].dtype == torch.long
    assert b["labels"].shape == (1, 8192) and b["position_ids"].dtype == torch.long
    assert b["cu_seqlens"].dtype == torch.int32 and int(b["cu_seqlens"][-1]) == 8192
    assert isinstance(b["batch_hash"], str) and len(b["batch_hash"]) == 64


# ------------------------------------------------------------------ packing invariants
def _check_batch(b, L, mb):
    inp, lab, cu, pos = b["input_ids"][0], b["labels"][0], b["cu_seqlens"], b["position_ids"][0]
    N = L * mb
    assert inp.shape == (N,) and lab.shape == (N,) and pos.shape == (N,)  # exact token count
    assert cu[0] == 0 and cu[-1] == N and np.all(np.diff(cu) > 0)
    assert set(range(0, N, L)) <= set(cu.tolist())  # every sequence start is a boundary
    # every EOT is at a boundary; every boundary that is not a sequence start is an EOT
    eot = set(np.flatnonzero(inp == 0).tolist())
    bounds = set(cu[:-1].tolist())
    assert eot <= bounds
    assert {x for x in bounds if x % L} <= eot
    # positions restart at each boundary
    seg = np.diff(cu)
    exp = np.concatenate([np.arange(n) for n in seg])
    assert np.array_equal(pos, exp)
    assert b["max_seqlen"] == seg.max()
    # labels are next-token within each sequence
    for j in range(mb):
        assert np.array_equal(lab[j * L:(j + 1) * L - 1], inp[j * L + 1:(j + 1) * L])


def test_packing_invariants_stage_A_and_B(root):
    a = small_A(root)
    for s in range(6):
        for m in range(4):
            _check_batch(a.get_numpy(s, m), 4096, 2)
    b = StageStream("B", 0, root=root)
    assert (b.seq_len, b.micro_batch, b.accum) == (32768, 1, 4)
    for s in range(5):
        for m in range(4):
            _check_batch(b.get_numpy(s, m), 32768, 1)


def test_fwe_stream_reconstructs_permuted_documents(root):
    """FWE slots, concatenated in run order, are exactly the docs in permuted order: no token
    dropped or inserted; long documents are cut at sequence boundaries; labels continue."""
    a = small_A(root)
    L = a.seq_len
    store = get_store("fwe", "train", str(root))
    seqs, labs = [], []
    g = 0
    while len(seqs) < 300:
        sl = a.slot(g)
        if sl.kind == "fwe":
            assert sl.index == len(seqs)
            toks, st, _ = a.sequence(g)
            seqs.append(toks[:L]); labs.append(toks[1:])
        g += 1
    got = np.concatenate(seqs)
    perm = R.generator(0, R.STREAMS["data_A"], 0, 0).permutation(store.n_docs)
    ref = np.concatenate([store.doc(i) for i in perm[:2000]])[:len(got) + 1]
    assert np.array_equal(got, ref[:-1])
    assert np.array_equal(np.concatenate(labs), ref[1:])


def test_pg19_windows_book_bounded(root):
    a = small_A(root)
    store = get_store("pg19", "train", str(root))
    for b_ in (a, StageStream("B", 0, root=root)):
        L = b_.seq_len
        ws = b_.pg19
        for k in range(min(ws.n, 200)):
            wid = ws.window_id(k)
            s = int(ws.starts[wid]); book = int(ws.book[wid])
            assert store.idx[book] <= s and s + L <= store.idx[book + 1]
            assert (s - store.idx[book]) % L == 0
            t = ws.take(k)
            assert not np.any(t[1:L] == 0)  # no separator inside a window
        assert ws.n == int((store.lengths // L).sum())
        # the 2x32K book contributes exactly 2 windows at 32K, the short book none
        if L == 32768:
            assert int((ws.book == 0).sum()) == 2 and int((ws.book == 1).sum()) == 0


def test_mixing_exact_per_window(root):
    a = small_A(root)
    W, P = 160, 24
    for w in range(5):
        kinds = []
        for step in range(w * a.window_steps, (w + 1) * a.window_steps):
            for m in range(a.accum):
                kinds += a.get_numpy(step, m)["sources"]
        assert len(kinds) == W and kinds.count("pg19") == P
    # protocol default shapes
    A = StageStream("A", 0, root=root)
    assert (A.seq_len, A.micro_batch, A.accum, A.window_steps) == (4096, 2, 16, 20)
    for w in range(3):
        is_pg, _ = A._window_layout(w)
        assert is_pg.sum() == 96 and len(is_pg) == 640
    B = StageStream("B", 0, root=root)
    assert B.window_steps == 5 and B._window_layout(0)[0].sum() == 12
    # PG19 and FWE slot indices are consecutive (no reuse, no gaps)
    sl = [A.slot(g) for g in range(640 * 3)]
    for kind in ("pg19", "fwe"):
        idx = [s.index for s in sl if s.kind == kind]
        assert idx == list(range(len(idx)))
    with pytest.raises(ValueError):
        StageStream("A", 0, micro_batch=2, accum=16, mix=dict(pg19_per_window=96, window_sequences=600), root=root)


def test_lm2_fixed_and_seed_independent(root):
    x, y = StageStream("LM2", 0, root=root), StageStream("LM2", 5, root=root)
    assert x.num_steps() == 1000
    assert x.get_numpy(7, 0)["batch_hash"] == y.get_numpy(7, 0)["batch_hash"]
    _check_batch(x.get_numpy(999, 0), 4096, 2)
    with pytest.raises(IndexError):
        x.get_numpy(1000, 0)


def test_epoch_wrap(root):
    a = small_A(root)
    ep = a.fwe.epoch_tokens
    toks, st = a.fwe.take(ep - 100, 300)
    assert 100 in st.tolist()  # first doc of epoch 1 starts exactly at the wrap
    assert toks[100] == 0


def test_effective_context_histogram():
    h, e = effective_context_histogram(np.array([0, 3, 10], dtype=np.int32))
    assert h.sum() == 10 and h[0] == 2  # context 1 occurs at two segment starts


def test_seed_contract_matches_utils():
    ss = R.seed_sequence(3, 2, 1, 0).generate_state(4)
    assert np.array_equal(ss, R.fallback_seed_sequence(3, 2, 1, 0).generate_state(4))


def test_ignore_index_is_pad():
    assert IGNORE_INDEX == 1


# ------------------------------------------------------------------ real data
@realdata
def test_heldout_hash_intersections_empty():
    """Training docs never appear in held-out splits (doc-id hash and text hash), and test/dev are disjoint."""
    def hashes(c, s, key):
        z = np.load(S.TOKENS_DIR / c / f"{s}.docs.npz")[key]
        return set(map(bytes, z))

    for key in ("id_hash", "text_hash"):
        fwe_tr, fwe_te, fwe_dv = (hashes("fwe", s, key) for s in ("train", "test", "dev"))
        pg_tr, pg_te, pg_va = (hashes("pg19", s, key) for s in ("train", "test", "validation"))
        train = fwe_tr | pg_tr
        for name, held in [("fwe_test", fwe_te), ("fwe_dev", fwe_dv), ("pg19_test", pg_te), ("pg19_val", pg_va)]:
            assert not (train & held), (key, name, len(train & held))
        assert not (fwe_te & fwe_dv) and not (pg_te & pg_va)


@realdata
def test_real_heldout_parity_split():
    m = json.loads(S.MANIFEST_PATH.read_text())
    assert m["split_assignment"]["fwe"]["test"]["rule"].startswith("sha256(doc_id) even")


@realdata
def test_real_streams_and_throughput():
    for st in ("A", "B"):
        s = StageStream(st, 0)
        _check_batch(s.get_numpy(0, 0), s.seq_len, s.micro_batch)
        t0 = time.perf_counter(); n = 0
        for step in range(3):
            for m in range(s.accum):
                n += s.get_numpy(step, m)["input_ids"].size
        rate = n / (time.perf_counter() - t0)
        assert rate > 1.5e6, f"stage {st}: {rate:.0f} tok/s"  # 10x the ~150K tok/s training rate


@needs_tokenizer
def test_tokenizer_contract():
    from hybridlc.data.tokenizer import load_tokenizer

    tok = load_tokenizer()
    assert len(tok) == 16384
    assert [tok.convert_ids_to_tokens(i) for i in range(8)] == \
        ["<|endoftext|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(6)]
    assert tok.convert_ids_to_tokens(tok.encode(" 4821", add_special_tokens=False)) == ["Ġ", "4", "8", "2", "1"]
    assert tok.decode(tok.encode("key 0042 is x", add_special_tokens=False)) == "key 0042 is x"


@realdata
def test_real_manifest_and_store_hashes():
    """MANIFEST.json lists every store with the SHA-256 recorded at build time; small stores re-hash equal."""
    import hashlib

    m = json.loads(S.MANIFEST_PATH.read_text())
    assert m["pins"][S.FWE_REPO]["revision"] == S.FWE_REVISION
    assert m["pins"][S.PG19_REPO]["revision"] == S.PG19_REVISION
    assert all(r["verified"] for r in m["raw_files"]) and len(m["raw_files"]) == 7 + 25
    for sp in S.SPLITS:
        e = m["tokens"][f"{sp.corpus}/{sp.split}"]
        meta = json.loads((S.TOKENS_DIR / sp.corpus / f"{sp.split}.meta.json").read_text())
        assert e["bin_sha256"] == meta["bin_sha256"] and e["n_tokens"] == meta["length_stats"]["n_tokens"]
        store = get_store(sp.corpus, sp.split)
        assert store.n_tokens == e["n_tokens"] and store.n_docs == e["n_docs"]
    for split in ("validation", "test"):  # cheap enough to re-hash
        p = S.TOKENS_DIR / "pg19" / f"{split}.bin"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == m["tokens"][f"pg19/{split}"]["bin_sha256"]
    assert m["tokens"]["pg19/validation"]["n_docs"] == 50 and m["tokens"]["pg19/test"]["n_docs"] == 100


@realdata
def test_real_stores_eot_only_at_doc_starts():
    """Spot check (pretokenisation asserts this for every document): id 0 exactly at document starts,
    ids 1..7 never."""
    g = np.random.default_rng(0)
    for sp in S.SPLITS:
        st = get_store(sp.corpus, sp.split)
        for i in g.choice(st.n_docs, size=min(200, st.n_docs), replace=False):
            d = st.doc(int(i))
            assert d[0] == 0 and not np.any(d[1:] <= 7)


@realdata
def test_real_stage_A_full_window_exact():
    """Protocol shapes on the real corpus: one full 20-step Stage-A window has exactly 96/640 PG19
    sequences and 32 x 4,096 tokens per step; one 5-step Stage-B window has 12/20."""
    for stage, steps, n_pg, n_seq in (("A", 20, 96, 640), ("B", 5, 12, 20)):
        s = StageStream(stage, 0)
        kinds = []
        for step in range(steps):
            ntok = 0
            for m in range(s.accum):
                b = s.get_numpy(step, m)
                _check_batch(b, s.seq_len, s.micro_batch)
                ntok += b["input_ids"].size
                kinds += b["sources"]
            assert ntok == 131072
        assert len(kinds) == n_seq and kinds.count("pg19") == n_pg


@realdata
def test_real_lr_and_lm2_streams():
    lr = StageStream("LR", 100)
    a = StageStream("A", 100)
    assert lr.get_numpy(3, 5)["batch_hash"] == a.get_numpy(3, 5)["batch_hash"]  # LR sweep = Stage-A recipe
    assert lr.get_numpy(3, 5)["batch_hash"] != StageStream("A", 0).get_numpy(3, 5)["batch_hash"]
    lm2 = StageStream("LM2", 0)
    assert lm2.num_steps() * lm2.seqs_per_step == 2000
    _check_batch(lm2.get_numpy(0, 0), 4096, 2)
    _check_batch(lm2.get_numpy(999, 0), 4096, 2)


@realdata
def test_real_parity_assignment_against_raw_ids():
    """Kept shard-000 documents sit in the split given by SHA-256(doc_id) parity (checked against the raw
    parquet ids via src_row), and the stored id_hash matches."""
    import hashlib

    import pyarrow.parquet as pq

    ids = pq.read_table(S.raw_path(S.FWE_REPO, S.FWE_HELDOUT_FILE), columns=["id"]).column("id").to_pylist()
    g = np.random.default_rng(1)
    for split in ("test", "dev"):
        z = np.load(S.TOKENS_DIR / "fwe" / f"{split}.docs.npz")
        for i in g.choice(len(z["src_row"]), size=500, replace=False):
            doc_id = ids[int(z["src_row"][i])]
            assert S.fwe_heldout_split(doc_id) == split
            assert hashlib.sha256(doc_id.encode()).digest()[:16] == bytes(z["id_hash"][i])
        meta = json.loads((S.TOKENS_DIR / "fwe" / f"{split}.meta.json").read_text())
        assert meta["heldout_filter"]["n_kept"] == len(z["src_row"])


@realdata
def test_real_store_tokens_match_raw_text():
    """Stored documents equal [EOT] + encode(raw text) for random FWE-test and PG19-validation rows,
    and decode back to the raw text."""
    import pyarrow.parquet as pq

    from hybridlc.data.tokenizer import load_raw_tokenizer

    tok = load_raw_tokenizer(plain_specials=True)
    g = np.random.default_rng(2)
    pg_val = next(S.raw_path(S.PG19_REPO, "data").glob("validation-*.parquet"))
    for corpus, split, path, n in (("fwe", "test", S.raw_path(S.FWE_REPO, S.FWE_HELDOUT_FILE), 30),
                                   ("pg19", "validation", pg_val, 3)):
        store = get_store(corpus, split)
        z = np.load(S.TOKENS_DIR / corpus / f"{split}.docs.npz")
        texts = pq.read_table(path, columns=["text"]).column("text")
        for i in g.choice(store.n_docs, size=n, replace=False):
            text = texts[int(z["src_row"][i])].as_py()
            ids = tok.encode(text, add_special_tokens=False).ids
            d = store.doc(int(i))
            assert d[0] == 0 and np.array_equal(d[1:], np.asarray(ids, np.uint16))
            assert tok.decode(d[1:].tolist(), skip_special_tokens=False) == text
