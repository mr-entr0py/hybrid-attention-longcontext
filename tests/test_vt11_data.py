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


@realdata
def test_tokenizer_contract():
    from hybridlc.data.tokenizer import load_tokenizer

    tok = load_tokenizer()
    assert len(tok) == 16384
    assert [tok.convert_ids_to_tokens(i) for i in range(8)] == \
        ["<|endoftext|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(6)]
    assert tok.convert_ids_to_tokens(tok.encode(" 4821", add_special_tokens=False)) == ["Ġ", "4", "8", "2", "1"]
    assert tok.decode(tok.encode("key 0042 is x", add_special_tokens=False)) == "key 0042 is x"
