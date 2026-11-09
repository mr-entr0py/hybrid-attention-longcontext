"""VT-12: evaluation items (protocol §9, §16).

Pass thresholds (all must hold): len == T exactly; each answer is 4 single-digit tokens (or
1 token for A-freq/A-count); train/eval (key, value, template) hash overlap = 0; no
separator inside contexts; recorded positions match; item files byte-identical across models.

Runs against ``data/tokenizer`` when it exists, else a dev BPE with the protocol's
pre-tokenizer (``eval.devtools``); filler from ``data/tokens`` when built, else the C4
dev-text stand-in (the generator code path is identical: only the DocPool differs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hybridlc.eval import templates as TP
from hybridlc.eval.constants import MAX_FACT_FRACTION
from hybridlc.eval.devtools import InductionLM, resolve_test_tokenizer
from hybridlc.eval.filler import StoreFiller, dev_text_filler
from hybridlc.eval.generators import ROLE_LINK, generate
from hybridlc.eval.items import load_items, save_items, sha256_file, verify_manifest, write_manifest
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.scorer import score_file
from hybridlc.eval.suites import SUITES, suite_cells
from hybridlc.eval.tokens import token_space
from hybridlc.eval.words import word_list

K = 1024
TOKENS_REAL = (Path(__file__).resolve().parents[1] / "data" / "tokens" / "pg19" / "test.bin").exists()


@pytest.fixture(scope="module")
def tok():
    t, kind = resolve_test_tokenizer()
    print(f"tokenizer: {kind}")
    return t


@pytest.fixture(scope="module")
def ts(tok):
    return token_space(tok)


@pytest.fixture(scope="module")
def fillers(ts):
    def mk(ns):
        if TOKENS_REAL:
            return StoreFiller(ts, ns)
        return dev_text_filler(ts, ns)
    return {ns: mk(ns) for ns in ("test", "dev", "train")}


# Representative cells: every generator, every variant, the extreme lengths/levels.
CELLS = [
    ("R1", {"D": 3}, 1 * K), ("R1", {"D": 15}, 4 * K), ("R1", {"D": 15}, 32 * K),
    ("R-load", {"N": 8}, 1 * K), ("R-load", {"N": 512}, 8 * K), ("R-load", {"N": 1024}, 16 * K),
    ("R-load", {"N": 2048}, 32 * K),
    ("M-hop", {"h": 1}, 1 * K), ("M-hop", {"h": 4}, 8 * K), ("M-hop", {"h": 5}, 32 * K),
    ("M-hop", {"h": 2, "order": "forward"}, 16 * K), ("M-hop", {"h": 2, "order": "reversed"}, 16 * K),
    ("A-freq", {"rho": 4.0}, 1 * K), ("A-freq", {"rho": 1.25}, 8 * K), ("A-freq", {"rho": 1.1}, 32 * K),
    ("A-count", {}, 8 * K), ("R3", {"U": 4}, 4 * K), ("R3", {"U": 2}, 32 * K),
    ("R-dist", {"filler": "natural"}, 32 * K), ("R-dist", {"filler": "repetitive"}, 32 * K),
    ("R0", {}, 4 * K),
    ("R1", {"D": 3, "placement": "oracle"}, 4 * K), ("R-load", {"N": 32, "placement": "oracle"}, 32 * K),
    ("M-hop", {"h": 2, "placement": "oracle"}, 4 * K), ("A-freq", {"rho": 2.0, "placement": "oracle"}, 4 * K),
    ("R1", {"D": 3, "template": "T4"}, 8 * K), ("M-hop", {"h": 2, "template": "T4"}, 8 * K),
    ("A-freq", {"rho": 2.0, "template": "T4"}, 8 * K),
    ("R1", {"D": 15, "placement": "nofill"}, None), ("R-load", {"N": 1024, "placement": "nofill"}, None),
    ("M-hop", {"h": 4, "placement": "nofill"}, None), ("A-freq", {"rho": 1.25, "placement": "nofill"}, None),
]


@pytest.fixture(scope="module")
def generated(tok, fillers):
    out = {}
    for ns in ("test", "dev"):
        for task, lvl, T in CELLS:
            n = 3 if (T or 0) >= 16 * K else 6
            key = (ns, task, json.dumps(lvl, sort_keys=True), T)
            out[key] = generate(task, lvl, T, n, eval_rng(ns, task, json.dumps(lvl, sort_keys=True), str(T)), ns, tok,
                                fillers[ns])
    return out


def _all_items(generated):
    for key, items in generated.items():
        for it in items:
            yield key, it


# ----------------------------------------------------------------------------- exact length
def test_len_exactly_T(generated):
    n = 0
    for (ns, task, lvl, T), it in _all_items(generated):
        ctx = it["context_ids"]
        assert ctx.dtype == np.int32
        if T is None:  # C0: [EOT] + needles only
            continue
        for q in it["queries"]:
            assert len(ctx) + len(q["prompt_ids"]) + len(q["answer_ids"]) == T, (task, lvl)
            n += 1
    assert n > 300


# ----------------------------------------------------------------------------- answers
def test_answer_encoding(generated, ts):
    for (ns, task, lvl, T), it in _all_items(generated):
        for q in it["queries"]:
            a = np.asarray(q["answer_ids"])
            if task == "A-freq":
                assert len(a) == 1
                assert [int(a[0])] == q["candidates"][0]
            elif task == "A-count":
                assert len(a) == 1 and int(a[0]) in ts.digit_set
                assert int(q["prompt_ids"][-1]) == ts.space_id  # whitespace belongs to the prompt
            else:
                assert len(a) == 4 and all(int(x) in ts.digit_set for x in a), (task, a)
                assert int(q["prompt_ids"][-1]) == ts.space_id
                assert ts.decode(a) == q["meta"]["value"]
            # candidates: gold first, all distinct, all of the answer's length
            cs = [tuple(c) for c in q["candidates"]]
            assert cs[0] == tuple(int(x) for x in a)
            assert len(set(cs)) == len(cs)
            assert all(len(c) == len(a) for c in cs)
            assert len(cs) <= 1 + 32 + 4


def test_value_tokenisation_natural(ts):
    """The protocol tokenizer must encode ' 1234' as [' ', 1, 2, 3, 4] (§6.2): items are built in token
    space either way, so this only checks that token-space items equal what the tokenizer would produce."""
    if not ts.natural_value_ok():
        pytest.skip("tokenizer without digit isolation (dev GPT-2 only)")
    tpl = TP.CODE_TEMPLATES["T1"]
    wl = word_list(ts, "B")
    k = wl.key_ids(12345 % wl.n_keys)
    fact = TP.fact_ids(ts, tpl, k, 42)
    assert ts.enc(ts.decode(fact)).tolist() == fact.tolist()


# ----------------------------------------------------------------------------- separators
def test_no_separator_inside_contexts(generated, ts):
    for (ns, task, lvl, T), it in _all_items(generated):
        ctx = it["context_ids"]
        assert ctx[0] == ts.eot_id
        assert not (ctx[1:] == ts.eot_id).any(), task
        for q in it["queries"]:
            assert not (np.asarray(q["prompt_ids"]) == ts.eot_id).any()


# ----------------------------------------------------------------------------- positions
def test_recorded_positions_match(generated, ts):
    for (ns, task, lvl, T), it in _all_items(generated):
        ctx, arr, meta = it["context_ids"], it["arrays"], it["meta"]
        tname = meta["template"]
        if task in ("A-freq", "A-count"):
            for q in it["queries"]:
                pos = q["meta"]["needle_start"]
                assert len(pos) > 0
                if task == "A-count":
                    assert len(pos) == int(q["meta"]["value"])
            continue
        tpl = TP.PASSKEY if tname == "PK" else TP.CODE_TEMPLATES[tname]
        for j in range(len(arr["fact_value"])):
            s, e = int(arr["fact_start"][j]), int(arr["fact_end"][j])
            seg = ctx[s:e]
            if task == "R0":
                exp = TP.fact_ids(ts, tpl, np.zeros(0, np.int32), int(arr["fact_value"][j]))
            elif arr["fact_role"][j] == ROLE_LINK:
                # link k_i -> k_{i-1}: must equal the rendering with exactly one other key of the item
                hits = [m for m in range(len(arr["fact_key"]))
                        if seg.tolist() == TP.link_ids(ts, tpl, arr["fact_key"][j], arr["fact_key"][m]).tolist()]
                assert len(hits) == 1, (task, j)
                continue
            else:
                exp = TP.fact_ids(ts, tpl, arr["fact_key"][j], int(arr["fact_value"][j]))
            assert seg.tolist() == exp.tolist(), (task, j)
        for q in it["queries"]:
            m = q["meta"]
            s, e = m["needle_start"][0], m["needle_end"][0]
            assert m["distance"] == len(ctx) - e
            assert ts.decode(ctx[s:e]).find(m["value"]) >= 0
            if T is not None and m["depth"] is not None:
                assert 0.0 <= m["depth"] <= 1.0
        if meta["placement"] == "oracle":  # C2: every needle right before the query
            assert int(arr["fact_end"].max()) == len(ctx)
            assert int(arr["fact_start"].min()) == len(ctx) - sum(
                int(e - s) for s, e in zip(arr["fact_start"], arr["fact_end"]))


def test_depth_deciles_and_snapping(tok, fillers, ts):
    items = generate("R1", {"D": 3}, 4 * K, 50, eval_rng("dev", "decile-test"), "dev", tok, fillers["dev"])
    d = np.asarray([q["meta"]["depth"] for it in items for q in it["queries"]])
    counts = np.bincount(np.minimum((d * 10).astype(int), 9), minlength=10)
    # 200 queried needles -> 20 per decile before snapping; snapping (<= 32 tokens) can move a few across edges
    assert counts.min() >= 14 and counts.max() <= 26, counts
    snapped = np.mean([s for it in items for q in it["queries"] for s in q["meta"]["snapped"]])
    assert snapped > 0.9
    # inserted needles follow a sentence end
    for it in items:
        for s in it["arrays"]["fact_start"]:
            prev = int(it["context_ids"][s - 1])
            assert prev == ts.eot_id or ts.ends_sentence[prev] or not all(
                q["meta"]["snapped"][0] for q in it["queries"])


def test_mhop_order(generated):
    for (ns, task, lvl, T), it in _all_items(generated):
        if task != "M-hop":
            continue
        order = json.loads(lvl).get("order", "shuffled")
        for q in it["queries"]:
            st = q["meta"]["needle_start"]  # value fact, link 1..h
            if order == "forward":
                assert st == sorted(st)
            elif order == "reversed":
                assert st == sorted(st, reverse=True)


def test_rload_fact_budget(generated):
    for (ns, task, lvl, T), it in _all_items(generated):
        if task == "R-load" and T is not None:
            arr = it["arrays"]
            tot = int((arr["fact_end"] - arr["fact_start"]).sum())
            assert tot <= MAX_FACT_FRACTION * T
            assert len(set(map(tuple, arr["fact_key"].tolist()))) == len(arr["fact_key"])  # distinct keys


def test_afreq_counts(generated, ts):
    comma = int(ts.enc(",")[0])
    for (ns, task, lvl, T), it in _all_items(generated):
        if task != "A-freq":
            continue
        ctx = it["context_ids"]
        rho = json.loads(lvl)["rho"]
        x, y = it["arrays"]["probe_x"], it["arrays"]["probe_y"]
        for a, b in zip(x, y):
            assert (ctx == a).sum() == round(8 * rho) and (ctx == b).sum() == 8
        # split the list into words; every non-probe (background) word occurs <= 3 times
        words, cur = [], []
        for t in ctx[1:].tolist():
            if t == comma:
                words.append(tuple(cur))
                cur = []
            else:
                cur.append(t)
        words.append(tuple(cur))
        probes = {(int(v),) for v in np.r_[x, y]}
        bg = [w for w in words if w not in probes]
        if it["meta"]["placement"] == "nofill":
            assert bg == []
        else:
            vals, cnt = np.unique(np.asarray([hash(w) for w in bg]), return_counts=True)
            assert cnt.max() <= 3 and len(bg) > 0
        assert len(words) - len(bg) == 4 * (round(8 * rho) + 8)


def test_prefix_stability_and_determinism(tok, fillers):
    a = generate("R-load", {"N": 32}, 4 * K, 12, eval_rng("dev", "R-load", "N32", "4K"), "dev", tok, fillers["dev"])
    b = generate("R-load", {"N": 32}, 4 * K, 5, eval_rng("dev", "R-load", "N32", "4K"), "dev", tok, fillers["dev"])
    for x, y in zip(a, b):
        assert x["context_ids"].tolist() == y["context_ids"].tolist()
        assert [q["answer_ids"].tolist() for q in x["queries"]] == [q["answer_ids"].tolist() for q in y["queries"]]


# ----------------------------------------------------------------------------- leakage
def _tuples(items, ts):
    out = set()
    for it in items:
        arr = it.get("arrays", {})
        if "fact_key" not in arr:
            continue
        for k, v in zip(arr["fact_key"], arr["fact_value"]):
            if v < 0 or k[0] < 0:
                continue
            key = ts.decode(k).strip()
            out.add(hashlib.sha256(f"{key}|{int(v):04d}|{it['meta']['template']}".encode()).hexdigest())
    return out


def test_train_eval_tuple_overlap_zero(generated, tok, ts, fillers):
    ev = set()
    for _, items in generated.items():
        ev |= _tuples(items, ts)
    tr = set()
    for task, lvl in (("R1", {"D": 20}), ("R-load", {"N": 256}), ("R3", {"U": 5}), ("M-hop", {"h": 6})):
        for tname in ("T1", "T2", "T3"):
            tr |= _tuples(generate(task, {**lvl, "template": tname}, 8 * K, 4, eval_rng("dev", "train-sim", task, tname),
                                   "train", tok, fillers["train"]), ts)
    assert len(ev) > 1000 and len(tr) > 1000
    assert ev.isdisjoint(tr)
    # structural guarantee: the key vocabularies of lists A and B are disjoint
    A, B = word_list(ts, "A"), word_list(ts, "B")
    assert not set(A.adjectives) & set(B.adjectives) and not set(A.nouns) & set(B.nouns)
    assert not set(A.probe_words) & set(B.probe_words) and not set(A.background_words) & set(B.background_words)


def test_namespace_template_rules(tok, fillers):
    with pytest.raises(ValueError):
        generate("R1", {"D": 3, "template": "T2"}, 1 * K, 1, eval_rng("test", "x"), "test", tok, fillers["test"])
    with pytest.raises(ValueError):
        generate("R1", {"D": 3, "template": "T4"}, 1 * K, 1, eval_rng("test", "x"), "train", tok, fillers["train"])


# ----------------------------------------------------------------------------- suites
def test_suite_definitions():
    cs = suite_cells("CS")
    get = lambda task, lvl, T, v=None: [c for c in cs if c.task == task and c.level_dict == lvl and c.T == T  # noqa: E731
                                        and c.variant == v]
    assert get("R1", {"D": 15}, 32 * K)[0].n == 300 and get("R1", {"D": 3}, 8 * K)[0].n == 200
    assert get("R-load", {"N": 2048}, 32 * K)[0].n == 150
    assert {c.level_dict["N"] for c in cs if c.task == "R-load" and c.T == 8 * K} == {8, 32, 128, 512}
    assert {c.level_dict["N"] for c in cs if c.task == "R-load" and c.T == 4 * K} == {8, 32}
    assert {c.level_dict["h"] for c in cs if c.task == "M-hop" and c.T == 16 * K} == {1, 2, 3, 4}
    assert {c.level_dict["rho"] for c in cs if c.task == "A-freq" and c.T == 2 * K} == {4.0, 2.0}
    c0 = [c for c in cs if c.variant == "C0"]
    assert len(c0) == 2 + 5 + 4 + 4 and all(c.n == 200 and c.T is None for c in c0)
    assert suite_cells("CS", u15=True)[10].n in (200, 600)
    ms = suite_cells("MS")
    assert sum(c.task == "R-dist" for c in ms) == 2 and sum(c.variant == "C2" for c in ms) == 8
    assert sum(c.variant == "T4" for c in ms) == 6 and all(c.n == 100 for c in ms if c.variant == "T4")
    assert all(c.namespace == "dev" for nm in ("DG", "DC", "FD") for c in suite_cells(nm))
    assert len(suite_cells("DC")) == 3 * 5 and all(c.n == 100 for c in suite_cells("DC"))
    t0 = suite_cells("Tier0")
    assert sum(c.task == "LM-4" for c in t0) == 10 and sum(c.task == "R0" for c in t0) == 2
    for nm in SUITES:
        suite_cells(nm)  # no duplicate relpaths


# ----------------------------------------------------------------------------- files
def test_item_files_byte_identical(tmp_path, tok, fillers, ts):
    rng_key = ("R-load", "N32", "4K")
    a = generate("R-load", {"N": 32}, 4 * K, 8, eval_rng("dev", *rng_key), "dev", tok, fillers["dev"])
    b = generate("R-load", {"N": 32}, 4 * K, 8, eval_rng("dev", *rng_key), "dev", tok, fillers["dev"])
    meta = dict(task="R-load", level={"N": 32}, T=4 * K, tokenizer_sha256=ts.fingerprint)
    s1 = save_items(tmp_path / "x" / "a.npz", a, meta)
    s2 = save_items(tmp_path / "y" / "a.npz", b, meta)
    assert s1 == s2  # regeneration is byte-identical
    with pytest.raises(FileExistsError):
        save_items(tmp_path / "x" / "a.npz", a, meta)
    items, cell = load_items(tmp_path / "x" / "a.npz")
    assert cell == meta
    s3 = save_items(tmp_path / "z" / "a.npz", items, cell)  # load -> save round trip
    assert s3 == s1
    for x, y in zip(a, items):
        assert x["context_ids"].tolist() == y["context_ids"].tolist()
        for qa, qb in zip(x["queries"], y["queries"]):
            assert qa["prompt_ids"].tolist() == qb["prompt_ids"].tolist()
            assert qa["candidates"] == qb["candidates"]
        for k in x["arrays"]:
            assert np.array_equal(x["arrays"][k], y["arrays"][k])
    np.load(tmp_path / "x" / "a.npz")  # also a valid .npz
    m = write_manifest(tmp_path / "x")
    assert json.loads(m.read_text())["files"]["a.npz"]["sha256"] == s1
    assert verify_manifest(tmp_path / "x") == []
    # "across models": two different models score the same bytes (header records the same SHA)
    p = tmp_path / "x" / "a.npz"
    o1 = score_file(InductionLM(ts.vocab_size, order=4), p, tmp_path / "r" / "m1.jsonl", score_candidates=False)
    o2 = score_file(InductionLM(ts.vocab_size, order=2), p, tmp_path / "r" / "m2.jsonl", score_candidates=False)
    h1 = json.loads(o1.with_name(o1.stem + "_meta.json").read_text())
    h2 = json.loads(o2.with_name(o2.stem + "_meta.json").read_text())
    assert h1["item_sha256"] == h2["item_sha256"] == sha256_file(p)


# ----------------------------------------------------------------------------- scorer
class Toy(torch.nn.Module):
    def __init__(self, V, d=16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Parameter(torch.randn(V, d, generator=g) * 0.5)
        self.mix = torch.nn.Parameter(torch.randn(d, d, generator=g) * 0.3)

    @property
    def lm_head_weight(self):
        return self.emb

    def forward(self, ids, cu, pos):  # causal cumulative mean (a linear-attention-like toy)
        x = self.emb[ids[0]]
        h = torch.cumsum(x, 0) / torch.arange(1, x.shape[0] + 1)[:, None]
        return torch.tanh(h @ self.mix + x)[None]


def test_scorer_end_to_end_induction(tmp_path, tok, fillers, ts):
    """An induction (copy) model solves distinct-key retrieval and R3 (latest value) but not
    multi-hop; the scorer must report exactly that, with consistent ranks and error types."""
    lm = InductionLM(ts.vocab_size, order=6)
    res = {}
    for task, lvl, T in (("R1", {"D": 15}, 2 * K), ("R-load", {"N": 32}, 2 * K), ("R3", {"U": 2}, 2 * K),
                         ("M-hop", {"h": 2}, 2 * K), ("R1", {"D": 3, "placement": "nofill"}, None)):
        items = generate(task, lvl, T, 4, eval_rng("dev", "scorer", task), "dev", tok, fillers["dev"])
        p = tmp_path / f"{task}_{T}.npz"
        save_items(p, items, dict(task=task, level=lvl, T=T))
        out = score_file(lm, p, tmp_path / "res" / f"{task}_{T}.jsonl")
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        assert len(rows) == 16
        need = {"item_id", "context_id", "query_idx", "template", "filler_source", "T", "depth", "distance",
                "needle_start", "gold_ids", "argmax_ids", "EM_tf", "logp_gold", "logp_gold_tokens", "cand_logp",
                "rank", "error_type", "scoring_path"}
        assert need <= set(rows[0])
        res[task] = rows
        # never overwrite: a second call writes a suffixed file
        out2 = score_file(lm, p, tmp_path / "res" / f"{task}_{T}.jsonl", score_candidates=False)
        assert out2 != out and out2.name.endswith(".r1.jsonl")
    for t in ("R1", "R-load", "R3"):
        assert all(r["EM_tf"] and r["rank"] == 1 and r["error_type"] == "none" for r in res[t]), t
        assert all(abs(r["logp_gold"] - 4 * np.log(0.9)) < 1e-4 for r in res[t])
    assert not any(r["EM_tf"] for r in res["M-hop"])
    assert all(r["error_type"] in ("wrong_chain", "absent", "non_candidate") for r in res["M-hop"])


def test_scorer_matches_single_pass(tmp_path, tok, fillers, ts):
    """Branched scoring with a toy torch LM equals the single-pass reference (VT-07 style, CPU fp32)."""
    from hybridlc.eval.adapter import ForwardOnlyAdapter

    model = Toy(ts.vocab_size)
    ad = ForwardOnlyAdapter(model)
    items = generate("R1", {"D": 3}, 1 * K, 3, eval_rng("dev", "toy"), "dev", tok, fillers["dev"])
    p = tmp_path / "toy.npz"
    save_items(p, items, dict(task="R1", level={"D": 3}, T=1 * K))
    out = score_file(ad, p, tmp_path / "toy.jsonl")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    for r in rows:
        it = items[r["context_id"]]
        q = it["queries"][r["query_idx"]]
        full = np.concatenate([it["context_ids"], q["prompt_ids"], q["answer_ids"]])
        lg = ad.logits(torch.from_numpy(full.astype(np.int64)))
        lp = torch.log_softmax(lg, -1)
        a0 = len(it["context_ids"]) + len(q["prompt_ids"])
        ref = sum(float(lp[a0 - 1 + j, int(q["answer_ids"][j])]) for j in range(4))
        assert abs(ref - r["logp_gold"]) < 1e-4
        assert len(full) == 1 * K


# ----------------------------------------------------------------------------- real model (GPU)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scorer_with_hybridlm_gpu(tmp_path, tok, fillers, ts):
    """score_file on the real HybridLM (random init, C3 pattern, small width): branched scoring agrees
    with the single-pass re-score (EM identical, |dlogp| small) in fp32; candidate logps from the
    post-prompt cache equal the gold logp for the gold candidate."""
    from hybridlc.utils.gpu import claim

    claim(0.44)
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.eval.scorer import rescore_check

    cfg = ModelConfig.for_scale("D384", "C3", vocab_size=len(tok))
    model = build_model(cfg, 900).cuda().eval()
    items = generate("R1", {"D": 15}, 2 * K, 4, eval_rng("dev", "gpu-scorer"), "dev", tok, fillers["dev"])
    p = tmp_path / "R1_D15_2K.npz"
    save_items(p, items, dict(task="R1", level={"D": 15}, T=2 * K))
    out = score_file(model, p, tmp_path / "res.jsonl", autocast=False)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 16 and all(len(r["cand_logp"]) == r["n_cand"] for r in rows)
    for r in rows:  # candidate 0 is the gold: both scoring routes must agree
        assert abs(r["cand_logp"][0] - r["logp_gold"]) < 1e-6
    chk = rescore_check(model, p, out, frac=1.0)
    print("fp32 rescore", chk)
    # interface-level check (the precise branched-vs-separate tolerance is the model builder's VT-07):
    # chunk (single pass) vs fused-recurrent (branched) GDN kernels differ at ~1e-3 relative
    scale = max(abs(r["logp_gold"]) for r in rows)
    assert chk["flips"] == 0 and chk["max_abs_dlogp"] < 3e-3 * scale
    # bf16 autocast (canonical precision) runs and records itself in the header
    out2 = score_file(model, p, tmp_path / "res_bf16.jsonl")
    hdr = json.loads(out2.with_name(out2.stem + "_meta.json").read_text())
    assert hdr["autocast_bf16"] is True and hdr["frozen_batch_size"] == 8
    rows2 = [json.loads(x) for x in out2.read_text().splitlines()]
    flips = sum(a["EM_tf"] != b["EM_tf"] for a, b in zip(rows, rows2))
    print("bf16 vs fp32 EM flips", flips, "max dlogp", max(abs(a["logp_gold"] - b["logp_gold"]) for a, b in zip(rows, rows2)))


# ----------------------------------------------------------------------------- Tier 0
def test_tier0_items_and_lm_scoring(tmp_path, tok, fillers, ts):
    from hybridlc.eval import tier0
    from hybridlc.eval.constants import LM1_BUCKET_EDGES

    f = fillers["test"]
    # LM-4: [EOT] + P + G + P, exactly 1 + 512 + G tokens, both P copies identical
    for var in ("a", "b"):
        for G in (0, 512):
            its = tier0.gen_lm4({"variant": var, "G": G}, None, 3, eval_rng("test", "LM-4", var, G), "test", tok, f)
            for it in its:
                c, m = it["context_ids"], it["meta"]
                assert len(c) == 1 + 512 + G and c[0] == ts.eot_id and not (c[1:] == ts.eot_id).any()
                assert (c[1:257] == c[m["p2_start"]:m["p2_start"] + 256]).all()
                if var == "b":
                    assert set(c[1:257].tolist()) <= set(tier0.lm4_vocab(tok, "test").tolist())
    assert len(tier0.lm4_vocab(tok, "test")) == 4096
    # LM-3 annotations: positions whose 4-gram repeats at Delta in [64, 32K); banned set excludes
    x = np.r_[np.arange(10, 110), np.arange(500, 600), np.arange(10, 110)].astype(np.int32)  # repeat at Delta=200
    pos, delta = tier0.lm3_annotations(x, None)
    assert len(pos) == 97 and (delta == 200).all() and pos.min() == 203
    banned = tier0.ngram_codes(x)[:5]
    pos2, _ = tier0.lm3_annotations(x, banned)
    assert len(pos2) == 97 - 5 and pos2.min() == 208  # codes of the 4-grams ending at 3..7 -> repeats at 203..207
    top = tier0.build_lm3_banned(np.r_[np.tile([5, 6, 7, 8], 50), np.arange(100, 400)].astype(np.int64))
    assert tier0.ngram_codes(np.asarray([5, 6, 7, 8]))[0] in set(top.tolist())
    # LM-1 (short T for speed): docs start with EOT, carry LM-3 annotations
    its = tier0.gen_lm1({"n_pg19": 3}, 4096, 0, eval_rng("test", "LM-1"), "test", tok, f)
    assert len(its) >= 3 and all(len(it["context_ids"]) == 4096 and it["context_ids"][0] == ts.eot_id for it in its)
    p = tmp_path / "LM-1_all_4K.npz"
    save_items(p, its[:4] + its[-2:], dict(task="LM-1", level={}, T=4096))
    p4 = tmp_path / "LM-4_a-G512.npz"
    save_items(p4, tier0.gen_lm4({"variant": "a", "G": 512}, None, 2, eval_rng("test", "x"), "test", tok, f),
               dict(task="LM-4", level={"variant": "a", "G": 512}, T=1025))
    model = Toy(ts.vocab_size)
    rows = [json.loads(l) for l in score_file(model, p, tmp_path / "lm1.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert sum(rows[0]["pos_count"]) == 4095 and rows[0]["pos_bucket_edges"] == list(LM1_BUCKET_EDGES)
    assert sum(rows[0]["lm3_count"]) == int((np.asarray(its[0]["arrays"]["lm3_delta"]) < 32768).sum())
    rows4 = [json.loads(l) for l in score_file(model, p4, tmp_path / "lm4.jsonl").read_text().splitlines()]
    assert all(r["n_scored"] == 248 and np.isfinite(r["copy_gain"]) for r in rows4)
    # LM-2 needs the held-out FineWeb-Edu store
    if TOKENS_REAL:
        its2 = tier0.gen_lm2({}, 4096, 3, None, "test", tok)
        assert len(its2) == 3 and all(len(it["context_ids"]) == 4096 for it in its2)
        p2 = tmp_path / "LM-2_all_4K.npz"
        save_items(p2, its2, dict(task="LM-2", level={}, T=4096))
        r2 = [json.loads(l) for l in score_file(model, p2, tmp_path / "lm2.jsonl").read_text().splitlines()]
        assert all(r["n_tokens"] >= 4095 for r in r2)
