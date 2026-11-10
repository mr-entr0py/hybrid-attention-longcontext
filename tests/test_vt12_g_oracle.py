"""VT-12 extension for rule G-ORACLE (VT-24r v2, PI ruling D074; experiments/prereg/VT-24r.v2.yaml).

* K6: synthetic contexts with planted collisions (T1 same noun + same 3-digit prefix in a later fact;
  T4 same 2-digit prefix in a later fact; a filler span containing the window) are detected by the
  independent checker and resolved by the repair;
* K5a: a planted collision in an item generated through the same ``generate()`` call the TEF uses is
  NOT repaired in namespace 'train' (byte-identical to the pre-fix generator's output, whose hash was
  recorded on the pre-fix tree in experiments/repro/VT-24r/k5a_baseline.json) and IS repaired in
  namespace 'dev';
* the checker's import graph contains none of eval.devtools, eval.scorer or any model / LM module.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hybridlc.eval import templates as TP
from hybridlc.eval.devtools import resolve_test_tokenizer
from hybridlc.eval.generators import RepairError, repair_item
from hybridlc.eval.oracle_check import check_item, item_ok, most_recent_occurrence
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.tokens import token_space
from hybridlc.eval.words import namespace_list, word_list

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ts():
    tok, _ = resolve_test_tokenizer()
    return token_space(tok)


def _filler(ts, text: str) -> np.ndarray:
    return np.asarray(ts.enc(text), np.int32)


def _make_item(ts, tname: str, keys: list[np.ndarray], values: list[int], extra_after: np.ndarray | None = None,
               queried: int = 0) -> dict:
    """[EOT] filler fact_0 filler fact_1 ... [extra_after] filler; one query on fact ``queried``."""
    tpl = TP.CODE_TEMPLATES[tname]
    fill = _filler(ts, " It was a quiet morning in the valley and nobody spoke for a while.")
    parts, starts, ends = [np.asarray([ts.eot_id], np.int32), fill], [], []
    pos = 1 + len(fill)
    for k, v in zip(keys, values):
        f = TP.fact_ids(ts, tpl, k, v)
        starts.append(pos)
        parts.append(f)
        pos += len(f)
        ends.append(pos)
        parts.append(fill)
        pos += len(fill)
    if extra_after is not None:
        parts.append(np.asarray(extra_after, np.int32))
        parts.append(fill)
    ctx = np.concatenate(parts).astype(np.int32)
    cvals = list(values) + [9999 - values[0] if 9999 - values[0] not in values else 1234]
    q = dict(prompt_ids=TP.query_prompt_ids(ts, tpl, keys[queried]), answer_ids=TP.value_ids(ts, values[queried]),
             candidates=[TP.value_ids(ts, v).tolist() for v in [values[queried]] + [v for v in cvals if v != values[queried]]],
             meta=dict(query_idx=0, needle_start=[starts[queried]], needle_end=[ends[queried]],
                       value=f"{values[queried]:04d}",
                       cand_values=[f"{v:04d}" for v in [values[queried]] + [v for v in cvals if v != values[queried]]]))
    n = len(keys)
    arrays = dict(fact_key=np.asarray(keys, np.int32).reshape(-1, 2), fact_value=np.asarray(values, np.int32),
                  fact_start=np.asarray(starts, np.int32), fact_end=np.asarray(ends, np.int32),
                  fact_role=np.asarray([1 if j == queried else 0 for j in range(n)], np.int8))
    meta = dict(task="R1", namespace="dev", template=tname, query_mode="branch", context_id=0)
    return dict(context_ids=ctx, queries=[q], meta=meta, arrays=arrays)


def _two_keys_same_noun(ts) -> tuple[np.ndarray, np.ndarray]:
    wl = word_list(ts, namespace_list("dev"))
    k0 = wl.key_ids(0)
    for i in range(1, wl.n_keys):
        k = wl.key_ids(i)
        if k[1] == k0[1] and k[0] != k0[0]:
            return k0, k
    # construct one: keep the noun, swap in another key's adjective
    k1 = wl.key_ids(1)
    return k0, np.asarray([k1[0] if k1[0] != k0[0] else wl.key_ids(2)[0], k0[1]], np.int32)


def _assert_repaired(ts, it: dict, bad_digit: int) -> None:
    before = check_item(it)
    assert not before[0]["loc_ok"], "planted collision not detected"
    assert not before[0]["digits"][bad_digit]["loc_ok"]
    assert all(before[0]["digits"][j]["loc_ok"] for j in range(bad_digit))
    ctx0 = it["context_ids"].copy()
    others = [int(v) for v in it["arrays"]["fact_value"][1:]]
    recs = repair_item(it, ts, eval_rng("dev", "K6", 0), 0)
    assert recs and item_ok(it)
    # only the queried fact's 4 value digits changed; other facts untouched
    diff = np.flatnonzero(it["context_ids"] != ctx0)
    d0 = recs[-1]["digit_start"]
    assert set(diff.tolist()) <= set(range(d0, d0 + 4))
    assert [int(v) for v in it["arrays"]["fact_value"][1:]] == others
    new = int(it["arrays"]["fact_value"][0])
    assert np.array_equal(it["queries"][0]["answer_ids"], TP.value_ids(ts, new))
    assert it["queries"][0]["candidates"][0] == TP.value_ids(ts, new).tolist()
    assert new not in others and it["meta"]["vt24r_repair"][-1]["new_value"] == f"{new:04d}"


def test_k6_t1_same_noun_three_digit_prefix(ts):
    ka, kb = _two_keys_same_noun(ts)
    it = _make_item(ts, "T1", [ka, kb], [5176, 5179])  # later fact: same noun, same "517"
    _assert_repaired(ts, it, bad_digit=3)


def test_k6_t4_two_digit_prefix(ts):
    wl = word_list(ts, namespace_list("dev"))
    it = _make_item(ts, "T4", [wl.key_ids(3), wl.key_ids(7)], [3938, 3908])  # later fact shares "39"
    _assert_repaired(ts, it, bad_digit=2)


def test_k6_filler_window(ts):
    wl = word_list(ts, namespace_list("dev"))
    v = 6865
    # a filler span after the fact containing T4's key-free window "'s code: 68" followed by another digit
    span = np.concatenate([_filler(ts, " the old ship"), TP.render(ts, ("'s code:",)), [ts.space_id],
                           TP.value_ids(ts, 6850)]).astype(np.int32)
    it = _make_item(ts, "T4", [wl.key_ids(5), wl.key_ids(9)], [v, 1203], extra_after=span)
    rec = check_item(it)[0]
    s = rec["digits"][2]["occ_end"]
    assert s >= int(it["arrays"]["fact_end"][1])  # the colliding occurrence is the filler span
    _assert_repaired(ts, it, bad_digit=2)
    assert it["meta"]["vt24r_repair"][0]["trigger"] in ("filler", "mixed")


def test_k6_clean_item_untouched_and_train_refused(ts):
    wl = word_list(ts, namespace_list("dev"))
    it = _make_item(ts, "T1", [wl.key_ids(11), wl.key_ids(12)], [1111, 8642])
    snap = (it["context_ids"].copy(), json.dumps(it["meta"], sort_keys=True))
    assert item_ok(it)
    assert repair_item(it, ts, eval_rng("dev", "K6", 1), 0) == []
    assert np.array_equal(it["context_ids"], snap[0]) and json.dumps(it["meta"], sort_keys=True) == snap[1]
    it["meta"]["namespace"] = "train"
    with pytest.raises(ValueError):
        repair_item(it, ts, eval_rng("dev", "K6", 1), 0)


def test_k6_unrepairable_raises(ts):
    """A collision in the window before digit 1 (key-only window) cannot be fixed by resampling the
    value: the repair must raise after the attempt budget, never fall back silently."""
    wl = word_list(ts, namespace_list("dev"))
    k = wl.key_ids(4)
    span = np.concatenate([TP.render(ts, ("'s code:",), ), [ts.space_id]]).astype(np.int32)
    span = np.concatenate([k, span, TP.value_ids(ts, 1)]).astype(np.int32)  # "<key>'s code: 0001" in filler
    it = _make_item(ts, "T4", [k, wl.key_ids(8)], [4321, 9876], extra_after=span)
    assert not check_item(it)[0]["digits"][0]["loc_ok"]
    with pytest.raises(RepairError):
        repair_item(it, ts, eval_rng("dev", "K6", 2), 0, max_attempts=20)


def test_checker_matches_brute_force(ts):
    """most_recent_occurrence equals a brute-force scan on random sequences."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = rng.integers(0, 4, size=60)
        pos = int(rng.integers(6, 60))
        w = seq[pos - 6:pos]
        brute = max((s for s in range(5, pos - 1) if np.array_equal(seq[s - 5:s + 1], w)), default=-1)
        assert most_recent_occurrence(seq, pos) == brute


# ----------------------------------------------------------------------------------------------- K5a
def test_k5a_train_not_repaired_dev_repaired():
    sys.path.insert(0, str(REPO / "scripts" / "repro"))
    import vt24r_baselines as B

    base_p = REPO / "experiments" / "repro" / "VT-24r" / "k5a_baseline.json"
    if not base_p.exists():
        pytest.skip("k5a baseline (pre-fix) not present")
    base = json.loads(base_p.read_text())
    assert base["provenance"]["pre_fix_tree"] is True
    tok, kind, ts, fillers = B.k5a_env()
    if ts.fingerprint != base["tokenizer_fingerprint"]:
        pytest.skip(f"tokenizer {kind} differs from the baseline's")
    tr = B.k5a_item("train", base["cases"]["train"]["attempt"], tok, fillers["train"])
    assert B.item_sha(tr) == base["cases"]["train"]["item_sha256"]  # byte-identical to pre-fix output
    assert base["cases"]["train"]["n_loc_miss_queries"] > 0
    assert not item_ok(tr) and "vt24r_repair" not in tr["meta"]  # the collision is still there
    dv = B.k5a_item("dev", base["cases"]["dev"]["attempt"], tok, fillers["dev"])
    assert base["cases"]["dev"]["n_loc_miss_queries"] > 0
    assert B.item_sha(dv) != base["cases"]["dev"]["item_sha256"]
    assert item_ok(dv) and dv["meta"].get("vt24r_repair")


# ----------------------------------------------------------------------------------------------- imports
FORBIDDEN = ("hybridlc.eval.devtools", "hybridlc.eval.scorer", "hybridlc.eval.adapter", "hybridlc.models",
             "hybridlc.integration", "torch", "transformers", "fla")


def test_oracle_check_import_graph():
    src = (REPO / "src" / "hybridlc" / "eval" / "oracle_check.py").read_text()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert mods <= {"__future__", "typing", "numpy"}, mods
    code = ("import sys; import hybridlc.eval.oracle_check; "
            "print('\\n'.join(sorted(m for m in sys.modules if m.startswith(('hybridlc', 'torch', 'transformers', 'fla')))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}).stdout.split()
    bad = [m for m in out if m.startswith(FORBIDDEN)]
    assert not bad, bad
    assert "score_items(" not in src and "score_file(" not in src
