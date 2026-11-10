"""VT-17 regression (verification round 1, TEF-REPLAY-CROSS-DOCUMENT / DTS-01).

In the TEF stream, ``<|endoftext|>`` (id 0) must occur only at ``cu_seqlens`` starts and
positions must restart there (§6.4 intra-document masking in both mixers, RoPE reset per
document; §9.1 attention never crosses a separator). The S0 code registered a FineWeb-Edu
replay chunk (several whole documents) as one segment, so 5-13 separators per chunk were
crossed. Steps 3 and 500 are where the verifiers observed the violation on real data.
Also checks that the D014 step weights still sum to 1 (the extra EOT-led starts add no
ignored labels, so the planned counts stay exact).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridlc.data.storeio import synthetic_corpus
from hybridlc.data.tef import IGNORE_INDEX, StoreReplay, TEFStream
from hybridlc.eval.devtools import resolve_test_tokenizer
from hybridlc.eval.filler import StoreFiller, dev_text_filler
from hybridlc.eval.tokens import token_space

REPO = Path(__file__).resolve().parents[1]
REAL = (REPO / "data" / "tokens" / "fwe" / "train.bin").exists() and (REPO / "data" / "tokenizer" / "tokenizer.json").exists()


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    tok, _ = resolve_test_tokenizer()
    ts = token_space(tok)
    if REAL:
        filler, replay = StoreFiller(ts, "train"), StoreReplay()
    else:
        root = tmp_path_factory.mktemp("tokens")
        synthetic_corpus(root, n_fwe=1500, n_books=12)
        filler, replay = dev_text_filler(ts, "train"), StoreReplay(root)
    return TEFStream(0, 1000, tok=tok, filler=filler, replay=replay), ts.eot_id


@pytest.mark.parametrize("step", [3, 500])
def test_eot_only_at_segment_starts(stream, step):
    s, eot = stream
    n_fwe_docs = 0
    w_sum = 0.0
    for micro in range(s.accum):
        b = s.get_numpy(step, micro)
        x, cu, pos, lab = b["input_ids"][0], b["cu_seqlens"], b["position_ids"][0], b["labels"][0]
        eots = np.flatnonzero(x == eot)
        starts = set(cu[:-1].tolist())
        off = [int(t) for t in eots if int(t) not in starts]
        assert not off, ("EOT not at a segment start", off[:10])
        assert (pos[eots] == 0).all()
        seglen = np.diff(cu)
        assert (pos == np.arange(len(x)) - np.repeat(cu[:-1], seglen)).all()
        # separators are still predicted: the label before an EOT-led start is the EOT
        for t in eots[eots > 0]:
            assert lab[t - 1] == eot
        assert lab[-1] == IGNORE_INDEX
        n_fwe_docs += sum(sg.get("n_docs", 0) for sg in b["segments"][0] if sg.get("replay") == "fwe")
        w_sum += float(b["loss_weights"][0].astype(np.float64).sum())
    # step weight mass under the configured D014 normalisation (block by default since round 3)
    na, no = s.step_counts(step)
    wa, wo = s.step_weight_scale(step)
    assert abs(w_sum - (na * wa + no * wo)) < 1e-4, w_sum
    if s.answer_norm == "step":
        assert abs(w_sum - 1.0) < 1e-4, w_sum
    print(f"step {step}: fwe replay documents = {n_fwe_docs}")
