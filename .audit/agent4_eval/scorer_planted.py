#!/usr/bin/env python
"""Agent-4: scorer correctness with planted models (CPU only, READ-ONLY on the repo).

A. PlantedLM: per query, the model "answers" a planned 4-digit value (gold / in-context distractor /
   absent / stale / other-chain / random non-candidate) with p_hit on each planned digit.
   Checks EM_tf, rank, rank_correct and error_type against the plan, through scorer.score_items with
   and without the ``extend`` candidate path.
B. EM_tf == greedy decoding: a tiny random torch LM wrapped in ForwardOnlyAdapter; items whose gold
   answer is set to the model's own greedy continuation (half) or perturbed (half).
C. Tie handling: an indifferent model (all candidates equal) -> which error type is reported.
"""
from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))
from hybridlc.eval.adapter import ForwardOnlyAdapter, RecomputeCache  # noqa: E402
from hybridlc.eval.items import load_items  # noqa: E402
from hybridlc.eval.scorer import score_items  # noqa: E402

V = 16384


class PlantedLM:
    scoring_path = "test-double:planted"

    def __init__(self, plan: dict, p_hit: float = 0.9, with_extend: bool = True):
        self.plan = plan  # key: (ctx bytes hash, prompt tuple) -> list of 4 planned token ids
        self.lp_hit = float(np.log(p_hit))
        self.lp_miss = float(np.log((1 - p_hit) / (V - 1)))
        self.device = torch.device("cpu")
        if not with_extend:
            self.extend = None  # noqa
            del self.extend

    def prefill(self, ids):
        return torch.full((ids.shape[0], V), self.lp_miss), RecomputeCache(ids.clone())

    def extend(self, tok, cache):  # noqa: F811
        return None, RecomputeCache(torch.cat([cache.ids, tok.cpu()], 1))

    def _pred(self, full: np.ndarray, T0: int, M: int) -> np.ndarray:
        """Predicted token for positions predicting full[T0 + j], j < M."""
        pred = np.zeros(M, np.int64)
        # find which (context, prompt) this is: try every split point >= T0 where a plan key matches
        for (ctx_len, ctx_h, prompt), planned in self.plan.items():
            if len(full) < ctx_len + len(prompt):
                continue
            if hash(full[:ctx_len].tobytes()) != ctx_h:
                continue
            if tuple(full[ctx_len:ctx_len + len(prompt)].tolist()) != prompt:
                continue
            a0 = ctx_len + len(prompt)  # first answer token index in full
            for j in range(M):
                pos = T0 + j  # token being predicted
                if a0 <= pos < a0 + len(planned):
                    pred[j] = planned[pos - a0]
            return pred
        raise KeyError("no plan for this sequence")

    def score_continuation(self, cache, cont):
        T = cache.ids.shape[1]
        c = cont.cpu().numpy()
        lps, ams = [], []
        for b in range(c.shape[0]):
            full = np.concatenate([cache.ids[b].numpy(), c[b]])
            pred = self._pred(full, T, c.shape[1])
            lps.append(np.where(pred == c[b], self.lp_hit, self.lp_miss))
            ams.append(pred)
        return torch.tensor(np.stack(lps), dtype=torch.float32), torch.tensor(np.stack(ams))


def make_plan(items, rng, kinds):
    plan, expect = {}, {}
    for i, it in enumerate(items):
        ctx = np.asarray(it["context_ids"], np.int64)
        for q in it["queries"]:
            ct = q["meta"]["cand_types"]
            kind = kinds[rng.integers(len(kinds))]
            cands = q["candidates"]
            idx = [j for j, t in enumerate(ct) if t == kind]
            if kind == "gold":
                planned, exp_err = list(cands[0]), "none"
            elif kind == "random":
                # 4 digit tokens not equal to any candidate
                while True:
                    planned = list(rng.choice(np.asarray(sorted({t for c in cands for t in c})), 4))
                    if planned not in [list(c) for c in cands]:
                        break
                exp_err = None  # depends on digit overlap
            elif idx:
                planned = list(cands[idx[0]])
                exp_err = {"context": "binding", "absent": "absent", "stale": "stale",
                           "other_chain": "wrong_chain"}[kind]
            else:
                planned, exp_err = list(cands[0]), "none"
            key = (len(ctx), hash(ctx.tobytes()), tuple(np.asarray(q["prompt_ids"]).tolist()))
            plan[key] = [int(x) for x in planned]
            expect[(i, q["meta"]["query_idx"])] = (kind if exp_err is not None or kind == "random" else "gold", exp_err)
    return plan, expect


def run_A():
    rng = np.random.default_rng(0)
    out = {}
    for f, kinds in (("test/CS/R1_D15_1K.npz", ["gold", "context", "absent", "random"]),
                     ("test/MS/R3_U2_4K.npz", ["gold", "stale", "context", "absent"]),
                     ("test/CS/M-hop_h2_1K.npz", ["gold", "other_chain", "absent", "random"])):
        items, cell = load_items(REPO / "experiments" / "items" / f)
        items = items[:12]
        plan, expect = make_plan(items, rng, kinds)
        for ext in (True, False):
            m = PlantedLM(plan, with_extend=ext)
            rows = score_items(m, items, batch_size=4, cell_name=cell["name"])
            bad = Counter()
            for r in rows:
                kind, exp_err = expect[(r["context_id"], r["query_idx"])]
                if kind == "gold":
                    ok = r["EM_tf"] and r["rank"] == 1 and r["error_type"] == "none"
                elif kind == "random":
                    ok = (not r["EM_tf"]) and r["error_type"] != "none"
                else:
                    ok = (not r["EM_tf"]) and r["rank"] > 1 and r["error_type"] == exp_err
                if not ok:
                    bad[(kind, r["error_type"], r["EM_tf"], r["rank"])] += 1
            out[f"{f} extend={ext}"] = dict(n=len(rows), mismatches=dict(bad),
                                            kinds=dict(Counter(expect[(r['context_id'], r['query_idx'])][0] for r in rows)))
    return out


class TinyLM(torch.nn.Module):
    """Random 1-layer causal transformer with the training forward contract."""

    def __init__(self, d=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(V, d, generator=g) * 0.5)
        self.q = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)
        self.k = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)
        self.v = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5)

    @property
    def lm_head_weight(self):
        return self.emb

    def forward(self, ids, cu, pos):
        x = self.emb[ids[0]]
        n = x.shape[0]
        att = (x @ self.q) @ (x @ self.k).T / x.shape[1] ** 0.5
        att = att.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        h = x + torch.softmax(att, -1) @ (x @ self.v)
        return h[None]


def run_B():
    items, cell = load_items(REPO / "experiments" / "items" / "test/CS/R1_D3_1K.npz")
    items = copy.deepcopy(items[:6])
    model = TinyLM()
    ad = ForwardOnlyAdapter(model)
    rng = np.random.default_rng(1)
    want = {}
    for i, it in enumerate(items):
        ctx = torch.as_tensor(np.asarray(it["context_ids"], np.int64))[None]
        for q in it["queries"]:
            p = torch.as_tensor(np.asarray(q["prompt_ids"], np.int64))[None]
            _, cache = ad.prefill(torch.cat([ctx, p], 1))
            seq = []
            last = ad.logits(torch.cat([ctx, p], 1)[0])[-1]
            ids = torch.cat([ctx, p], 1)
            for _ in range(len(q["answer_ids"])):
                t = int(last.argmax())
                seq.append(t)
                ids = torch.cat([ids, torch.tensor([[t]])], 1)
                last = ad.logits(ids[0])[-1]
            greedy = np.asarray(seq, np.int32)
            if rng.random() < 0.5:
                q["answer_ids"] = greedy
                want[(i, q["meta"]["query_idx"])] = True
            else:
                g2 = greedy.copy()
                j = int(rng.integers(len(g2)))
                g2[j] = (g2[j] + 1) % V
                q["answer_ids"] = g2
                want[(i, q["meta"]["query_idx"])] = False
            q["candidates"][0] = q["answer_ids"].tolist()
    rows = score_items(ad, items, batch_size=2, cell_name="tiny", score_candidates=False, autocast=False)
    mism = sum(1 for r in rows if r["EM_tf"] != want[(r["context_id"], r["query_idx"])])
    return dict(n=len(rows), em_true=sum(want.values()), mismatches=mism)


class Indifferent(PlantedLM):
    def __init__(self):
        self.lp = float(np.log(1.0 / V))
        self.lp_miss = self.lp
        self.device = torch.device("cpu")

    def score_continuation(self, cache, cont):
        c = cont.cpu().numpy()
        return torch.full(c.shape, self.lp), torch.zeros(c.shape, dtype=torch.long)


def run_C():
    out = {}
    for f in ("test/CS/R1_D15_1K.npz", "test/CS/M-hop_h2_1K.npz", "test/CS/A-freq_rho2_1K.npz"):
        items, cell = load_items(REPO / "experiments" / "items" / f)
        rows = score_items(Indifferent(), items[:10], batch_size=5, cell_name=cell["name"])
        out[f] = dict(error_types=dict(Counter(r["error_type"] for r in rows)),
                      rank=dict(Counter(r["rank"] for r in rows)))
    return out


if __name__ == "__main__":
    import json
    res = dict(A_planted=run_A(), B_em_equals_greedy=run_B(), C_ties=run_C())
    print(json.dumps(res, indent=1, default=str))
    (REPO / ".audit" / "agent4_eval" / "scorer_planted.json").write_text(json.dumps(res, indent=1, default=str))
