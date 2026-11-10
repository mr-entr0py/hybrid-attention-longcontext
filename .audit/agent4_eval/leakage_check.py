#!/usr/bin/env python
"""Agent-4 leakage audit (READ-ONLY): train namespace (TEF / pretraining) vs eval items.

1. Document-level: text_hash / id_hash intersections between every held-out store used for eval
   filler (fwe/test, fwe/dev, pg19/test, pg19/validation) and the training stores (fwe/train,
   pg19/train); and between test and dev stores.
2. Documents actually used in the frozen item files (filler_source.doc / docs): their text hashes
   vs training; within-context duplicate FWE documents.
3. Word lists: token-level disjointness of list A (TEF) vs list B (eval) for adjectives, nouns,
   probes, animals; case-insensitive word collisions; list-A background multi-token words that
   contain list-B key/probe tokens as sub-tokens.
4. Key bigrams of eval items vs list-A key space (must be impossible by construction).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))

from hybridlc.data.tokenizer import load_tokenizer  # noqa: E402
from hybridlc.eval.items import load_items  # noqa: E402
from hybridlc.eval.tokens import token_space  # noqa: E402
from hybridlc.eval.words import word_list  # noqa: E402

out = {}


def hashes(corpus, split, field="text_hash"):
    z = np.load(REPO / "data" / "tokens" / corpus / f"{split}.docs.npz")
    h = z[field]
    return {bytes(r) for r in h}, h


# ---------------------------------------------------------------- 1. store-level intersections
stores = {"fwe/train": ("fwe", "train"), "pg19/train": ("pg19", "train"), "fwe/test": ("fwe", "test"),
          "fwe/dev": ("fwe", "dev"), "pg19/test": ("pg19", "test"), "pg19/validation": ("pg19", "validation")}
TH, IH, ARR = {}, {}, {}
for name, (c, s) in stores.items():
    TH[name], ARR[name] = hashes(c, s, "text_hash")
    IH[name], _ = hashes(c, s, "id_hash")
inter = {}
for ev in ("fwe/test", "fwe/dev", "pg19/test", "pg19/validation"):
    for tr in ("fwe/train", "pg19/train"):
        inter[f"{ev} & {tr} text"] = len(TH[ev] & TH[tr])
        inter[f"{ev} & {tr} id"] = len(IH[ev] & IH[tr])
inter["fwe/test & fwe/dev text"] = len(TH["fwe/test"] & TH["fwe/dev"])
inter["pg19/test & pg19/validation text"] = len(TH["pg19/test"] & TH["pg19/validation"])
inter["fwe/test & pg19 any text"] = len(TH["fwe/test"] & (TH["pg19/train"] | TH["pg19/test"]))
out["store_intersections"] = inter
out["store_sizes"] = {k: len(v) for k, v in TH.items()}
# within-split duplicates (by text hash)
out["within_split_dup_text"] = {k: int(len(ARR[k]) - len(TH[k])) for k in ARR}

# ---------------------------------------------------------------- 2. docs used by items
used = {"test": Counter(), "dev": Counter()}
dup_ctx = Counter()
n_ctx_fwe = Counter()
pool_of_ns = {"test": {"pg19": "pg19/test", "fwe": "fwe/test"}, "dev": {"pg19": "pg19/validation", "fwe": "fwe/dev"}}
bad_used = Counter()
for ns in ("test", "dev"):
    root = REPO / "experiments" / "items" / ns
    man = json.loads((root / "MANIFEST.json").read_text())
    for rel in sorted(man["files"]):
        if not rel.endswith(".npz"):
            continue
        items, cell = load_items(root / rel)
        for it in items:
            fs = it["meta"].get("filler_source") or {}
            k = fs.get("kind")
            if k == "pg19":
                pool = fs.get("pool")
                d = fs.get("doc")
                if d is not None and d >= 0:
                    used[ns][(pool, int(d))] += 1
            elif k == "fwe":
                pool = fs.get("pool")
                docs = fs.get("docs", [])
                n_ctx_fwe[ns] += 1
                if len(set(docs)) != len(docs):
                    dup_ctx[ns] += 1
                for d in docs:
                    used[ns][(pool, int(d))] += 1
            elif k in ("list", "repetitive", "none", None):
                pass
for ns in ("test", "dev"):
    for (pool, d), c in used[ns].items():
        th = bytes(ARR[pool][d])
        if th in TH["fwe/train"] or th in TH["pg19/train"]:
            bad_used[ns] += 1
out["used_docs"] = {ns: dict(n_distinct=len(used[ns]), in_training_text=bad_used[ns],
                             pools=dict(Counter(p for p, _ in used[ns]))) for ns in used}
out["fwe_contexts_with_repeated_doc"] = {ns: dict(n=dup_ctx[ns], of=n_ctx_fwe[ns]) for ns in n_ctx_fwe}
# test/dev reuse of the same documents? (pools are disjoint by construction)
out["test_dev_doc_overlap"] = len({d for d in used["test"]} & {d for d in used["dev"]})

# ---------------------------------------------------------------- 3. word lists
tok = load_tokenizer(REPO / "data" / "tokenizer")
ts = token_space(tok)
A, B = word_list(ts, "A"), word_list(ts, "B")
wl = {}
for f in ("adj_ids", "noun_ids", "probe_ids", "animal_ids"):
    a, b = set(getattr(A, f).tolist()), set(getattr(B, f).tolist())
    wl[f + "_overlap"] = len(a & b)
# cross-role overlap: A adjective token == B noun token etc.
allA = set(A.adj_ids.tolist()) | set(A.noun_ids.tolist())
allB = set(B.adj_ids.tolist()) | set(B.noun_ids.tolist())
wl["A_keytokens_vs_B_keytokens"] = len(allA & allB)
# case-insensitive collisions between word lists
la = {w.lower() for w in A.adjectives + A.nouns}
lb = {w.lower() for w in B.adjectives + B.nouns}
wl["case_insensitive_word_collisions"] = len(la & lb)
wl["case_insensitive_examples"] = sorted(la & lb)[:15]
# capitalised words in key lists
wl["capitalised_B_adjectives"] = sum(1 for w in B.adjectives if w[:1].isupper())
wl["capitalised_B_nouns"] = sum(1 for w in B.nouns if w[:1].isupper())
wl["capitalised_B_examples"] = [w for w in B.adjectives + B.nouns if w[:1].isupper()][:15]
# list-A background words containing list-B key/probe tokens as sub-tokens
Bset = allB | set(B.probe_ids.tolist())
n_bg_multi = sum(1 for e in A.background if len(e) > 1)
n_bg_contains = sum(1 for e in A.background if set(e.tolist()) & Bset)
wl["A_background_n"] = len(A.background)
wl["A_background_multitoken_n"] = n_bg_multi
wl["A_background_containing_B_key_or_probe_token"] = n_bg_contains
# list B probes that are also list-B key nouns (A-freq probes = nouns minus animals): by design
wl["B_probe_in_B_nouns"] = len(set(B.probe_ids.tolist()) & set(B.noun_ids.tolist()))
wl["A_bg_tokens_equal_B_probe"] = len({int(e[0]) for e in A.background if len(e) == 1} & set(B.probe_ids.tolist()))
out["word_lists"] = wl

# ---------------------------------------------------------------- 4. eval key bigrams vs list A
adjA, nounA = set(A.adj_ids.tolist()), set(A.noun_ids.tolist())
n_keys, n_A = 0, 0
for ns in ("test", "dev"):
    root = REPO / "experiments" / "items" / ns
    man = json.loads((root / "MANIFEST.json").read_text())
    for rel in sorted(man["files"]):
        if not rel.endswith(".npz"):
            continue
        items, cell = load_items(root / rel)
        for it in items:
            fk = it["arrays"].get("fact_key")
            if fk is None:
                continue
            for a_, n_ in fk.reshape(-1, 2).tolist():
                if a_ < 0:
                    continue
                n_keys += 1
                if a_ in adjA or n_ in nounA:
                    n_A += 1
out["eval_keys_checked"] = n_keys
out["eval_keys_using_listA_token"] = n_A

print(json.dumps(out, indent=1, default=str))
(REPO / ".audit" / "agent4_eval" / "leakage.json").write_text(json.dumps(out, indent=1, default=str))
