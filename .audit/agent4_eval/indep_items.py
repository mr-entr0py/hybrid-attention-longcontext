#!/usr/bin/env python
"""Agent-4 (resumed session): INDEPENDENT item checks that do not use hybridlc.eval.items.load_items.

Reads the frozen .npz members directly and checks, on a sample of contexts per cell:
  1. exact T per branched query (len(ctx) + len(prompt) + len(answer) == T), EOT only at position 0;
  2. filler provenance: the context minus its needle spans equals the span re-extracted from the
     recorded held-out document(s) in data/tokens (pg19: one doc span; fwe: docs joined by "\\n\\n",
     last n tokens); the documents' text hashes are absent from the training stores;
  3. canonical tokenisation: decode(fact tokens) re-encodes to the same ids (fact interior), and the
     needle/filler junctions re-encode identically (first 3 needles per context);
  4. PG19 book reuse within / across cells (dependence between contexts).
READ-ONLY on the repo. Writes JSON to stdout and to indep_items.json next to this file.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
OUT = Path(__file__).with_name("indep_items.json")


def read_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with zipfile.ZipFile(path) as z:
        arrs = {}
        for name in z.namelist():
            if name.endswith(".npy"):
                arrs[name[:-4]] = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
        meta = json.loads(z.read("meta.json"))
    return arrs, meta


class Store:
    def __init__(self, corpus: str, split: str):
        base = REPO / "data" / "tokens" / corpus / split
        self.tok = np.memmap(str(base) + ".bin", dtype=np.uint16, mode="r")
        self.idx = np.load(str(base) + ".idx.npy")
        self.th = np.load(str(base) + ".docs.npz")["text_hash"]

    def doc_body(self, i: int) -> np.ndarray:  # without the leading EOT
        return np.asarray(self.tok[self.idx[i] + 1:self.idx[i + 1]]).astype(np.int64)


def main() -> None:
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(REPO / "data" / "tokenizer"))
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    joiner = np.asarray(enc("\n\n"), np.int64)
    EOT = 0
    stores = {}

    def store(pool: str) -> Store:
        if pool not in stores:
            c, s = pool.split("/")
            stores[pool] = Store(c, s)
        return stores[pool]

    train_th = set()
    for c in ("fwe", "pg19"):
        th = np.load(REPO / "data" / "tokens" / c / "train.docs.npz")["text_hash"]
        train_th |= {bytes(r) for r in th}

    rng = np.random.default_rng(20261110)
    res = defaultdict(Counter)
    book_use = defaultdict(Counter)  # (ns, T) -> Counter(book id) over contexts
    book_use_cell = {}
    noncanon_examples = []
    per_cell = {}
    for ns in ("test", "dev"):
        root = REPO / "experiments" / "items" / ns
        man = json.loads((root / "MANIFEST.json").read_text())
        for rel in sorted(man["files"]):
            if not rel.endswith(".npz"):
                continue
            arrs, meta = read_npz(root / rel)
            cell = meta["cell"]
            task = cell["task"]
            if task.startswith("LM-"):
                continue
            T = cell.get("T")
            n = meta["n_items"]
            ctx_off = arrs["ctx_off"]
            items_meta = meta["items"]
            q_item = arrs["q_item"]
            # pg19 book use (all contexts)
            bc = Counter()
            for im in items_meta:
                fs = im.get("filler_source") or {}
                if fs.get("kind") == "pg19":
                    book_use[(ns, T)][int(fs["doc"])] += 1
                    bc[int(fs["doc"])] += 1
            if bc:
                book_use_cell[f"{ns}/{rel}"] = dict(n_pg19_ctx=sum(bc.values()), n_books=len(bc),
                                                    max_per_book=max(bc.values()))
            sample = sorted(rng.choice(n, size=min(n, 25), replace=False).tolist())
            c = Counter()
            for i in sample:
                ctx = arrs["ctx"][ctx_off[i]:ctx_off[i + 1]].astype(np.int64)
                im = items_meta[i]
                c["contexts"] += 1
                if ctx[0] != EOT:
                    c["fail_bos"] += 1
                if (ctx[1:] == EOT).any():
                    c["fail_eot_inside"] += 1
                qs = np.flatnonzero(q_item == i)
                lens = set()
                for q in qs:
                    p = arrs["q_prompt"][arrs["q_prompt_off"][q]:arrs["q_prompt_off"][q + 1]]
                    a = arrs["q_answer"][arrs["q_answer_off"][q]:arrs["q_answer_off"][q + 1]]
                    lens.add(len(p) + len(a))
                    if T is not None and len(ctx) + len(p) + len(a) != T:
                        c["fail_T"] += 1
                    if (p == EOT).any() or (a == EOT).any():
                        c["fail_eot_query"] += 1
                    c["queries"] += 1
                if len(lens) != 1:
                    c["fail_branch_len"] += 1
                # ---- filler provenance
                fs = im.get("filler_source") or {}
                kind = fs.get("kind")
                if kind in ("pg19", "fwe") and "arr.fact_start" in arrs:
                    o0, o1 = arrs["arr.fact_start.off"][i], arrs["arr.fact_start.off"][i + 1]
                    st = arrs["arr.fact_start"][o0:o1].astype(np.int64)
                    en = arrs["arr.fact_end"][o0:o1].astype(np.int64)
                    keep = np.ones(len(ctx), bool)
                    keep[0] = False
                    for s_, e_ in zip(st, en):
                        keep[s_:e_] = False
                    fill = ctx[keep]
                    s = store(fs["pool"])
                    if kind == "pg19":
                        body = s.doc_body(int(fs["doc"]))
                        ref = body[int(fs["start"]):int(fs["end"])]
                        docs = [int(fs["doc"])]
                    else:
                        parts = []
                        docs = [int(d) for d in fs["docs"]]
                        for j, d in enumerate(docs):
                            if j:
                                parts.append(joiner)
                            parts.append(s.doc_body(d))
                        cat = np.concatenate(parts)
                        ref = cat[int(fs["first_offset"]):]
                    if len(ref) == len(fill) and np.array_equal(ref, fill):
                        c["filler_match"] += 1
                    else:
                        c["filler_MISMATCH"] += 1
                    if any(bytes(s.th[d]) in train_th for d in docs):
                        c["filler_doc_in_train"] += 1
                    expected_pool = {"test": {"pg19/test", "fwe/test"}, "dev": {"pg19/validation", "fwe/dev"}}[ns]
                    if fs["pool"] not in expected_pool:
                        c["wrong_pool"] += 1
                    # ---- canonical tokenisation of the first 3 needles and their junctions
                    for f in range(min(3, len(st))):
                        s_, e_ = int(st[f]), int(en[f])
                        span = ctx[s_:e_]
                        if enc(tok.decode(span.tolist())) != span.tolist():
                            c["noncanonical_fact"] += 1
                            if len(noncanon_examples) < 5:
                                noncanon_examples.append(dict(file=f"{ns}/{rel}", item=i, text=tok.decode(span.tolist())))
                        lo, hi = max(1, s_ - 4), min(len(ctx), e_ + 4)
                        win = ctx[lo:hi]
                        if enc(tok.decode(win.tolist())) != win.tolist():
                            c["noncanonical_junction"] += 1
                        c["facts_tok_checked"] += 1
                elif kind == "repetitive":
                    c["repetitive_ctx"] += 1
                elif kind == "list":
                    c["list_ctx"] += 1
                else:
                    c["nofill_ctx"] += 1
            per_cell[f"{ns}/{rel}"] = dict(c)
            for k, v in c.items():
                res["total"][k] += v
    out = dict(total=dict(res["total"]),
               cells_with_failures={k: v for k, v in per_cell.items()
                                    if any(kk.startswith("fail") or kk in ("filler_MISMATCH", "filler_doc_in_train", "wrong_pool") for kk in v)},
               noncanonical_examples=noncanon_examples,
               cells_with_noncanonical={k: {kk: vv for kk, vv in v.items() if kk.startswith("noncanonical") or kk == "facts_tok_checked"}
                                        for k, v in per_cell.items() if v.get("noncanonical_fact") or v.get("noncanonical_junction")},
               pg19_books_by_ns_T={f"{ns}/{T}": dict(n_contexts=sum(cnt.values()), n_books=len(cnt), max_contexts_per_book=max(cnt.values()),
                                                     top5=cnt.most_common(5))
                                   for (ns, T), cnt in sorted(book_use.items(), key=lambda x: (x[0][0], x[0][1] or 0))},
               pg19_books_per_cell_examples={k: v for k, v in book_use_cell.items() if k.endswith("_32K.npz") and "/CS/" in k})
    print(json.dumps(out, indent=1, default=str))
    OUT.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
