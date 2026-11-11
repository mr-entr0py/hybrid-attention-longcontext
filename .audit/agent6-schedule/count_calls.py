"""Exact scorer call counts per suite from the frozen item files (CPU only; reads meta + offsets).

Mirrors src/hybridlc/eval/scorer.py::_score_items: contexts are grouped by (T_ctx, n_queries) and batched at
frozen_batch_size(T) (8 for T <= 4K, 2 for 8K-16K, 1 for 32K); per batch one prefill; per query one gold
score_continuation call, plus (if candidates are scored and the batch max n_cand > 1) one extend + (nc - 1)
candidate calls. Output: per-suite totals of batches, prefill tokens, calls (with and without candidates) and
the per-(T, B) breakdown used by the cost model.
"""
import io, json, sys, zipfile, collections, glob, os
import numpy as np
REPO = "/home/sedrica-lynx/Downloads/outeropt"
K = 1024
def fbs(T):
    return 8 if T <= 4 * K else (2 if T <= 16 * K else 1)
def arr(zf, name):
    return np.lib.format.read_array(io.BytesIO(zf.read(f"{name}.npy")), allow_pickle=False)
out = {}
for suite_dir in sorted(glob.glob(f"{REPO}/experiments/items/*/*")):
    if not os.path.isdir(suite_dir):
        continue
    suite = "/".join(suite_dir.split("/")[-2:])
    tot = collections.Counter()
    cells = []
    for f in sorted(glob.glob(f"{suite_dir}/*.npz")):
        with zipfile.ZipFile(f) as zf:
            meta = json.loads(zf.read("meta.json"))
            ctx_off = arr(zf, "ctx_off"); q_item = arr(zf, "q_item"); qco = arr(zf, "q_cand_off")
        n_items = meta["n_items"]; nq_total = meta["n_queries"]
        Tc = np.diff(ctx_off)
        ncand = np.diff(qco)
        q_by_item = collections.defaultdict(list)
        for j in range(nq_total):
            q_by_item[int(q_item[j])].append(int(ncand[j]))
        groups = collections.defaultdict(list)
        for i in range(n_items):
            groups[(int(Tc[i]), len(q_by_item[i]))].append(i)
        c_b = c_pre_tok = c_calls_cand = c_calls_nocand = 0
        for (T, nq), idxs in groups.items():
            B = fbs(T)
            for a in range(0, len(idxs), B):
                b = idxs[a:a + B]
                c_b += 1; c_pre_tok += T * len(b)
                for q in range(nq):
                    nc = max(q_by_item[i][q] for i in b)
                    c_calls_nocand += 1
                    c_calls_cand += 1 + ((1 + (nc - 1)) if nc > 1 else 0)
        Tnom = max(Tc) if n_items else 0
        cells.append(dict(cell=os.path.basename(f)[:-4], n_ctx=n_items, n_q=nq_total, T_max=int(Tnom), B=fbs(int(Tnom)),
                          batches=c_b, calls_cand=c_calls_cand, calls_nocand=c_calls_nocand,
                          mean_ncand=float(ncand.mean()) if len(ncand) else 0.0))
        tot.update(dict(cells=1, contexts=n_items, queries=nq_total, batches=c_b, prefill_tok=c_pre_tok,
                        calls_cand=c_calls_cand, calls_nocand=c_calls_nocand))
    out[suite] = dict(total=dict(tot), cells=cells)
    print(f"{suite:10s} cells {tot['cells']:3d}  contexts {tot['contexts']:6d}  queries {tot['queries']:6d}  batches {tot['batches']:6d}  "
          f"prefill tokens {tot['prefill_tok']/1e6:7.1f}M  calls w/ cand {tot['calls_cand']:7d}  w/o cand {tot['calls_nocand']:6d}")
json.dump(out, open(f"{REPO}/.audit/agent6-schedule/call_counts.json", "w"), indent=1)
