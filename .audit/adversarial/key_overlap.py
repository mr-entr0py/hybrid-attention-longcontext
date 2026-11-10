"""Read-only: how often does a queried fact's key share a token (adjective or noun) with another fact in the
same context (frozen CS items)? Also: is the query prompt a literal copy of the fact prefix?"""
import sys, numpy as np
sys.path.insert(0, "/home/sedrica-lynx/Downloads/outeropt/src")
from hybridlc.eval.items import load_items
root = "/home/sedrica-lynx/Downloads/outeropt/experiments/items/test/CS/"
for cell in ("R1_D3_32K", "R1_D15_32K", "R-load_N32_32K", "R-load_N128_32K", "R-load_N512_32K", "R-load_N1024_32K", "R-load_N1024_16K"):
    items, meta = load_items(root + cell + ".npz")
    n_q = 0; share_noun = 0; share_adj = 0; unique_noun = 0; unique_adj = 0; prefix_copy = 0
    for it in items:
        a = it["arrays"]; fk = np.asarray(a["fact_key"]); role = np.asarray(a["fact_role"])
        ctx = np.asarray(it["context_ids"])
        for q in it["queries"]:
            key = None
            # recover queried key ids from the prompt: tokens right before " is" ... use fact_key match
            p = np.asarray(q["prompt_ids"])
            # the key is the 2 tokens at positions where fact_key matches a subsequence of the prompt
            for j in range(len(fk)):
                kk = fk[j]
                for s in range(len(p) - 1):
                    if p[s] == kk[0] and p[s + 1] == kk[1]:
                        key = j; break
                if key is not None: break
            if key is None: continue
            n_q += 1
            others = np.delete(np.arange(len(fk)), key)
            sn = int(np.sum(fk[others, 1] == fk[key, 1])); sa = int(np.sum(fk[others, 0] == fk[key, 0]))
            share_noun += sn > 0; share_adj += sa > 0
            # literal prefix: prompt (minus trailing space) occurs in context right before the gold value
            pr = p[:-1]
            L = len(pr)
            found = any(np.array_equal(ctx[s:s + L], pr) for s in range(max(0, int(a["fact_start"][key]) - 2), int(a["fact_start"][key]) + 3))
            prefix_copy += found
    print(f"{cell}: queries={n_q}  P(another fact shares the noun)={share_noun/n_q:.3f}  "
          f"P(another fact shares the adjective)={share_adj/n_q:.3f}  P(prompt is a literal copy of the fact prefix)={prefix_copy/n_q:.3f}")
