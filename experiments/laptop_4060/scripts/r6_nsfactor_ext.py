"""R6 namespace-factor diagnostic (NON-GATE, descriptive; never changes the registered R6 verdict).

Question: the R6 AA model scores ~0.9 EM on train-namespace items but ~0.4 on dev-namespace items. A dev item
differs from a train item in (a) the key word list (train = list A, dev = list B, disjoint by SHA parity) and
(b) the filler corpus split (train = pg19/fwe train, dev = pg19 validation / fwe dev). The eval template T1 is
also a training template, so templates are not a factor for these cells.
Design: generate the two primary R6 cells (R1 D=3 @1K, R-load N=8 @1K; 200 contexts x 4 queries) in namespace
'dev' (eval semantics, G-ORACLE repair on) with the SAME seeds for every variant and swap only the word list
and/or the filler split: (B, dev) = registered dev items; (A, dev); (B, train); (A, train).
Also reports embedding-row norms of list-A vs list-B key tokens (tied embedding) per checkpoint.
Run under the GPU lock with TRITON_CACHE_DIR = .cache/triton-diag (D069).
"""
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path.home() / "outeropt-4060" / "wt-gates"; sys.path.insert(0, str(REPO / "src"))
import torch
from hybridlc.utils.gpu_lock import guard
from hybridlc.utils.gpu import claim
guard("r6-nsfactor"); dev = claim(0.9)
from hybridlc.data.tokenizer import load_tokenizer
from hybridlc.eval.tokens import token_space
from hybridlc.eval.filler import StoreFiller
from hybridlc.eval import generators as G
from hybridlc.eval.words import word_list
from hybridlc.eval.scorer import score_items
from hybridlc.integration.evaluate import load_checkpoint

tok = load_tokenizer(REPO / "data" / "tokenizer"); ts = token_space(tok)
fills = {ns: StoreFiller(ts, ns, str(REPO / "data" / "tokens")) for ns in ("dev", "train")}
CELLS = [("R1", {"D": 3}), ("R-load", {"N": 8})]
VARIANTS = [("B", "dev"), ("A", "dev"), ("B", "train"), ("A", "train")]
orig_nl = G.namespace_list
items = {}
for wl, fns in VARIANTS:
    G.namespace_list = (lambda ns, _wl=wl: _wl)
    for ci, (task, lv) in enumerate(CELLS):
        rng = np.random.default_rng([2026, 11, 25, ci])          # identical seed across variants
        items[(wl, fns, task)] = G.generate(task, dict(lv), 1024, 200, rng, "dev", tok, fills[fns])
G.namespace_list = orig_nl
wlA, wlB = word_list(ts, "A"), word_list(ts, "B")
key_ids = {"A": np.concatenate([wlA.adj_ids, wlA.noun_ids]), "B": np.concatenate([wlB.adj_ids, wlB.noun_ids])}

R = Path.home() / "outeropt-4060"
CK = []
sd = int(sys.argv[1])
for p in sorted((R / f"r6ext/s{sd}/ckpt").glob("*/milestones/step_*_bf16.pt")):
    CK.append((f"AA ext s{sd} step {int(p.stem.split('_')[1])}", p))
out = dict(kind="R6 namespace-factor diagnostic (non-gate, descriptive)", ts=time.strftime("%FT%T%z"),
           variants=[f"words {w} / filler {f}" for w, f in VARIANTS], rows=[], emb=[])
for lab, path in CK:
    if not Path(path).exists():
        print("missing", path, flush=True); continue
    ck = load_checkpoint(path, dev, runs_root=None, canonical_bf16=True)
    for (wl, fns, task), its in items.items():
        rows = score_items(ck.model, its, 8, "nsdiag", False, autocast=True)
        em = float(np.mean([bool(r["EM_tf"]) for r in rows]))
        out["rows"].append(dict(ckpt=lab, words=wl, filler=fns, task=task, n_queries=len(rows), EM_tf=em))
        print(f"{lab:<30} words {wl} filler {fns:<5} {task:<6} EM={em:.3f} (n={len(rows)})", flush=True)
    E = ck.model.lm_head_weight.detach().float().cpu()
    nrm = E.norm(dim=1).numpy()
    out["emb"].append(dict(ckpt=lab, mean_norm_keysA=float(nrm[key_ids["A"]].mean()), mean_norm_keysB=float(nrm[key_ids["B"]].mean()),
                           mean_norm_all=float(nrm.mean()), n_keysA=int(len(key_ids["A"])), n_keysB=int(len(key_ids["B"]))))
    print("   emb norms", out["emb"][-1], flush=True)
    del ck; torch.cuda.empty_cache()
(R / f"laptop_ext_s{sd}_namespace_factors.json").write_text(json.dumps(out, indent=1))
print("wrote")
import collections
agg = collections.defaultdict(list)
for r in out["rows"]:
    agg[(r["ckpt"].split()[-1], r["words"], r["filler"], r["task"])].append(r["EM_tf"])
for k in sorted(agg): print("MEAN step=%s words=%s filler=%-5s %-6s" % k, "n=%d mean=%.3f sd=%.3f" % (len(agg[k]), np.mean(agg[k]), np.std(agg[k])))
