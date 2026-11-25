"""R6 replicate scoring (non-gate, descriptive): dev-namespace EM_tf on the two primary 1K cells for every seed final.pt + ext ckpts."""
import json, sys, glob
from pathlib import Path
import numpy as np
REPO = Path.home() / "Downloads" / "outeropt"; sys.path.insert(0, str(REPO / "src"))
from hybridlc.data.tokenizer import load_tokenizer
from hybridlc.eval.tokens import token_space
from hybridlc.eval.filler import StoreFiller
from hybridlc.eval.generators import generate
from hybridlc.eval.scorer import score_items
from hybridlc.integration.evaluate import load_checkpoint
from hybridlc.utils.gpu_lock import guard
from hybridlc.utils.gpu import claim
guard("r6-score-seeds"); dev = claim(0.44)
tok = load_tokenizer(REPO / "data" / "tokenizer"); ts = token_space(tok)
fill = StoreFiller(ts, "dev", str(REPO / "data" / "tokens"))
CELLS = [("R1", {"D": 3}), ("R-load", {"N": 8})]
items = {f"{t}{lv}": generate(t, dict(lv), 1024, 200, np.random.default_rng([7, i]), "dev", tok, fill) for i, (t, lv) in enumerate(CELLS)}
R = REPO / "experiments/repro/R6"
paths = sorted(glob.glob(str(R / "seeds/s*/ckpt/*/final.pt"))) + sorted(glob.glob(str(R / "ext/ckpt/*/milestones/step_*_bf16.pt"))) + sorted(glob.glob(str(R / "ext/ckpt/*/final.pt")))
out = []
for p in paths:
    ck = load_checkpoint(p, dev, runs_root=None, canonical_bf16=True)
    ems = {k: float(np.mean([bool(r["EM_tf"]) for r in score_items(ck.model, its, 8, "seed", False, autocast=True)])) for k, its in items.items()}
    kind = "GDN" if "R6-GDN" in p else "AA"
    out.append(dict(path=p.replace(str(R) + "/", ""), kind=kind, score=float(np.mean(list(ems.values()))), **ems))
    print(out[-1]["path"], kind, round(out[-1]["score"], 3), flush=True)
    del ck
(R / "replicate_scores.json").write_text(json.dumps(out, indent=1))
for kind in ("AA", "GDN"):
    s = [o["score"] for o in out if o["kind"] == kind and "seeds/" in o["path"]]
    if s: print(kind, "n=%d mean=%.3f sd=%.3f min=%.3f max=%.3f" % (len(s), np.mean(s), np.std(s), min(s), max(s)))
