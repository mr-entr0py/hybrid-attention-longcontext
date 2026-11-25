"""R6 diagnosis (non-gate, descriptive): is the dev-eval EM gap vs training CE a train/dev distribution shift?
Scores R6 checkpoints on the 4 primary 1K cells + C0 generated in namespace 'train' (what the model was trained on) and 'dev' (R6 eval)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path.home() / "Downloads" / "outeropt"; sys.path.insert(0, str(REPO / "src"))
from hybridlc.data.tokenizer import load_tokenizer
from hybridlc.eval.tokens import token_space
from hybridlc.eval.filler import StoreFiller
from hybridlc.eval.generators import generate
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.scorer import score_items
from hybridlc.integration.evaluate import load_checkpoint
from hybridlc.utils.gpu_lock import guard
from hybridlc.utils.gpu import claim
guard("r6-diag"); dev = claim(0.44)
tok = load_tokenizer(REPO / "data" / "tokenizer"); ts = token_space(tok)
CELLS = [("R1", {"D": 3}, 1024, None), ("R1", {"D": 15}, 1024, None), ("R-load", {"N": 8}, 1024, None),
         ("R-load", {"N": 32}, 1024, None), ("R1", {"D": 3}, None, "C0"), ("R-load", {"N": 8}, None, "C0")]
items = {}
for ns in ("train", "dev"):
    fill = StoreFiller(ts, ns, str(REPO / "data" / "tokens"))
    for task, lv, T, var in CELLS:
        lv2 = dict(lv, placement="nofill") if var == "C0" else dict(lv)
        rng = np.random.default_rng([abs(hash(0)) + 11, {"train": 1, "dev": 2}[ns], len(task), int(T or 0), len(str(lv)), len(var or "")])
        items[(ns, task, str(lv), T, var)] = generate(task, lv2, T, 200, rng, ns, tok, fill)
out = []
for lab, path in zip(sys.argv[1::2], sys.argv[2::2]):
    ck = load_checkpoint(path, dev, runs_root=None, canonical_bf16=True)
    for key, its in items.items():
        rows = score_items(ck.model, its, 8, "diag", False, autocast=True)
        em = float(np.mean([bool(r["EM_tf"]) for r in rows]))
        out.append(dict(ckpt=lab, ns=key[0], task=key[1], level=key[2], T=key[3], variant=key[4], EM_tf=em))
        print(f"{lab:>10} {key[0]:>5} {key[1]:>6} {key[2]:<12} T={key[3]} {key[4] or '':>3} EM={em:.3f}", flush=True)
Path(REPO / "experiments/repro/R6/diag_train_vs_dev.json").write_text(json.dumps(out, indent=1))
