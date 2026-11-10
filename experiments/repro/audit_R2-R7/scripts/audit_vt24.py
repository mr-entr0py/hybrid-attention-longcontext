"""Auditor's re-run of VT-24 on the three files with misses plus two clean controls (CPU only).

Uses the frozen oracle definition (InductionLM(16384, order=6, p_hit=0.9) through score_file),
with the CURRENT src/hybridlc (S0 has edited eval/ since the R3 run), writing to the scratchpad.
"""
import json
import math
import sys
from pathlib import Path

from hybridlc.eval.devtools import InductionLM
from hybridlc.eval.scorer import score_file

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
FILES = [
    "experiments/items/test/MS/R1-T4_D3_32K.npz",
    "experiments/items/dev/DG/R-load_N1024_16K.npz",
    "experiments/items/dev/FD/R-load_N1024_16K.npz",
    "experiments/items/test/CS/R1_D3_32K.npz",          # control expected clean
    "experiments/items/test/MS/M-hop_h5_32K.npz",       # negative control
]
target = 4 * math.log(0.9)
for f in FILES:
    p = Path("/home/sedrica-lynx/Downloads/outeropt") / f
    if not p.exists():
        print(json.dumps({"file": f, "missing": True}))
        continue
    lm = InductionLM(16384, order=6, p_hit=0.9)
    out = OUT / (p.parent.parent.name + "_" + p.parent.name + "_" + p.stem + ".jsonl")
    res = score_file(lm, p, out, score_candidates=False, autocast=False)
    rows = [json.loads(x) for x in Path(res).read_text().splitlines()]
    rows = [r for r in rows if "EM_tf" in r]
    miss = [(r["context_id"], r["query_idx"]) for r in rows
            if not (bool(r["EM_tf"]) and abs(float(r["logp_gold"]) - target) <= 1e-4)]
    em = sum(bool(r["EM_tf"]) for r in rows) / len(rows)
    print(json.dumps({"file": f, "n_queries": len(rows), "n_miss": len(miss), "em_mean": em, "misses": miss[:10]}))
