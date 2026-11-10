"""VT-24 post-verdict diagnosis (report-only; the frozen verdict is not re-computed).

For every miss in a VT-24 run, record the colliding n-gram occurrence that the order-6 oracle copied
from, the text around both occurrences, and whether InductionLM at orders 8/10/12 answers the query.
    python scripts/repro/r3_vt24_diagnose.py <run_dir>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_common as RC
from hybridlc.eval.items import load_items
from hybridlc.eval.devtools import InductionLM
from hybridlc.data.tokenizer import load_tokenizer

run = Path(sys.argv[1]); tok = load_tokenizer(); out = []
for line in open(run / "vt24_files.jsonl"):
    r = json.loads(line)
    if r["pass"] or r["role"] != "retrieval":
        continue
    items, cell = load_items(RC.REPO / r["path"])
    for m in r["misses_first20"]:
        it = items[m["context_id"]]; q = it["queries"][m["query_idx"]]
        ctx = np.asarray(it["context_ids"]); full = np.concatenate([ctx, q["prompt_ids"], q["answer_ids"]])
        a0 = len(ctx) + len(q["prompt_ids"])
        j = next(i for i in range(4) if m["argmax_ids"][i] != m["gold_ids"][i])
        t = a0 + j - 1; g = full[t - 5:t + 1]
        occ = [i for i in range(5, t) if np.array_equal(full[i - 5:i + 1], g)]
        rec = {"file": r["path"], "context_id": m["context_id"], "query_idx": m["query_idx"], "key": q["meta"]["key"],
               "value": q["meta"]["value"], "wrong_digit_index": j, "gram": tok.decode(g),
               "gram_occurrences": occ, "copied_from_text": tok.decode(full[max(0, occ[-1] - 12):occ[-1] + 4]),
               "needle_text": tok.decode(full[q["meta"]["needle_start"][-1]:q["meta"]["needle_end"][-1]]),
               "collision_inside_needles": any(s <= occ[-1] < e for s, e in zip(*(lambda mm: (mm.get("needle_start", []), mm.get("needle_end", [])))(q["meta"]))),
               "higher_order_em": {}}
        for order in (8, 10, 12):
            lm = InductionLM(16384, order=order, p_hit=0.9)
            pred = lm._predict(full)[a0 - 1:a0 + 3]
            rec["higher_order_em"][order] = bool((pred == np.asarray(q["answer_ids"])).all())
        out.append(rec)
(run / "diagnosis_misses.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
