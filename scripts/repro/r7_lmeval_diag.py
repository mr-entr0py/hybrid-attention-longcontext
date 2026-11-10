"""R7 failure-tree step 3 (DIAGNOSTIC ONLY; run in .venv-lmeval, never in .venv).

Runs lm-eval @d6de8164 (``HFLM``, zero-shot, fp32, batch 1, tokenizer-default special tokens)
on a pinned local snapshot and dumps per-item ``(loglikelihood, is_greedy)`` so that it can be
compared item-by-item with our scorer's JSONL (scripts/repro/r7_compare.py).

    .venv-lmeval/bin/python -I scripts/repro/r7_lmeval_diag.py --snapshot <dir> --task lambada_openai --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--task", default="lambada_openai")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=1)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.44)
    torch.set_float32_matmul_precision("highest")
    import lm_eval
    import transformers
    from lm_eval.models.huggingface import HFLM

    t0 = time.time()
    lm = HFLM(pretrained=a.snapshot, dtype=a.dtype, batch_size=a.batch_size, device="cuda")
    res = lm_eval.simple_evaluate(model=lm, tasks=[a.task], log_samples=True, limit=a.limit, num_fewshot=0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in sorted(res["samples"][a.task], key=lambda s: s["doc_id"]):
            ll, greedy = s["resps"][0][0]
            f.write(json.dumps(dict(doc_id=s["doc_id"], text=s["doc"]["text"], ll=float(ll), is_greedy=bool(greedy),
                                    arguments=s["arguments"][0] if s.get("arguments") else None)) + "\n")
    summ = dict(task=a.task, snapshot=a.snapshot, dtype=a.dtype, batch_size=a.batch_size, results=res["results"][a.task], lm_eval=lm_eval.__version__,
                transformers=transformers.__version__, torch=torch.__version__, wall_s=time.time() - t0,
                config={k: str(v) for k, v in res.get("config", {}).items()},
                add_bos_token=getattr(lm, "add_bos_token", None))
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    print(json.dumps(summ["results"], default=str))


if __name__ == "__main__":
    main()
