"""Auditor's independent LAMBADA (openai) scorer for the pinned Pythia checkpoints.

Plain transformers + tokenizers; imports neither hybridlc nor lm-eval. Definition follows
lm-eval lambada_openai: context = text minus its last space-delimited word, target = ' ' + last word,
continuation tokens = enc(ctx + target)[len(enc(ctx)):] (lm-eval _encode_pair), acc = all greedy.
Usage: audit_lambada.py <snapshot_dir> <dtype: fp32|fp16> [limit]
"""
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.44)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DATA = ("/home/sedrica-lynx/Downloads/outeropt/experiments/repro/R7/hf_cache/datasets--EleutherAI--lambada_openai/"
        "snapshots/900124bf3b8235c6daf21033af9948b3f07346c4/data/lambada_test.jsonl")
snap, dt = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
dtype = {"fp32": torch.float32, "fp16": torch.float16}[dt]

tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, dtype=dtype).to(DEV).eval()
docs = [json.loads(x)["text"] for x in open(DATA)]
if limit:
    docs = docs[:limit]
n_ok, ll_sum, n_tok = 0, 0.0, 0
per_item = []
with torch.no_grad():
    for i, text in enumerate(docs):
        ctx, last = text.rsplit(" ", 1)
        cont = " " + last
        whole = tok(ctx + cont, add_special_tokens=False)["input_ids"]
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        t_ids = whole[len(c_ids):]
        inp = torch.tensor([whole[:-1]], device=DEV)
        logits = model(inp).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        pos = torch.arange(len(c_ids) - 1, len(whole) - 1, device=DEV)
        tgt = torch.tensor(t_ids, device=DEV)
        greedy = bool((logits[pos].argmax(-1) == tgt).all())
        ll = float(lp[pos, tgt].sum())
        n_ok += greedy
        ll_sum += ll
        per_item.append(greedy)
acc = n_ok / len(docs)
print(json.dumps({"snapshot": snap.split("/")[-3], "dtype": dt, "n": len(docs), "correct": n_ok, "acc": acc,
                  "se": math.sqrt(acc * (1 - acc) / (len(docs) - 1)), "ppl": math.exp(-ll_sum / len(docs))}))
json.dump(per_item, open(f"lambada_{snap.split('/')[-3]}_{dt}.json", "w"))
