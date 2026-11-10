"""LAMBADA item writer (reproduction_plan.md §2.7, R7), following lm-eval @d6de8164.

Definition (third_party/lm-evaluation-harness @d6de8164):
* ``tasks/lambada/lambada_openai.yaml`` / ``lambada_standard.yaml``:
  ``doc_to_text = text.split(' ')[:-1]|join(' ')``, ``doc_to_target = ' ' + text.split(' ')[-1]``;
  ``api/task.py`` L1370: loglikelihood request = ``(ctx, doc_to_target(doc))`` (no extra delimiter).
* ``api/model.py::_encode_pair`` (causal): trailing spaces of the context move to the
  continuation; ``whole = enc(ctx + cont)``, ``ctx_enc = enc(ctx)``, ``cont_enc = whole[len(ctx_enc):]``.
  The model sees ``ctx_enc + cont_enc``; ``acc = is_greedy`` over ``cont_enc``.
* ``models/huggingface.py::tok_encode``: with ``add_bos_token=None`` the tokenizer's default
  special tokens are used (GPT-NeoX: none; the Wang Llama tokenizer: ``<s>``).

Mapping onto our item format (``hybridlc.eval.items``): ``context_ids = ctx_enc[:-1]``,
one query with ``prompt_ids = [ctx_enc[-1]]`` (so len(prompt) = 1 >= 1, as the scorer requires)
and ``answer_ids = cont_enc``; candidates = [gold] only. The scorer's EM_tf over the answer
positions is then exactly lm-eval's ``is_greedy``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_texts(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [str(t) for t in pq.read_table(str(path), columns=["text"]).column("text").to_pylist()]
    raise ValueError(path)


def doc_split(text: str) -> tuple[str, str]:
    parts = text.split(" ")
    return " ".join(parts[:-1]), " " + parts[-1]


def encode_pair(tok, context: str, continuation: str, bos: str = "default") -> tuple[list[int], list[int], dict]:
    """lm-eval ``_encode_pair`` (causal). ``bos``: 'default' (tokenizer default), 'on', 'off'."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    def enc(s: str) -> list[int]:
        if bos == "default":
            return list(tok.encode(s))
        ids = list(tok.encode(s, add_special_tokens=False))
        if bos == "on":
            ids = [tok.bos_token_id] + ids
        return ids

    whole = enc(context + continuation)
    ctx = enc(context)
    cont = whole[len(ctx):]
    diag = dict(prefix_consistent=whole[:len(ctx)] == ctx, n_spaces_moved=n_spaces)
    return ctx, cont, diag


def build_items(tok, texts: list[str], bos: str = "default") -> tuple[list[dict], dict]:
    items, n_incons = [], 0
    for i, text in enumerate(texts):
        c, t = doc_split(text)
        ctx, cont, d = encode_pair(tok, c, t, bos)
        if len(ctx) < 2 or len(cont) < 1:
            raise ValueError(f"item {i}: ctx {len(ctx)} tokens, cont {len(cont)} tokens")
        n_incons += int(not d["prefix_consistent"])
        items.append(dict(
            context_ids=np.asarray(ctx[:-1], dtype=np.int64),
            queries=[dict(prompt_ids=np.asarray(ctx[-1:], dtype=np.int64), answer_ids=np.asarray(cont, dtype=np.int64),
                          candidates=[list(map(int, cont))], meta=dict(query_idx=0, cand_types=["gold"]))],
            meta=dict(context_id=i, task="LAMBADA", T=len(ctx) + len(cont), T_ctx=len(ctx) - 1,
                      prefix_consistent=d["prefix_consistent"], n_spaces_moved=d["n_spaces_moved"])))
    stats = dict(n=len(items), n_prefix_inconsistent=n_incons,
                 ctx_len_max=max(len(it["context_ids"]) + 1 for it in items),
                 cont_len_hist={int(k): int(v) for k, v in zip(*np.unique([len(it["queries"][0]["answer_ids"]) for it in items],
                                                                           return_counts=True))})
    return items, stats
