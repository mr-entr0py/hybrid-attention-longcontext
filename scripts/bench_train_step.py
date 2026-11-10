#!/usr/bin/env python
"""Measured train-step memory and throughput for the production model code (shared GPU; indicative).

Shapes (protocol §6.3):
  * ``4k`` : micro-batch 2 × 4,096 flattened into one packed row (N = 8,192), no checkpointing;
  * ``32k``: micro-batch 1 × 32,768 with per-block activation checkpointing.
One step = forward (bf16 autocast) + chunked CE/z-loss + backward + fused AdamW step (grad clip 1.0),
i.e. one micro-batch of the real training loop.  Documents are packed with ``cu_seqlens``; their
lengths are drawn from the ``bench`` stream (entropy 0) as a mixture of short web-like documents and
one long document per sequence, so both varlen kernels see realistic boundaries.

Usage::

    python scripts/bench_train_step.py                       # D384/D512 × C0/C3/C12 × 4k/32k
    python scripts/bench_train_step.py --scales D384 --configs C3 --shapes 4k --steps 10

Writes one JSON line per (scale, config, shape) to ``--out`` (default
``runs/bench_train_step/<UTC timestamp>.jsonl``) and prints a table.  Every process caps itself at
``--mem-fraction`` (0.44 ≈ 7 GiB) of the GPU; a shape that OOMs is recorded as such.
"""

from __future__ import annotations

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[1] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridlc.models.build import build_model  # noqa: E402
from hybridlc.models.config import ModelConfig  # noqa: E402
from hybridlc.models.loss import chunked_ce  # noqa: E402
from hybridlc.utils.gpu import claim, set_production_env, snapshot  # noqa: E402
from hybridlc.utils.provenance import env_info  # noqa: E402
from hybridlc.utils.seeds import BENCH_ENTROPY, entropy_sequence  # noqa: E402

SHAPES = {"4k": dict(seq_len=4096, n_seq=2, ckpt=False), "32k": dict(seq_len=32768, n_seq=1, ckpt=True)}


def packed_batch(seq_len: int, n_seq: int, vocab: int, device, key: int):
    """Packed row of n_seq sequences, each split into documents (bench stream)."""
    rng = np.random.default_rng(entropy_sequence(BENCH_ENTROPY, key))
    lens: list[int] = []
    for _ in range(n_seq):
        left = seq_len
        long_doc = int(rng.integers(seq_len // 4, seq_len // 2))  # one long document
        parts = [long_doc]
        left -= long_doc
        while left > 0:
            n = int(min(left, max(16, rng.lognormal(6.3, 0.9))))  # median ≈ 545 tokens
            parts.append(n)
            left -= n
        rng.shuffle(parts)
        lens += parts
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    pos = np.concatenate([np.arange(n) for n in lens])
    N = seq_len * n_seq
    ids = rng.integers(8, vocab, N)
    labels = np.roll(ids, -1)
    labels[-1] = 1
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return (t(ids, torch.long)[None], t(cu, torch.int32), t(pos, torch.long)[None], t(labels, torch.long)[None],
            int(max(lens)), len(lens))


def bench_one(scale: str, cfg_name: str, shape: str, steps: int, warmup: int, device) -> dict:
    sh = SHAPES[shape]
    cfg = ModelConfig.for_scale(scale, cfg_name)
    rec = dict(scale=scale, config=cfg_name, pattern=cfg.pattern, shape=shape, micro=f"{sh['n_seq']}x{sh['seq_len']}",
               activation_checkpointing=sh["ckpt"])
    model = opt = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, seed_index=0).to(device)
        model.train()
        model.set_activation_checkpointing(sh["ckpt"])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
        ids, cu, pos, labels, max_len, n_docs = packed_batch(sh["seq_len"], sh["n_seq"], cfg.vocab_size, device, 1)
        rec.update(n_docs=n_docs, max_doc=max_len)
        times = []
        for i in range(warmup + steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", torch.bfloat16):
                h = model(ids, cu, pos)
                loss, z, n = chunked_ce(h, model.lm_head_weight, labels, chunk=4096)
            (loss + z).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
        N = ids.shape[1]
        rec.update(status="ok", step_s_min=min(times), step_s_median=float(np.median(times)),
                   tok_per_s_median=N / float(np.median(times)), tok_per_s_best=N / min(times),
                   peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, final_loss=loss.item())
    except torch.OutOfMemoryError as e:
        rec.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del model, opt
        torch.cuda.empty_cache()
    return rec


def wait_for_free(gib: float, timeout_s: float) -> float:
    """Poll ``cudaMemGetInfo`` until ≥ ``gib`` GiB are free (or timeout); returns the free GiB seen."""
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free >= gib or time.time() - t0 > timeout_s:
            return free
        time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", nargs="+", default=["D384", "D512"])
    ap.add_argument("--configs", nargs="+", default=["C0", "C3", "C12"])
    ap.add_argument("--shapes", nargs="+", default=["4k", "32k"], choices=list(SHAPES))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mem-fraction", type=float, default=0.44)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wait-free-gib", type=float, default=0.0,
                    help="before each shape, wait until the device has this much free memory (shared GPU)")
    ap.add_argument("--wait-timeout-s", type=float, default=1200.0)
    args = ap.parse_args()
    set_production_env()
    device = claim(args.mem_fraction)
    if device.type != "cuda":
        sys.exit("needs CUDA")
    out = args.out or REPO / "runs" / "bench_train_step" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = env_info()
    with out.open("a") as f:
        f.write(json.dumps({"_meta": env, "args": {k: str(v) for k, v in vars(args).items()}}) + "\n")
    print(f"{'scale':5} {'cfg':4} {'shape':4} {'status':6} {'step ms(med)':>12} {'tok/s(med)':>11} {'peak GiB':>8}")
    for scale in args.scales:
        for c in args.configs:
            for shape in args.shapes:
                free = wait_for_free(args.wait_free_gib, args.wait_timeout_s) if args.wait_free_gib else None
                rec = bench_one(scale, c, shape, args.steps, args.warmup, device)
                rec["free_gib_at_start"] = free
                rec["gpu"] = snapshot()
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if rec["status"] == "ok":
                    print(f"{scale:5} {c:4} {shape:4} ok     {rec['step_s_median']*1e3:12.1f} {rec['tok_per_s_median']:11.0f} "
                          f"{rec['peak_alloc_gib']:8.2f}", flush=True)
                else:
                    print(f"{scale:5} {c:4} {shape:4} {rec['status']}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
