#!/usr/bin/env python
"""R2(c) 32K kernel profile (reproduction plan §2.2): longest single CUDA kernel in

* one Stage-B 1 x 32K training step: forward (bf16 autocast) + chunked CE/z-loss + backward with
  per-block activation checkpointing + grad clip + fused AdamW step (the production micro-step, same
  code path as ``scripts/bench_train_step.py`` / ``training/trainer.py``);
* one 32K prefill: ``HybridLM.prefill(ids[1, 32768])`` under bf16 autocast (frozen batch size 1 at
  32K, eval/scorer path).

Kernel durations are device-side CUPTI kernel records from ``torch.profiler`` (chrome-trace events with
``cat == "kernel"``), not CPU op times. Each profiled call is synchronised before and after.

Phases per (scale, config):
  ``train_cold``   first training step in the process (includes Triton JIT/autotune runs), single doc;
  ``train_warm``   after 2 untraced steps: single 32,768-token document (worst case for attention;
                   PG19 32K windows are 60 % of Stage-B sequences);
  ``train_warm_stageB``  real Stage-B micro-batch from ``StageStream('B', 0, 32768, 1, 4).get(0, 0)``;
  ``prefill_cold`` first prefill call; ``prefill_warm`` after one untraced prefill.

Pre-registered reading (results_R2.md §0): the criterion (< 1 s) applies to the warm phases; the cold
phases are reported as a risk note.
"""
from __future__ import annotations

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hybridlc.utils.gpu import claim, set_production_env  # noqa: E402

set_production_env()

from hybridlc.models.build import build_model  # noqa: E402
from hybridlc.models.config import ModelConfig  # noqa: E402
from hybridlc.models.loss import chunked_ce  # noqa: E402
from hybridlc.utils.provenance import env_info  # noqa: E402

T = 32768


def single_doc_batch(vocab: int, device, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = torch.as_tensor(rng.integers(8, vocab, T), dtype=torch.long, device=device)[None]
    labels = torch.roll(ids, -1, dims=1)
    labels[0, -1] = 1
    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    pos = torch.arange(T, device=device)[None]
    return dict(input_ids=ids, labels=labels, cu_seqlens=cu, position_ids=pos, max_seqlen=T, n_docs=1)


def stage_b_batch(device):
    from hybridlc.data.loader import StageStream
    s = StageStream("B", 0, seq_len=T, micro_batch=1, accum=4)
    b = s.get(0, 0)
    cu = b["cu_seqlens"]
    lens = (cu[1:] - cu[:-1])
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()
           if k in ("input_ids", "labels", "cu_seqlens", "position_ids", "max_seqlen")}
    out["n_docs"] = int(len(lens))
    out["max_doc"] = int(lens.max())
    out["sources"] = b.get("sources")
    return out


def kernel_stats(trace_path: Path, top: int = 10) -> dict:
    with gzip.open(trace_path, "rt") if trace_path.suffix == ".gz" else open(trace_path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    k = [e for e in evs if e.get("cat") == "kernel" and "dur" in e]
    mc = [e for e in evs if e.get("cat") in ("gpu_memcpy", "gpu_memset") and "dur" in e]
    k.sort(key=lambda e: -e["dur"])
    return {
        "n_kernels": len(k),
        "max_kernel_us": k[0]["dur"] if k else None,
        "max_kernel_name": k[0]["name"][:300] if k else None,
        "sum_kernel_us": float(sum(e["dur"] for e in k)),
        "top_kernels": [{"us": e["dur"], "name": e["name"][:200],
                         "grid": e.get("args", {}).get("grid"), "block": e.get("args", {}).get("block")} for e in k[:top]],
        "max_memcpy_memset_us": max((e["dur"] for e in mc), default=None),
    }


def profiled(fn, trace_path: Path) -> dict:
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    tmp = trace_path.with_suffix("")  # .json
    prof.export_chrome_trace(str(tmp))
    with open(tmp, "rb") as fi, gzip.open(trace_path, "wb") as fo:
        fo.write(fi.read())
    tmp.unlink()
    st = kernel_stats(trace_path)
    st["wall_s_profiled"] = wall
    st["trace"] = str(trace_path.relative_to(REPO))
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", required=True, choices=["D384", "D512"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO / "experiments" / "repro" / "r2" / "profile")
    ap.add_argument("--mem-fraction", type=float, default=0.44)
    ap.add_argument("--phases", nargs="+",
                    default=["train_cold", "train_warm", "train_warm_stageB", "prefill_cold", "prefill_warm"])
    args = ap.parse_args()
    device = claim(args.mem_fraction)
    assert device.type == "cuda", "needs CUDA"
    torch.manual_seed(0)
    name = f"{args.scale}-{args.config}"
    out_dir = args.out_dir.resolve()
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    res: dict = {"cell": name, "scale": args.scale, "config": args.config, "T": T, "env": env_info(),
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": {}}
    cfg = ModelConfig.for_scale(args.scale, args.config)
    res["pattern"] = cfg.pattern
    model = build_model(cfg, seed_index=0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
    single = single_doc_batch(cfg.vocab_size, device)

    def train_step(b):
        model.train()
        model.set_activation_checkpointing(True)
        with torch.autocast("cuda", torch.bfloat16):
            h = model(b["input_ids"], b["cu_seqlens"], b["position_ids"], max_seqlen=b["max_seqlen"])
            loss, z, _ = chunked_ce(h, model.lm_head_weight, b["labels"], chunk=4096)
        (loss + z).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss)

    def prefill():
        model.eval()
        with torch.autocast("cuda", torch.bfloat16):
            logits, cache = model.prefill(single["input_ids"])
        return logits

    def save():
        res["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        warm = [p for k, p in res["phases"].items() if "warm" in k and p.get("max_kernel_us") is not None]
        res["max_kernel_us_warm"] = max((p["max_kernel_us"] for p in warm), default=None)
        cold = [p for k, p in res["phases"].items() if "cold" in k and p.get("max_kernel_us") is not None]
        res["max_kernel_us_cold"] = max((p["max_kernel_us"] for p in cold), default=None)
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))

    tr = lambda ph: out_dir / "traces" / f"{name}.{ph}.json.gz"  # noqa: E731
    for ph in args.phases:
        t0 = time.time()
        if ph == "train_cold":
            st = profiled(lambda: train_step(single), tr(ph))
            st["layout"] = {"n_docs": 1, "max_doc": T}
        elif ph == "train_warm":
            for _ in range(2):
                train_step(single)
            st = profiled(lambda: train_step(single), tr(ph))
            st["layout"] = {"n_docs": 1, "max_doc": T}
        elif ph == "train_warm_stageB":
            b = stage_b_batch(device)
            train_step(b)  # untraced warm-up for this document layout (new varlen shapes)
            st = profiled(lambda: train_step(b), tr(ph))
            st["layout"] = {"n_docs": b["n_docs"], "max_doc": b["max_doc"], "sources": b["sources"],
                            "source": "StageStream('B', data_seed=0, 32768, 1, 4).get(0, 0)"}
        elif ph == "prefill_cold":
            st = profiled(prefill, tr(ph))
        elif ph == "prefill_warm":
            prefill()
            st = profiled(prefill, tr(ph))
        else:
            raise SystemExit(f"unknown phase {ph}")
        st["phase_wall_s"] = time.time() - t0
        res["phases"][ph] = st
        print(f"[r2c] {name} {ph:18s} max kernel {st['max_kernel_us']/1e3:9.2f} ms  ({st['max_kernel_name'][:70]})  "
              f"n={st['n_kernels']} wall={st['wall_s_profiled']:.2f}s", flush=True)
        save()
    print(f"[r2c] {name} done: warm max {res['max_kernel_us_warm']/1e3:.2f} ms, cold max "
          f"{(res['max_kernel_us_cold'] or 0)/1e3:.2f} ms -> {out_dir / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
