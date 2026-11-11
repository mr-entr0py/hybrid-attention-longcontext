"""Agent-6 probe 2 (report-only; writes only to .audit/agent6-schedule/). Run under scripts/repro/gpu_lock.py.

Measures, with CARLA co-resident (whatever its load is at run time; per-process SM is logged by an
`nvidia-smi pmon -o DT` side-car so foreign SM can be attributed per measurement window):
  T1  train step (bench_train_step.py recipe: fwd bf16 autocast + chunked CE/z + bwd + clip + fused AdamW) on
      (a) the bench's packed 4K batch (2x4096), (b) the bench's packed 32K batch, (c) a single-document 32K batch
      (cu_seqlens [0, 32768]); D384 C0/C3/C12 and D512 C0/C12 (D512 32K only: Stage A does not fit 0.35).
  T2  eval calls: prefill at the frozen batch size, then one score_continuation call (4 candidate tokens) and one
      extend (6 prompt tokens), eager, as the production scorer issues them; D384 C0/C3/C12 at T in {4K (B=8), 32K (B=1)}.
Memory capped at fraction 0.35. Timing does not depend on weights (random init, seed index 0).
"""
import importlib.util, json, os, subprocess, sys, time
REPO = "/home/sedrica-lynx/Downloads/outeropt"
OUTD = os.path.join(REPO, ".audit/agent6-schedule")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(OUTD, "triton"))
os.environ.setdefault("FLA_FLASH_QLA", "0"); os.environ.setdefault("FLA_CACHE_RESULTS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); os.environ.setdefault("PYTHONHASHSEED", "0")
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np, torch
FRAC = float(os.environ.get("PROBE_FRAC", "0.35"))
torch.cuda.set_per_process_memory_fraction(FRAC)
spec = importlib.util.spec_from_file_location("btsmod", os.path.join(REPO, "scripts/bench_train_step.py"))
bts = importlib.util.module_from_spec(spec); spec.loader.exec_module(bts)   # guard only runs under __main__
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.loss import chunked_ce
OUT = os.path.join(OUTD, "probe2.jsonl")
dev = torch.device("cuda")
pm = subprocess.Popen(["nvidia-smi", "pmon", "-s", "um", "-d", "1", "-o", "DT"],
                      stdout=open(os.path.join(OUTD, "probe2_pmon.txt"), "w"), stderr=subprocess.STDOUT)

def emit(rec):
    rec["pid"] = os.getpid(); rec["wall_end"] = time.time()
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")

def single_doc_batch(T, vocab, device):
    rng = np.random.default_rng(12345)
    ids = rng.integers(8, vocab, T); labels = np.roll(ids, -1); labels[-1] = 1
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return (t(ids, torch.long)[None], t(np.array([0, T]), torch.int32), t(np.arange(T), torch.long)[None],
            t(labels, torch.long)[None], T, 1)

def train_cell(scale, cfg_name, shape, kind, steps, warmup):
    sh = bts.SHAPES[shape]; cfg = ModelConfig.for_scale(scale, cfg_name)
    rec = dict(test="T1", scale=scale, config=cfg_name, shape=shape, batch=kind)
    model = opt = None
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, seed_index=0).to(dev); model.train(); model.set_activation_checkpointing(sh["ckpt"])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
        if kind == "packed":
            ids, cu, pos, labels, max_len, n_docs = bts.packed_batch(sh["seq_len"], sh["n_seq"], cfg.vocab_size, dev, 1)
        else:
            ids, cu, pos, labels, max_len, n_docs = single_doc_batch(sh["seq_len"] * sh["n_seq"], cfg.vocab_size, dev)
        rec.update(n_docs=n_docs, max_doc=max_len)
        times = []; t_start = None
        for i in range(warmup + steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if i == warmup: t_start = time.time()
            with torch.autocast("cuda", torch.bfloat16):
                h = model(ids, cu, pos)
                loss, z, n = chunked_ce(h, model.lm_head_weight, labels, chunk=4096)
            (loss + z).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= warmup: times.append(time.perf_counter() - t0)
        N = ids.shape[1]
        rec.update(status="ok", wall_start=t_start, step_s_min=min(times), step_s_median=float(np.median(times)),
                   tok_per_s_median=N / float(np.median(times)), tok_per_s_best=N / min(times),
                   peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    except torch.OutOfMemoryError as e:
        rec.update(status="oom", error=str(e).splitlines()[0][:200])
    finally:
        del model, opt; torch.cuda.empty_cache()
    emit(rec)

def sync_t(fn, n):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

def eval_cell(scale, cfg_name, T, B):
    rec = dict(test="T2", scale=scale, config=cfg_name, T=T, B=B)
    try:
        model = build_model(ModelConfig.for_scale(scale, cfg_name), 0).to(dev).eval()
        ids = torch.randint(10, 16000, (B, T), device=dev)
        cand = torch.randint(10, 16000, (B, 4), device=dev); prm = torch.randint(10, 16000, (B, 6), device=dev)
        gold = torch.randint(10, 16000, (B, 10), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, cache = model.prefill(ids); torch.cuda.synchronize()
            rec["wall_start"] = time.time()
            rec["prefill_ms"] = 1e3 * min(sync_t(lambda: model.prefill(ids), 1) for _ in range(3))
            _, cache = model.prefill(ids)
            model.score_continuation(cache, cand)
            rec["cand_call_ms"] = 1e3 * min(sync_t(lambda: model.score_continuation(cache, cand), 10) for _ in range(3))
            rec["gold_call_ms"] = 1e3 * min(sync_t(lambda: model.score_continuation(cache, gold), 10) for _ in range(3))
            rec["extend6_ms"] = 1e3 * min(sync_t(lambda: model.extend(prm, cache), 10) for _ in range(3))
        rec["max_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        rec["status"] = "ok"
        del cache, model
    except torch.OutOfMemoryError as e:
        rec.update(status="oom", error=str(e).splitlines()[0][:200])
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    emit(rec)

emit(dict(test="start", frac=FRAC, torch=torch.__version__, triton_cache=os.environ["TRITON_CACHE_DIR"]))
for cfg in ("C0", "C3", "C12"):
    train_cell("D384", cfg, "4k", "packed", 16, 4)
    train_cell("D384", cfg, "32k", "packed", 8, 2)
    train_cell("D384", cfg, "32k", "single", 8, 2)
for cfg in ("C0", "C12"):
    train_cell("D512", cfg, "32k", "packed", 6, 2)
    train_cell("D512", cfg, "32k", "single", 6, 2)
for cfg in ("C0", "C3", "C12"):
    eval_cell("D384", cfg, 4096, 8)
    eval_cell("D384", cfg, 32768, 1)
emit(dict(test="end"))
pm.terminate()
