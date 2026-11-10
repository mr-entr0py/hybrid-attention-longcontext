"""Agent-6 eval-cost probe (report-only diagnostic; writes only to .audit/agent6-schedule/).

Times, on random-init D384 models (timing does not depend on weights), with bf16 autocast:
  (1) prefill B=1 at T in {4K,16K,32K};
  (2) one candidate call = model.score_continuation(cache, 4 tokens);
  (3) the real scorer path (scorer.score_items) on 3 contexts of DC R1_D3 and R-load_N32 cells.
Memory is capped low (fraction 0.10) to protect the co-tenant CARLA lane; cells that do not fit are skipped.
"""
import json, os, sys, time, subprocess, traceback
REPO = "/home/sedrica-lynx/Downloads/outeropt"
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PROBE_FRAC", "0.10")))
from hybridlc.models.config import ModelConfig
from hybridlc.models.build import build_model
from hybridlc.eval import scorer
from hybridlc.eval.items import load_items

OUT = os.path.join(REPO, ".audit/agent6-schedule/eval_cost_probe.jsonl")
dev = torch.device("cuda")
def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa
        return str(e)
def sync_time(fn, n=1):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
def emit(rec):
    rec["ts"] = time.time(); rec["smi"] = smi()
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")

plan = [("C0", (4096, 16384, 32768)), ("C3", (4096, 16384, 32768)), ("C12", (4096, 16384))]
Tname = {4096: "4K", 16384: "16K", 32768: "32K"}
emit(dict(kind="start", torch=torch.__version__, frac=os.environ.get("PROBE_FRAC", "0.10")))
for cfgname, Ts in plan:
    model = build_model(ModelConfig.for_scale("D384", cfgname), 0).to(dev).eval()
    for T in Ts:
        rec = dict(kind="cell", config=cfgname, T=T)
        try:
            ids = torch.randint(10, 16000, (1, T), device=dev)
            cont = torch.randint(10, 16000, (1, 4), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, cache = model.prefill(ids)             # warm-up
                rec["prefill_ms"] = 1e3 * min(sync_time(lambda: model.prefill(ids)) for _ in range(3))
                _, cache = model.prefill(ids)
                model.score_continuation(cache, cont)    # warm-up
                rec["cand_call_ms"] = 1e3 * min(sync_time(lambda: model.score_continuation(cache, cont), n=5) for _ in range(3))
                _, pc = model.extend(cont[:, :3], cache)
                rec["extend3_ms"] = 1e3 * min(sync_time(lambda: model.extend(cont[:, :3], cache), n=5) for _ in range(3))
                del cache, pc
            rec["max_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
            for task in ("R1_D3", "R-load_N32"):
                p = os.path.join(REPO, f"experiments/items/dev/DC/{task}_{Tname[T]}.npz")
                items, _ = load_items(p)
                sub = items[:3]
                scorer.score_items(model, sub[:1], 1, task, True, autocast=True)  # warm-up (shapes)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                rows = scorer.score_items(model, sub, 1, task, True, autocast=True)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                rec[f"{task}_s_per_ctx"] = dt / len(sub)
                rec[f"{task}_n_cand_mean"] = float(np.mean([r["n_cand"] for r in rows]))
        except Exception as e:  # noqa
            rec["error"] = repr(e)[:300]
            traceback.print_exc()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        emit(rec)
    del model; torch.cuda.empty_cache()
emit(dict(kind="end"))
