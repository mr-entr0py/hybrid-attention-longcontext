"""R7 driver: LAMBADA on published checkpoints through OUR scorer (reproduction_plan.md §2.7).

    python scripts/repro/r7_run.py --model pythia-160m --variant openai --bos default --precision fp32

Pipeline: pinned HF files (r7_download.py) -> item file (lambada_items.py, lm-eval
``_encode_pair`` split) -> ``hybridlc.eval.scorer.score_file(ForwardOnlyAdapter(HFLogitsWrapper(m)))``
-> acc = mean(EM_tf) (= lm-eval ``is_greedy``). Writes, under experiments/repro/R7/:
  items/<tok>_<variant>_bos-<bos>.npz            (frozen item file; reused if identical)
  scores/<run>.jsonl + <run>_meta.json            (scorer output, never overwritten)
  summary/<run>.json                               (acc, SE, ppl, pass check, metadata)
Skips a run whose summary exists (resumable).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "experiments/repro/R7"
CACHE = OUT / "hf_cache"

MODELS = {
    "pythia-160m": dict(repo="EleutherAI/pythia-160m", rev="50f5173d932e8e61f858120bcb800b97af589f46", loader="pythia", target=0.328, tol=0.015, tok="neox"),
    "pythia-70m": dict(repo="EleutherAI/pythia-70m", rev="a39f36b100fe8a5377810d56c3f4789b9c53ac42", loader="pythia", target=0.185, tol=0.015, tok="neox"),
    "gdn340m": dict(repo="m-a-p/340M-20B-GatedDeltaNet-pure-baseline", rev="14661a98768e1f5d5493ff7b18f2d6809717a4c8",
                    loader="fla_gdn", target=0.348, tol=0.02,
                    tok="wang-llama"),
}
DATA = {
    "openai": ("EleutherAI/lambada_openai", "data/lambada_test.jsonl"),
    "standard": ("cimec/lambada", "plain_text/test-00000-of-00001.parquet"),
}


def snap(repo: str, kind: str = "models", rev: str | None = None) -> Path:
    """Pinned snapshot dir; ``rev`` defaults to the pin in MODELS (other revisions may share the cache)."""
    if rev is None and kind == "models":
        rev = next(v["rev"] for v in MODELS.values() if v["repo"] == repo)
    hits = glob.glob(str(CACHE / f"{kind}--{repo.replace('/', '--')}" / "snapshots" / (rev or "*")))
    if len(hits) != 1:
        raise SystemExit(f"expected one pinned snapshot for {repo}, got {hits}; run r7_download.py")
    return Path(hits[0])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def kernel_events(since: str) -> list[str]:
    try:
        r = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-o", "short-iso"],
                           capture_output=True, text=True, timeout=30)
        return [ln for ln in r.stdout.splitlines() if any(t in ln for t in ("Xid", "NVRM", "nvidia", "Out of memory"))]
    except Exception as e:  # noqa: BLE001
        return [f"journalctl failed: {e!r}"]


def smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,"
                               "power.draw", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)


def build_item_file(tok_key: str, tok, variant: str, bos: str) -> tuple[Path, dict]:
    from hybridlc.eval.items import items_to_members, save_items  # noqa: F401
    from lambada_items import build_items, read_texts

    repo, fn = DATA[variant]
    texts = read_texts(snap(repo, "datasets") / fn)
    items, stats = build_items(tok, texts, bos)
    cell = dict(task="LAMBADA", variant=f"lambada_{variant}", tokenizer=tok_key, bos=bos, n=len(items),
                T=int(stats["ctx_len_max"] + max(stats["cont_len_hist"])), source=f"{repo}/{fn}",
                definition="lm-eval@d6de8164 lambada_*.yaml + api/model.py::_encode_pair")
    path = OUT / "items" / f"{tok_key}_{variant}_bos-{bos}.npz"
    if path.exists():
        tmp = path.with_name(path.stem + ".check.npz")
        tmp.unlink(missing_ok=True)
        save_items(tmp, items, cell)
        same = sha256(tmp) == sha256(path)
        tmp.unlink()
        if not same:
            raise SystemExit(f"{path} exists but differs from a rebuild (item files are frozen)")
    else:
        save_items(path, items, cell)
        # human-inspection dump of the first 20 items (failure tree step 1 material)
        dump = []
        for it in items[:20]:
            q = it["queries"][0]
            ctx = list(it["context_ids"]) + list(q["prompt_ids"])
            dump.append(dict(context_id=it["meta"]["context_id"], ctx_tail=tok.decode(ctx[-12:]),
                             ctx_head_ids=[int(x) for x in ctx[:3]], answer_ids=[int(x) for x in q["answer_ids"]],
                             answer_tokens=tok.convert_ids_to_tokens([int(x) for x in q["answer_ids"]]),
                             target_text=" " + texts[it["meta"]["context_id"]].split(" ")[-1]))
        (path.with_suffix(".inspect.json")).write_text(json.dumps(dump, indent=1, ensure_ascii=False) + "\n")
    stats["item_sha256"] = sha256(path)
    return path, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--variant", choices=sorted(DATA), default="openai")
    ap.add_argument("--bos", choices=["default", "on", "off"], default="default")
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16w"], default="fp32",
                    help="fp32: fp32 weights, no autocast (pre-registered); bf16: scorer's bf16 autocast; "
                         "fp16w: fp16 weights + fp16 compute, no autocast (diagnostic: the published-table setting)")
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N items (never a result)")
    ap.add_argument("--mem-frac", type=float, default=0.44)
    a = ap.parse_args()

    run = f"{a.model}_{a.variant}_bos-{a.bos}_{a.precision}" + (f"_limit{a.limit}" if a.limit else "")
    summ = OUT / "summary" / f"{run}.json"
    if summ.exists():
        print("exists, skipping:", summ)
        return
    t_start = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    smi_pre = smi()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    from transformers import AutoTokenizer

    from hf_adapter import load_fla_gdn, load_pythia
    from hybridlc.eval.adapter import ForwardOnlyAdapter
    from hybridlc.eval.items import load_items, save_items
    from hybridlc.eval.scorer import score_file
    from hybridlc.utils.provenance import env_info

    spec = MODELS[a.model]
    mdir = snap(spec["repo"])
    tok = AutoTokenizer.from_pretrained(str(mdir))
    item_path, istats = build_item_file(spec["tok"], tok, a.variant, a.bos)
    if a.limit:
        items, cell = load_items(item_path)
        item_path = OUT / "items" / f"_debug_{spec['tok']}_{a.variant}_bos-{a.bos}_limit{a.limit}.npz"
        item_path.unlink(missing_ok=True)
        save_items(item_path, items[:a.limit], dict(cell, n=a.limit, debug_limit=a.limit))

    t0 = time.time()
    wrapper, load_rep = (load_pythia if spec["loader"] == "pythia" else load_fla_gdn)(mdir)
    if a.precision == "fp16w":
        wrapper.half()
    wrapper.eval()
    model = ForwardOnlyAdapter(wrapper)
    t_load = time.time() - t0
    # batch_size=1: the scorer requires equal answer lengths within a batch and groups only by
    # context length; LAMBADA answers vary (1-3 tokens). ForwardOnlyAdapter recomputes per row
    # anyway, so this changes no number (recorded by the scorer as batch_size_deviation).
    out = score_file(model, item_path, OUT / "scores" / f"{run}.jsonl", batch_size=1, score_candidates=False,
                     autocast=(a.precision == "bf16"),
                     extra_meta=dict(r7_run=run, r7_model=a.model, r7_repo=spec["repo"], r7_snapshot=mdir.name))
    t_score = time.time() - t0 - t_load

    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    em = np.array([r["EM_tf"] for r in rows], dtype=float)
    lp = np.array([r["logp_gold"] for r in rows], dtype=float)
    n = len(em)
    acc = float(em.mean())
    se = float(math.sqrt(acc * (1 - acc) / (n - 1))) if n > 1 else float("nan")  # lm-eval mean_stderr
    ppl = float(math.exp(-lp.mean()))
    lo, hi = spec["target"] - spec["tol"], spec["target"] + spec["tol"]
    primary = a.variant == "openai" and a.bos == "default" and a.precision == "fp32" and not a.limit
    wang_std_crit = a.model == "gdn340m" and a.variant == "standard" and a.bos == "default" and a.precision == "fp32"
    res = dict(
        run=run, model=a.model, repo=spec["repo"], snapshot=mdir.name, variant=a.variant, bos=a.bos,
        precision=a.precision, n=n, acc=acc, acc_se=se, ppl=ppl, n_correct=int(em.sum()),
        target=spec["target"], tol=spec["tol"], interval=[round(lo, 6), round(hi, 6)],
        within_interval=bool(lo <= acc <= hi),
        role=("PRIMARY (pass/fail)" if primary else
              "criterion alternative (Wang lambada_standard)" if wang_std_crit and not a.limit else
              "debug (not a result)" if a.limit else "sensitivity (not a criterion)"),
        item_file=str(item_path.relative_to(ROOT)), item_stats=istats, scores_jsonl=str(out.relative_to(ROOT)),
        load_report={k: v for k, v in load_rep.items() if k not in ("missing", "unexpected")},
        load_missing=load_rep.get("missing"), load_unexpected=load_rep.get("unexpected"),
        gpu_oom_retries=len(wrapper.oom_events), gpu_oom_events=wrapper.oom_events[:200],
        wall_load_s=round(t_load, 2), wall_score_s=round(t_score, 2),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), seed="n/a (deterministic eval; torch.manual_seed(0))",
        max_mem_alloc_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), mem_frac=a.mem_frac,
        started=t_start, finished=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        nvidia_smi_pre=smi_pre, nvidia_smi_post=smi(), kernel_events_during_run=kernel_events(t_start),
        env=env_info(), command=" ".join([sys.executable] + sys.argv),
    )
    summ.parent.mkdir(parents=True, exist_ok=True)
    summ.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("run", "n", "acc", "acc_se", "ppl", "interval", "within_interval", "role",
                                          "wall_score_s")}))


if __name__ == "__main__":
    main()
