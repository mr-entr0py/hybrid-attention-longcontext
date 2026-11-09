#!/usr/bin/env python
"""VT-19: warm the Triton cache for every training, evaluation and benchmark shape, then archive it.

Coverage (protocol §16 VT-19: ``IS_VARLEN`` true and false; every length):

* **mixers**: one config per mixer kind at both scales -- C0 (GDN-A), C3 (F_g + GDN-A), C12 (F_g),
  X-gc3 (F_gc), X-E3 (GDN-E), O-gla3 (GLA), O-nc3 (GDN no-conv) -- i.e. every head geometry used;
* **training** (varlen, fwd + bwd + chunked CE): Stage A 2 x 4K packed (no checkpointing), Stage B /
  TEF 1 x 32K packed with per-block checkpointing;
* **evaluation** (dense rows, ``IS_VARLEN`` false): one frozen-size batch of one item cell per distinct
  (task family, length) of the frozen item files -- prefill (chunk), prompt ``extend`` and candidate
  ``score_continuation`` (fused_recurrent) -- plus the Tier-0 varlen forwards (LM-1/2/4);
* **benchmarks** (§10.2): prefill B in {1, 8} x T in {1K, 4K, 16K, 32K} within the memory cap and
  decode steps at L in {1K, 4K, 16K, 32K}, B in {1, 8}.

Then writes a deterministic tarball (sorted members, mtime 0, uid/gid 0) of ``$TRITON_CACHE_DIR`` to
``--archive`` and its SHA-256 to ``<archive>.sha256`` and to the registry (one ``note`` row).

    python scripts/warm_triton.py [--scales D384 D512] [--skip-eval] [--archive runs/_artifacts/triton_cache.tar]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO / ".cache" / "triton"))

CONFIGS = ["C0", "C3", "C12", "X-gc3", "X-E3", "O-gla3", "O-nc3"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_shapes(model, device) -> None:
    import torch

    from hybridlc.models.loss import chunked_ce

    V = model.cfg.vocab_size
    for n_seq, T, ckpt in ((2, 4096, False), (1, 32768, True)):
        g = torch.Generator().manual_seed(T)
        lens = [T // 3, T // 5, T - T // 3 - T // 5 - 7, 7] * n_seq
        N = sum(lens)
        ids = torch.randint(8, V, (1, N), generator=g).to(device)
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        pos = torch.cat([torch.arange(n) for n in lens])[None].to(device)
        labels = torch.roll(ids, -1, 1)
        model.set_activation_checkpointing(ckpt)
        model.train()
        with torch.autocast("cuda", torch.bfloat16):
            h = model(ids, cu, pos, max_seqlen=max(lens))
            loss, z, _ = chunked_ce(h, model.lm_head_weight, labels)
        (loss + z).backward()
        model.zero_grad(set_to_none=True)
        model.set_activation_checkpointing(False)
        torch.cuda.synchronize()


def eval_cells(items_root: Path) -> list[Path]:
    """One item file per distinct (task-with-variant family, length tag)."""
    seen, out = set(), []
    for f in sorted(items_root.rglob("*.npz")):
        task, _level, t = f.stem.rsplit("_", 2)
        key = (task, t)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def eval_shapes(model, cells: list[Path], scratch: Path) -> None:
    import torch

    from hybridlc.eval.constants import frozen_batch_size
    from hybridlc.eval.items import load_items
    from hybridlc.integration.evaluate import score_cell

    model.eval()
    for f in cells:
        _, cell = load_items(f)
        T = cell.get("T") or 512
        n = frozen_batch_size(int(T)) if cell.get("task") not in ("LM-1", "LM-2", "LM-4") else 1
        out = scratch / f"{f.parent.name}_{f.stem}.jsonl"
        score_cell(model, f, out, n_contexts=n)
        out.unlink(missing_ok=True)
        out.with_name(out.stem + "_meta.json").unlink(missing_ok=True)
    torch.cuda.synchronize()


def bench_shapes(model, device, cap_gib: float = 6.0) -> None:
    import torch

    V = model.cfg.vocab_size
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for B in (1, 8):
            for T in (1024, 4096, 16384, 32768):
                if B * T > 65536:   # beyond the memory budget at our scale (§10.2: skip such shapes)
                    continue
                ids = torch.randint(8, V, (B, T), device=device)
                _, cache = model.prefill(ids)
                tok = torch.randint(8, V, (B, 1), device=device)
                for _ in range(2):
                    _, cache = model.decode_step(tok, cache)
                del cache
                torch.cuda.empty_cache()
    torch.cuda.synchronize()


def deterministic_tar(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.is_file() and not p.name.endswith(".tmp") and ".tmp." not in p.name)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tf:
        for p in files:
            data = p.read_bytes()
            ti = tarfile.TarInfo(str(p.relative_to(src)))
            ti.size, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = len(data), 0, 0, 0, "", "", 0o644
            tf.addfile(ti, io.BytesIO(data))
    os.replace(tmp, dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    Path(str(dst) + ".sha256").write_text(f"{h}  {dst.name}\n")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", nargs="+", default=["D384", "D512"])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--items-root", type=Path, default=REPO / "experiments" / "items")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--archive", type=Path, default=REPO / "runs" / "_artifacts" / "triton_cache.tar")
    ap.add_argument("--registry", type=Path, default=REPO / "experiments" / "registry.jsonl")
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()

    import torch

    from hybridlc.models.build import build_model
    from hybridlc.training.compat import triton_cache_hash
    from hybridlc.training.config import load_yaml
    from hybridlc.utils.gpu import claim, set_production_env

    set_production_env()
    claim(0.44)
    dev = torch.device("cuda")
    scratch = REPO / "runs" / "_warm_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cells = eval_cells(a.items_root) if not a.skip_eval else []
    log(f"cache {os.environ['TRITON_CACHE_DIR']} hash before {triton_cache_hash()}; {len(cells)} eval cells")
    t_all = time.time()
    for scale in a.scales:
        for name in a.configs:
            t0 = time.time()
            fields = load_yaml(REPO / "configs" / "model" / scale / f"{name}.yaml")["fields"]
            model = build_model(fields, 0).to(dev)
            if not a.skip_train:
                train_shapes(model, dev)
            if not a.skip_eval:
                eval_shapes(model, cells, scratch)
            if not a.skip_bench:
                bench_shapes(model, dev)
            del model
            torch.cuda.empty_cache()
            log(f"{scale} {name}: {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
    th = triton_cache_hash()
    rec = dict(kind="triton_cache", triton_cache_hash=th, wall_s=time.time() - t_all, scales=a.scales,
               configs=a.configs, n_eval_cells=len(cells))
    if not a.no_archive:
        sha = deterministic_tar(Path(os.environ["TRITON_CACHE_DIR"]), a.archive)
        rec.update(archive=str(a.archive), archive_sha256=sha, archive_bytes=a.archive.stat().st_size)
        from hybridlc.training.registry import Registry

        Registry(a.registry).append("S0-triton-cache", "note", **rec)
    log(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
