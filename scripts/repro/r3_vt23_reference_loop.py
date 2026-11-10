"""VT-23 (R3d) independent reference training loop, written from protocol §7.1–§7.2 text only.

    python scripts/repro/r3_vt23_reference_loop.py --out <dir> [--steps 50] [--seed-index 900]

Does NOT import hybridlc.training (no optim.py, schedules.py, trainer.py, loss helpers). It uses
the model builder and the data stream only, so that both sides see the same weights and batches.
Spec used (protocol §7.1 table; reproduction_plan.md §2.3 R3d):
  AdamW(fused=True), betas (0.9, 0.95), eps 1e-8; decoupled weight decay 0.1 on every parameter with
  ndim >= 2 (incl. tied embedding, conv kernels, W_g); none on 1-D tensors, on `_no_weight_decay`
  tensors (A_log, dt_bias) or SSMax; global L2 clipping at 1.0; z-loss 1e-4·mean(lse²) over loss
  tokens; loss = mean next-token CE over non-ignored tokens (pad id 1 ignored), fp32 logits;
  exact full-step mean over the accumulated micro-batches; LR: linear warm-up over W steps then
  constant, lr(s) = η·min(1, (s+1)/W) for 0-based step s (prereg reading).
Outputs: <out>/ref_trace.jsonl (per step: loss, z, gnorm (pre-clip), lr, n_tokens),
         <out>/ref_final.pt (state_dict), <out>/ref_groups.json, <out>/ref_meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import r3_common as RC  # noqa: E402

PAD = 1
Z_COEF = 1e-4
CLIP = 1.0
WD = 0.1


def _ce_z_sums(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum over valid tokens of CE and of lse² for one chunk (fp32 logits)."""
    logits = h.float() @ w.float().t()
    lse = torch.logsumexp(logits, dim=-1)
    valid = (y != PAD) & (y >= 0)
    tgt = logits.gather(1, y.clamp_min(0)[:, None])[:, 0]
    ce = torch.where(valid, lse - tgt, torch.zeros_like(lse))
    z = torch.where(valid, lse * lse, torch.zeros_like(lse))
    return ce.sum(), z.sum()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--seed-index", type=int, default=900)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=4096)
    # Post-verdict isolation diagnostics only (defaults = the pre-registered independent loop):
    ap.add_argument("--loss", choices=["own", "chunked_ce"], default="own",
                    help="chunked_ce: use hybridlc.models.loss.chunked_ce for the per-micro loss (diagnostic)")
    ap.add_argument("--clip", choices=["manual", "torch"], default="manual",
                    help="torch: torch.nn.utils.clip_grad_norm_(foreach=True) (diagnostic)")
    ap.add_argument("--ce-chunk", type=int, default=4096, help="CE chunk length of the own loss (diagnostic)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.utils.gpu import claim, set_production_env

    set_production_env(overwrite=False)            # same kernel env (FLA_FLASH_QLA=0, ...) as any project run
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_cudnn_sdp(False)    # protocol §18: no cuDNN SDPA in training
    dev = claim(0.44)
    meta = RC.run_meta("VT-23-reference", {"args": vars(a)})

    model = build_model(ModelConfig.for_scale("D384", "C3"), a.seed_index).to(dev).train()
    # Same per-block activation checkpointing as the trainer side (memory only; see training_vt23.yaml).
    model.set_activation_checkpointing(True)
    from hybridlc.utils.provenance import sha256_tensor
    meta["init_param_sha256"] = {n: sha256_tensor(p.detach().cpu()) for n, p in model.named_parameters()}
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False) and "ssmax" not in n) else no_decay).append((n, p))
    (out / "ref_groups.json").write_text(json.dumps({"decay": [n for n, _ in decay], "no_decay": [n for n, _ in no_decay]}, indent=1))
    opt = torch.optim.AdamW([{"params": [p for _, p in decay], "weight_decay": WD},
                             {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
                            lr=0.0, betas=(0.9, 0.95), eps=1e-8, fused=True)
    params = [p for p in model.parameters()]
    stream = StageStream("A", a.seed_index, a.seq_len, a.micro, a.accum, None)
    W = model.embed.weight                          # tied LM head
    with open(out / "ref_trace.jsonl", "x") as tr:
        for s in range(a.steps):
            lr = a.lr * min(1.0, (s + 1) / a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            batches = [stream.get(s, m) for m in range(a.accum)]
            N = sum(int(((b["labels"] != PAD) & (b["labels"] >= 0)).sum()) for b in batches)
            ce_tot = torch.zeros((), dtype=torch.float64, device=dev)
            z_tot = torch.zeros((), dtype=torch.float64, device=dev)
            for b in batches:
                ids, y = b["input_ids"].to(dev), b["labels"].to(dev)[0]
                h = model(ids, b["cu_seqlens"].to(dev), b["position_ids"].to(dev))[0]
                if a.loss == "chunked_ce":   # diagnostic: the production loss function, our step assembly
                    from hybridlc.models.loss import chunked_ce
                    loss_m, zl_m, n_m = chunked_ce(h[None], W, y[None], chunk=4096, z_loss_coef=Z_COEF, ignore_index=PAD)
                    n_m = int(((y != PAD) & (y >= 0)).sum())
                    ce_s, z_s = loss_m * n_m, zl_m * n_m / Z_COEF
                    (loss_m.float() * (n_m / N) + zl_m.float() * (n_m / N)).backward()
                else:
                    ce_s = torch.zeros((), device=dev)
                    z_s = torch.zeros((), device=dev)
                    C = a.ce_chunk
                    for c0 in range(0, h.shape[0], C):
                        ce_c, z_c = checkpoint(_ce_z_sums, h[c0:c0 + C], W, y[c0:c0 + C], use_reentrant=False)
                        ce_s, z_s = ce_s + ce_c, z_s + z_c
                    ((ce_s + Z_COEF * z_s) / N).backward()
                ce_tot += ce_s.detach().double()
                z_tot += z_s.detach().double()
            if a.clip == "torch":        # diagnostic
                gn = torch.nn.utils.clip_grad_norm_(params, CLIP, foreach=True)
                coef = min(1.0, CLIP / (float(gn) + 1e-6))
            else:
                gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
                coef = min(1.0, CLIP / (float(gn) + 1e-6))
                if coef < 1.0:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(coef)
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": s, "loss": float(ce_tot / N), "z": float(Z_COEF * z_tot / N), "gnorm": float(gn),
                   "lr": lr, "n_tokens": N, "clipped": coef < 1.0, "batch_hashes": [b.get("batch_hash") for b in batches]}
            tr.write(json.dumps(rec) + "\n")
            tr.flush()
            if not math.isfinite(rec["loss"]):
                break
    torch.save({n: p.detach().cpu() for n, p in model.state_dict().items()}, out / "ref_final.pt")
    meta["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    RC.finish_meta(meta)
    (out / "ref_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


if __name__ == "__main__":
    main()
