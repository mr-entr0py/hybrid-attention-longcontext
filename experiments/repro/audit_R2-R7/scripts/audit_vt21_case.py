"""Auditor's independent re-check of one VT-21 fp32 dw case (T=32768, dense, no h0, no act).

Own float64 reference (direct shifted sums), own seeds; does not import the VT-21 test.
Prints forward / dx / dw max-abs and relative errors for fla's Triton causal_conv1d.
"""
import json
import sys

import torch
from fla.modules.causal_conv1d.ops import causal_conv1d

torch.cuda.set_per_process_memory_fraction(0.44)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

out = []
for D in (int(a) for a in sys.argv[1:] or ["1536", "384"]):
    B, T, W = 2, 32768, 4
    g = torch.Generator().manual_seed(4242 + D)
    x = torch.randn(B, T, D, generator=g)
    w = (torch.rand(D, W, generator=g) * 2 - 1) * (1.0 / W ** 0.5)
    dy = torch.randn(B, T, D, generator=g)

    xc = x.cuda().requires_grad_(True)
    wc = w.cuda().requires_grad_(True)
    y, _ = causal_conv1d(xc, wc, activation=None, backend="triton")
    y.backward(dy.cuda())
    y_k, dx_k, dw_k = y.detach().double().cpu(), xc.grad.double().cpu(), wc.grad.double().cpu()

    # float64 reference: y[t] = sum_k w[k] * x[t-(W-1)+k]  (causal, left zero padding)
    x64, w64, dy64 = x.double(), w.double(), dy.double()
    xp = torch.nn.functional.pad(x64, (0, 0, W - 1, 0))  # pad time dim on the left
    y_ref = sum(w64[:, k] * xp[:, k:k + T, :] for k in range(W))
    dw_ref = torch.stack([(dy64 * xp[:, k:k + T, :]).sum(dim=(0, 1)) for k in range(W)], dim=1)
    dyp = torch.nn.functional.pad(dy64, (0, 0, 0, W - 1))  # pad right
    dx_ref = sum(w64[:, k] * dyp[:, (W - 1 - k):(W - 1 - k) + T, :] for k in range(W))

    def m(a, r):
        d = (a - r).abs()
        return {"max_abs": float(d.max()), "ref_maxabs": float(r.abs().max()),
                "rel_max": float(d.max() / r.abs().max()),
                "rel_rms": float((a - r).norm() / r.norm())}

    rec = {"D": D, "T": T, "B": B, "y": m(y_k, y_ref), "dx": m(dx_k, dx_ref), "dw": m(dw_k, dw_ref)}
    rec["frozen_fp32_pass"] = {"y<=1e-5": rec["y"]["max_abs"] <= 1e-5, "dx<=1e-4": rec["dx"]["max_abs"] <= 1e-4,
                               "dw<=1e-4": rec["dw"]["max_abs"] <= 1e-4}
    out.append(rec)
    print(json.dumps(rec))
    del xc, wc, y
    torch.cuda.empty_cache()
