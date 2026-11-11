"""VT-21 (R3b, DEV-01): fla 0.6.0 Triton ``causal_conv1d`` vs an independent reference.

Pre-registration: ``experiments/repro/R3/prereg.yaml`` (copied from reproduction_plan.md §2.3 R3b).
Reference: per-document ``F.conv1d`` (groups = D, K−1 left padding from ``initial_state[..., 1:]``
or zeros) in float64 on the CPU, written here without touching hybridlc or fla code.

Pass (frozen):
  fp32: max abs ≤ 1e-5 (forward, final state), ≤ 1e-4 (dx, dw);
  bf16: relative RMS ≤ 1e-2 (y, final state, dx, dw);
  cross-document leakage exactly zero; final conv state matches the reference.

Raw output: ``experiments/repro/R3/vt21/vt21_conv.jsonl`` (+ ``_meta.json``).
Run: ``pytest -q -s tests/test_vt21_conv_reference.py`` (GPU, ≈ 5–10 min).
"""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "repro"))
import r3_common as RC  # noqa: E402

W = 4
CHANNELS = {"D384": {"gdn_qk": 192, "gdn_v": 384, "fgc_3d": 1152},
            "D512": {"gdn_qk": 256, "gdn_v": 512, "fgc_3d": 1536}}
TS = [37, 300, 4096, 32768]
PACKED_CU = {37: [0, 13, 37], 300: [0, 77, 190, 300], 4096: [0, 1000, 2731, 4096], 32768: [0, 5003, 16411, 32768]}
TOL_FP32_FWD, TOL_FP32_GRAD, TOL_BF16 = 1e-5, 1e-4, 1e-2
REF_BLOCK = 256   # channels per reference block (memory only)


# ---------------------------------------------------------------------------- reference
def ref_conv(x: torch.Tensor, w: torch.Tensor, seqs: list[tuple[int, int, int]], h0: torch.Tensor | None,
             act: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B, T, D] float64 (CPU); w [D, W]; seqs = [(row, start, end)] one entry per sequence;
    h0 [N, D, W] (h0[n, :, W-1] = most recent input) or None. Returns (y [B,T,D], final [N,D,W])."""
    finals = []
    D = x.shape[-1]
    rows: dict[int, list[torch.Tensor]] = {}
    for n, (b, a, e) in enumerate(seqs):
        seg = x[b, a:e]                                                     # [L, D]
        hist = h0[n].t() if h0 is not None else x.new_zeros(W, D)           # [W, D], last row = newest
        xp = torch.cat([hist[1:], seg], 0)                                  # W-1 rows of history
        ys = F.conv1d(xp.t()[None], w[:, None, :], groups=D)[0].t()        # [L, D]; w[:, W-1] hits x_t
        if act == "silu":
            ys = F.silu(ys)
        rows.setdefault(b, []).append(ys)
        finals.append(torch.cat([hist, seg], 0)[-W:].t())                  # last W raw inputs
    # sequences tile each row contiguously and in order (dense: one per row; packed: all in row 0)
    y = torch.stack([torch.cat(rows[b], 0) for b in sorted(rows)], 0)
    assert y.shape == x.shape
    return y, torch.stack(finals, 0)


def _fla():
    from fla.modules.causal_conv1d.ops import causal_conv1d

    return causal_conv1d


def _run_case(dev, D: int, T: int, packed: bool, use_h0: bool, act: str | None, dtype: torch.dtype,
              key: tuple[int, ...], cu_dtype: torch.dtype, cu_list: list[int] | None = None) -> dict:
    conv = _fla()
    g = RC.torch_gen(21, *key)
    if packed:
        cu_list = cu_list or PACKED_CU[T]
        B, seqs = 1, [(0, cu_list[i], cu_list[i + 1]) for i in range(len(cu_list) - 1)]
    else:
        B, seqs = 2, [(0, 0, T), (1, 0, T)]
    N = len(seqs)
    x32 = torch.randn(B, T, D, generator=g)
    w32 = (torch.rand(D, W, generator=g) * 2 - 1) * (1.0 / W ** 0.5)
    h032 = torch.randn(N, D, W, generator=g) if use_h0 else None
    dy32 = torch.randn(B, T, D, generator=g)
    # round to the kernel dtype once; the reference sees exactly these values
    x, w, dy = x32.to(dtype), w32.to(dtype), dy32.to(dtype)
    h0 = h032.to(dtype) if use_h0 else None

    xg = x.to(dev).requires_grad_(True)
    wg = w.to(dev).requires_grad_(True)
    h0g = h0.to(dev).requires_grad_(True) if use_h0 else None
    cu = torch.tensor(cu_list, dtype=cu_dtype, device=dev) if packed else None
    y, fin = conv(x=xg, weight=wg, bias=None, initial_state=h0g, output_final_state=True, activation=act,
                  backend="triton", cu_seqlens=cu)
    y.backward(dy.to(dev))
    out = {"y": y.detach(), "fin": fin.detach(), "dx": xg.grad, "dw": wg.grad, "dh0": h0g.grad if use_h0 else None}

    # reference: float64 autograd on the GPU, in independent channel blocks (depthwise conv), with
    # streaming error accumulators (exact max / sum-of-squares, so the metrics equal full-tensor ones)
    acc = {k: [0.0, 0.0, 0.0, 0.0, 0] for k in out if out[k] is not None}   # maxdiff, maxref, ssd, ssr, n
    hyb = [0.0, 0.0]
    want_hyb = act is None and not (packed and use_h0)
    cdim = {"y": -1, "dx": -1, "fin": 1, "dh0": 1, "dw": 0}
    for c0 in range(0, D, REF_BLOCK):
        c1 = min(D, c0 + REF_BLOCK)
        xr = x[..., c0:c1].to(dev).double().requires_grad_(True)
        wr = w[c0:c1].to(dev).double().requires_grad_(True)
        h0r = h0[:, c0:c1].to(dev).double().requires_grad_(True) if use_h0 else None
        yr, finr = ref_conv(xr, wr, seqs, h0r, act)
        yr.backward(dy[..., c0:c1].to(dev).double())
        ref = {"y": yr.detach(), "fin": finr.detach(), "dx": xr.grad, "dw": wr.grad, "dh0": h0r.grad if use_h0 else None}
        for k, a_ in acc.items():
            o = out[k].narrow(cdim[k] % out[k].dim(), c0, c1 - c0).double()
            r = ref[k]
            d_ = o - r
            a_[0] = max(a_[0], float(d_.abs().max()))
            a_[1] = max(a_[1], float(r.abs().max()))
            a_[2] += float(d_.pow(2).sum())
            a_[3] += float(r.pow(2).sum())
            a_[4] += r.numel()
        if want_hyb:   # cross-check of hybridlc's own torch reference where its contract applies (report-only)
            from hybridlc.models.attention import causal_dwconv_reference

            yh, fh = causal_dwconv_reference(x[..., c0:c1].to(dev).double(), w[c0:c1].to(dev).double(),
                                             cu_seqlens=torch.tensor(cu_list, device=dev) if packed else None,
                                             initial_state=h0[:, c0:c1].to(dev).double() if use_h0 else None)
            hyb[0] = max(hyb[0], float((yh - ref["y"]).abs().max()))
            hyb[1] = max(hyb[1], float((fh - ref["fin"]).abs().max()))
        del xr, wr, h0r, yr, finr, ref

    row = {"D": D, "T": T, "packed": packed, "cu": cu_list if packed else None, "h0": use_h0, "act": act or "none",
           "dtype": str(dtype).replace("torch.", ""), "cu_dtype": str(cu_dtype).replace("torch.", "") if packed else None,
           "ref_device": "cuda-fp64-blocked"}
    ok = True
    for name in ("y", "fin", "dx", "dw", "dh0"):
        if name not in acc:
            continue
        md, mr, ssd, ssr, n = acc[name]
        row[f"{name}_max_abs"] = md
        row[f"{name}_rel_max"] = md / max(mr, 1e-300)
        row[f"{name}_rel_rms"] = (ssd / max(ssr, 1e-300)) ** 0.5
        row[f"{name}_ref_maxabs"] = mr
        if name == "dh0":
            continue  # report-only
        if dtype == torch.float32:
            tol = TOL_FP32_FWD if name in ("y", "fin") else TOL_FP32_GRAD
            passed = row[f"{name}_max_abs"] <= tol
        else:
            passed = row[f"{name}_rel_rms"] <= TOL_BF16
        row[f"{name}_pass"] = bool(passed)
        ok &= passed
    if want_hyb:
        row["hybridlc_ref_y_max_abs"], row["hybridlc_ref_fin_max_abs"] = hyb

    # leakage (packed only): perturb document 0, everything else must be bitwise unchanged
    if packed:
        a0, e0 = cu_list[0], cu_list[1]
        with torch.no_grad():
            xp = xg.detach().clone()
            xp[0, a0:e0] += 3.0
            hp = None
            if use_h0:
                hp = h0g.detach().clone()
                hp[0] += 3.0
            y2, fin2 = conv(x=xp, weight=wg.detach(), bias=None, initial_state=hp, output_final_state=True,
                            activation=act, backend="triton", cu_seqlens=cu)
            y1, fin1 = conv(x=xg.detach(), weight=wg.detach(), bias=None, initial_state=h0g.detach() if use_h0 else None,
                            output_final_state=True, activation=act, backend="triton", cu_seqlens=cu)
            leak_fwd = bool(torch.equal(y2[0, e0:], y1[0, e0:]) and torch.equal(fin2[1:], fin1[1:])
                            and not torch.equal(y2[0, a0:e0], y1[0, a0:e0]))
        # backward: dy only on the last document -> dx exactly 0 on all others
        aL, eL = cu_list[-2], cu_list[-1]
        xb = xg.detach().clone().requires_grad_(True)
        hb = h0g.detach().clone().requires_grad_(True) if use_h0 else None
        yb, _ = conv(x=xb, weight=wg.detach(), bias=None, initial_state=hb, output_final_state=False, activation=act,
                     backend="triton", cu_seqlens=cu)
        mask = torch.zeros_like(yb)
        mask[0, aL:eL] = 1
        yb.backward(dy.to(dev) * mask)
        leak_bwd = bool((xb.grad[0, :aL] == 0).all()) and (not use_h0 or bool((hb.grad[:-1] == 0).all()))
        row["leak_fwd_zero"], row["leak_bwd_zero"] = leak_fwd, leak_bwd
        ok &= leak_fwd and leak_bwd
    row["pass"] = bool(ok)
    return row


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-VT21; PI ruling D071; criterion unchanged")
def test_vt21_fla_conv_vs_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from hybridlc.utils.gpu import claim

    dev = claim(0.44)
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, rec):  # fla backend dispatch decisions
            records.append(rec.getMessage())

    lg = logging.getLogger("fla")
    lg.setLevel(logging.INFO)
    lg.addHandler(_H())
    import fla

    meta = RC.run_meta("VT-21", {"fla_version": fla.__version__, "criteria": {
        "fp32_fwd_max_abs": TOL_FP32_FWD, "fp32_grad_max_abs": TOL_FP32_GRAD, "bf16_rel_rms": TOL_BF16}})
    # Resumable: R3_VT21_RESUME_FROM=<partial raw jsonl> keeps its completed main-grid rows (same case ids,
    # same seeds) and runs only the missing cases; the summary covers the union.
    import json as _json
    import os as _os

    prev = {}
    rp = _os.environ.get("R3_VT21_RESUME_FROM")
    if rp:
        for line in open(rp):
            r = _json.loads(line)
            if r.get("grid") == "main":
                prev[r["case"]] = r
        meta["resumed_from"] = {"path": rp, "n_rows_kept": len(prev)}
    wr = RC.RawWriter("vt21", "vt21_conv", meta)
    fails = [r for r in prev.values() if not r["pass"]]
    i = 0
    for scale, chans in CHANNELS.items():
        for (path, D), T, packed, use_h0, act, dtype in itertools.product(
                chans.items(), TS, (False, True), (False, True), (None, "silu"), (torch.float32, torch.bfloat16)):
            cu_dtype = torch.int32 if path == "fgc_3d" else torch.int64   # F_gc passes int32, linear.py .long()
            if i in prev:
                i += 1
                continue
            row = _run_case(dev, D, T, packed, use_h0, act, dtype, key=(i,), cu_dtype=cu_dtype)
            row.update(scale=scale, path=path, case=i, grid="main")
            wr.write(row)
            if not row["pass"]:
                fails.append(row)
            i += 1
            torch.cuda.empty_cache()
    # supplementary (beyond pre-registration; reported separately, not in the verdict)
    supp = []
    for use_h0, act, dtype in itertools.product((False, True), (None, "silu"), (torch.float32, torch.bfloat16)):
        row = _run_case(dev, 384, 300, True, use_h0, act, dtype, key=(10_000 + len(supp),), cu_dtype=torch.int64,
                        cu_list=[0, 2, 5, 150, 300])
        row.update(scale="D384", path="gdn_v", case=10_000 + len(supp), grid="supplementary_short_docs")
        wr.write(row)
        supp.append(row)
    summary = {"n_main": i, "n_main_fail": len(fails), "n_supp": len(supp),
               "n_supp_fail": sum(not r["pass"] for r in supp), "fla_backend_log": sorted(set(records)),
               "worst": {}}
    main_rows = list(prev.values()) + [r for r in wr.rows if r["grid"] == "main"]
    summary["n_main_rows_union"] = len(main_rows)
    for k in ("y", "fin", "dx", "dw"):
        f32 = [r for r in main_rows if r["dtype"] == "float32"]
        b16 = [r for r in main_rows if r["dtype"] == "bfloat16"]
        summary["worst"][f"fp32_{k}_max_abs"] = max(r[f"{k}_max_abs"] for r in f32)
        summary["worst"][f"bf16_{k}_rel_rms"] = max(r[f"{k}_rel_rms"] for r in b16)
    summary["leak_all_zero"] = all(r.get("leak_fwd_zero", True) and r.get("leak_bwd_zero", True)
                                   for r in main_rows)
    summary["verdict"] = "PASS" if not fails else "FAIL"
    wr.close(summary)
    print("VT-21 summary:", summary)
    assert not fails, f"{len(fails)} of {i} VT-21 cases failed; first: {fails[0]}"
