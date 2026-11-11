"""VT-05: recurrent vs chunked equivalence (protocol §16).

(a) chunk prefill vs token-by-token ``fused_recurrent`` with state carry;
(b) chunk prefill + recurrent tail vs one chunk pass.
Pass: outputs relative RMS ≤ 3e-3 (fp32) / ≤ 1e-2 (bf16); final state relative max ≤ 2e-3 (fp32) /
≤ 8e-3 (bf16).

Checked at op level (with the exact in-kernel gating keywords GDNMixer uses) and at layer level
(``GDNMixer.prefill`` / ``extend``, incl. the short-conv cache), for GDN-A, GDN-E and GLA.
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models import linear as L
from hybridlc.models.config import ModelConfig
from hybridlc.models.mixers import build_mixer
from hybridlc.models.testing import cuda_device_or_skip, rel_max, rel_rms

OUT_TOL = {torch.float32: 3e-3, torch.bfloat16: 1e-2}
STATE_TOL = {torch.float32: 2e-3, torch.bfloat16: 8e-3}


def _inputs(dev, dtype, T, H, K, V, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, T, H, K, generator=gen)
    k = torch.randn(1, T, H, K, generator=gen)
    v = torch.randn(1, T, H, V, generator=gen)
    a = torch.randn(1, T, H, generator=gen)
    b = torch.randn(1, T, H, generator=gen)
    A_log = torch.log(torch.empty(H).uniform_(0, 16, generator=gen))
    dt = torch.exp(torch.rand(H, generator=gen) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                   + torch.log(torch.tensor(0.001)))
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    to = lambda t: t.to(dev, dtype)  # noqa: E731
    return to(q), to(k), to(v), to(a), to(b), A_log.to(dev), dt_bias.to(dev)


def _kw(A_log, dt_bias):
    return dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
                state_v_first=True, use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)


#: DEVIATION (flagged in the S0 report): in bf16 the fla *chunk* kernel's final state carries a
#: relative-max error of 3.8e-3 … 8.03e-3 vs fp64 (16 seeds, T = 384), while the recurrent kernel is
#: within 1.1e-5 of fp64.  The protocol bound 8e-3 is therefore asserted on the median over seeds,
#: with a per-seed ceiling of 1e-2, and the error is attributed to the chunk side explicitly.
BF16_STATE_PER_SEED_CEIL = 1e-2
N_SEEDS = {torch.float32: 2, torch.bfloat16: 8}


GEOS = [(4, 64, 128), (3, 64, 128), (1, 256, 512), (1, 192, 384)]
RELAXED = pytest.mark.relaxed_guard(reason="VT-05 bf16 final state: median <= 8e-3 and per-seed <= 1e-2 (registered: every seed-case <= 8e-3)")
_OP_CACHE: dict = {}


def _op_case(geo, dtype):
    """(out rel-RMS per seed, final-state rel-max per seed, recurrent-vs-fp64 per seed); computed once per
    (geo, dtype) and shared by the regression guard and the registered-criterion test."""
    key = (tuple(geo), dtype)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    import torch.nn.functional as F

    from hybridlc.models.testing import naive_gated_delta_fp64

    dev = cuda_device_or_skip()
    H, K, V = geo
    T = 384
    out_errs, st_errs, rec_vs_fp64 = [], [], []
    for seed in range(N_SEEDS[dtype]):
        q, k, v, a, b, A_log, dt_bias = _inputs(dev, dtype, T, H, K, V, seed=seed * 101 + K)
        kw = _kw(A_log, dt_bias)
        o_c, s_c = L.chunk_gated_delta_rule(q=q, k=k, v=v, g=a, beta=b, initial_state=None,
                                            output_final_state=True, **kw)
        state = None
        outs = []
        for t in range(T):
            o_t, state = L.fused_recurrent_gated_delta_rule(q=q[:, t:t + 1], k=k[:, t:t + 1], v=v[:, t:t + 1],
                                                            g=a[:, t:t + 1], beta=b[:, t:t + 1], initial_state=state,
                                                            output_final_state=True, **kw)
            outs.append(o_t)
        o_r = torch.cat(outs, 1)
        out_errs.append(rel_rms(o_r, o_c))
        st_errs.append(rel_max(state, s_c))
        g = -A_log.double().exp() * F.softplus(a.double() + dt_bias.double())
        _, href = naive_gated_delta_fp64(F.normalize(q.double(), dim=-1), F.normalize(k.double(), dim=-1), v, g,
                                         torch.sigmoid(b.double()))
        rec_vs_fp64.append(rel_max(state, href.transpose(-1, -2)))
    _OP_CACHE[key] = (out_errs, st_errs, rec_vs_fp64)
    return _OP_CACHE[key]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, pytest.param(torch.bfloat16, marks=RELAXED)])
@pytest.mark.parametrize("geo", GEOS, ids=lambda g: f"{g[0]}x{g[1]}-{g[2]}")
def test_op_token_by_token_vs_chunk(geo, dtype) -> None:
    """fp32: the registered criteria. bf16: outputs and the recurrent kernel at the registered bounds; the
    final-state part is a **regression guard, not the registered criterion** (median over 8 seeds <= 8e-3,
    per-seed ceiling 1e-2; PI ruling D059 / RR-G0-1; registered: ``test_vt05_registered_bf16_state_every_case``)."""
    import statistics

    out_errs, st_errs, rec_vs_fp64 = _op_case(geo, dtype)
    print(f"VT-05 {geo} {dtype}: out rel-RMS max {max(out_errs):.2e}; state rel-max per seed "
          f"{[f'{e:.2e}' for e in st_errs]}; recurrent-vs-fp64 max {max(rec_vs_fp64):.2e}")
    assert max(out_errs) <= OUT_TOL[dtype], out_errs
    assert max(rec_vs_fp64) <= 1e-4  # the recurrent (decode/scoring) kernel is exact to fp32 level
    if dtype == torch.float32:
        assert max(st_errs) <= STATE_TOL[dtype], st_errs
    else:
        assert statistics.median(st_errs) <= STATE_TOL[dtype], st_errs
        assert max(st_errs) <= BF16_STATE_PER_SEED_CEIL, st_errs


@pytest.mark.gpu
@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-G0-1; PI ruling D059; criterion unchanged")
def test_vt05_registered_bf16_state_every_case(record_property) -> None:
    """Registered criterion (protocol §16 VT-05): bf16 final state relative max <= 8e-3 for **every** seed-case
    (4 geometries x 8 seeds = 32). S0 measured 2 of 32 above 8e-3 (max 8.47e-3). XFAIL expected; an XPASS is
    recorded, never a PASS of VT-05."""
    per = {f"{g[0]}x{g[1]}-{g[2]}": _op_case(g, torch.bfloat16)[1] for g in GEOS}
    worst = max(max(v) for v in per.values())
    n_over = sum(e > STATE_TOL[torch.bfloat16] for v in per.values() for e in v)
    record_property("vt05_bf16_state_rel_max_worst", worst)
    record_property("vt05_bf16_seed_cases_over_8e-3", n_over)
    print(f"VT-05 registered (bf16 state <= 8e-3 every case): worst {worst:.3e}, {n_over}/32 over")
    assert n_over == 0, per


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("T,tail", [(1024, 64), (4096, 32), (2048, 1)])
def test_op_chunk_prefill_plus_recurrent_tail(T, tail, dtype) -> None:
    dev = cuda_device_or_skip()
    H, K, V = 4, 64, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(dev, dtype, T, H, K, V, seed=T + tail)
    kw = _kw(A_log, dt_bias)
    o_full, s_full = L.chunk_gated_delta_rule(q=q, k=k, v=v, g=a, beta=b, initial_state=None, output_final_state=True, **kw)
    P = T - tail
    _, s_p = L.chunk_gated_delta_rule(q=q[:, :P], k=k[:, :P], v=v[:, :P], g=a[:, :P], beta=b[:, :P], initial_state=None,
                                      output_final_state=True, **kw)
    o_t, s_t = L.fused_recurrent_gated_delta_rule(q=q[:, P:], k=k[:, P:], v=v[:, P:], g=a[:, P:], beta=b[:, P:],
                                                  initial_state=s_p, output_final_state=True, **kw)
    assert rel_rms(o_t, o_full[:, P:]) <= OUT_TOL[dtype], rel_rms(o_t, o_full[:, P:])
    assert rel_max(s_t, s_full) <= STATE_TOL[dtype], rel_max(s_t, s_full)


def _layer(kind: str, dev, dtype):
    cfg = ModelConfig(d_model=256, d_ff=512, n_layers=1, vocab_size=128, pattern=kind)
    torch.manual_seed(1)
    return build_mixer(kind, cfg, 0).to(dev, dtype).eval()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["-", "E", "N", "G"])
def test_layer_prefill_vs_token_by_token_extend(kind, dtype) -> None:
    """Layer level, including the short-conv cache: prefill(P) + M single-token extends and one
    M-token extend vs one prefill over P + M (outputs and final state)."""
    dev = cuda_device_or_skip()
    m = _layer(kind, dev, dtype)
    gen = torch.Generator().manual_seed(2)
    T, P = 300, 200
    x = torch.randn(2, T, 256, generator=gen).to(dev, dtype)
    y_full, st_full = m.prefill(x)
    _, st = m.prefill(x[:, :P])
    ys = []
    st_step = st
    for t in range(P, T):
        y_t, st_step = m.extend(x[:, t:t + 1], st_step, t)
        ys.append(y_t)
    y_steps = torch.cat(ys, 1)
    y_multi, st_multi = m.extend(x[:, P:], st, P)
    for y_r, st_r in ((y_steps, st_step), (y_multi, st_multi)):
        assert rel_rms(y_r, y_full[:, P:]) <= OUT_TOL[dtype], (kind, rel_rms(y_r, y_full[:, P:]))
        assert rel_max(st_r["state"], st_full["state"]) <= STATE_TOL[dtype]
        for key in ("conv_q", "conv_k", "conv_v"):
            if key in st_full:  # raw projection outputs: identical up to the GEMM's row tiling
                assert rel_max(st_r[key], st_full[key]) <= STATE_TOL[dtype], key
    # the input state was not modified by extend (functional cache)
    _, st2 = m.prefill(x[:, :P])
    for key in st2:
        assert torch.equal(st[key], st2[key]), key
