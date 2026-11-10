"""Full-attention mixers F_g and F_gc (protocol §3.3, §6.4–6.6, §9.2).

F_g:
  q, k, v = x W_Qᵀ, x W_Kᵀ, x W_Vᵀ               (W_* ∈ ℝ^{d×d}, no bias)
  q, k   ← per-head *unshared* RMSNorm over head_dim (gains 2·H·64, init 1)
  q, k   ← RoPE θ = 5e5 (full rotary dim 64, HF rotate_half layout), positions reset per document
  q      ← q · s_h · ln(t + 1)                      (optional SSMax, §6.6; t = in-document position)
  o_h    = causal softmax(q_h k_hᵀ / √64) v_h        (within the document only)
  o_h    ← σ(x W_gᵀ)_h · o_h                        (headwise sigmoid gate, W_g ∈ ℝ^{H×d})
  out    = W_O concat_h(o_h)

F_gc = F_g plus a causal depthwise ShortConv(4) (no bias) on q, k, v right after the projections
and before QK-norm/RoPE, SiLU on v only.  The conv respects document boundaries.

Kernel paths (every one is explicit; nothing auto-switches):
  * training, bf16/fp16 on CUDA   : ``torch.nn.attention.varlen.varlen_attn`` (FA2 inside torch),
                                     causal via ``window_size=(-1, 0)``, one call over the packed row;
  * training, fp32/fp64 or CPU    : per-document SDPA with the MATH backend (reference path);
  * inference prefill  (Tq == Tk) : SDPA ``is_causal=True`` pinned to FLASH (bf16) / MATH (fp32);
  * inference decode   (Tq == 1)  : SDPA over the whole cache, no mask, same pinning;
  * inference extend   (1 < Tq < Tk, teacher-forced scoring from a snapshot): SDPA with the
    ``causal_lower_right`` bias pinned to FLASH (bf16) or an explicit lower-right boolean mask under
    MATH (fp32).
Pinning uses ``sdpa_kernel([backend])``; if the pinned backend cannot run, SDPA raises instead of
silently falling back (this is the runtime backend assertion of VT-08 / §10.2).

R-ATT (FEAS-PREC v2, PI ruling D083): an *explicitly selected* fp32 reference attention for the
full-sequence forward and for inference prefill / extend / decode, entered only through
``fp32_reference_attention(mode)``. The MATH backend materialises an [H, T, T] fp32 score matrix
(~25.8 GB per call at 32K with 6 heads), which cannot run under the 0.44 memory cap. Modes:
  * ``"tiled"``     : exact query-tiled attention in fp32 (matmul + softmax over every allowed key, one
                      query tile at a time; lower-right causal; per document under cu_seqlens);
  * ``"efficient"`` : SDPA pinned to EFFICIENT_ATTENTION in fp32 (raises if it cannot run).
In the reference mode every attention call must be fp32 (a half-precision call raises), and nothing
auto-switches; the default (bf16 FLASH / cuDNN, fp32 MATH) paths are unchanged when the mode is off.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from .layers import apply_rope, rope_tables

try:  # torch ≥ 2.10
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except Exception:  # pragma: no cover - environment dependent
    _varlen_attn = None

try:
    from fla.modules.causal_conv1d.ops import causal_conv1d as _fla_causal_conv1d
except Exception:  # pragma: no cover - environment dependent
    _fla_causal_conv1d = None

#: Backend pinned for bf16/fp16 inference SDPA calls (prefill, decode, extend). The benchmark
#: harness may set this module attribute to ``SDPBackend.CUDNN_ATTENTION`` for decode timing.
INFERENCE_HALF_BACKEND: SDPBackend = SDPBackend.FLASH_ATTENTION

#: Test hook: force the pure-torch reference paths (per-document SDPA-math attention and the
#: torch causal conv) even on CUDA in half precision.
FORCE_REFERENCE: bool = False

#: Pointer to SDPA used by every inference call in this module (tests wrap it with a spy).
_sdpa = F.scaled_dot_product_attention

#: R-ATT mode (None = off). Set only through :func:`fp32_reference_attention`.
REFERENCE_FP32_MODE: str | None = None
REFERENCE_FP32_MODES = ("tiled", "efficient")
#: fp32 bytes of one score tile [B, H, tile, Tk] in the tiled path (the softmax output doubles it).
REFERENCE_TILE_BYTES: int = 1 << 30


@contextlib.contextmanager
def fp32_reference_attention(mode: str = "tiled") -> Iterator[str]:
    """Select the R-ATT fp32 reference attention for the duration of the block (see module docstring)."""
    global REFERENCE_FP32_MODE
    if mode not in REFERENCE_FP32_MODES:
        raise ValueError(f"R-ATT mode must be one of {REFERENCE_FP32_MODES}, got {mode!r}")
    prev = REFERENCE_FP32_MODE
    REFERENCE_FP32_MODE = mode
    try:
        yield mode
    finally:
        REFERENCE_FP32_MODE = prev


def reference_backend_name() -> str:
    """Attention backend label for result metadata (FEAS-PREC v2 ``_meta``)."""
    if REFERENCE_FP32_MODE == "tiled":
        return "R-ATT:tiled-fp32"
    if REFERENCE_FP32_MODE == "efficient":
        return "R-ATT:sdpa-efficient-fp32"
    return f"sdpa:{INFERENCE_HALF_BACKEND.name} (half) / MATH (fp32)"


def _check_reference_dtype(*ts: torch.Tensor) -> None:
    bad = [t.dtype for t in ts if t.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"R-ATT fp32 reference attention ({REFERENCE_FP32_MODE}) called with {bad}: the "
                           "reference mode is fp32 only (no autocast); refusing instead of switching precision")


def attn_reference_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                        mode: str | None = None) -> torch.Tensor:
    """fp32 reference attention of Tq queries over Tk keys, lower-right causal (query i sits at key
    Tk-Tq+i; Tq == Tk is ordinary causal, Tq == 1 attends to every key). q: [B, H, Tq, D]."""
    mode = mode or REFERENCE_FP32_MODE or "tiled"
    _check_reference_dtype(q, k, v)
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    if mode == "efficient":
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            if Tq == Tk:
                return _sdpa(q, k, v, is_causal=True, scale=scale)
            if Tq == 1:
                return _sdpa(q, k, v, scale=scale)
            return _sdpa(q, k, v, attn_mask=causal_lower_right(Tq, Tk), scale=scale)
    if mode != "tiled":
        raise ValueError(f"unknown R-ATT mode {mode!r}")
    off = Tk - Tq
    out = torch.empty_like(q)
    tile = max(1, min(Tq, REFERENCE_TILE_BYTES // max(1, 4 * B * H * Tk)))
    for a in range(0, Tq, tile):
        b = min(Tq, a + tile)
        kend = off + b  # keys visible to the last query of the tile
        s = torch.matmul(q[:, :, a:b], k[:, :, :kend].transpose(-1, -2)) * scale  # [B, H, t, kend]
        if kend > off + a + 1:  # mask keys after each query's own position
            qi = torch.arange(off + a, off + b, device=q.device)[:, None]
            kj = torch.arange(kend, device=q.device)[None, :]
            s.masked_fill_(kj > qi, float("-inf"))
        p = torch.softmax(s, dim=-1)
        del s
        out[:, :, a:b] = torch.matmul(p, v[:, :, :kend])
        del p
    return out


def _is_half(t: torch.Tensor) -> bool:
    return t.dtype in (torch.bfloat16, torch.float16)


def _pinned_backend(t: torch.Tensor) -> SDPBackend:
    """The SDPA backend this tensor's inference call is pinned to."""
    if t.is_cuda and _is_half(t) and not FORCE_REFERENCE:
        return INFERENCE_HALF_BACKEND
    return SDPBackend.MATH


# ---------------------------------------------------------------------------------------
# Head-wise RMSNorm with per-head unshared gains
# ---------------------------------------------------------------------------------------

def head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, n_heads: int, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim of ``x[..., H, D]`` with per-head gains ``weight[H*D]``.

    Computed in fp32 (at least), result cast back to ``x.dtype``.
    """
    D = x.shape[-1]
    ct = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(ct)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.to(ct).view(n_heads, D)).to(x.dtype)


# ---------------------------------------------------------------------------------------
# Causal depthwise conv (F_gc); fla Triton kernel or a torch reference
# ---------------------------------------------------------------------------------------

def causal_dwconv_reference(x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                            initial_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch causal depthwise conv with fla's tap convention (``weight[:, W-1]`` multiplies the
    current token) that never crosses a document boundary.

    x: [B, T, D]; weight: [D, W]; cu_seqlens: int [n+1] (then B == 1); initial_state: [B, D, W]
    (dense only).  Returns (y, final_state[N, D, W]) where final_state holds the last W raw inputs
    of each sequence (zero-padded on the left), matching fla's conv cache layout.
    """
    B, T, D = x.shape
    W = weight.shape[1]
    if cu_seqlens is not None:
        assert B == 1 and initial_state is None
        cu = cu_seqlens.long()
        lens = cu[1:] - cu[:-1]
        doc_start = torch.repeat_interleave(cu[:-1], lens)
        pos = torch.arange(T, device=x.device) - doc_start  # in-document position
    else:
        pos = torch.arange(T, device=x.device)
    # history buffer: [B, W-1 + T, D]
    if initial_state is not None:
        hist = initial_state[..., 1:].transpose(1, 2).to(x.dtype)  # last W-1 inputs
    else:
        hist = x.new_zeros(B, W - 1, D)
    xp = torch.cat([hist, x], dim=1)
    y = torch.zeros_like(x)
    for j in range(W):  # j = lag
        xs = xp[:, W - 1 - j: W - 1 - j + T]
        if cu_seqlens is not None:
            keep = (pos >= j).to(x.dtype)[None, :, None]
            xs = xs * keep
        y = y + xs * weight[:, W - 1 - j].to(x.dtype)
    # final state: last W inputs of each sequence (raw), [N, D, W]
    if cu_seqlens is not None:
        cu = cu_seqlens.long()
        finals = []
        for i in range(len(cu) - 1):
            a, b = int(cu[i]), int(cu[i + 1])
            seg = x[0, max(a, b - W):b]  # [<=W, D]
            pad = x.new_zeros(W - seg.shape[0], D)
            finals.append(torch.cat([pad, seg], 0).t())
        final = torch.stack(finals, 0)
    else:
        final = xp[:, -W:].transpose(1, 2)
    return y, final.contiguous()


class CausalDWConv(nn.Module):
    """Depthwise causal conv, kernel ``W``, no bias; parameter ``weight[D, 1, W]`` like nn.Conv1d.

    The constructor uses nn.Conv1d's default (kaiming-uniform a=√5 → U(−1/√W, 1/√W)); the model
    builder (``build.py``) redraws it N(0, 0.02) from the project init stream, as fla's model-level
    init does for GDN's ShortConvolution (verification round 3), never from the global RNG.
    """

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        # nn.Conv1d's default distribution (global RNG; build_model redraws from the project stream).
        if self.weight.device.type != "meta":
            nn.init.uniform_(self.weight, -self.init_bound(), self.init_bound())

    def init_bound(self) -> float:
        return 1.0 / math.sqrt(self.kernel_size)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                initial_state: torch.Tensor | None = None, output_final_state: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        w = self.weight[:, 0, :]
        use_fla = (_fla_causal_conv1d is not None and x.is_cuda and not FORCE_REFERENCE
                   and x.dtype in (torch.bfloat16, torch.float16, torch.float32))
        if use_fla:
            init = initial_state.to(x.dtype) if initial_state is not None else None
            y, fin = _fla_causal_conv1d(x=x, weight=w.to(x.dtype) if x.dtype != w.dtype else w, bias=None,
                                        initial_state=init, output_final_state=output_final_state,
                                        activation=None, cu_seqlens=cu_seqlens)
            return y, fin
        y, fin = causal_dwconv_reference(x, w, cu_seqlens=cu_seqlens, initial_state=initial_state)
        return y, (fin if output_final_state else None)


# ---------------------------------------------------------------------------------------
# Attention cores
# ---------------------------------------------------------------------------------------

def _attn_reference_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor,
                           scale: float) -> torch.Tensor:
    """Per-document causal SDPA (MATH backend).  q, k, v: [N, H, D]; cu int [n+1]."""
    out = torch.empty_like(q)
    bounds = cu.tolist()
    with sdpa_kernel([SDPBackend.MATH]):
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            qs, ks, vs = (t[a:b].transpose(0, 1)[None] for t in (q, k, v))  # [1, H, T, D]
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
            out[a:b] = o[0].transpose(0, 1)
    return out


def _attn_reference_fp32_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor,
                                scale: float) -> torch.Tensor:
    """R-ATT per document of a packed row (document isolation as in the MATH reference)."""
    out = torch.empty_like(q)
    bounds = cu.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        qs, ks, vs = (t[a:b].transpose(0, 1)[None] for t in (q, k, v))  # [1, H, T, D]
        out[a:b] = attn_reference_fp32(qs, ks, vs, scale)[0].transpose(0, 1)
    return out


def attn_varlen_train(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                      max_seqlen: int, scale: float) -> torch.Tensor:
    """Causal attention within documents of a packed row.  q, k, v: [N, H, D]."""
    if REFERENCE_FP32_MODE is not None:
        _check_reference_dtype(q, k, v)
        return _attn_reference_fp32_varlen(q, k, v, cu_seqlens, scale)
    if q.is_cuda and _is_half(q) and _varlen_attn is not None and not FORCE_REFERENCE:
        cu = cu_seqlens.to(torch.int32)
        return _varlen_attn(q, k, v, cu, cu, int(max_seqlen), int(max_seqlen), scale=scale, window_size=(-1, 0))
    return _attn_reference_varlen(q, k, v, cu_seqlens, scale)


def attn_inference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention of Tq new queries over Tk cached+new keys (query i sits at key Tk−Tq+i).

    q: [B, H, Tq, D]; k, v: [B, H, Tk, D].  The backend is pinned (FLASH for half on CUDA, MATH
    otherwise); a backend that cannot run raises.
    """
    if REFERENCE_FP32_MODE is not None:
        return attn_reference_fp32(q, k, v, scale)
    Tq, Tk = q.shape[2], k.shape[2]
    be = _pinned_backend(q)
    with sdpa_kernel([be]):
        if Tq == Tk:
            return _sdpa(q, k, v, is_causal=True, scale=scale)
        if Tq == 1:
            return _sdpa(q, k, v, scale=scale)
        if be == SDPBackend.MATH:
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
            return _sdpa(q, k, v, attn_mask=mask, scale=scale)
        return _sdpa(q, k, v, attn_mask=causal_lower_right(Tq, Tk), scale=scale)


# ---------------------------------------------------------------------------------------
# The mixer
# ---------------------------------------------------------------------------------------

class AttnMixer(nn.Module):
    """F_g (``conv=False``) or F_gc (``conv=True``).  See the module docstring for the definition.

    Parameters (names chosen to transplant 1:1 onto ``transformers`` Qwen3 attention):
      q_proj, k_proj, v_proj, o_proj : nn.Linear(d, H·hd) / (H·hd, d), no bias
      q_norm, k_norm                 : 1-D gains [H·hd] (per-head unshared), init 1, no weight decay
      g_proj                         : nn.Linear(d, H) headwise gate W_g (if ``out_gate``)
      ssmax_s                        : [H] SSMax scale (if ``ssmax``), init 1/ln 4096, no weight decay
      conv                           : CausalDWConv(3·H·hd, 4) on concat(q, k, v) (F_gc only)
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int = 64, *, qk_norm: bool = True, rope: bool = True,
                 rope_theta: float = 5e5, out_gate: bool = True, ssmax: bool = False, conv: bool = False,
                 conv_size: int = 4, norm_eps: float = 1e-5, layer_idx: int = 0) -> None:
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.inner = n_heads * head_dim
        self.use_qk_norm, self.use_rope, self.rope_theta = qk_norm, rope, rope_theta
        self.use_out_gate, self.use_ssmax, self.use_conv = out_gate, ssmax, conv
        self.norm_eps = norm_eps
        self.layer_idx = layer_idx
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, self.inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner, bias=False)
        self.o_proj = nn.Linear(self.inner, d_model, bias=False)
        self.o_proj._is_residual_out = True  # type: ignore[attr-defined]  (init std 0.02/√24)
        if qk_norm:
            self.q_norm = nn.Parameter(torch.ones(self.inner))
            self.k_norm = nn.Parameter(torch.ones(self.inner))
            self.q_norm._no_weight_decay = True  # type: ignore[attr-defined]
            self.k_norm._no_weight_decay = True  # type: ignore[attr-defined]
        if out_gate:
            self.g_proj = nn.Linear(d_model, n_heads, bias=False)
        if ssmax:
            self.ssmax_s = nn.Parameter(torch.full((n_heads,), 1.0 / math.log(4096.0)))
            self.ssmax_s._no_weight_decay = True  # type: ignore[attr-defined]
        if conv:
            self.conv = CausalDWConv(3 * self.inner, conv_size)
        #: Test flag for VT-09: skip the gate multiplication (parameters stay).
        self.gate_bypass: bool = False
        #: Optional attention-statistics hook (training logs): when True, forward stores
        #: ``attn_stats = {"max_logit", "entropy"}`` computed on the first ≤ 512 tokens of the
        #: first document of the row (dense, no grad).
        self.collect_attn_stats: bool = False
        self.attn_stats: dict[str, float] | None = None

    # ---- shared projection pipeline ---------------------------------------------------
    def _qkv(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None, conv_state: torch.Tensor | None,
             want_conv_state: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Projections (+ conv for F_gc).  x: [B, T, d] → q, k, v: [B, T, H, hd]."""
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        new_conv = None
        if self.use_conv:
            qkv = torch.cat([q, k, v], dim=-1)
            qkv, new_conv = self.conv(qkv, cu_seqlens=cu_seqlens, initial_state=conv_state,
                                      output_final_state=want_conv_state)
            q, k, v = qkv.split(self.inner, dim=-1)
            v = F.silu(v)
        B, T = x.shape[:2]
        shp = (B, T, self.n_heads, self.head_dim)
        return q.reshape(shp), k.reshape(shp), v.reshape(shp), new_conv

    def _norm_rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
                   rope: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """QK-norm, RoPE and SSMax.  q, k: [B, T, H, hd]; positions: [B, T] or [T] (in-document)."""
        if self.use_qk_norm:
            q = head_rmsnorm(q, self.q_norm, self.n_heads, self.norm_eps)
            k = head_rmsnorm(k, self.k_norm, self.n_heads, self.norm_eps)
        if self.use_rope:
            if rope is None:
                rope = rope_tables(positions, self.head_dim, self.rope_theta)
            cos, sin = rope
            if cos.dim() == 2:  # [T, hd] → broadcast over batch
                cos, sin = cos[None], sin[None]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.use_ssmax:
            pos = positions if positions.dim() == 2 else positions[None]
            mult = torch.log1p(pos.to(torch.float32))[..., None] * self.ssmax_s.float()  # [B, T, H]
            q = (q.float() * mult[..., None]).to(q.dtype)
        return q, k

    def _gate_out(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Headwise gate and output projection.  o: [B, T, H, hd]; x: mixer input [B, T, d]."""
        if self.use_out_gate and not self.gate_bypass:
            gate = torch.sigmoid(self.g_proj(x))  # [B, T, H]
            o = o * gate[..., None].to(o.dtype)
        B, T = o.shape[:2]
        return self.o_proj(o.reshape(B, T, self.inner))

    # ---- training / full-sequence forward ------------------------------------------------
    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                *, max_seqlen: int | None = None, rope: tuple[torch.Tensor, torch.Tensor] | None = None,
                **_: Any) -> torch.Tensor:
        """x: [B, T, d].  With ``cu_seqlens`` (B == 1) documents are isolated and ``position_ids``
        [1, T] give in-document positions; without it each row is one sequence at positions 0..T−1."""
        B, T, _d = x.shape
        if cu_seqlens is None:
            cu = torch.arange(0, (B + 1) * T, T, device=x.device, dtype=torch.int32)
            max_seqlen = T
            if position_ids is None:
                position_ids = torch.arange(T, device=x.device)[None].expand(B, T)
        else:
            assert B == 1, "packed varlen input must have batch size 1"
            cu = cu_seqlens
            if position_ids is None:
                raise ValueError("position_ids are required with cu_seqlens")
            if max_seqlen is None:
                max_seqlen = int((cu[1:] - cu[:-1]).max())
        conv_cu = cu_seqlens  # dense rows need no cu for the conv (it is per-row causal)
        q, k, v, _ = self._qkv(x, conv_cu, None, False)
        q, k = self._norm_rope(q, k, position_ids, rope)
        N = B * T
        qf, kf, vf = (t.reshape(N, self.n_heads, self.head_dim) for t in (q, k, v))
        if self.collect_attn_stats:
            self._record_stats(qf, kf, cu)
        o = attn_varlen_train(qf, kf, vf, cu, max_seqlen, self.scale)
        return self._gate_out(o.reshape(B, T, self.n_heads, self.head_dim), x)

    @torch.no_grad()
    def _record_stats(self, q: torch.Tensor, k: torch.Tensor, cu: torch.Tensor) -> None:
        n = int(min(int(cu[1] - cu[0]), 512))
        qs = q[:n].float().transpose(0, 1)  # [H, n, D]
        ks = k[:n].float().transpose(0, 1)
        logits = (qs @ ks.transpose(-1, -2)) * self.scale
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        p = logits.softmax(-1)
        ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean()
        self.attn_stats = {"max_logit": float(logits[:, mask].max()), "entropy": float(ent)}

    # ---- inference --------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x: [B, T, d] (one document per row, no padding) → (y, state).

        state = {"k": [B, T, H, hd], "v": [B, T, H, hd]} (post-norm/RoPE keys, in the compute dtype —
        bf16 under autocast) plus ``"conv": [B, 3·H·hd, W]`` fp32 for F_gc.
        """
        B, T, _ = x.shape
        q, k, v, conv = self._qkv(x, None, None, self.use_conv)
        pos = torch.arange(T, device=x.device)[None].expand(B, T)
        q, k = self._norm_rope(q, k, pos, None)
        o = attn_inference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.scale)
        y = self._gate_out(o.transpose(1, 2), x)
        state = {"k": k.contiguous(), "v": v.contiguous()}
        if self.use_conv:
            state["conv"] = conv.float().contiguous()
        return y, state

    @torch.no_grad()
    def extend(self, x: torch.Tensor, state: dict[str, torch.Tensor], pos0: int
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Teacher-forced continuation of M tokens from ``state`` (cache length ``pos0``).

        Functional: the input ``state`` tensors are never written; a new state dict is returned.
        """
        B, M, _ = x.shape
        q, k, v, conv = self._qkv(x, None, state.get("conv"), self.use_conv)
        pos = torch.arange(pos0, pos0 + M, device=x.device)[None].expand(B, M)
        q, k = self._norm_rope(q, k, pos, None)
        K = torch.cat([state["k"], k.to(state["k"].dtype)], dim=1)
        V = torch.cat([state["v"], v.to(state["v"].dtype)], dim=1)
        qh = q.transpose(1, 2).to(K.dtype)
        o = attn_inference(qh, K.transpose(1, 2), V.transpose(1, 2), self.scale)
        y = self._gate_out(o.transpose(1, 2).to(v.dtype), x)
        new = {"k": K, "v": V}
        if self.use_conv:
            new["conv"] = conv.float()
        return y, new
