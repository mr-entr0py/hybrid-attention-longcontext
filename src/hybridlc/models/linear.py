"""Linear mixers: GDN-A / GDN-E / GDN-nc (fla ``GatedDeltaNet``) and GLA (fla ``GatedLinearAttention``).

The modules *subclass* the fla 0.6.0 layers, so parameters, parameter names and constructor-time
initialisation are fla's own (protocol §3.3, §7.1), and a state dict transplants 1:1 onto fla's
``GatedDeltaNetForCausalLM`` (VT-09).  Only ``forward`` is replaced, by an explicit re-statement of
fla's layer forward in which **every fla op is called with keyword arguments only** and the kernel
is chosen by us, never by fla's automatic ``q_len <= 64 → fused_recurrent`` switch
(``fla/layers/gated_deltanet.py`` L227–230; protocol §9.2, VT-08):

* training / full forward : ``chunk_gated_delta_rule`` (varlen via ``cu_seqlens``; resets state and
  short-conv state at document boundaries);
* inference prefill       : ``chunk_gated_delta_rule`` with ``output_final_state=True``;
* inference extend/decode : ``fused_recurrent_gated_delta_rule`` from the cached state.

The short convs use ``fla.modules.causal_conv1d.ops.causal_conv1d`` (Triton) with the same weights,
activation (SiLU) and tap convention as ``ShortConvolution``.  Inference never mutates an input
state: every call returns new tensors (``causal_conv1d`` with ``initial_state`` is out-of-place).

Inference state layout (per layer, per row; dtypes as stored):
  ``state``  fp32 [B, HV, d_v, d_k]  (``state_v_first=True``, fla's layer layout)
  ``conv_q`` / ``conv_k`` / ``conv_v`` fp32 [B, D, W]  (last W raw projection outputs)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from fla.layers import GatedDeltaNet, GatedLinearAttention
from fla.modules.causal_conv1d import ops as _conv_ops
from fla.ops import gated_delta_rule as _gdr_ops
from fla.ops import gla as _gla_ops

# Module-level indirection so tests can spy on the exact call (keyword-only check, VT-08).
chunk_gated_delta_rule = _gdr_ops.chunk_gated_delta_rule
fused_recurrent_gated_delta_rule = _gdr_ops.fused_recurrent_gated_delta_rule
chunk_gla = _gla_ops.chunk_gla
fused_recurrent_gla = _gla_ops.fused_recurrent_gla
causal_conv1d = _conv_ops.causal_conv1d


def _ops():
    """Resolve the op pointers at call time (so monkeypatching this module's names takes effect)."""
    import sys

    return sys.modules[__name__]


def _conv(module: Any, x: torch.Tensor, cu_seqlens: torch.Tensor | None, initial_state: torch.Tensor | None,
          want_final: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one fla ``ShortConvolution``'s weights through ``causal_conv1d`` (explicit keywords)."""
    w = rearrange(module.weight, "d 1 w -> d w")
    init = initial_state.to(x.dtype) if initial_state is not None else None
    y, fin = _ops().causal_conv1d(x=x, weight=w, bias=module.bias, initial_state=init,
                                  output_final_state=want_final, activation=module.activation,
                                  backend=module.backend, cu_seqlens=cu_seqlens)
    return y, (fin.float() if fin is not None else None)


class _LinearInferenceMixin:
    """Shared prefill/extend wrappers around ``_core``."""

    @torch.no_grad()
    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x: [B, T, d] (one document per row) → (y, state) via the chunk kernel."""
        y, st = self._core(x, cu_seqlens=None, state=None, want_state=True, recurrent=False)  # type: ignore[attr-defined]
        return y, st

    @torch.no_grad()
    def extend(self, x: torch.Tensor, state: dict[str, torch.Tensor], pos0: int = 0
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Continue M tokens from ``state`` with the explicit fused_recurrent kernel (functional)."""
        y, st = self._core(x, cu_seqlens=None, state=state, want_state=True, recurrent=True)  # type: ignore[attr-defined]
        return y, st


class GDNMixer(_LinearInferenceMixin, GatedDeltaNet):
    """fla ``GatedDeltaNet`` with an explicit forward.  Construct with the protocol kwargs
    (``ModelConfig.linear_geometry``), e.g. GDN-A at d = 512::

        GDNMixer(hidden_size=512, head_dim=64, num_heads=4, expand_v=2, use_gate=True,
                 use_short_conv=True, conv_size=4, conv_bias=False, allow_neg_eigval=False,
                 norm_eps=1e-5, mode='chunk')
    """

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None,  # type: ignore[override]
                position_ids: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        """x: [B, T, d]; with ``cu_seqlens`` (B == 1) every document starts from a zero state and a
        zero conv state.  ``position_ids`` are unused (no positional encoding in linear layers)."""
        y, _ = self._core(x, cu_seqlens=cu_seqlens, state=None, want_state=False, recurrent=False)
        return y

    def _core(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor | None, state: dict[str, torch.Tensor] | None,
              want_state: bool, recurrent: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        cu = cu_seqlens.long() if cu_seqlens is not None else None
        new: dict[str, torch.Tensor] = {}
        if self.use_short_conv:
            cs = state or {}
            q, new["conv_q"] = _conv(self.q_conv1d, self.q_proj(x), cu, cs.get("conv_q"), want_state)
            k, new["conv_k"] = _conv(self.k_conv1d, self.k_proj(x), cu, cs.get("conv_k"), want_state)
            v, new["conv_v"] = _conv(self.v_conv1d, self.v_proj(x), cu, cs.get("conv_v"), want_state)
        else:
            q, k, v = F.silu(self.q_proj(x)), F.silu(self.k_proj(x)), F.silu(self.v_proj(x))
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        beta = self.b_proj(x)
        g = self.a_proj(x)
        s0 = state["state"] if state is not None else None
        ops = _ops()
        if recurrent:
            o, s1 = ops.fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=s0, output_final_state=want_state,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias,
                use_beta_sigmoid_in_kernel=True, allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
                cu_seqlens=cu)
        else:
            o, s1 = ops.chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=s0, output_final_state=want_state,
                use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=self.allow_neg_eigval,
                state_v_first=True, cu_seqlens=cu, use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias)
        if self.use_gate:
            gate = rearrange(self.g_proj(x), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)
        y = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        if not want_state:
            return y, None
        new["state"] = s1.float()
        return y, {k_: v_ for k_, v_ in new.items() if v_ is not None}


class GLAMixer(_LinearInferenceMixin, GatedLinearAttention):
    """fla ``GatedLinearAttention`` (Tier C) with an explicit forward (chunk for training/prefill,
    fused_recurrent for extend).  Protocol kwargs: ``hidden_size=d, num_heads=d//128, expand_k=0.5,
    expand_v=1.0, use_short_conv=True, conv_size=4, use_output_gate=True`` (fla defaults otherwise)."""

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None,  # type: ignore[override]
                position_ids: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        y, _ = self._core(x, cu_seqlens=cu_seqlens, state=None, want_state=False, recurrent=False)
        return y

    def _core(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor | None, state: dict[str, torch.Tensor] | None,
              want_state: bool, recurrent: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.num_kv_groups != 1 or self.feature_map_fn is not None:
            raise NotImplementedError("GLAMixer supports the protocol geometry only (no GQA, no feature map)")
        cu = cu_seqlens.long() if cu_seqlens is not None else None
        new: dict[str, torch.Tensor] = {}
        if self.use_short_conv:
            cs = state or {}
            q, new["conv_q"] = _conv(self.q_conv1d, self.q_proj(x), cu, cs.get("conv_q"), want_state)
            k, new["conv_k"] = _conv(self.k_conv1d, self.k_proj(x), cu, cs.get("conv_k"), want_state)
            v, new["conv_v"] = _conv(self.v_conv1d, self.v_proj(x), cu, cs.get("conv_v"), want_state)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        gk = self.gk_proj(x)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        gk = rearrange(gk, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)
        s0 = state["state"] if state is not None else None
        ops = _ops()
        if recurrent:
            o, s1 = ops.fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=s0, output_final_state=want_state,
                                            state_v_first=True, cu_seqlens=cu)
        else:
            o, s1 = ops.chunk_gla(q=q, k=k, v=v, g=gk, initial_state=s0, output_final_state=want_state,
                                  state_v_first=True, cu_seqlens=cu)
        if self.use_output_gate:
            g = self.g_proj(x)
            if self.fuse_norm_and_gate:
                g = rearrange(g, "... (h d) -> ... h d", d=self.head_v_dim)
                o = rearrange(self.g_norm_swish_gate(o, g), "... h d -> ... (h d)")
            else:
                o = rearrange(self.g_norm(o), "... h d -> ... (h d)") * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), "... h d -> ... (h d)")
        y = self.o_proj(o)
        if not want_state:
            return y, None
        new["state"] = s1.float()
        return y, {k_: v_ for k_, v_ in new.items() if v_ is not None}
