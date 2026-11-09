"""Mixer factory (INTERFACES.md ``models/mixers.py``; used by ``build_model`` and by S1(b)).

``build_mixer(kind, cfg, layer_idx, **overrides)`` returns the sequence mixer for one pattern char:

====  =========================================================================================
kind  module
====  =========================================================================================
A     :class:`~hybridlc.models.attention.AttnMixer` F_g (conv=False)
C     :class:`~hybridlc.models.attention.AttnMixer` F_gc (conv=True)
-     :class:`~hybridlc.models.linear.GDNMixer` GDN-A
N     :class:`~hybridlc.models.linear.GDNMixer` GDN-nc (``use_short_conv=False``)
E     :class:`~hybridlc.models.linear.GDNMixer` GDN-E (1 head × d/2 / d)
G     :class:`~hybridlc.models.linear.GLAMixer` GLA
====  =========================================================================================

Every mixer has ``forward(x[B,T,d], cu_seqlens=None, position_ids=None, **ctx) -> [B,T,d]`` plus the
inference methods ``prefill(x) -> (y, state)`` and ``extend(x, state, pos0) -> (y, new_state)``.

Construction draws from the *global* torch RNG (nn.Linear defaults, fla's ``A_log``/``dt_bias``
draws); ``build_model`` therefore always calls this inside ``seeds.scoped_torch_seed`` with a seed
taken from the protocol ``init_mixer`` stream and then re-initialises the projections.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch.nn as nn

from .attention import AttnMixer
from .config import FULL_MIXERS, ModelConfig

#: Keyword arguments of ``AttnMixer`` that ``overrides`` may set for ``A``/``C``.
_ATTN_OVERRIDES = {"n_heads", "head_dim", "qk_norm", "rope", "out_gate", "ssmax", "rope_theta"}


def build_mixer(kind: str, cfg: ModelConfig, layer_idx: int, **overrides: Any) -> nn.Module:
    """Construct the mixer for pattern char ``kind`` (defaults = protocol §3.3; see module doc)."""
    if kind in FULL_MIXERS:
        bad = set(overrides) - _ATTN_OVERRIDES
        if bad:
            raise TypeError(f"unknown F_g/F_gc overrides {sorted(bad)}")
        n_heads = overrides.pop("n_heads", cfg.n_attn_heads)
        head_dim = overrides.pop("head_dim", cfg.attn_head_dim)
        return AttnMixer(
            cfg.d_model, n_heads, head_dim,
            qk_norm=overrides.pop("qk_norm", True),
            rope=overrides.pop("rope", not cfg.nope),
            rope_theta=overrides.pop("rope_theta", cfg.rope_theta),
            out_gate=overrides.pop("out_gate", True),
            ssmax=overrides.pop("ssmax", cfg.ssmax),
            conv=(kind == "C"),
            conv_size=cfg.conv_size,
            norm_eps=cfg.norm_eps,
            layer_idx=layer_idx,
        )
    geo = cfg.linear_geometry(kind)
    family = geo.pop("kind")
    geo.update(overrides)
    geo["layer_idx"] = layer_idx
    if family == "gdn":
        from .linear import GDNMixer

        with warnings.catch_warnings():
            # fla warns when use_short_conv=False (GDN-nc is a deliberate Tier-C arm).
            warnings.filterwarnings("ignore", message="ShortConvolution is crucial")
            return GDNMixer(**geo)
    if family == "gla":
        from .linear import GLAMixer

        return GLAMixer(**geo)
    raise ValueError(f"unknown mixer kind {kind!r}")  # pragma: no cover
