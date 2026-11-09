"""S1 part (b) skeleton: OUR mixers inside Zoology's MQAR harness (protocol §17, G-RP (b)).

Status: SKELETON. It is written against the models/ contracts in ``src/hybridlc/INTERFACES.md``
(``ModelConfig`` plus the mixer-level factory ``models/mixers.py: build_mixer`` that this task
added to INTERFACES.md, because the pre-existing contract only exposed whole models). It
will run once ``hybridlc.models`` implements that factory; until then constructing a wrapper
raises a clear ``NotImplementedError``.

How Zoology uses a sequence mixer
---------------------------------
``zoology.mixers.hybrid.Hybrid`` instantiates ``configs[layer_idx % 2]`` as
``import_from_str(name)(d_model=d, layer_idx=i, **kwargs)``. In the original MQAR config,
layer 0 is ``BaseConv(kernel 3)`` and layer 1 is the sequence mixer under test. The mixer is
called as ``mixer(u)`` with ``u: float32[B, T, d]`` (dense batch, one sequence per row, no
padding, no position ids) and must return ``float32[B, T, d]``. ``state_size(sequence_length)``
is optional (only logged). The harness then re-initialises every ``nn.Linear`` /
``nn.Embedding`` with N(0, 0.02) and rescales parameters whose name contains
``out_proj.weight`` (``zoology/model.py::_init_weights``); that applies to our wrappers too.

Wrappers (dotted paths for Zoology's ModuleConfig; put scripts/s1_zoology on sys.path)
-------------------------------------------------------------------------------------
* ``our_mixers.OurFgZoology``: F_g configured to match Zoology's MHA (protocol §17): 2 heads,
  head_dim = d/2, **no output gate, no QK-norm, no RoPE**, causal softmax, scale 1/sqrt(head_dim).
  Known residual differences (inherent to F_g, protocol §3.3, not configurable, reported with
  the results): no biases (Zoology's Wqkv/out_proj have bias=True) and no attention dropout
  (Zoology: 0.1).
* ``our_mixers.OurGdnZoology``: our fla-GDN wrapper (GDN-A code path) in Zoology's geometry:
  2 heads, head_dim = d/2, expand_v = 2, use_gate = False, short conv 4 (same as
  ``zoology.mixers.gated_delta_net`` in ``add_gated_delta_net``).
* ``our_mixers.OurGdnConfigA``: exploratory (protocol §17, 2 runs): GDN-A head geometry
  d_k = 64, d_v = 128, H = d/128, use_gate = True, at d ∈ {128, 256}.

Run list (4 + 2 runs, protocol §17) is produced by :func:`part_b_runs` from Zoology's
original config objects, with only the sequence-mixer entry swapped.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

# Zoology geometry (models_repo.py add_attention / add_gated_delta_net at the pinned commit).
ZOOLOGY_NUM_HEADS = 2
ZOOLOGY_GDN_EXPAND_V = 2
ZOOLOGY_GDN_CONV = 4
ZOOLOGY_VOCAB = 8192
# GDN-A geometry (protocol §3.3).
GDN_A_HEAD_DIM = 64
GDN_A_EXPAND_V = 2

# Part (b) widths (protocol §17) and the exploratory config-A widths.
PART_B_WIDTHS = {"our_fg": (64, 128), "our_gdn": (128, 256)}
CONFIG_A_WIDTHS = (128, 256)


def _build_mixer(kind: str, d_model: int, layer_idx: int | None, **overrides: Any) -> nn.Module:
    """Call the hybridlc mixer factory (INTERFACES.md models/mixers.py)."""
    try:
        from hybridlc.models.config import ModelConfig  # type: ignore
        from hybridlc.models.mixers import build_mixer  # type: ignore
    except ImportError as exc:  # models/ not built yet
        raise NotImplementedError(
            "S1(b) needs hybridlc.models.config.ModelConfig and hybridlc.models.mixers.build_mixer "
            "(INTERFACES.md). They are not implemented yet."
        ) from exc
    # A 2-layer config purely to carry shared constants (norm eps, init std); the pattern is
    # irrelevant because we build a single mixer. d_ff is unused here (Zoology has no MLP).
    cfg = ModelConfig(d_model=d_model, d_ff=4 * d_model, n_layers=2, vocab_size=ZOOLOGY_VOCAB,
                      pattern="-" + kind if kind != "-" else "--")
    return build_mixer(kind, cfg, layer_idx=0 if layer_idx is None else layer_idx, **overrides)


class _ZoologyMixerAdapter(nn.Module):
    """Adapts a hybridlc mixer (forward(x, cu_seqlens=None, position_ids=None)) to Zoology's call."""

    def __init__(self, inner: nn.Module, d_model: int, state_fn) -> None:
        super().__init__()
        self.inner = inner
        self.d_model = d_model
        self._state_fn = state_fn

    def forward(self, u: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Dense batch of independent sequences: no cu_seqlens; positions are implicit 0..T-1
        # (irrelevant here because RoPE is off for F_g and GDN has no positions).
        return self.inner(u, cu_seqlens=None, position_ids=None)

    def state_size(self, sequence_length: int = 2048, **_: Any) -> int:
        return int(self._state_fn(sequence_length))


class OurFgZoology(_ZoologyMixerAdapter):
    """Our F_g as Zoology MHA: 2 × (d/2) heads, no gate, no QK-norm, no RoPE."""

    def __init__(self, d_model: int, layer_idx: int | None = None, num_heads: int = ZOOLOGY_NUM_HEADS,
                 **_: Any) -> None:
        head_dim = d_model // num_heads
        inner = _build_mixer("A", d_model, layer_idx, n_heads=num_heads, head_dim=head_dim,
                             qk_norm=False, rope=False, out_gate=False)
        # Same accounting as zoology.mixers.attention.MHA.state_size: K and V per token.
        super().__init__(inner, d_model, lambda T: 2 * d_model * T)


class OurGdnZoology(_ZoologyMixerAdapter):
    """Our fla-GDN wrapper in Zoology's geometry (2 heads, d_k = d/2, d_v = d, no gate, conv 4)."""

    def __init__(self, d_model: int, layer_idx: int | None = None, num_heads: int = ZOOLOGY_NUM_HEADS,
                 **_: Any) -> None:
        head_dim = d_model // num_heads
        inner = _build_mixer("-", d_model, layer_idx, num_heads=num_heads, head_dim=head_dim,
                             expand_v=ZOOLOGY_GDN_EXPAND_V, use_gate=False, use_short_conv=True,
                             conv_size=ZOOLOGY_GDN_CONV)
        dk, dv = head_dim, head_dim * ZOOLOGY_GDN_EXPAND_V
        super().__init__(inner, d_model, lambda T: num_heads * dk * dv)


class OurGdnConfigA(_ZoologyMixerAdapter):
    """Exploratory: GDN-A head geometry (d_k 64, d_v 128, H = d/128, output gate on)."""

    def __init__(self, d_model: int, layer_idx: int | None = None, **_: Any) -> None:
        h = d_model // (GDN_A_HEAD_DIM * GDN_A_EXPAND_V)
        assert h >= 1, f"config-A geometry needs d_model >= 128, got {d_model}"
        inner = _build_mixer("-", d_model, layer_idx)  # defaults = GDN-A (protocol §3.3)
        super().__init__(inner, d_model, lambda T: h * GDN_A_HEAD_DIM * GDN_A_HEAD_DIM * GDN_A_EXPAND_V)


# ---------------------------------------------------------------------------------------------
# Run list for part (b). Imports Zoology lazily (via zoology_compat) so this module stays
# importable as a mixer path without side effects.
# ---------------------------------------------------------------------------------------------
_SWAP = {"our_fg": ("attention", "our_mixers.OurFgZoology"),
         "our_gdn": ("gated_delta_net", "our_mixers.OurGdnZoology"),
         "our_gdn_configA": ("gated_delta_net", "our_mixers.OurGdnConfigA")}


def part_b_runs(best_lr: dict[tuple[str, int], float]) -> list[Any]:
    """Build the 4 + 2 part-(b) TrainConfigs.

    ``best_lr[(zoology_family, d_model)]`` is Zoology's best LR from S1(a) at that width
    (protocol §17 "at Zoology's best LR"; G-RP (b) compares "at matched d_model and LR", so the
    best LR is taken per width). The exploratory config-A runs reuse GDN's best LR at that width.
    """
    from s1_configs import build_runs  # Zoology's original configs (data, model, LR grid)

    base = {(r.family, r.d_model): r for r in build_runs()
            if (r.family, r.d_model) in best_lr and abs(r.lr - best_lr[(r.family, r.d_model)]) < 1e-12}
    out = []
    for tag, (fam, mixer_path) in _SWAP.items():
        widths = CONFIG_A_WIDTHS if tag == "our_gdn_configA" else PART_B_WIDTHS[tag]
        for d in widths:
            cfg = base[(fam, d)].config.model_copy(deep=True)
            seq = cfg.model.sequence_mixer.kwargs["configs"]
            assert seq[0]["name"].endswith("BaseConv"), seq
            seq[1] = {"name": mixer_path, "kwargs": {}}
            cfg.model.name = tag
            cfg.run_id = f"{tag}-d{d}-lr{cfg.learning_rate:.1e}"
            out.append(cfg)
    return out
