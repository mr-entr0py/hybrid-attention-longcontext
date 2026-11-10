"""The hybrid LM (protocol §3.1), its initialisation (§7.1, §8.1) and its inference API (§9.1–9.2).

    h⁰ = E[x]
    u   = h^l + Mix_{a_l}(RMSNorm₁(h^l));   h^{l+1} = u + SwiGLU(RMSNorm₂(u))
    logits = RMSNorm_f(h¹²) · Eᵀ            (tied embedding, no biases anywhere)

Initialisation (``HybridLM.__init__`` with a ``seed_index``; never touches the caller's global RNG):

* **shared modules** (embedding, every MLP, every norm, the final norm) draw from stream
  ``init_shared`` with spawn key ``(s, 0, crc32(parameter_name))`` → bit-identical across patterns
  for the same seed index (common random numbers).  Embedding and W_gate/W_up ~ N(0, 0.02);
  W_down ~ N(0, 0.02/√24); norm gains = 1.
* **mixers** draw from stream ``init_mixer`` with key ``(s, 1, layer_idx, MIXER_TYPE_ID[char])``.
  The first draw seeds the (scoped) global RNG for the mixer constructor, so fla's own tensors
  (``A_log`` ~ log U(0,16), ``dt_bias``, ``o_norm``) keep fla's distribution; then every
  ``nn.Linear`` weight is redrawn N(0, 0.02), except the residual output projection (F_g W_O, GDN /
  GLA ``o_proj``) N(0, 0.02/√24).  W_g ~ N(0, 0.02) (gate starts at ≈ 0.5).  QK-norm gains = 1.
  The only bias in the family (GLA ``gk_proj[1].bias``, an fla default) is zeroed as fla's
  ``_init_weights``.
* **short-conv kernels** (GDN / GLA ``ShortConvolution``, F_gc ``CausalDWConv``) are redrawn
  N(0, 0.02) from the same ``init_mixer`` generator, *after* all linears (so the linear draws are
  unchanged).  This is fla's own *model* initialisation (``GatedDeltaNetPreTrainedModel._init_weights``
  re-draws every ``nn.Conv1d`` N(0, initializer_range=0.02), as in the VT-09 transplant target
  ``GatedDeltaNetForCausalLM``), not the layer constructor's nn.Conv1d default U(−1/√4, 1/√4)
  (std 0.289, ≈ 14× larger), which no fla model trains with.  F_gc mirrors it.  [IMPL] reading of
  §7.1 "conv kernels keep fla's own initialisation" (verification round 3,
  GDN-CONV-INIT-NOT-FLA-MODEL-INIT; the whole §7.1 init table is fla's ``_init_weights``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..utils.seeds import MIXER_TYPE_ID, STREAMS, crc32, scoped_torch_seed, torch_generator, torch_seed
from .attention import AttnMixer, CausalDWConv
from .cache import InferenceCache
from .config import FULL_MIXERS, ModelConfig
from .layers import RMSNorm, SwiGLU, init_normal_, logits_fp32, output_proj_std, rope_tables
from .mixers import build_mixer


class Block(nn.Module):
    """Pre-norm block: mixer then SwiGLU MLP, both residual."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, kind: str, mixer: nn.Module) -> None:
        super().__init__()
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = mixer
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, h: torch.Tensor, cu_seqlens: torch.Tensor | None, position_ids: torch.Tensor | None,
                ctx: dict[str, Any]) -> torch.Tensor:
        u = h + self.mixer(self.norm1(h), cu_seqlens, position_ids, **ctx)
        return u + self.mlp(self.norm2(u))

    # inference -------------------------------------------------------------------------
    def prefill(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, st = self.mixer.prefill(self.norm1(h))
        u = h + y
        return u + self.mlp(self.norm2(u)), st

    def extend(self, h: torch.Tensor, state: dict[str, torch.Tensor], pos0: int
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, st = self.mixer.extend(self.norm1(h), state, pos0)
        u = h + y
        return u + self.mlp(self.norm2(u)), st


class HybridLM(nn.Module):
    """12-layer hybrid LM.  See the module docstring and INTERFACES.md (``HybridLM``)."""

    def __init__(self, cfg: ModelConfig, seed_index: int | None = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed_index = seed_index
        self.activation_checkpointing = False
        # Construct under a scoped global seed so the caller's RNG state is untouched and every
        # constructor draw is a deterministic function of seed_index.
        base_seed = torch_seed(seed_index if seed_index is not None else 0, STREAMS["init_shared"], 0xC0175)
        with scoped_torch_seed(base_seed):
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            blocks = []
            for l, ch in enumerate(cfg.pattern):
                gen = self._mixer_generator(l, ch)
                ctor_seed = int(torch.randint(0, 2**62, (1,), generator=gen)) if gen is not None else base_seed + l
                with scoped_torch_seed(ctor_seed):
                    mixer = build_mixer(ch, cfg, l)
                blocks.append(Block(cfg, l, ch, mixer))
                if gen is not None:
                    self._init_mixer(mixer, gen)
            self.layers = nn.ModuleList(blocks)
            self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        if seed_index is not None:
            self._init_shared()

    # ---- initialisation ---------------------------------------------------------------
    def _mixer_generator(self, layer_idx: int, ch: str) -> torch.Generator | None:
        if self.seed_index is None:
            return None
        return torch_generator(self.seed_index, STREAMS["init_mixer"], layer_idx, MIXER_TYPE_ID[ch])

    def _init_mixer(self, mixer: nn.Module, gen: torch.Generator) -> None:
        std = self.cfg.init_std
        out_std = output_proj_std(std, self.cfg.n_layers)
        with torch.no_grad():
            for name, mod in mixer.named_modules():
                if isinstance(mod, nn.Linear):
                    is_out = name == "o_proj"
                    init_normal_(mod.weight, out_std if is_out else std, gen)
                    if mod.bias is not None:
                        mod.bias.zero_()
            # Short-conv kernels: N(0, init_std) like fla's model-level ``_init_weights`` (see the
            # module docstring), drawn after every linear so the linear draws do not move.
            for _, mod in mixer.named_modules():
                if isinstance(mod, (nn.Conv1d, CausalDWConv)):
                    init_normal_(mod.weight, std, gen)
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.zero_()

    def _init_shared(self) -> None:
        s = self.seed_index
        std = self.cfg.init_std
        out_std = output_proj_std(std, self.cfg.n_layers)

        def gen(name: str) -> torch.Generator:
            return torch_generator(s, STREAMS["init_shared"], crc32(name))

        with torch.no_grad():
            init_normal_(self.embed.weight, std, gen("embed.weight"))
            for l, blk in enumerate(self.layers):
                for pname in ("gate_proj", "up_proj", "down_proj"):
                    full = f"layers.{l}.mlp.{pname}.weight"
                    init_normal_(getattr(blk.mlp, pname).weight, out_std if pname == "down_proj" else std, gen(full))
                blk.norm1.weight.fill_(1.0)
                blk.norm2.weight.fill_(1.0)
            self.norm_f.weight.fill_(1.0)

    def shared_parameter_names(self) -> list[str]:
        """Names of the shared (pattern-independent) parameters: embedding, MLPs, norms."""
        return [n for n, _ in self.named_parameters() if ".mixer." not in n]

    # ---- misc ---------------------------------------------------------------------------
    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.embed.weight

    def set_activation_checkpointing(self, on: bool = True) -> None:
        """Per-block non-reentrant activation checkpointing (Stage B / TEF at 32K)."""
        self.activation_checkpointing = bool(on)

    def set_gate_bypass(self, on: bool = True) -> None:
        """Test flag (VT-09): skip the headwise gate multiplication in every full-attention mixer."""
        for m in self.modules():
            if isinstance(m, AttnMixer):
                m.gate_bypass = bool(on)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """fp32 logits for already-normalised hidden states [..., d]."""
        return logits_fp32(hidden, self.embed.weight)

    def _needs_rope(self) -> bool:
        return any(isinstance(b.mixer, AttnMixer) and b.mixer.use_rope for b in self.layers)

    def _has_attention(self) -> bool:
        return any(c in FULL_MIXERS for c in self.cfg.pattern)

    # ---- training forward ---------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, max_seqlen: int | None = None) -> torch.Tensor:
        """Varlen training path.  input_ids [1, N] long; cu_seqlens int32 [n_docs+1]; position_ids
        [1, N] long (reset per document).  Returns final-normed hidden states [1, N, d].

        With ``cu_seqlens=None`` the input may be a dense batch [B, T] of independent sequences.
        """
        ctx: dict[str, Any] = {}
        if self._has_attention():
            if cu_seqlens is not None:
                ctx["max_seqlen"] = int(max_seqlen) if max_seqlen is not None else int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            if self._needs_rope():
                pos = position_ids if position_ids is not None else torch.arange(input_ids.shape[1], device=input_ids.device)[None]
                cos, sin = rope_tables(pos, self.cfg.attn_head_dim, self.cfg.rope_theta)  # [B|1, N, hd]
                ctx["rope"] = (cos, sin)
        h = self.embed(input_ids)
        for blk in self.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                h = checkpoint(blk, h, cu_seqlens, position_ids, ctx, use_reentrant=False, preserve_rng_state=False)
            else:
                h = blk(h, cu_seqlens, position_ids, ctx)
        return self.norm_f(h)

    # ---- inference --------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, InferenceCache]:
        """input_ids [B, T] (one document per row, no padding) → (last_logits [B, V] fp32, cache).

        Uses the chunk GDN kernel and SDPA (FLASH pinned in bf16) — run it under bf16 autocast for
        the canonical path; the cache then stores bf16 KV and fp32 GDN/conv state.
        """
        h = self.embed(input_ids)
        states = []
        for blk in self.layers:
            h, st = blk.prefill(h)
            states.append(st)
        last = self.logits(self.norm_f(h[:, -1]))
        cache = InferenceCache(self.cfg.pattern, states, int(input_ids.shape[1]), last)
        return last, cache

    @torch.no_grad()
    def extend(self, token_ids: torch.Tensor, cache: InferenceCache) -> tuple[torch.Tensor, InferenceCache]:
        """Teacher-forced forward of ``token_ids`` [B, M] from ``cache`` (not mutated).

        Returns (logits [B, M, V] fp32, new cache).  GDN layers use the explicit
        ``fused_recurrent_gated_delta_rule`` call; attention layers SDPA over the cached KV.
        """
        h = self.embed(token_ids)
        pos0 = cache.seq_len
        new_states = []
        for blk, st in zip(self.layers, cache.layers):
            h, st2 = blk.extend(h, st, pos0)
            new_states.append(st2)
        logits = self.logits(self.norm_f(h))
        new = InferenceCache(cache.pattern, new_states, pos0 + int(token_ids.shape[1]), logits[:, -1], dict(cache.meta))
        return logits, new

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, cache: InferenceCache) -> tuple[torch.Tensor, InferenceCache]:
        """token_ids [B, 1] → (logits [B, V] fp32, cache advanced by one token, updated in place)."""
        logits, new = self.extend(token_ids, cache)
        cache.assign(new)
        return logits[:, -1], cache

    @torch.no_grad()
    def score_continuation(self, cache: InferenceCache, cont_ids: torch.Tensor, return_logits: bool = False):
        """Teacher-forced scoring of ``cont_ids`` [B, M] from a cache snapshot (cache not mutated).

        Returns ``(logprobs [B, M] fp32, argmax [B, M] long)`` where ``logprobs[:, j]`` is
        log p(cont_j | prefix, cont_<j) and ``argmax[:, j]`` the model's top token at that position
        (EM_tf = all(argmax == cont) over the answer tokens).  With ``return_logits=True`` the fp32
        logits [B, M, V] are returned as a third element.
        """
        if cache.last_logits is None:
            raise ValueError("cache has no last_logits (create it with prefill())")
        first = cache.last_logits[:, None].float()
        if cont_ids.shape[1] > 1:
            rest, _ = self.extend(cont_ids[:, :-1], cache)
            logits = torch.cat([first, rest.float()], dim=1)
        else:
            logits = first
        lp = F.log_softmax(logits, dim=-1).gather(-1, cont_ids[..., None].long())[..., 0]
        am = logits.argmax(-1)
        return (lp, am, logits) if return_logits else (lp, am)

    @torch.no_grad()
    def full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Single-pass fp32 logits [B, T, V] for dense rows (re-scoring check, §9.2; tests)."""
        return self.logits(self.forward(input_ids))


def build_model(cfg: ModelConfig | dict, seed_index: int) -> HybridLM:
    """INTERFACES.md: ``build_model(cfg, seed_index) -> HybridLM`` (fp32, on CPU)."""
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return HybridLM(cfg, seed_index)
