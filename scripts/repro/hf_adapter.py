"""Thin HF-checkpoint wrapper for the scoring contract (reproduction_plan.md §2.7, R7).

``HFLogitsWrapper(hf_causal_lm)`` exposes exactly what ``hybridlc.eval.adapter.ForwardOnlyAdapter``
needs: ``lm_head_weight`` (device probe) and ``full_logits(ids[1, T]) -> fp32 logits[1, T, V]``.
``ForwardOnlyAdapter(HFLogitsWrapper(m))`` then provides ``prefill`` / ``score_continuation``
/ ``decode_step`` by recomputation, and ``hybridlc.eval.scorer.score_file`` scores it like
any HybridLM. Nothing in ``src/`` is modified.

Loaders:
* ``load_pythia(repo_dir)``  -> transformers ``GPTNeoXForCausalLM`` (fp16 weights cast to fp32).
* ``load_fla_gdn(repo_dir)`` -> fla 0.6.0 ``GatedDeltaNetForCausalLM`` (the class our
  ``GDNMixer`` subclasses the layer of), loaded from the safetensors file with a strict
  key check (zero missing / unexpected keys required; legacy config keys are reported).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class HFLogitsWrapper(torch.nn.Module):
    """Single-sequence fp32-logit view of a HF / fla causal LM (no cache, no padding)."""

    def __init__(self, lm: torch.nn.Module, lm_head: torch.nn.Linear, max_oom_retries: int = 60):
        super().__init__()
        self.lm = lm
        self._lm_head = lm_head
        self.max_oom_retries = max_oom_retries
        self.oom_events: list[dict] = []  # GPU events (shared GPU): each retried CUDA OOM is logged

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self._lm_head.weight

    @torch.no_grad()
    def full_logits(self, ids: torch.Tensor) -> torch.Tensor:
        assert ids.dim() == 2 and ids.shape[0] == 1, "one sequence per forward (plan §2.7)"
        import time

        for attempt in range(self.max_oom_retries + 1):
            try:
                out = self.lm(input_ids=ids, use_cache=False)
                return out.logits.float()
            except (torch.OutOfMemoryError, RuntimeError) as e:  # Triton launch OOM is a RuntimeError
                if "out of memory" not in str(e).lower() or attempt == self.max_oom_retries:
                    raise
                free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else -1
                self.oom_events.append(dict(t=time.time(), T=int(ids.shape[1]), attempt=attempt,
                                            free_bytes=int(free), err=str(e)[:160]))
                torch.cuda.empty_cache()
                time.sleep(2.0)
        raise AssertionError("unreachable")


def _strict_load(model: torch.nn.Module, st_path: Path, key_map=None) -> dict:
    from safetensors.torch import load_file

    sd = load_file(str(st_path))
    map_rep = {}
    if key_map is not None:
        sd, map_rep = key_map(model, sd)
    own = model.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shape_bad = sorted(k for k in set(own) & set(sd) if tuple(own[k].shape) != tuple(sd[k].shape))
    rep = dict(n_ckpt_keys=len(sd), n_model_keys=len(own), missing=missing, unexpected=unexpected,
               shape_mismatch=shape_bad, ckpt_dtypes=sorted({str(v.dtype) for v in sd.values()}), key_map=map_rep)
    if missing or unexpected or shape_bad:
        raise RuntimeError(f"strict load failed: {json.dumps(rep)[:2000]}")
    model.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()}, strict=True)
    return rep


def pythia_key_map(model: torch.nn.Module, sd: dict) -> tuple[dict, dict]:
    """Documented key map for transformers-4.24 GPT-NeoX checkpoints under transformers 5.x.

    * ``embed_out.weight`` -> ``lm_head.weight`` (module renamed; untied head).
    * Per-layer legacy *buffers* are dropped only after checking they equal what the current
      model computes: ``attention.bias`` must be the lower-triangular causal mask,
      ``attention.masked_bias`` must be -inf / a large negative fill (unused by SDPA causal
      masking), ``attention.rotary_emb.inv_freq`` must equal the model's shared
      ``gpt_neox.rotary_emb.inv_freq`` to fp16 precision.
    """
    out, dropped, checks = {}, [], {}
    ref_inv = model.gpt_neox.rotary_emb.inv_freq.detach().float().cpu()
    for k, v in sd.items():
        if k == "embed_out.weight":
            out["lm_head.weight"] = v
        elif k.endswith(".attention.bias"):
            n = v.shape[-1]
            ok = bool(torch.equal(v.reshape(n, n).bool(), torch.ones(n, n, dtype=torch.bool).tril()))
            checks.setdefault("causal_mask_is_tril", True)
            checks["causal_mask_is_tril"] &= ok
            dropped.append(k)
        elif k.endswith(".attention.masked_bias"):
            fv = float(v)
            checks.setdefault("masked_bias_values", set()).add(fv)
            dropped.append(k)
        elif k.endswith(".attention.rotary_emb.inv_freq"):
            d = float((v.float() - ref_inv).abs().max()) if v.shape == ref_inv.shape else float("inf")
            checks["inv_freq_max_abs_diff"] = max(checks.get("inv_freq_max_abs_diff", 0.0), d)
            dropped.append(k)
        else:
            out[k] = v
    checks["masked_bias_values"] = sorted(checks.get("masked_bias_values", []))
    if not checks.get("causal_mask_is_tril", True) or checks.get("inv_freq_max_abs_diff", 0.0) > 1e-3 \
            or any(x > -1e4 for x in checks["masked_bias_values"]):
        raise RuntimeError(f"legacy buffer check failed: {checks}")
    return out, dict(renamed={"embed_out.weight": "lm_head.weight"}, n_dropped_buffers=len(dropped), checks=checks)


def load_pythia(repo_dir: str | Path, device: str = "cuda") -> tuple[HFLogitsWrapper, dict]:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    repo_dir = Path(repo_dir)
    cfg = GPTNeoXConfig.from_pretrained(str(repo_dir))
    m = GPTNeoXForCausalLM(cfg).float()
    rep = _strict_load(m, repo_dir / "model.safetensors", key_map=pythia_key_map)
    m = m.to(device).eval()
    head = m.get_output_embeddings()
    rep.update(cls=type(m).__name__, attn_implementation=getattr(m.config, "_attn_implementation", None),
               tie_word_embeddings=bool(cfg.tie_word_embeddings),
               lm_head_is_embed=bool(head.weight.data_ptr() == m.gpt_neox.embed_in.weight.data_ptr()))
    if rep["lm_head_is_embed"]:
        raise RuntimeError("lm_head is tied to the embedding but the checkpoint is untied")
    return HFLogitsWrapper(m, head).eval(), rep


FLA_GDN_KNOWN_LEGACY = ("qk_activation", "qk_norm", "use_beta", "expand_k", "use_output_norm", "norm_first", "attn")


def fla_gdn_key_map(model: torch.nn.Module, sd: dict) -> tuple[dict, dict]:
    """Documented key map for pre-2025-03-12 fla GatedDeltaNet checkpoints under fla 0.6.0.

    ``model.layers.*.attn.D`` is dropped. Provenance: fla commit 834989cc (2025-03-12,
    "[Gated DeltaNet] Remove redundant parameter (fix #226)") deletes only the declaration
    ``self.D = nn.Parameter(torch.ones(num_heads))``; in the parent file (d30c0833) ``self.D``
    is referenced nowhere else, i.e. it never entered the forward. Extra check (revised after
    the first load showed D == 0.0, not the 1.0 I had assumed as init): every D must be exactly
    0.0, which makes dropping it an exact no-op even under the NVlabs reading ``o + D * v``.
    """
    out, dropped, vals = {}, [], set()
    for k, v in sd.items():
        if k.startswith("model.layers.") and k.endswith(".attn.D"):
            vals.update(float(x) for x in v.flatten().unique())
            dropped.append(k)
        else:
            out[k] = v
    if vals and vals != {0.0}:
        raise RuntimeError(f"attn.D values are not all 0.0 ({sorted(vals)[:8]}): dropping D may not be a no-op")
    return out, dict(dropped=len(dropped), dropped_pattern="model.layers.*.attn.D", D_unique_values=sorted(vals),
                     provenance="fla 834989cc removes unused self.D (declared, never used in forward)")


def load_fla_gdn(repo_dir: str | Path, device: str = "cuda") -> tuple[HFLogitsWrapper, dict]:
    import inspect

    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    repo_dir = Path(repo_dir)
    raw = json.loads((repo_dir / "config.json").read_text())
    sig = set(inspect.signature(GatedDeltaNetConfig.__init__).parameters)
    hf_generic = {"architectures", "model_type", "torch_dtype", "transformers_version", "_name_or_path"}
    ignored = sorted(k for k in raw if k not in sig and k not in hf_generic)
    cfg = GatedDeltaNetConfig.from_pretrained(str(repo_dir))
    m = GatedDeltaNetForCausalLM(cfg).float()
    rep = _strict_load(m, repo_dir / "model.safetensors", key_map=fla_gdn_key_map)
    m = m.to(device).eval()
    rep.update(cls=type(m).__name__, config_keys_not_in_fla_signature=ignored,
               legacy_values={k: raw.get(k) for k in FLA_GDN_KNOWN_LEGACY if k in raw},
               tie_word_embeddings=bool(cfg.tie_word_embeddings),
               lm_head_is_embed=bool(m.lm_head.weight.data_ptr() == m.model.embeddings.weight.data_ptr()),
               attn_mode=cfg.attn_mode, fuse_norm=cfg.fuse_norm, fuse_swiglu=cfg.fuse_swiglu)
    if rep["lm_head_is_embed"]:
        raise RuntimeError("lm_head is tied to the embedding but the checkpoint is untied")
    return HFLogitsWrapper(m, m.lm_head).eval(), rep
