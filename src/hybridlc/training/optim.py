"""AdamW with the weight-decay groups of protocol §7.1.

Decay (0.1, decoupled) applies to every parameter with ``ndim >= 2`` — including the tied embedding
(seen once, since ``named_parameters`` de-duplicates shared tensors), conv kernels and the headwise
gate W_g. **No** decay on 1-D tensors (all norm gains incl. QK-norm and GDN o_norm), on anything
flagged ``_no_weight_decay`` (fla's ``A_log``, ``dt_bias``), or on SSMax s_h (matched by name, in
addition to being 1-D).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .config import OptimConfig


def wants_decay(name: str, p: torch.Tensor, no_wd_name_patterns: Iterable[str]) -> bool:
    if p.ndim < 2:
        return False
    if getattr(p, "_no_weight_decay", False):
        return False
    return not any(pat in name for pat in no_wd_name_patterns)


def param_groups(model: nn.Module, cfg: OptimConfig) -> tuple[list[dict], dict[str, list[str]]]:
    """Return (torch param groups, {"decay": names, "no_decay": names}) in a deterministic order."""
    decay, no_decay, dn, nn_ = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if wants_decay(name, p, cfg.no_wd_name_patterns):
            decay.append(p)
            dn.append(name)
        else:
            no_decay.append(p)
            nn_.append(name)
    groups = [dict(params=decay, weight_decay=cfg.weight_decay, group_name="decay"),
              dict(params=no_decay, weight_decay=0.0, group_name="no_decay")]
    return groups, {"decay": dn, "no_decay": nn_}


def build_optimizer(model: nn.Module, cfg: OptimConfig, lr: float = 0.0) -> tuple[torch.optim.Optimizer, dict]:
    if cfg.name.lower() != "adamw":
        raise ValueError(f"only AdamW is part of the protocol, got {cfg.name}")
    groups, names = param_groups(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    if n_params != n_grouped:
        raise AssertionError(f"param groups cover {n_grouped} of {n_params} parameters")
    on_cuda = all(p.is_cuda for g in groups for p in g["params"])
    # fused=True is the protocol setting. CPU fused AdamW exists in torch>=2.4, but we only use the
    # CPU path in unit tests; fall back to foreach there if fused is unavailable.
    kwargs = dict(lr=lr, betas=tuple(cfg.betas), eps=cfg.eps)
    try:
        opt = torch.optim.AdamW(groups, fused=cfg.fused, **kwargs)
    except (RuntimeError, ValueError):
        if on_cuda:
            raise
        opt = torch.optim.AdamW(groups, foreach=False, **kwargs)
    summary = {
        "n_decay_tensors": len(names["decay"]), "n_no_decay_tensors": len(names["no_decay"]),
        "n_decay_params": sum(p.numel() for p in groups[0]["params"]),
        "n_no_decay_params": sum(p.numel() for p in groups[1]["params"]),
        "no_decay_names": names["no_decay"], "fused": bool(opt.defaults.get("fused")),
    }
    return opt, summary


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr
