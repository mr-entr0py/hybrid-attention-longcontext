"""Learning-rate schedules (protocol §6.3, §7.2, §7.3).

All schedules are pure functions of the (0-based) optimiser step, so a resumed run reproduces them
exactly. ``step`` counts optimiser updates within the run (stage-local); the LR returned is the one
used for the update performed *at* that step.

* ``warmup_constant`` — Stage A (500-step linear warm-up to eta*, then constant) and the TEF
  (50-step linear re-warm 0 -> 3e-4, then constant). Warm-up uses lr = peak·(step+1)/W so that the
  first update is not a no-op and step W-1 reaches the peak.
* ``linear_decay`` — Stage B: linear decay eta* -> 0 over the B steps from the branch point S,
  lr = peak·(B - step)/B (step 0 uses eta*, the last step eta*/B, and the value at step B is 0).
* ``wsd`` — LR-sweep compressed WSD (115 warm-up, constant to step 916, linear decay to 0 over the
  final 229 steps of 1,145): lr = peak·(N - step)/(N - stable_until) for step >= stable_until.
"""
from __future__ import annotations

from .config import ScheduleConfig

KINDS = ("warmup_constant", "linear_decay", "wsd")


def _warm(step: int, warmup: int) -> float:
    if warmup <= 0 or step >= warmup:
        return 1.0
    return (step + 1) / warmup


def lr_at(step: int, s: ScheduleConfig) -> float:
    if step < 0:
        raise ValueError("negative step")
    if s.kind == "warmup_constant":
        return s.peak_lr * _warm(step, s.warmup_steps)
    if s.kind == "linear_decay":
        if step >= s.steps:
            return 0.0
        return s.peak_lr * _warm(step, s.warmup_steps) * (s.steps - step) / s.steps
    if s.kind == "wsd":
        if s.stable_until is None:
            raise ValueError("wsd needs stable_until")
        if step >= s.steps:
            return 0.0
        if step >= s.stable_until:
            return s.peak_lr * (s.steps - step) / (s.steps - s.stable_until)
        return s.peak_lr * _warm(step, s.warmup_steps)
    raise ValueError(f"unknown schedule kind {s.kind!r}; expected one of {KINDS}")
