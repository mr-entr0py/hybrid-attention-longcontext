"""G-div divergence monitor (protocol §13.3).

Trigger: the step loss is NaN/inf, **or** the loss exceeds EMA_0.99(loss) + 1.0 for more than 50
consecutive steps. The EMA is the plain exponential moving average of the step loss (initialised
with the first finite loss and updated every step, including during an excursion — the literal
reading of the protocol). Its state is part of every full checkpoint, so detection is resume-exact.

Near-miss logging (PI ruling D067 / PR-10; log-only, never an intervention): beside the literal rule,
a *frozen-EMA excursion* is tracked: when the loss first exceeds EMA + margin, the pre-excursion EMA is
frozen; the excursion lasts while loss > frozen EMA + margin. If it lasts more than ``patience`` steps
without the literal trigger firing, one ``gdiv_near_miss`` event is emitted (``pop_near_miss``), which
the trainer writes as a registry note (FEAS verdict flag F-nearmiss, DTS10-AT v2). The literal
trigger, its EMA and every restart decision are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import GDivConfig


class DivergenceError(RuntimeError):
    def __init__(self, step: int, reason: str, loss: float, ema: float | None):
        super().__init__(f"G-div at step {step}: {reason} (loss={loss}, ema={ema})")
        self.step, self.reason, self.loss, self.ema = step, reason, loss, ema


@dataclass
class GDivState:
    ema: float | None = None
    above: int = 0          # consecutive steps with loss > ema + margin
    max_above: int = 0
    # near-miss (log-only): frozen pre-excursion EMA, excursion length, whether it was already noted
    nm_ref: float | None = None
    nm_start: int | None = None
    nm_above: int = 0
    nm_noted: bool = False
    nm_max_excess: float = 0.0


class GDivMonitor:
    def __init__(self, cfg: GDivConfig, state: dict | None = None):
        self.cfg = cfg
        self.s = GDivState(**state) if state else GDivState()
        self._pending: dict | None = None

    def pop_near_miss(self) -> dict | None:
        """The near-miss event emitted by the last ``update`` (once per excursion), else None."""
        ev, self._pending = self._pending, None
        return ev

    def _track_near_miss(self, step: int, loss: float, ema_before: float) -> None:
        s, m = self.s, self.cfg.margin
        if s.nm_ref is None:
            if loss > ema_before + m:
                s.nm_ref, s.nm_start, s.nm_above, s.nm_noted, s.nm_max_excess = ema_before, step, 1, False, loss - ema_before
            return
        if loss > s.nm_ref + m:
            s.nm_above += 1
            s.nm_max_excess = max(s.nm_max_excess, loss - s.nm_ref)
            if s.nm_above > self.cfg.patience and not s.nm_noted:
                s.nm_noted = True
                self._pending = {"frozen_ema": s.nm_ref, "excursion_start": s.nm_start, "excursion_steps": s.nm_above,
                                 "max_excess": s.nm_max_excess, "margin": m, "patience": self.cfg.patience,
                                 "literal_above": s.above, "literal_max_above": s.max_above}
        else:
            s.nm_ref, s.nm_start, s.nm_above, s.nm_noted, s.nm_max_excess = None, None, 0, False, 0.0

    def state_dict(self) -> dict:
        return asdict(self.s)

    def update(self, step: int, loss: float) -> None:
        """Feed one step's loss; raise :class:`DivergenceError` if the gate fires."""
        if not math.isfinite(loss):
            raise DivergenceError(step, "non-finite loss", loss, self.s.ema)
        if self.s.ema is None:
            self.s.ema = loss
            return
        self._track_near_miss(step, loss, self.s.ema)
        if loss > self.s.ema + self.cfg.margin:
            self.s.above += 1
            self.s.max_above = max(self.s.max_above, self.s.above)
        else:
            self.s.above = 0
        # check against the EMA *before* this step's update, then update the EMA
        self.s.ema = self.cfg.ema_decay * self.s.ema + (1.0 - self.cfg.ema_decay) * loss
        if self.s.above > self.cfg.patience:
            self._pending = None  # the literal trigger fired: not a near miss
            raise DivergenceError(step, f"loss > EMA+{self.cfg.margin} for {self.s.above} steps", loss, self.s.ema)
