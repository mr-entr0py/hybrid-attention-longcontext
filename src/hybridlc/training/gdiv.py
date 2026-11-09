"""G-div divergence monitor (protocol §13.3).

Trigger: the step loss is NaN/inf, **or** the loss exceeds EMA_0.99(loss) + 1.0 for more than 50
consecutive steps. The EMA is the plain exponential moving average of the step loss (initialised
with the first finite loss and updated every step, including during an excursion — the literal
reading of the protocol). Its state is part of every full checkpoint, so detection is resume-exact.
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


class GDivMonitor:
    def __init__(self, cfg: GDivConfig, state: dict | None = None):
        self.cfg = cfg
        self.s = GDivState(**state) if state else GDivState()

    def state_dict(self) -> dict:
        return asdict(self.s)

    def update(self, step: int, loss: float) -> None:
        """Feed one step's loss; raise :class:`DivergenceError` if the gate fires."""
        if not math.isfinite(loss):
            raise DivergenceError(step, "non-finite loss", loss, self.s.ema)
        if self.s.ema is None:
            self.s.ema = loss
            return
        if loss > self.s.ema + self.cfg.margin:
            self.s.above += 1
            self.s.max_above = max(self.s.max_above, self.s.above)
        else:
            self.s.above = 0
        # check against the EMA *before* this step's update, then update the EMA
        self.s.ema = self.cfg.ema_decay * self.s.ema + (1.0 - self.cfg.ema_decay) * loss
        if self.s.above > self.cfg.patience:
            raise DivergenceError(step, f"loss > EMA+{self.cfg.margin} for {self.s.above} steps", loss, self.s.ema)
