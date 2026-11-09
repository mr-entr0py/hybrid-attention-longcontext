"""Evaluation suites as data (protocol §9.3-9.6).

A ``Cell`` is one item file: (namespace, suite, task, level, T, n contexts, variant). The
RNG of a cell is ``eval_rng(namespace, task, level_name, T_name, variant)`` (§8.1: eval_test
entropy 20261109 / eval_dev 77, keyed by (task, level, T)). Because items are prefix-stable,
dev cells that appear in several dev suites with different n (e.g. DG 200 vs DC/FD 100)
contain identical leading items.

Variants: ``C0`` (no filler, T=None), ``C2`` (oracle position), ``T4`` (unseen template),
``natural`` / ``repetitive`` (R-dist filler), ``forward`` / ``reversed`` (M-hop order).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybridlc.eval.items import cell_name, level_name, t_name

K = 1024


@dataclass(frozen=True)
class Cell:
    namespace: str  # 'test' | 'dev'
    suite: str  # CS | MS | DG | DC | FD | Tier0
    task: str  # R1 | R-load | R-dist | R3 | M-hop | A-freq | A-count | R0 | LM-1 | LM-2 | LM-4
    level: tuple[tuple[str, Any], ...]  # frozen dict
    T: int | None
    n: int
    variant: str | None = None
    purpose: str = ""

    @property
    def level_dict(self) -> dict:
        return dict(self.level)

    @property
    def name(self) -> str:
        return cell_name(self.task, self.level_dict, self.T, self.variant)

    @property
    def relpath(self) -> str:
        return f"{self.suite}/{self.name}.npz"

    def gen_level(self) -> dict:
        """The ``task_level`` dict passed to the generator (level + variant switches)."""
        lv = dict(self.level)
        v = self.variant
        if v == "C0":
            lv["placement"] = "nofill"
        elif v == "C2":
            lv["placement"] = "oracle"
        elif v == "T4":
            lv["template"] = "T4"
        elif v in ("natural", "repetitive"):
            lv["filler"] = v
        elif v in ("forward", "reversed"):
            lv["order"] = v
        return lv

    def rng_key(self) -> tuple[str, ...]:
        return (self.task, level_name(self.level_dict), t_name(self.T), self.variant or "")


def _c(ns: str, suite: str, task: str, level: dict, T: int | None, n: int, variant: str | None = None,
       purpose: str = "") -> Cell:
    return Cell(ns, suite, task, tuple(sorted(level.items())), T, n, variant, purpose)


L_1_4 = (1 * K, 2 * K, 4 * K)
L_8_32 = (8 * K, 16 * K, 32 * K)
RLOAD_8K = (8, 32, 128, 512)
RLOAD_16_32 = (8, 32, 128, 512, 1024)
HOPS_ALL = (1, 2, 3, 4)
RHO_ALL = (4.0, 2.0, 1.5, 1.25)


def confirmatory(u15: bool = False) -> list[Cell]:
    """CS (§9.3), test namespace. ``u15``: raise the 16K/32K retrieval cells to 600 contexts."""
    ns, s = "test", "CS"
    big = 600 if u15 else 300
    out: list[Cell] = []
    for D in (3, 15):
        for T in (1 * K, 2 * K, 4 * K, 8 * K):
            out.append(_c(ns, s, "R1", {"D": D}, T, 200))
        for T in (16 * K, 32 * K):
            out.append(_c(ns, s, "R1", {"D": D}, T, big))
    for T in L_1_4:
        for N in (8, 32):
            out.append(_c(ns, s, "R-load", {"N": N}, T, 200))
    for N in RLOAD_8K:
        out.append(_c(ns, s, "R-load", {"N": N}, 8 * K, 200))
    for T in (16 * K, 32 * K):
        for N in RLOAD_16_32:
            out.append(_c(ns, s, "R-load", {"N": N}, T, big))
    out.append(_c(ns, s, "R-load", {"N": 2048}, 32 * K, 150, purpose="N50 only"))
    for T in L_1_4:
        for h in (1, 2):
            out.append(_c(ns, s, "M-hop", {"h": h}, T, 200))
    for T in L_8_32:
        for h in HOPS_ALL:
            out.append(_c(ns, s, "M-hop", {"h": h}, T, 200))
    for T in L_1_4:
        for rho in (4.0, 2.0):
            out.append(_c(ns, s, "A-freq", {"rho": rho}, T, 200))
    for T in L_8_32:
        for rho in RHO_ALL:
            out.append(_c(ns, s, "A-freq", {"rho": rho}, T, 200))
    # C0: every CS level except N = 2,048, per level (not per length)
    for D in (3, 15):
        out.append(_c(ns, s, "R1", {"D": D}, None, 200, "C0"))
    for N in RLOAD_16_32:
        out.append(_c(ns, s, "R-load", {"N": N}, None, 200, "C0"))
    for h in HOPS_ALL:
        out.append(_c(ns, s, "M-hop", {"h": h}, None, 200, "C0"))
    for rho in RHO_ALL:
        out.append(_c(ns, s, "A-freq", {"rho": rho}, None, 200, "C0"))
    return out


def mechanism() -> list[Cell]:
    """MS (§9.4). Load x length re-uses CS cells (no new items). Attention diagnostics use the
    first 50 items of the dev R1 D=3 32K cell; pathway diagnostics the first 100 items of the
    R-dist natural cell (both listed here as their own cells for convenience)."""
    ns, s = "test", "MS"
    out: list[Cell] = []
    for v in ("natural", "repetitive"):
        out.append(_c(ns, s, "R-dist", {}, 32 * K, 200, v, "H5 mechanism"))
    for T in (4 * K, 16 * K, 32 * K):
        for U in (2, 4):
            out.append(_c(ns, s, "R3", {"U": U}, T, 150))
    for T in (4 * K, 32 * K):
        out.append(_c(ns, s, "R1", {"D": 3}, T, 150, "C2"))
        out.append(_c(ns, s, "R-load", {"N": 32}, T, 150, "C2"))
        out.append(_c(ns, s, "M-hop", {"h": 2}, T, 150, "C2"))
        out.append(_c(ns, s, "A-freq", {"rho": 2.0}, T, 150, "C2"))
    for T in (8 * K, 32 * K):
        out.append(_c(ns, s, "A-count", {}, T, 150))
    for v in ("forward", "reversed"):
        out.append(_c(ns, s, "M-hop", {"h": 2}, 16 * K, 150, v, "M-order"))
    out.append(_c(ns, s, "M-hop", {"h": 5}, 32 * K, 150, purpose="top level"))
    out.append(_c(ns, s, "A-freq", {"rho": 1.1}, 32 * K, 150, purpose="top level"))
    for T in (8 * K, 32 * K):
        out.append(_c(ns, s, "R1", {"D": 3}, T, 100, "T4"))
        out.append(_c(ns, s, "M-hop", {"h": 2}, T, 100, "T4"))
        out.append(_c(ns, s, "A-freq", {"rho": 2.0}, T, 100, "T4"))
    out.append(_c("dev", s, "R1", {"D": 3}, 32 * K, 50, purpose="attention diagnostics (k >= 1)"))
    return out


def dev_gate() -> list[Cell]:
    """DG (§9.5), dev namespace."""
    ns, s = "dev", "DG"
    out: list[Cell] = []
    for T in (16 * K, 32 * K):
        for D in (3, 15):
            out.append(_c(ns, s, "R1", {"D": D}, T, 200))
        for N in RLOAD_16_32:
            out.append(_c(ns, s, "R-load", {"N": N}, T, 200))
        for h in HOPS_ALL:
            out.append(_c(ns, s, "M-hop", {"h": h}, T, 200, purpose="C12 only (U-11)"))
    for D in (3, 15):
        out.append(_c(ns, s, "R1", {"D": D}, None, 200, "C0"))
    for N in RLOAD_16_32:
        out.append(_c(ns, s, "R-load", {"N": N}, None, 200, "C0"))
    return out


def dev_checkpoint() -> list[Cell]:
    """DC (§9.5), dev namespace, 100 x 4."""
    ns, s = "dev", "DC"
    out: list[Cell] = []
    for T in (4 * K, 16 * K, 32 * K):
        out.append(_c(ns, s, "R1", {"D": 3}, T, 100))
        for N in (32, 128):
            out.append(_c(ns, s, "R-load", {"N": N}, T, 100))
        out.append(_c(ns, s, "M-hop", {"h": 2}, T, 100))
        out.append(_c(ns, s, "A-freq", {"rho": 2.0}, T, 100))
    return out


def feas_dev() -> list[Cell]:
    """FD (§9.5), dev namespace, 100 x 4 (LM-2 and attention diagnostics come from Tier0 / MS)."""
    ns, s = "dev", "FD"
    out: list[Cell] = []
    for T in (4 * K, 8 * K, 16 * K, 32 * K):
        for D in (3, 15):
            out.append(_c(ns, s, "R1", {"D": D}, T, 100))
    for N in (8, 32):
        out.append(_c(ns, s, "R-load", {"N": N}, 4 * K, 100))
    for N in RLOAD_8K:
        out.append(_c(ns, s, "R-load", {"N": N}, 8 * K, 100))
    for T in (16 * K, 32 * K):
        for N in RLOAD_16_32:
            out.append(_c(ns, s, "R-load", {"N": N}, T, 100))
    for D in (3, 15):
        out.append(_c(ns, s, "R1", {"D": D}, None, 100, "C0"))
    for N in RLOAD_16_32:
        out.append(_c(ns, s, "R-load", {"N": N}, None, 100, "C0"))
    for T in (8 * K, 32 * K):
        for h in HOPS_ALL:
            out.append(_c(ns, s, "M-hop", {"h": h}, T, 100))
        for rho in RHO_ALL:
            out.append(_c(ns, s, "A-freq", {"rho": rho}, T, 100))
    return out


def tier0() -> list[Cell]:
    """Tier 0 (§9.6), test namespace (LM-2 uses the fixed held-out LM2 stream)."""
    ns, s = "test", "Tier0"
    out = [_c(ns, s, "LM-1", {}, 32 * K, 400, purpose="LM-1 + LM-3 annotations"),
           _c(ns, s, "LM-2", {}, 4 * K, 2000)]
    for var in ("a", "b"):
        for G in (0, 512, 2 * K, 8 * K, 30 * K):
            out.append(_c(ns, s, "LM-4", {"variant": var, "G": G}, 1 + 2 * 256 + G, 200))
    for T in (4 * K, 32 * K):
        out.append(_c(ns, s, "R0", {}, T, 100))
    return out


SUITES = {"CS": confirmatory, "MS": mechanism, "DG": dev_gate, "DC": dev_checkpoint, "FD": feas_dev, "Tier0": tier0}


def suite_cells(name: str, **kw: Any) -> list[Cell]:
    cells = SUITES[name](**kw) if name == "CS" else SUITES[name]()
    names = [c.relpath for c in cells]
    assert len(names) == len(set(names)), f"duplicate cells in {name}"
    return cells


def all_cells(u15: bool = False) -> list[Cell]:
    return [c for nm in SUITES for c in suite_cells(nm, **({"u15": u15} if nm == "CS" else {}))]


def tokens_per_suite(cells: list[Cell], q: int = 4) -> dict[str, int]:
    """Approximate prefill tokens per suite (contexts x T), for budgeting."""
    out: dict[str, int] = {}
    for c in cells:
        out[c.suite] = out.get(c.suite, 0) + c.n * (c.T or 0)
    return out
