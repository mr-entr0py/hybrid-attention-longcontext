"""Layer patterns and named configurations (protocol §3.2).

Notation: ``-`` GDN-A, ``A`` F_g, ``C`` F_gc, ``E`` GDN-E, ``G`` GLA, ``N`` GDN-nc.
"""

from __future__ import annotations

import re
from typing import Any

L_DEFAULT = 12

#: Composition ks of the study (C4 optional, Tier C).
COMPOSITION_KS = (0, 1, 2, 3, 4, 6, 12)


def centred_positions(k: int, L: int = L_DEFAULT) -> list[int]:
    """Centred-uniform positions p_j = ⌊(j + ½)·L/k⌋ (all layers when k == L)."""
    if not 0 <= k <= L:
        raise ValueError(f"k must be in [0, {L}]")
    if k == L:
        return list(range(L))
    # exact rational arithmetic: floor((2j+1)·L / (2k))
    return [((2 * j + 1) * L) // (2 * k) for j in range(k)]


def pattern_from_positions(positions: list[int] | set[int], L: int = L_DEFAULT, full: str = "A", linear: str = "-") -> str:
    s = [linear] * L
    for p in positions:
        s[p] = full
    return "".join(s)


def centred(k: int, L: int = L_DEFAULT, full: str = "A", linear: str = "-") -> str:
    """Composition pattern with k full layers at the centred-uniform positions."""
    return pattern_from_positions(centred_positions(k, L), L, full, linear)


#: Placement arms at k* (protocol §3.2 table).  "middle" equals the composition arm.
PLACEMENT_POSITIONS: dict[int, dict[str, list[int]]] = {
    1: {"early": [1], "middle": [6], "late": [11]},
    2: {"early": [1, 7], "middle": [3, 9], "late": [5, 11]},
    3: {"early": [1, 5, 9], "middle": [2, 6, 10], "late": [3, 7, 11]},
}


def placement(kstar: int, where: str) -> str:
    """Placement pattern at k* ∈ {1, 2, 3} and where ∈ {'early', 'middle', 'late'}."""
    if kstar not in PLACEMENT_POSITIONS:
        raise ValueError("k* must be 1, 2 or 3")
    return pattern_from_positions(PLACEMENT_POSITIONS[kstar][where])


def named_config(name: str) -> tuple[str, dict[str, Any]]:
    """Return ``(pattern, overrides)`` for a named configuration.

    Supported names:
      * ``C{k}`` for k ∈ {0, 1, 2, 3, 4, 6, 12} (composition, centred-uniform);
      * ``P{k*}-early|middle|late`` (placement);
      * ``X-gc{k}`` (F_g → F_gc), ``X-nope{k}`` (NoPE on all full layers),
        ``X-E{k}`` (GDN-A → GDN-E), ``O-gla{k}`` (GDN-A → GLA), ``O-nc{k}`` (GDN-A → GDN-nc).

    ``O-twin12`` (Primer-style twin, Tier C) is not implemented.
    """
    m = re.fullmatch(r"C(\d+)", name)
    if m:
        return centred(int(m.group(1))), {}
    m = re.fullmatch(r"P([123])-(early|middle|late)", name)
    if m:
        return placement(int(m.group(1)), m.group(2)), {}
    m = re.fullmatch(r"X-gc(\d+)", name)
    if m:
        return centred(int(m.group(1)), full="C"), {}
    m = re.fullmatch(r"X-nope(\d+)", name)
    if m:
        return centred(int(m.group(1))), {"nope": True}
    m = re.fullmatch(r"X-E(\d+)", name)
    if m:
        return centred(int(m.group(1)), linear="E"), {}
    m = re.fullmatch(r"O-gla(\d+)", name)
    if m:
        return centred(int(m.group(1)), linear="G"), {}
    m = re.fullmatch(r"O-nc(\d+)", name)
    if m:
        return centred(int(m.group(1)), linear="N"), {}
    if name == "O-twin12":
        raise NotImplementedError("O-twin12 (Tier C Primer-style twin) is not implemented in this package")
    raise KeyError(f"unknown named config {name!r}")


#: The named configurations of protocol §3.4 (both scales).
TABLE_CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C6", "C12", "X-gc12", "X-E0"]

#: Convenience dict of raw patterns for the composition configs.
CONFIGS: dict[str, str] = {f"C{k}": centred(k) for k in COMPOSITION_KS}
