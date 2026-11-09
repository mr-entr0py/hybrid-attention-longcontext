"""Protocol-derived constants for the pre-registered analysis (protocol v1.0, §2, §9, §11).

Every number in this module is copied from ``reports/experimental_protocol.md`` (frozen). The section
is cited next to each constant. All accuracy-type quantities are on the **probability scale** in
[0, 1]: "15 pp" is written ``0.15``. Recovery quantities are unitless; slopes are per doubling of T
(i.e. per unit of log2 T), so "2 pp per doubling" is ``0.02``.

Nothing in here is an experimental *setting* that may be tuned; changing a value is a logged
deviation (DEV-xx in ``reports/decisions.md``).
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------------------------
# Units and naming
# --------------------------------------------------------------------------------------------
PP: Final[float] = 0.01
"""One percentage point on the probability scale."""

CONTROL_T: Final[int] = 0
"""Sentinel context length used for the no-filler C0 control cells (§9.3: "per level, not per
length"). Files are named ``<task>_<level>_C0.jsonl``."""

ALL_LENGTHS: Final[tuple[int, ...]] = (1024, 2048, 4096, 8192, 16384, 32768)  # §9.3
LONG_LENGTHS: Final[tuple[int, ...]] = (16384, 32768)  # §11.2 long band
FALLBACK_LENGTH: Final[int] = 8192  # §11.2 / I-05
Q_PER_CONTEXT: Final[int] = 4  # §9.1 query branching: Q = 4 queries per context

# Configuration name -> k (number of F_g layers of 12), §3.2.
CONFIG_K: Final[dict[str, int]] = {
    "C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C6": 6, "C12": 12,
}
ANCHOR_CONFIGS: Final[tuple[str, ...]] = ("C0", "C1", "C3", "C12")  # §8.2 core table (4 x n)
CURVE_CONFIGS: Final[tuple[str, ...]] = ("C0", "C1", "C2", "C3", "C6", "C12")  # H2 curve
L_LAYERS: Final[int] = 12

# --------------------------------------------------------------------------------------------
# Families, tasks and levels (§2 definitions, §9.3)
# --------------------------------------------------------------------------------------------
FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "retrieval": ("R1", "R-load"),
    "multihop": ("M-hop",),
    "aggregation": ("A-freq",),
}
"""Primary tasks of each family. Family score = equal-weight mean of task ladder means, each ladder
mean being the equal-weight mean over levels (§2)."""

LEVEL_PARAM: Final[dict[str, str]] = {
    "R1": "D", "R-load": "N", "M-hop": "h", "A-freq": "rho", "R3": "U", "A-count": "c",
    "R-dist": "d", "R0": "", "M-order": "h",
}
"""Canonical level prefix per task: a level is written ``f"{param}{value}"`` (e.g. ``D3``, ``N32``,
``h2``, ``rho1.5``). :func:`hybridlc.analysis.results.normalise_level` maps loose spellings."""

N50_ONLY_LEVELS: Final[frozenset[tuple[str, str]]] = frozenset({("R-load", "N2048")})
"""Levels that exist only for the N50 fit and never enter a ladder mean (§9.3)."""

H5_FIXED_LEVELS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    # §2 H5(a): fixed level set {R1 D in {3, 15}; R-load N in {8, 32}} at all six T.
    "retrieval": {"R1": ("D3", "D15"), "R-load": ("N8", "N32")},
    # §2 H5(b): multi-hop fixed set h in {1, 2}.
    "multihop": {"M-hop": ("h1", "h2")},
}
H3_MULTIHOP_LEVELS: Final[tuple[str, ...]] = ("h1", "h2", "h3", "h4")  # §11.7 H3

# --------------------------------------------------------------------------------------------
# Confirmatory tests (§2, §11.3)
# --------------------------------------------------------------------------------------------
ALPHA_FW: Final[float] = 0.05
R_TARGET: Final[float] = 0.8  # H1: R(k) >= 0.8
L_WEIGHT_S0: Final[float] = 1.0 - R_TARGET  # 0.2
L_WEIGHT_S12: Final[float] = R_TARGET  # 0.8
SUPPORT_FAMILY: Final[tuple[str, ...]] = ("H1", "H1-min", "H3", "H5a")
FALSIFICATION_FAMILY: Final[tuple[str, ...]] = ("H1", "H1-min")
SECONDARY_FAMILY: Final[tuple[str, ...]] = ("H4-c1", "H4-c2", "H6prime", "H5b")
H1_K: Final[int] = 3
H1MIN_K: Final[int] = 1
H3_MIN_GAP: Final[float] = 15 * PP  # every seed G >= 15 pp in both families, else p = 1

# --------------------------------------------------------------------------------------------
# G-REF and primary cell (§11.2)
# --------------------------------------------------------------------------------------------
GREF_ADEQUACY: Final[float] = 0.80  # (a) S12(T) >= 0.80 * S12(C0)
GREF_DISCRIMINABILITY: Final[float] = 15 * PP  # (b) S12(T) - S0(T) >= 15 pp
REF_INVERSION_MARGIN: Final[float] = 5 * PP  # hybrid exceeds S12 by >= 5 pp on DG

# --------------------------------------------------------------------------------------------
# Seed trigger U-05 (§11.4)
# --------------------------------------------------------------------------------------------
TRIGGER_RESID_SD: Final[float] = 2.5 * PP
TRIGGER_SEED_RANGE: Final[float] = 12 * PP
TRIGGER_EXTRA_SEEDS: Final[int] = 2

# --------------------------------------------------------------------------------------------
# I-22 falsification guards (§11.6)
# --------------------------------------------------------------------------------------------
DOSE_PLATEAU: Final[float] = 3 * PP  # |S3(4000) - S3(2000)| < 3 pp
SATURATION_MAX: Final[float] = 5 * PP  # 2x Stage-B changes S12 by <= 5 pp
REQUIRED_GATES: Final[tuple[str, ...]] = ("G0", "G-RP", "FEAS-1", "FEAS-2", "G-FULL")

# --------------------------------------------------------------------------------------------
# Sensitivity analyses (§11.3)
# --------------------------------------------------------------------------------------------
BOOTSTRAP_B: Final[int] = 10_000
BOOTSTRAP_ENTROPY: Final[int] = 2026  # §8.1 stream "bootstrap"
CI_LEVEL: Final[float] = 0.95

# --------------------------------------------------------------------------------------------
# Equivalence margins and secondary thresholds (§2, §11.7)
# --------------------------------------------------------------------------------------------
TOST_ACCURACY: Final[float] = 5 * PP
TOST_RECOVERY: Final[float] = 0.10
TOST_SLOPE: Final[float] = 2 * PP  # per doubling
TOST_LM_LOSS: Final[float] = 0.01  # nats
TOST_LATENCY_REL: Final[float] = 0.10
NONINFERIORITY_MARGIN: Final[float] = 5 * PP  # Delta = S12 - S3
BH_Q: Final[float] = 0.10  # exploratory family

H2_LINEAR_BAND: Final[float] = 0.10  # "linear or convex" if |R_iso(k) - k/12| <= 0.1 for all k
KSTAR_CANDIDATES: Final[tuple[int, ...]] = (1, 2, 3)
KSTAR_MIN_R: Final[float] = 0.3
KSTAR_DEFAULT: Final[int] = 3

# Informative level (§9.7): S12 in [20%, 90%] and G >= 15 pp on dev items.
INFORMATIVE_S12: Final[tuple[float, float]] = (0.20, 0.90)
INFORMATIVE_GAP: Final[float] = 15 * PP
EQUATED_PAIR_MAX: Final[float] = 10 * PP  # U-11 difficulty-equated |dS12| <= 10 pp

# --------------------------------------------------------------------------------------------
# Tier-A decision-point triggers (§11.8)
# --------------------------------------------------------------------------------------------
U02_TIER_B: Final[float] = 3 * PP
U02_CO_PRIMARY: Final[float] = 6 * PP
U02_SEEDS: Final[tuple[int, ...]] = (0, 1)
U06_DELTA_R3: Final[float] = 0.10
U06_DELTA_S12: Final[float] = 3 * PP

# --------------------------------------------------------------------------------------------
# Wording (§11.9)
# --------------------------------------------------------------------------------------------
PACKAGE_CLAIM: Final[str] = "k F_g layers of 12, the rest GDN-A as built"
SCOPE_STATEMENT: Final[str] = (
    "Scope: the chosen scale, GDN-A, this recipe; capability elicited after identical exposure "
    "(TEF). Prior work is described as 'consistent with'."
)
