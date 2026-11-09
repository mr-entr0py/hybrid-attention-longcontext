"""VT-18 (part 2): operating characteristics of the H1 decision rules (debate record §6.2, PI-SIM).

The library's own decision code (``one_sample_t``, ``ols_contrast`` on ``additive_design``,
``seed_trigger_batch``, ``welch_contrast``) is driven by the simulation set-ups of
``scripts/review_measurements/{pi,revB2,sims}`` at **20,000 replicates** and compared with the numbers
those scripts produced (``hybridlc.analysis.opchar_reference``).

Criteria
--------
* Every cell tabulated in debate §6.2 (and quoted in protocol §11.3): ``|ours - ref| <= 0.01``
  (the VT-18 threshold). Expected-seed-count cells (E[n]) compare the trigger rate within ±0.01
  (i.e. E[n] within ±0.02), see the DEVIATION note in the analysis report.
* Every other printed cell of the scripts (R3 = 0.9 rows, ``bimodal13``, revB2, Reviewer A's tables):
  ``|ours - ref| <= max(0.01, 4 * combined Monte Carlo SE)``. Both sides are Monte Carlo estimates
  (the reference with 20k–200k replicates), so a strict ±0.01 at 20k is expected to be exceeded by
  ~1–2 of ~350 cells by chance alone; the test prints every cell beyond ±0.01 with its z-score.
* Exact quantities (no Monte Carlo): noncentral-t power and the H5 ladder artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridlc.analysis import opchar
from hybridlc.analysis.opchar_reference import H5_LADDER_SLOPES_PP, POWER_N_EXACT, REFERENCE
from hybridlc.analysis.protocol import PP
from hybridlc.analysis.stats import paired_t_power

NT = 20_000
REF_N = {"pi": 60_000, "revB2": 20_000, "rule_df6": 200_000, "rule_df11": 200_000}


def _debate_62_cells() -> list[tuple]:
    """Cells shown in the two §6.2 tables of reports/reviews/debate_record.md."""
    tests = [f"POOL_n{n}" for n in (3, 4, 5)] + [f"PAIR_n{n}" for n in (3, 4, 5)] + \
            ["ADAPT_3to5", "ADAPT_4to6", "ADAPT_5to7"]
    cells = []
    for scn in ("homo", "hetero", "bimodal"):  # sigma = 3 table
        for R3 in (0.8, 1.0):
            cells += [("pi", 3.0, scn, R3, t) for t in tests]
    for scn, R3s in (("homo", (0.8, 1.0)), ("hetero", (0.8,))):  # sigma = 2 table
        for R3 in R3s:
            cells += [("pi", 2.0, scn, R3, t) for t in tests]
    en = [("pi", s, "homo", 1.0, f"ADAPT_{a}_En") for s in (2.0, 3.0) for a in ("3to5", "4to6", "5to7")]
    return cells + en


@pytest.fixture(scope="module")
def ours() -> dict[tuple, float]:
    return opchar.run_all(NT)


def _ref_n(key: tuple) -> int:
    if key[0] == "pi" and key[-1].startswith("ADAPT"):
        return 40_000
    return REF_N[key[0]]


def _combined_se(key: tuple, p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return float(np.sqrt(p * (1 - p) * (1 / NT + 1 / _ref_n(key))) + 0.0005 / np.sqrt(3))  # + 3-dp rounding


def test_reference_values_match_debate_record_spot_checks():
    # values transcribed in reports/reviews/debate_record.md §6.2 and protocol §11.3
    assert REFERENCE[("pi", 3.0, "homo", 1.0, "ADAPT_4to6")] == 0.923
    assert REFERENCE[("pi", 2.0, "homo", 1.0, "ADAPT_4to6")] == 0.979
    assert REFERENCE[("pi", 3.0, "homo", 1.0, "ADAPT_5to7")] == 0.967
    assert REFERENCE[("pi", 2.0, "homo", 1.0, "ADAPT_5to7")] == 0.997
    assert REFERENCE[("pi", 3.0, "hetero", 0.8, "POOL_n3")] == 0.114
    assert REFERENCE[("revB2", "hetero", 0.8, "pooled")] == 0.145
    assert REFERENCE[("rule_df6", 3.0, 1.0, "A_sup")] == 0.952


def test_pisim_debate_62_within_001(ours):
    bad = []
    for key in _debate_62_cells():
        ref, got = REFERENCE[key], ours[key]
        if key[-1].endswith("_En"):
            # E[n] = n0 + 2 P(trigger): compare the trigger rate within +-0.01
            ok = abs(got - ref) / 2 <= 0.01 + 0.0025  # + rounding of the 2-dp E[n] (0.005 / 2)
        else:
            ok = abs(got - ref) <= 0.01
        if not ok:
            bad.append((key, ref, round(got, 4)))
    print(f"debate §6.2 cells checked: {len(_debate_62_cells())}")
    assert not bad, bad


def test_all_script_cells_within_mc_tolerance(ours):
    over_001, bad = [], []
    for key, ref in REFERENCE.items():
        got = ours[key]
        if key[-1].endswith("_En"):
            tol = max(0.02, 4 * 2 * _combined_se(key, (ref - int(key[-1][6])) / 2) + 0.005)
            if abs(got - ref) > tol:
                bad.append((key, ref, got))
            continue
        se = _combined_se(key, ref)
        if abs(got - ref) > 0.01:
            over_001.append((key, ref, round(got, 4), round((got - ref) / se, 2)))
        if abs(got - ref) > max(0.01, 4 * se):
            bad.append((key, ref, got))
    print(f"{len(REFERENCE)} cells; beyond ±0.01 (with z): {over_001}")
    assert not bad, bad
    assert len(over_001) <= 4  # ~1–2 expected by chance at 20k vs 20k–200k replicates


def test_paired_homoscedastic_cells_match_exact_power(ours):
    """PAIR under 'homo' has an exact answer: per-seed L3 ~ N((R3-0.8) G, 1.68 sigma^2)."""
    for sig in (2.0, 3.0):
        for R3 in (0.8, 0.9, 1.0):
            for n in (3, 4, 5):
                exact = paired_t_power((R3 - 0.8) * 30 * PP, sig * PP * np.sqrt(1.68), n) if R3 > 0.8 else 0.05
                assert abs(ours[("pi", sig, "homo", R3, f"PAIR_n{n}")] - exact) <= 0.015, (sig, R3, n)


def test_power_n_exact():
    for n, ref in POWER_N_EXACT.items():
        assert opchar.power_n_exact((n,))[("power_n", n)] == pytest.approx(ref, abs=5e-4)


def test_h5_ladder_artefact_exact():
    la = opchar.h5_ladder_artefact()
    assert la.r2_only == pytest.approx(H5_LADDER_SLOPES_PP["r2_only"], abs=0.005)
    assert la.r1_r2 == pytest.approx(H5_LADDER_SLOPES_PP["r1_r2"], abs=0.005)
    assert la.r1_r2_halved == pytest.approx(H5_LADDER_SLOPES_PP["r1_r2_halved"], abs=0.005)
    assert la.gaps_r2_only == pytest.approx([15.0, 70 / 3, 70 / 3, 32.5, 32.5, 32.5])
