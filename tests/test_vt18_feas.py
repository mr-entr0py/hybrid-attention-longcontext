"""T19: DTS-10 FEAS decision code acceptance tests (experiments/prereg/DTS10_acceptance.v2.yaml, every
``acceptance_tests`` item). Synthetic result trees only (CPU); no FEAS data exists or is read.

Tree builder: 9 TEF-1000 checkpoints (k in {0, 12} x seeds 101/102 x {30M, 90M} + SSMax k = 12 s101 30M), every FD
cell with 25 contexts x 4 queries (EM resolution 1/100), passes P0, P1, P4 as files, P2/P3 as re-score records of
the first 5 contexts. Planted EM values set exact ladder values (all ladder cells of a run at one length equal ->
ladder = that value)."""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from hybridlc.analysis import feas as FE

REPO = Path(__file__).resolve().parents[1]
N_CTX, M = 25, 4
N = N_CTX * M
SUB_CTX = 5


# ------------------------------------------------------------------------------------------ builder
def _design() -> list[FE.DesignCell]:
    out, h = [], 0
    for k, cfg in ((0, "C0"), (12, "C12")):
        for s in (101, 102):
            for b in ("30M", "90M"):
                h += 1
                rid = f"S2T-D384-{cfg}-s{s}-{h:08x}"
                out.append(FE.DesignCell(run_id=rid, config=cfg, k=k, arm="plain", seed=s, branch=b,
                                         lineage=(f"S2B-D384-{cfg}-s{s}-{h:08x}", f"S2-D384-{cfg}-s{s}-{k:08x}")))
    out.append(FE.DesignCell(run_id="S2T-D384-C12ssmax-s101-000000ff", config="C12ssmax", k=12, arm="ssmax", seed=101,
                             branch="30M", lineage=("S2B-D384-C12ssmax-s101-000000fe", "S2-D384-C12ssmax-s101-000000fd")))
    return out


def run_of(design, k, s, b, arm="plain"):
    return next(d.run_id for d in design if d.k == k and d.seed == s and d.branch == b and d.arm == arm)


def base_em(d: FE.DesignCell, cell: str) -> F:
    T = FE.cell_T(cell)
    task = cell.split("_")[0]
    if task in ("M-hop", "A-freq"):
        return F(50, 100)
    if T is None:                         # C0 controls
        return F(90, 100)
    if T == 4096:
        return F(90, 100)
    if T == 32768:
        if d.k == 0:
            return F(30, 100)
        return F(85, 100) if d.branch == "90M" else F(80, 100)   # Delta = 0.05 -> middle -> b* = 90M
    return F(50, 100)


class Tree:
    """Planted counts: cnt[(pass, run_id, cell)] = number of True among the N queries (first c items)."""

    def __init__(self, root: Path):
        self.root = root
        self.design = _design()
        self.cnt: dict[tuple[str, str, str], int] = {}
        self.flip_items: dict[tuple[str, str, str], set[int]] = {}   # explicit item flips vs P0 (pass P1/P2/P3/P4)
        self.lm2 = {d.run_id: {"before": 3.0, "after": 3.02} for d in self.design if d.arm == "plain"}
        self.registry: list[dict] = []
        self.skip: set[tuple[str, str, str]] = set()
        self.nan: set[tuple[str, str, str]] = set()

    def set_all(self, pass_id: str, run_id: str, cells, em: F) -> None:
        for c in cells:
            self.cnt[(pass_id, run_id, c)] = int(em * N)

    def set_ladder(self, pass_id: str, run_id: str, T: int, em: F, c0: bool = False) -> None:
        a, b = FE.ladder_cells(T, c0)
        self.set_all(pass_id, run_id, a + b, em)

    def count(self, pass_id: str, d, cell: str) -> int:
        if (pass_id, d.run_id, cell) in self.cnt:
            return self.cnt[(pass_id, d.run_id, cell)]
        if pass_id != "P0" and ("P0", d.run_id, cell) in self.cnt:
            return self.cnt[("P0", d.run_id, cell)]
        return int(base_em(d, cell) * N)

    def rows(self, pass_id: str, d, cell: str) -> list[dict]:
        c = self.count(pass_id, d, cell)
        flips = self.flip_items.get((pass_id, d.run_id, cell), set())
        out = []
        for i in range(N):
            ctx, q = divmod(i, M)
            em = (i < c) != (i in flips)
            lp = float("nan") if (pass_id, d.run_id, cell) in self.nan and i == 0 else -1.0
            out.append({"item_id": f"{cell}:{ctx}:{q}", "context_id": ctx, "query_idx": q, "EM_tf": em, "logp_gold": lp})
        return out

    def write(self) -> "Tree":
        for d in self.design:
            for pid, lab, cells in (("P0", "tef1000", FE.fd_cells()), ("P1", "tef1000-P1", FE.fd_cells()),
                                    ("P4", "tef1000-P4", FE.p4_cells())):
                dd = self.root / "results" / d.run_id / lab / "FD"
                dd.mkdir(parents=True, exist_ok=True)
                for c in cells:
                    if (pid, d.run_id, c) in self.skip:
                        continue
                    f = dd / f"{c}.jsonl"
                    f.write_text("".join(json.dumps(r) + "\n" for r in self.rows(pid, d, c)))
                    (dd / f"{c}_meta.json").write_text(json.dumps({"pass_id": pid}))
                    if pid == "P0" and ("P2P3", d.run_id, c) not in self.skip:
                        rec = {}
                        for rp in ("P2", "P3"):
                            rr = self.rows(rp, d, c)[:SUB_CTX * M]
                            rec[rp] = {"rows": [{"item_id": r["item_id"], "EM_tf": r["EM_tf"], "logp_gold": -1.0} for r in rr]}
                        (dd / f"{c}_rescore.json").write_text(json.dumps(rec))
        (self.root / "lm2.json").write_text(json.dumps(self.lm2))
        (self.root / "registry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.registry))
        (self.root / "design.json").write_text(json.dumps([{**d.__dict__, "lineage": list(d.lineage)} for d in self.design]))
        return self

    def inputs(self) -> FE.Inputs:
        return FE.load_inputs(self.root / "results", self.design, lm2=self.lm2, registry=self.registry)

    def decide(self) -> dict:
        return FE.decide(self.inputs())


@pytest.fixture
def tree(tmp_path) -> Tree:
    return Tree(tmp_path)


# ------------------------------------------------------------------------------------------ baseline
def test_fd_cells_match_items_manifest():
    man = json.loads((REPO / "experiments" / "items" / "dev" / "MANIFEST.json").read_text())
    from hybridlc.integration.evaluate import result_name

    fd = sorted(result_name(Path(k).stem) for k in man["files"] if k.startswith("FD/"))
    assert fd == sorted(FE.fd_cells())


def test_baseline_tree_decisions(tree):
    out = tree.write().decide()
    v = out["verdicts"]
    assert v["FEAS-3"]["Delta"] == F(5, 100) and v["FEAS-3"]["budget"] == "60M" and v["FEAS-3"]["b_star"] == "90M"
    assert v["U-01"]["outcome"] == "D384" and v["U-01"]["branch"] == "90M"
    assert v["FEAS-1/2"]["evaluated"] is False
    assert v["FEAS-5"]["branch"] == "30M" and v["FEAS-5"]["needs_pi_row"] is True       # R-ssmax-branch
    assert v["TEF-forgetting"]["outcome"] == "keep 90/10"
    assert v["elicitation"]["fires"] is False
    assert out["precision"]["Q4"]["selected"] == "P0" and not out["precision"]["Q1"]["fires"]
    assert out["precision"]["Q2"]["pass"] and not out["precision"]["Q3"]["fires"]
    assert out["header"]["held_for_pi_row"] == ["FEAS-5"]
    assert out["trace"][0] == "inputs complete" and "precision rules Q1-Q4 computed first" in out["trace"][1]


# ------------------------------------------------------------------------------------------ per-rule boundaries
@pytest.mark.parametrize("diff,choice,bstar", [(F(3, 100), "smallest", "30M"), (F(4, 100), "middle", "90M"),
                                               (F(10, 100), "middle", "90M"), (F(11, 100), "largest", "90M")])
def test_feas3_boundaries_and_branch(tree, diff, choice, bstar):
    for s in (101, 102):
        tree.set_ladder("P0", run_of(tree.design, 12, s, "30M"), 32768, F(60, 100))
        tree.set_ladder("P0", run_of(tree.design, 12, s, "90M"), 32768, F(60, 100) + diff)
    out = tree.write().decide()
    f3 = out["verdicts"]["FEAS-3"]
    assert f3["Delta"] == diff and f3["choice"] == choice and f3["b_star"] == bstar
    assert out["verdicts"]["U-01"]["branch"] == bstar           # the FEAS rule reads b*


@pytest.mark.parametrize("s12,c0,s0,ok", [
    (F(85, 100), F(100, 100), F(60, 100), True),     # ratio exactly 0.85, gap exactly 0.25 -> pass (>=)
    (F(84, 100), F(100, 100), F(59, 100), False),    # ratio 0.84 < 0.85
    (F(85, 100), F(100, 100), F(61, 100), False),    # gap 0.24 < 0.25
])
def test_u01_boundaries(tree, s12, c0, s0, ok):
    for s in (101, 102):
        for b in ("30M", "90M"):
            tree.set_ladder("P0", run_of(tree.design, 12, s, b), 32768, s12)
            tree.set_ladder("P0", run_of(tree.design, 12, s, b), 32768, c0, c0=True)
            tree.set_ladder("P0", run_of(tree.design, 0, s, b), 32768, s0)
    for pid in ("P1", "P4"):
        tree.cnt.update({(pid, *k[1:]): v for k, v in list(tree.cnt.items()) if k[0] == "P0"})
    out = tree.write().decide()
    assert out["verdicts"]["U-01"]["adopted"] is ok
    assert out["verdicts"]["U-01"]["outcome"] == ("D384" if ok else "FEAS-512")
    if not ok:
        assert out["verdicts"]["FEAS-1/2"]["outcome"].startswith("pending")


def test_u01_seed_disagreement_not_adopted(tree):
    b = "90M"
    tree.set_ladder("P0", run_of(tree.design, 0, 102, b), 32768, F(75, 100))    # seed 102 gap 0.10 -> fails
    for pid in ("P1", "P4"):
        tree.cnt.update({(pid, *k[1:]): v for k, v in list(tree.cnt.items()) if k[0] == "P0"})
    out = tree.write().decide()
    u = out["verdicts"]["U-01"]
    assert u["per_seed"][101]["pass"] is True and u["per_seed"][102]["pass"] is False and u["adopted"] is False


@pytest.mark.parametrize("gain,loss,adopt", [(F(3, 100), F(2, 100), True), (F(2, 100), F(0), False),
                                             (F(3, 100), F(3, 100), False)])
def test_feas5_boundaries(tree, gain, loss, adopt):
    d = tree.design
    ss, pl = run_of(d, 12, 101, "30M", "ssmax"), run_of(d, 12, 101, "30M")
    tree.set_ladder("P0", pl, 32768, F(50, 100))
    tree.set_ladder("P0", ss, 32768, F(50, 100) + gain)
    tree.set_ladder("P0", pl, 4096, F(80, 100))
    tree.set_ladder("P0", ss, 4096, F(80, 100) - loss)
    for pid in ("P1", "P4"):
        tree.cnt.update({(pid, *k[1:]): v for k, v in list(tree.cnt.items()) if k[0] == "P0"})
    out = tree.write().decide()
    f5 = out["verdicts"]["FEAS-5"]
    assert f5["gain_32K"] == gain and f5["loss_4K"] == loss and f5["adopted"] is adopt


@pytest.mark.parametrize("ems,fires", [((F(49, 100), F(49, 100)), True), ((F(50, 100), F(49, 100)), False),
                                       ((F(49, 100), F(60, 100)), False)])
def test_elicitation_both_seeds_literal(tree, ems, fires):
    for s, em in zip((101, 102), ems):
        for pid in ("P0", "P1", "P4"):
            tree.cnt[(pid, run_of(tree.design, 12, s, "90M"), "R1_D3_4096")] = int(em * N)
    out = tree.write().decide()
    assert out["verdicts"]["elicitation"]["fires"] is fires


@pytest.mark.parametrize("r0,r12,keep", [(0.05, 0.05, True), (0.05, 0.03, True), (0.05, 0.02, False),
                                         (0.06, 0.05, False), (0.0, 0.0, True)])
def test_tef_forgetting_boundaries(tree, r0, r12, keep):
    for d in tree.design:
        if d.arm == "plain":
            tree.lm2[d.run_id] = {"before": 3.0, "after": round(3.0 + (r0 if d.k == 0 else r12), 10)}
    out = tree.write().decide()
    g = out["verdicts"]["TEF-forgetting"]
    assert g["rise_0"] == F(repr(round(3.0 + r0, 10))) - 3 and g["keep"] is keep
    assert g["outcome"] == ("keep 90/10" if keep else "50/50 (evaluate the 50/50 TEF reruns by the same rule; reruns pending)")


def test_i08_floor_ceiling_mixed_triggers(tree):
    d = tree.design
    for s in (101, 102):
        for c in FE.fd_cells():
            if FE.cell_family(c) == "multi-hop":
                em = F(5, 100) if c.endswith("8192") else F(95, 100)         # floor and ceiling exactly: mixed
                for pid in ("P0", "P1", "P4"):
                    tree.cnt[(pid, run_of(d, 12, s, "90M"), c)] = int(em * N)
    out = tree.write().decide()
    i8 = out["verdicts"]["I-08"]
    assert i8["per_k"][12]["multi-hop"]["trigger"] is True and i8["triggered"] is True
    assert i8["per_k"][0]["multi-hop"]["trigger"] is False


def test_i08_informative_level_blocks_trigger(tree):
    d = tree.design
    for s in (101, 102):
        for c in FE.fd_cells():
            if FE.cell_family(c) == "multi-hop":
                em = F(6, 100) if c == "M-hop_h1_8192" else F(0)
                tree.cnt[("P0", run_of(d, 12, s, "90M"), c)] = int(em * N)
    out = tree.write().decide()
    assert out["verdicts"]["I-08"]["per_k"][12]["multi-hop"]["trigger"] is False


# ------------------------------------------------------------------------------------------ refusals
def _refused(tree) -> list[str]:
    with pytest.raises(FE.FeasInputError) as e:
        tree.inputs()
    return e.value.missing


def test_refuse_missing_cell(tree):
    rid = run_of(tree.design, 12, 101, "30M")
    tree.skip.add(("P0", rid, "R1_D15_32768"))
    tree.write()
    miss = _refused(tree)
    assert any(f"P0 {rid}/tef1000/FD/R1_D15_32768" in m for m in miss)


def test_refuse_missing_seed(tree):
    gone = [d for d in tree.design if d.seed == 102]
    tree.design = [d for d in tree.design if d.seed != 102]
    tree.write()
    miss = _refused(tree)
    assert sum("seed=102" in m for m in miss) == 4 and gone


def test_refuse_missing_pass(tree):
    rid = run_of(tree.design, 0, 102, "90M")
    for c in FE.p4_cells():
        tree.skip.add(("P4", rid, c))
    tree.skip.add(("P2P3", rid, "R1_D3_32768"))
    tree.write()
    miss = _refused(tree)
    assert sum(m.startswith(f"P4 {rid}/tef1000-P4/") for m in miss) == len(FE.p4_cells())
    assert any(m.startswith(f"P2/P3 {rid}/tef1000/FD/R1_D3_32768_rescore.json") for m in miss)


def test_refuse_missing_lm2_and_nan(tree):
    rid = run_of(tree.design, 12, 102, "30M")
    del tree.lm2[rid]
    rid2 = run_of(tree.design, 0, 101, "30M")
    tree.lm2[rid2] = {"before": 3.0, "after": float("nan")}
    rid3 = run_of(tree.design, 0, 101, "90M")
    tree.nan.add(("P1", rid3, "A-freq_rho2_8192"))
    tree.write()
    miss = _refused(tree)
    assert any(f"LM-2 before/after TEF for {rid}" in m for m in miss)
    assert any(f"LM-2 before/after TEF for {rid2}" in m for m in miss)
    assert any("A-freq_rho2_8192.jsonl:1 logp_gold NaN" in m for m in miss)


def test_refuse_wrong_pass_meta(tree):
    tree.write()
    rid = run_of(tree.design, 0, 101, "30M")
    (tree.root / "results" / rid / "tef1000-P1" / "FD" / "R1_D3_4096_meta.json").write_text(json.dumps({"pass_id": "P0"}))
    assert any("pass_id 'P0' where P1 expected" in m for m in _refused(tree))


# ------------------------------------------------------------------------------------------ ICC / U-15
def _sim_strata(rng, rho: float, n_strata=10, n_ctx=300, m=4):
    strata = {}
    for s in range(n_strata):
        mu = 0.2 + 0.6 * s / (n_strata - 1)
        if rho == 0:
            p = np.full(n_ctx, mu)
        else:
            ab = 1 / rho - 1
            p = rng.beta(mu * ab, (1 - mu) * ab, size=n_ctx)
        strata[s] = rng.binomial(m, p).tolist()
    return strata


@pytest.mark.parametrize("rho", [0.0, 0.15, 0.33, 0.5])
def test_icc_recovery_200_sims(rho):
    rng = np.random.default_rng(int(rho * 1000) + 7)
    est = [float(FE.icc_from_counts(_sim_strata(rng, rho))["ICC"]) for _ in range(200)]
    assert abs(np.mean(est) - rho) <= 0.03, (rho, np.mean(est))


def test_icc_de_boundary_exact():
    import itertools

    hit = None
    for n in (4, 5, 6):
        for counts in itertools.product(range(5), repeat=n):
            r = FE.icc_from_counts({"s": list(counts)})
            if r["ICC_raw"] == F(1, 3):
                hit = (counts, r)
                break
        if hit:
            break
    assert hit is not None
    counts, r = hit
    assert r["DE"] == 2 and r["contexts"] == 300                      # DE = 2 exactly -> 300 (DE > 2 for 600)
    up = FE.icc_from_counts({"s": list(counts), "t": [0, 4, 0, 4]})
    assert (up["DE"] > 2) == (up["contexts"] == 600)


def test_u15_in_decisions(tree):
    out = tree.write().decide()
    u = out["verdicts"]["U-15"]
    assert u["n_strata"] == 14 * 4 and u["n_contexts"] == 14 * 4 * N_CTX and u["source"] == "P0"


# ------------------------------------------------------------------------------------------ precision
def _spread(slots, n):
    """n item flips spread over slots [(pass, run, cell)], round-robin, on items 0..N-1."""
    per = {s: set() for s in slots}
    for i in range(n):
        s = slots[i % len(slots)]
        per[s].add(len(per[s]))
    return per


@pytest.mark.parametrize("n_flips,fires", [(188, True), (187, False)])
def test_q1_flip_rate_exactly_one_percent_triggers(tree, n_flips, fires):
    k0 = [d for d in tree.design if d.k == 0]
    total = len(k0) * len(FE.fd_cells()) * N
    assert total == 18800 and F(188, total) == F(1, 100)
    slots = [("P1", d.run_id, c) for d in k0 for c in FE.fd_cells() if c.startswith("M-hop") and c.endswith("8192")]
    tree.flip_items.update(_spread(slots, n_flips))
    out = tree.write().decide()
    q1 = out["precision"]["Q1"]
    assert q1["measure"]["pooled"][0]["rate"] == F(n_flips, total)
    assert ("pooled k=0" in q1["triggers"]) is fires and q1["fires"] is fires


def test_informative_cell_strict_and_symmetric(tree):
    d = tree.design
    rid = run_of(d, 0, 101, "30M")
    # EM exactly 0.05 under both passes -> NOT informative (strict); collapse (P0 0.0, P1 0.9) -> informative
    tree.cnt[("P0", rid, "R1_D3_16384")] = 5
    tree.cnt[("P1", rid, "R1_D3_16384")] = 5
    tree.cnt[("P0", rid, "R1_D15_16384")] = 0
    tree.cnt[("P1", rid, "R1_D15_16384")] = 90
    out = tree.write().decide()
    pc = out["precision"]["Q1"]["measure"]["per_cell"]
    assert pc[f"{rid}/R1_D3_16384"]["informative"] is False
    assert pc[f"{rid}/R1_D15_16384"]["informative"] is True
    assert f"{rid}/R1_D15_16384" in out["precision"]["Q1"]["measure"]["restricted"][0][16384]["informative_cells"]
    p4 = FE._pair_cell({str(i): False for i in range(10)}, {str(i): i < 9 for i in range(10)})  # P0 0.0 vs P4 0.9
    assert p4["informative"] is True


@pytest.mark.parametrize("extra_r1,switch", [(0, False), (1, True)])
def test_q4_delta_c_boundary(tree, extra_r1, switch):
    d = tree.design
    rid = run_of(d, 12, 101, "90M")
    a, b = FE.ladder_cells(32768)
    for c in a + b:
        tree.cnt[("P4", rid, c)] = int(F(85, 100) * N) + 2          # every ladder cell +2/100 -> dC = 0.02 exactly
    if extra_r1:
        tree.cnt[("P4", rid, "R1_D3_32768")] += 1                   # + 1/400 on the ladder -> dC = 0.02 + 1/400
    out = tree.write().decide()
    q4 = out["precision"]["Q4"]
    assert q4["abs_dC"][101] == F(2, 100) + (F(1, 400) if extra_r1 else 0)
    assert q4["outcome_differences"] == []
    assert q4["switch"] is switch and q4["selected"] == ("P4" if switch else "P0")
    assert out["flags"]["F-precision"]["fires"] is switch


def test_q4_planted_p4_flips_u01(tree):
    d = tree.design
    for s in (101, 102):
        tree.set_ladder("P4", run_of(d, 0, s, "90M"), 32768, F(70, 100))     # gap 0.15 under P4 -> U-01 fails
    out = tree.write().decide()
    q4 = out["precision"]["Q4"]
    assert "U-01" in q4["outcome_differences"] and q4["selected"] == "P4"
    assert any("outcome differs: U-01" in t for t in q4["triggers"])
    assert out["verdicts"]["U-01"]["outcome"] == "FEAS-512" and "F-precision" in out["verdicts"]["U-01"]["flags"]
    assert any("Q4 selected P4" in t and "U-01" in t for t in out["trace"])


@pytest.mark.parametrize("n_flips,fails", [(10, True), (9, False)])
def test_q2_fail_requires_single_pass_rescores(tree, n_flips, fails):
    rid = run_of(tree.design, 12, 101, "90M")
    sub = len(FE.fd_cells()) * SUB_CTX * M                       # 940 re-scored queries per checkpoint
    assert (F(n_flips, sub) >= F(1, 100)) is fails
    slots = [("P2", rid, c) for c in ("M-hop_h1_8192", "M-hop_h2_8192", "M-hop_h3_8192")]
    tree.flip_items.update(_spread(slots, n_flips))
    tree.write()
    if fails:
        miss = _refused(tree)
        assert any(m.startswith(f"P2full {rid}/tef1000-P2full/FD/") for m in miss)
    else:
        out = tree.decide()
        assert out["precision"]["Q2"]["pass"] is True


def test_q3_options(tree):
    k12 = [x for x in tree.design if x.k == 12]
    slots = [("P3", x.run_id, c) for x in k12 for c in ("R1_D3_32768",)]
    tree.flip_items.update({s: {0, 1, 2, 3, 4, 5} for s in slots})
    out = tree.write().decide()
    q3 = out["precision"]["Q3"]
    assert q3["fires"] and q3["options_allowed"]["a_fp32_compute"] is True
    assert q3["options_allowed"]["c_production_with_caveat"] is True     # no Q4 difference, dC 0
    assert q3["options_allowed"]["b_fp32_weights_only"] is False         # P1 == P0 so P1-P3 restricted >= 1%


# ------------------------------------------------------------------------------------------ flags
def test_near_miss_flags_exactly_the_reading_verdicts(tree):
    d = tree.design
    rid = run_of(d, 0, 102, "30M")                       # b* = 90M: no verdict reads the 30M k = 0 run
    ds = next(x for x in d if x.run_id == rid)
    tree.registry = [{"run_id": ds.lineage[0], "status": "note", "note": "gdiv_near_miss", "step": 12}]
    out = tree.write().decide()
    assert out["flags"]["F-nearmiss"]["flagged_verdicts"] == {}
    rid90 = run_of(d, 0, 102, "90M")
    ds90 = next(x for x in d if x.run_id == rid90)
    tree.registry = [{"run_id": ds90.lineage[1], "status": "note", "note": "gdiv_near_miss", "step": 40}]  # Stage A
    out = tree.write().decide()
    got = set(out["flags"]["F-nearmiss"]["flagged_verdicts"])
    assert got == {"U-01", "TEF-forgetting", "U-15", "I-08"}          # the verdicts that read k = 0 seed 102 at b*
    assert out["verdicts"]["U-01"]["hold_for_pi_row"] is True and out["verdicts"]["FEAS-3"]["flags"] == []


@pytest.mark.parametrize("n0,n12,fires", [(3, 0, True), (2, 0, False), (4, 1, True), (1, 3, False)])
def test_infra_flag_per_k(tree, n0, n12, fires):
    d = tree.design
    k0 = [x for x in d if x.k == 0]
    k12 = [x for x in d if x.k == 12]
    rows = []
    for i in range(n0):
        rows.append({"run_id": k0[i % len(k0)].run_id, "status": "infra_failed_attempt"})
    for i in range(n12):
        rows.append({"run_id": k12[i % len(k12)].lineage[0], "status": "suspect_sw_failure"})
    tree.registry = rows
    out = tree.write().decide()
    fl = out["flags"]["F-infra"]
    assert fl["n_per_k"] == {0: n0, 12: n12} and fl["fires"] is fires


def test_infra_flag_single_run_three(tree):
    rid = run_of(tree.design, 12, 101, "30M")
    tree.registry = [{"run_id": rid, "status": "infra_failed_attempt"}] * 2 + [{"run_id": rid, "status": "suspect_sw_failure"}] + \
                    [{"run_id": run_of(tree.design, 0, 101, "30M"), "status": "infra_failed_attempt"}] * 2
    out = tree.write().decide()
    assert out["flags"]["F-infra"]["runs_with_3_or_more"] == [rid] and out["flags"]["F-infra"]["fires"]
    assert all("F-infra" in v["flags"] for v in out["verdicts"].values())


# ------------------------------------------------------------------------------------------ CLI
def _cli(*args, cwd=REPO):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "feas_decide.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, timeout=600)


def test_cli_structure_only_on_s0_smoke_tree():
    r = _cli("--results", REPO / "experiments" / "s0" / "smoke" / "results", "--structure-only")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "structure: 1 run(s)" in r.stdout and "0 problem(s)" in r.stdout


def test_cli_refuses_s0_smoke_tree_naming_missing_inputs(tmp_path):
    r = _cli("--results", REPO / "experiments" / "s0" / "smoke" / "results",
             "--registry", REPO / "experiments" / "s0" / "smoke" / "registry.jsonl", "--out", tmp_path / "d.json")
    assert r.returncode == 2 and "REFUSED" in r.stderr and "design: k=0 seed=101 branch=30M" in r.stderr


def test_cli_full_synthetic_tree_blinded_and_unblinded(tree, tmp_path):
    tree.write()
    out = tmp_path / "feas" / "feas_decisions.json"
    args = ("--results", tree.root / "results", "--registry", tree.root / "registry.jsonl",
            "--design", tree.root / "design.json", "--lm2", tree.root / "lm2.json", "--out", out)
    r = _cli(*args)
    assert r.returncode == 0, r.stderr
    assert "BLINDED" in r.stdout and "D384" not in r.stdout and "U-01" not in r.stdout
    d = json.loads(out.read_text())
    assert d["verdicts"]["U-01"]["outcome"] == "D384" and set(d["verdicts"]) == set(FE.VERDICTS)
    assert all((out.parent / "precision" / f"q{i}.json").is_file() for i in (1, 2, 3, 4))
    assert set(d["flags"]) == {"F-nearmiss", "F-infra", "F-precision"}
    (tree.root / "registry.jsonl").write_text(json.dumps({"run_id": "DTS10-commit", "status": "note"}) + "\n")
    r = _cli(*args)
    assert r.returncode == 0 and "U-01: D384" in r.stdout and "BLINDED" not in r.stdout


def test_cli_design_from_registry_queued_rows(tree, tmp_path):
    """The CLI's default design: registry 'queued' rows whose run YAML is a generated FEAS TEF YAML."""
    from hybridlc.training.genconfigs import feas_design

    rows = []
    by = {(r["k"], r["arm"], r["seed"], r["branch"]): r["yaml"] for r in feas_design()}
    for d in tree.design:
        rows.append({"run_id": d.run_id, "status": "queued", "config": str(REPO / by[(d.k, d.arm, d.seed, d.branch)]),
                     "parent_run_id": d.lineage[0]})
        rows.append({"run_id": d.lineage[0], "status": "queued", "parent_run_id": d.lineage[1]})
    des = FE.design_from_registry(rows)
    assert {(x.run_id, x.k, x.arm, x.seed, x.branch, x.lineage) for x in des} == \
        {(x.run_id, x.k, x.arm, x.seed, x.branch, x.lineage) for x in tree.design}
