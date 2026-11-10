"""§7.2 LR selection by minimax regret (verification round 2, LR-RULE-NOT-IMPLEMENTED)."""
from __future__ import annotations

import json
import math

import pytest

from hybridlc.analysis import protocol as P
from hybridlc.analysis.lr_select import eta_for_config, eta_for_k, read_sweep_losses, select_lr

G = P.LR_GRID  # (1.0, 1.4, 2.0, 2.8) e-3


def test_shared_eta_when_max_regret_small():
    L = {0: dict(zip(G, [3.10, 3.07, 3.040, 3.07])),
         3: dict(zip(G, [3.00, 2.98, 2.950, 2.98])),
         12: dict(zip(G, [2.95, 2.90, 2.915, 2.96]))}  # k=12 optimum 1.4e-3, regret at 2.0e-3 = 0.015
    s = select_lr(L)
    assert s["eta_star"] == 2.0e-3 and s["shared"] and s["max_regret"] == pytest.approx(0.015)
    assert s["extend"] == [] and s["final"]
    assert s["eta_by_k"] == {0: 2.0e-3, 3: 2.0e-3, 12: 2.0e-3}
    assert eta_for_k(s, 1) == 2.0e-3 and eta_for_config(s, "X-gc12") == 2.0e-3


def test_per_config_eta_edge_extension_and_interpolation():
    L = {0: dict(zip(G, [3.10, 3.05, 3.04, 3.07])),
         3: dict(zip(G, [3.00, 2.96, 2.95, 2.98])),
         12: dict(zip(G, [2.85, 2.90, 2.95, 3.00]))}  # k=12 optimum at the low edge, regret 0.10 at 2e-3
    s = select_lr(L)
    assert not s["shared"] and s["per_k"] == {0: 2.0e-3, 3: 2.0e-3, 12: 1.0e-3}
    assert s["extend"] == [P.LR_EDGE_LOW] and not s["final"]
    # run the extension for all three k and re-apply
    L[0][0.7e-3], L[3][0.7e-3], L[12][0.7e-3] = 3.20, 3.10, 2.88
    s2 = select_lr(L)
    assert s2["extend"] == [] and s2["per_k"][12] == 1.0e-3 and not s2["shared"]
    # log-eta interpolation in k between grid k's; inheritance by k
    e1 = math.exp((2 / 3) * math.log(2.0e-3) + (1 / 3) * math.log(2.0e-3))
    assert eta_for_k(s2, 1) == pytest.approx(e1)
    e6 = math.exp((6 / 9) * math.log(2.0e-3) + (3 / 9) * math.log(1.0e-3))
    assert eta_for_k(s2, 6) == pytest.approx(e6)
    assert eta_for_config(s2, "X-nope12") == 1.0e-3 and eta_for_config(s2, "C3") == 2.0e-3


def test_divergence_is_infinite_regret():
    L = {0: dict(zip(G, [3.10, 3.05, 3.00, float("nan")])),
         3: dict(zip(G, [3.00, 2.96, 2.95, 2.94])),
         12: dict(zip(G, [2.95, 2.90, 3.45, 2.89]))}  # 3.45 > min 2.89 + 0.5 -> diverged
    s = select_lr(L)
    assert s["regrets"][0][repr(2.8e-3)] == "inf" and s["regrets"][12][repr(2.0e-3)] == "inf"
    assert s["eta_star"] == 1.4e-3  # 2.0e-3 and 2.8e-3 have infinite max regret
    with pytest.raises(ValueError, match="lacks k"):
        select_lr({0: L[0], 3: L[3]})


def test_read_sweep_losses(tmp_path):
    def write(rid, rows):
        d = tmp_path / rid
        d.mkdir()
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    write("a", [dict(kind="train", loss=4.0), dict(kind="val", val_loss=3.5, final=False),
                dict(kind="val", val_loss=3.1, final=True)])
    write("b", [dict(kind="train", loss=float("inf"))])  # diverged, no final val row
    out = read_sweep_losses({"a": (0, 2.0e-3), "b": (0, 2.8e-3), "c": (3, 2.0e-3)}, tmp_path)
    assert out == {0: {2.0e-3: 3.1, 2.8e-3: None}, 3: {2.0e-3: None}}
