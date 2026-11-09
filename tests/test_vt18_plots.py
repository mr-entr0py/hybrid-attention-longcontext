"""VT-18 (part 4): every plot template P1–P18 runs from a results tree + registry and writes
PDF + PNG + an input-hash JSON (protocol §12)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hybridlc.analysis.figures import FIGURES, FigureContext
from hybridlc.analysis.pipeline import AnalysisSpec
from hybridlc.analysis.synthetic import World, WorldSpec, default_plan, write_tree

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("vt18_plots")
    w = World(WorldSpec(n_contexts=10, configs={"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C6": 6, "C12": 12,
                                                "X-gc12": 12, "P3-early": 3, "P3-late": 3},
                        seeds=(0, 1, 2)))
    write_tree(w, default_plan(w), root)
    return root


def test_all_figures_write_pdf_png_and_hashes(tree, tmp_path):
    ctx = FigureContext(tree / "results", tree / "registry.jsonl", tmp_path / "out",
                        AnalysisSpec(B=100, kstar=3))
    statuses = {}
    for name, fn in FIGURES.items():
        meta = fn(ctx)
        for ext in ("pdf", "png", "inputs.json"):
            assert (tmp_path / "out" / f"{name}.{ext}").stat().st_size > 0, (name, ext)
        j = json.loads((tmp_path / "out" / f"{name}.inputs.json").read_text())
        assert j["n_inputs"] > 0 and len(j["inputs_sha256"]) == 64
        assert all(str(tree) in p for p in j["inputs"])  # only results/** and the registry are read
        statuses[name] = meta.get("status", "ok")
    # figures whose inputs exist in the synthetic tree must not fall back to placeholders
    for name in ("P1", "P2", "P3", "P4", "P5", "P6", "P10", "P11", "P12", "P13", "P14", "P18"):
        assert statuses[name] == "ok", (name, statuses[name])
    # inputs that the synthetic tree does not contain produce labelled placeholders
    for name in ("P8", "P9", "P16", "P17"):
        assert statuses[name] == "placeholder"


def test_template_script_runs(tree, tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "plots" / "make_p01.py"), "--results", str(tree / "results"),
                          "--registry", str(tree / "registry.jsonl"), "--out", str(tmp_path), "--B", "50"],
                         capture_output=True, text=True, check=True)
    assert "P1: ok" in out.stdout
    assert (tmp_path / "P1.pdf").read_bytes()[:4] == b"%PDF"
