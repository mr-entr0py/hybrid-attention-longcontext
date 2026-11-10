"""VT-12 regression, verification round 2 (ITEM-EVAL-CODE-SHA-STAMPS).

Item files embed the generating eval-code SHA in their cell meta, so regenerating an unchanged cell
after *any* eval-code change gives different bytes. Frozen items are not restamped; instead the
manifests record a stamp-independent ``content_sha256`` per file (``items.item_content_sha256``),
the top-level ``item_eval_code_sha256`` counts which code versions generated the frozen files, and
"regeneration is identical" is checked on content.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybridlc.eval.items import item_content_sha256, load_items, save_items, sha256_file, verify_manifest

REPO = Path(__file__).resolve().parents[1]
ITEMS = REPO / "experiments" / "items"


@pytest.mark.parametrize("ns", ["test", "dev"])
def test_manifest_content_hashes(ns):
    if not (ITEMS / ns / "MANIFEST.json").is_file():
        pytest.skip("frozen items not present")
    man = json.loads((ITEMS / ns / "MANIFEST.json").read_text())
    npz = [k for k in man["files"] if k.endswith(".npz")]
    assert npz and all("content_sha256" in man["files"][k] for k in npz)
    assert sum(man["item_eval_code_sha256"].values()) == len(npz)
    assert verify_manifest(ITEMS / ns, content=True) == []


def test_restamped_resave_same_content(tmp_path):
    src = ITEMS / "dev" / "DC" / "R1_D3_4K.npz"
    if not src.is_file():
        pytest.skip("frozen items not present")
    items, cell = load_items(src)
    out = tmp_path / "x.npz"
    save_items(out, items, dict(cell, eval_code_sha256="0" * 64))
    assert sha256_file(out) != sha256_file(src)  # only the stamp differs ...
    assert item_content_sha256(out) == item_content_sha256(src)  # ... and content identity holds
    out2 = tmp_path / "y.npz"
    items[0]["context_ids"] = items[0]["context_ids"].copy()
    items[0]["context_ids"][5] += 1
    save_items(out2, items, cell)
    assert item_content_sha256(out2) != item_content_sha256(src)


def test_verifier_regenerations_match_on_content():
    regen = REPO / ".verify" / "r2_model_eval" / "regen" / "dev" / "DC"
    files = sorted(regen.glob("*.npz")) if regen.is_dir() else []
    if not files:
        pytest.skip("no regenerated cells from the round-2 verifier")
    for f in files:
        frozen = ITEMS / "dev" / "DC" / f.name
        assert sha256_file(f) != sha256_file(frozen)
        assert item_content_sha256(f) == item_content_sha256(frozen), f.name
