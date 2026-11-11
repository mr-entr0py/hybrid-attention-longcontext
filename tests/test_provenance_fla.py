"""T22 / PI ruling D068 (PR-11): the fla wheel's RECORD and dist-info hashes are in env_info()["fla_install"]."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
from pathlib import Path

from hybridlc.utils import provenance as pv


def _dist_info_dir() -> Path:
    dist = md.distribution("flash-linear-attention")
    recs = [f for f in (dist.files or []) if f.name == "RECORD" and f.parent.name.endswith(".dist-info")]
    assert recs, "installed fla has no RECORD file"
    return Path(dist.locate_file(recs[0])).parent


def test_fla_install_records_wheel_hashes_and_they_recompute():
    info = pv.fla_install_info()
    assert "error" not in info, info
    di = _dist_info_dir()
    assert Path(info["dist_info"]) == di
    # independent recomputation (no use of the module's helpers)
    rec = hashlib.sha256((di / "RECORD").read_bytes()).hexdigest()
    lines = []
    for p in sorted(di.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            lines.append(f"{p.relative_to(di).as_posix()}\t{p.stat().st_size}\t"
                         f"{hashlib.sha256(p.read_bytes()).hexdigest()}")
    listing = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    assert info["record_sha256"] == rec
    assert info["dist_info_listing_sha256"] == listing
    assert len(rec) == 64 and len(listing) == 64


def test_env_info_carries_fla_install():
    info = pv.env_info()
    fi = info["fla_install"]
    assert fi.get("record_sha256") and fi.get("dist_info_listing_sha256")


def test_listing_changes_when_a_file_changes(tmp_path):
    d = tmp_path / "x-1.0.dist-info"
    d.mkdir()
    (d / "RECORD").write_text("a,b,c\n")
    (d / "METADATA").write_text("Name: x\n")
    h1 = pv.dist_info_listing(d)
    (d / "METADATA").write_text("Name: y\n")
    assert pv.dist_info_listing(d) != h1
