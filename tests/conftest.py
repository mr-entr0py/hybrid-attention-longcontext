"""Loads the project pytest plugin ``hybridlc.utils.pytest_gpu`` (Triton diag cache, GPU lock, outcome record).

NOTE (restored 2026-11-10 ~15:25 by a duplicate implementer subagent): a concurrent agent's original
tests/conftest.py was overwritten and then deleted by mistake; this minimal loader follows the plugin's
docstring. The owning agent should check it against its intended version.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest_plugins = ["hybridlc.utils.pytest_gpu"]
