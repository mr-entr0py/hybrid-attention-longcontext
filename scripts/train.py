#!/usr/bin/env python
"""Train one run (protocol §7, §18). The runner (scripts/runner.py) calls this under nohup.

Usage::

    python scripts/train.py --config configs/training/examples/S2_D384_C0_s101_A.yaml [--allow-dirty]
    python scripts/train.py --model configs/model/D384/C3.yaml --training configs/training/stage_A.yaml \\
        --exp-stage S4 --seed 0 --lr 2e-3 [-o checkpoint.every=500] [--no-resume]
    python scripts/train.py --config <yaml> --print-run-id      # resolve and print the run_id only

Resume is the default: if ``checkpoints/<run_id>/`` holds a full checkpoint the run continues from the
newest one (bitwise-identical data, LR and RNG state). Exit codes: 0 done, 75 OOM (runner retries),
76 diverged twice (failed), 77 refused (dirty tree, config/parent problem), other = crash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# Protocol §18 production environment: the pre-warmed Triton cache lives in the repo. Set before
# torch/triton/fla are imported so that every JIT lookup uses it.
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO / ".cache" / "triton"))

import yaml  # noqa: E402

from hybridlc.training.config import PathsConfig, load_run_config  # noqa: E402


def compose_run_yaml(a: argparse.Namespace) -> Path:
    """Write a run YAML from the composition flags (stored under runs/_composed/) and return its path."""
    doc = {"exp_stage": a.exp_stage, "model": a.model, "training": a.training, "seed_index": a.seed}
    if a.lr is not None:
        doc["lr"] = a.lr
    if a.steps_option is not None:
        doc["steps_option"] = a.steps_option
    if a.parent is not None:
        doc["parent"] = a.parent
    if a.config_name is not None:
        doc["config_name"] = a.config_name
    text = yaml.safe_dump(doc, sort_keys=True)
    out = REPO / "runs" / "_composed" / f"{hashlib.sha256(text.encode()).hexdigest()[:16]}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="run YAML (see hybridlc.training.config)")
    ap.add_argument("--model", help="model YAML (composition mode)")
    ap.add_argument("--training", help="training recipe YAML (composition mode)")
    ap.add_argument("--exp-stage", default="S0")
    ap.add_argument("--seed", type=int, default=0, help="seed index (protocol §8.1)")
    ap.add_argument("--lr", type=float, default=None, help="peak LR eta*")
    ap.add_argument("--steps-option", default=None)
    ap.add_argument("--parent", default=None, help="parent run YAML (branch point)")
    ap.add_argument("--config-name", default=None)
    ap.add_argument("-o", "--override", action="append", default=[], help="dotted key=value into the recipe")
    ap.add_argument("--tag", default=None, help="non-semantic run_id suffix")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--allow-dirty", action="store_true", help="run on a dirty git tree (recorded)")
    ap.add_argument("--print-run-id", action="store_true")
    ap.add_argument("--print-config", action="store_true")
    return ap


def resolve(a: argparse.Namespace):
    if not a.config and not (a.model and a.training):
        raise SystemExit("give --config, or --model and --training")
    if a.config:
        # composition-only flags would be silently ignored with --config (verification round 2)
        ignored = [f for f, v in (("--lr", a.lr), ("--model", a.model), ("--training", a.training),
                                  ("--steps-option", a.steps_option), ("--parent", a.parent),
                                  ("--config-name", a.config_name)) if v is not None]
        if ignored:
            raise SystemExit(f"{ignored} only apply in composition mode (--model/--training); with --config "
                             "edit the run YAML or use -o schedule.peak_lr=<value>")
    path = Path(a.config) if a.config else compose_run_yaml(a)
    paths = None
    if a.runs_root or a.ckpt_root or a.registry:
        d = PathsConfig()
        paths = PathsConfig(runs_root=a.runs_root or d.runs_root, ckpt_root=a.ckpt_root or d.ckpt_root,
                            registry=a.registry or d.registry)
    return load_run_config(path, cli_overrides=a.override, paths=paths, tag=a.tag)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = resolve(a)
    if a.print_run_id:
        print(cfg.run_id)
        return 0
    if a.print_config:
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0
    from hybridlc.training.trainer import main_exit_code, train

    print(f"[train] run_id={cfg.run_id} stage={cfg.stage} steps={cfg.schedule.steps} "
          f"peak_lr={cfg.schedule.peak_lr} resume={a.resume} allow_dirty={a.allow_dirty}", flush=True)
    result: dict = {}

    def _go() -> None:
        result.update(train(cfg, resume=a.resume, allow_dirty=a.allow_dirty))

    code = main_exit_code(_go)
    if code == 0:
        print("[train] done " + json.dumps({k: result.get(k) for k in ("steps_done", "final_val", "param_sha256")},
                                           default=str), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
