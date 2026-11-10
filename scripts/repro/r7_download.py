"""R7: download pinned checkpoints and LAMBADA data (reproduction_plan.md §2.7, Pins).

Everything goes to experiments/repro/R7/hf_cache (a dedicated cache dir) at the pinned
revisions; a manifest with sizes and SHA-256 of every downloaded file is written to
experiments/repro/R7/download_manifest.json. Idempotent (hf_hub_download skips cached files).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments/repro/R7/hf_cache"

PINS = {
    "EleutherAI/pythia-160m": ("model", "50f5173d932e8e61f858120bcb800b97af589f46",
                               ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
                                "tokenizer_config.json"]),
    "EleutherAI/pythia-70m": ("model", "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
                              ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
                               "tokenizer_config.json"]),
    "m-a-p/340M-20B-GatedDeltaNet-pure-baseline": ("model", "14661a98768e1f5d5493ff7b18f2d6809717a4c8",
                                                   ["config.json", "generation_config.json", "model.safetensors",
                                                    "special_tokens_map.json", "tokenizer.json",
                                                    "tokenizer_config.json"]),
    "EleutherAI/lambada_openai": ("dataset", "900124bf3b8235c6daf21033af9948b3f07346c4", ["data/lambada_test.jsonl"]),
    "cimec/lambada": ("dataset", "5953bd97664b64b95754f299b2309ecfbfbe81b9", ["plain_text/test-00000-of-00001.parquet"]),
}
EXPECT = {("EleutherAI/lambada_openai", "data/lambada_test.jsonl"):
          (1819752, "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226")}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def local_path(repo: str, fn: str) -> Path:
    kind, rev, _ = PINS[repo]
    return Path(hf_hub_download(repo, fn, revision=rev, repo_type=kind, cache_dir=str(CACHE)))


def main() -> None:
    man = {}
    for repo, (kind, rev, files) in PINS.items():
        for fn in files:
            p = local_path(repo, fn)
            s, d = p.stat().st_size, sha256(p)
            if (repo, fn) in EXPECT and EXPECT[(repo, fn)] != (s, d):
                raise SystemExit(f"pin mismatch for {repo}/{fn}: {(s, d)} != {EXPECT[(repo, fn)]}")
            man[f"{repo}@{rev}/{fn}"] = dict(path=str(p), size=s, sha256=d)
            print(repo, fn, s, d[:12], flush=True)
    out = ROOT / "experiments/repro/R7/download_manifest.json"
    out.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
