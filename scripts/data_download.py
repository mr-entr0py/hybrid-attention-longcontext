#!/usr/bin/env python
"""Download the pinned corpora (protocol §6.1) into data/raw/.

Usage: nohup python scripts/data_download.py > data/logs/download.log 2>&1 &
Idempotent: files already present with the expected size are skipped. After
download, each file's SHA-256 is checked against the LFS sha256 the HF API
reports at the pinned revision.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

from hybridlc.data import sources as S


def list_files() -> list[dict]:
    api = HfApi()
    out = []
    for repo, rev in [(S.FWE_REPO, S.FWE_REVISION), (S.PG19_REPO, S.PG19_REVISION)]:
        info = api.dataset_info(repo, revision=rev, files_metadata=True)
        for sib in info.siblings:
            f = sib.rfilename
            if repo == S.FWE_REPO and f not in S.FWE_FILES:
                continue
            if repo == S.PG19_REPO and not (f.startswith("data/") and f.endswith(".parquet")):
                continue
            lfs = sib.lfs
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            out.append(dict(repo=repo, revision=rev, file=f, size=sib.size, lfs_sha256=sha))
    return out


def sha256_file(path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict) -> dict:
    dst = S.raw_path(entry["repo"], entry["file"])
    if not (dst.exists() and dst.stat().st_size == entry["size"]):
        t0 = time.time()
        hf_hub_download(entry["repo"], entry["file"], repo_type="dataset", revision=entry["revision"],
                        local_dir=str(S.RAW_DIR / entry["repo"].replace("/", "__")))
        print(f"downloaded {entry['file']} in {time.time()-t0:.0f}s", flush=True)
    sha = sha256_file(dst)
    ok = sha == entry["lfs_sha256"]
    print(f"{'OK ' if ok else 'BAD'} {entry['repo']}/{entry['file']} {sha}", flush=True)
    return dict(entry, sha256=sha, ok=ok)


def main() -> int:
    entries = list_files()
    tot = sum(e["size"] for e in entries)
    print(f"{len(entries)} files, {tot/1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, entries))
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (S.RAW_DIR / "download_report.json").write_text(json.dumps(res, indent=1))
    bad = [r for r in res if not r["ok"]]
    print("ALL OK" if not bad else f"{len(bad)} BAD FILES", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
