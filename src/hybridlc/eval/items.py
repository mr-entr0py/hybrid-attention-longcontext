"""Frozen item files and their manifest (protocol §9.8, VT-12).

Layout: ``experiments/items/{test,dev}/<suite>/<cell>.npz`` with
``<cell> = <task>[-<variant>]_<level>_<T>`` (``T`` as 1K/4K/32K..., ``nofill`` for C0), plus
``experiments/items/{test,dev}/MANIFEST.json``.

File format: a valid ``.npz`` (zip, stored, members with a fixed 1980-01-01 timestamp in a
fixed order), so the bytes are a pure function of the items: regenerating a cell gives a
byte-identical file, and every model reads the same bytes (VT-12). Members:

* ``ctx`` (uint16 if every id < 65536, else int32) + ``ctx_off`` (int64[n+1]);
* per query: ``q_item`` (int32), ``q_prompt``/``q_prompt_off``, ``q_answer``/``q_answer_off``,
  ``q_cand_off`` (int64[nq+1], into the candidate list), candidates ``cand``/``cand_off``;
* per-item variable-length arrays ``arr.<name>`` + ``arr.<name>.off`` (concatenated on axis 0);
* ``meta.json``: canonical JSON with the cell meta, per-item meta and per-query meta.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "hybridlc-items-v1"
ITEMS_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "items"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------------- naming
def t_name(T: int | None) -> str:
    """'512', '1K' ... '32K'; 'nofill' for the C0 control (T=None); otherwise the integer."""
    if T is None:
        return "nofill"
    if T % 1024 == 0:
        return f"{T // 1024}K"
    return str(T)


def level_name(level: dict) -> str:
    """Canonical level string, e.g. {'D': 3} -> 'D3', {'rho': 1.25} -> 'rho1.25', {'variant': 'a', 'G': 512} -> 'a-G512'."""
    parts = []
    for k in sorted(level):
        if k in ("template", "placement", "filler", "order", "distances", "n_distractors", "variant", "n_pg19"):
            continue
        v = level[k]
        if isinstance(v, float) and v.is_integer() and k != "rho":
            v = int(v)
        parts.append(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}")
    if "variant" in level:
        parts.insert(0, str(level["variant"]))
    return "-".join(parts) if parts else "all"


def cell_name(task: str, level: dict, T: int | None, variant: str | None = None) -> str:
    return f"{task}{'-' + variant if variant else ''}_{level_name(level)}_{t_name(T)}"


# --------------------------------------------------------------------------- JSON
def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def canonical_json(obj: Any) -> bytes:
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


# --------------------------------------------------------------------------- write / read
def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def _cat(arrs: list[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    if arrs:
        off[1:] = np.cumsum([len(a) for a in arrs])
        cat = np.concatenate([np.asarray(a).astype(dtype, copy=False) for a in arrs]) if off[-1] else np.zeros(0, dtype)
    else:
        cat = np.zeros(0, dtype)
    return cat, off


def _tok_dtype(*arrs: np.ndarray) -> type:
    mx = max([int(a.max()) for a in arrs if len(a)] + [0])
    return np.uint16 if mx < 65536 else np.int32


def items_to_members(items: list[dict], cell_meta: dict) -> dict[str, bytes]:
    """Serialise items to ordered zip members (name -> bytes)."""
    ctx = [np.asarray(it["context_ids"]) for it in items]
    qs = [(i, q) for i, it in enumerate(items) for q in it["queries"]]
    prompts = [np.asarray(q["prompt_ids"]) for _, q in qs]
    answers = [np.asarray(q["answer_ids"]) for _, q in qs]
    cands = [np.asarray(c, dtype=np.int64) for _, q in qs for c in q["candidates"]]
    ncand = [len(q["candidates"]) for _, q in qs]
    allc = [np.concatenate(ctx) if ctx else np.zeros(0)] + prompts + answers + cands
    dt = _tok_dtype(*[a for a in allc if len(a)])
    m: dict[str, np.ndarray] = {}
    m["ctx"], m["ctx_off"] = _cat(ctx, dt)
    m["q_item"] = np.asarray([i for i, _ in qs], dtype=np.int32)
    m["q_prompt"], m["q_prompt_off"] = _cat(prompts, dt)
    m["q_answer"], m["q_answer_off"] = _cat(answers, dt)
    m["q_cand_off"] = np.concatenate([[0], np.cumsum(ncand)]).astype(np.int64)
    m["cand"], m["cand_off"] = _cat(cands, dt)
    names = sorted({k for it in items for k in it.get("arrays", {})})
    arr_info = {}
    for nm in names:
        parts = []
        tail, dtype = None, None
        for it in items:
            a = it.get("arrays", {}).get(nm)
            if a is None:
                raise ValueError(f"array {nm!r} missing from some items")
            a = np.asarray(a)
            tail = a.shape[1:] if tail is None else tail
            dtype = a.dtype if dtype is None else dtype
            if a.shape[1:] != tail or a.dtype != dtype:
                raise ValueError(f"array {nm!r} has inconsistent shape/dtype across items")
            parts.append(a)
        off = np.concatenate([[0], np.cumsum([len(a) for a in parts])]).astype(np.int64)
        m[f"arr.{nm}"] = np.concatenate(parts) if parts and off[-1] else np.zeros((0, *tail), dtype)
        m[f"arr.{nm}.off"] = off
        arr_info[nm] = dict(dtype=str(dtype), tail_shape=list(tail))
    meta = dict(format=FORMAT, cell=cell_meta, n_items=len(items), n_queries=len(qs), token_dtype=np.dtype(dt).name,
                arrays=arr_info, items=[it["meta"] for it in items], queries=[q["meta"] for _, q in qs])
    members = {f"{k}.npy": _npy_bytes(v) for k, v in m.items()}
    members["meta.json"] = canonical_json(meta)
    return members


def save_items(path: str | Path, items: list[dict], cell_meta: dict, overwrite: bool = False) -> str:
    """Write a deterministic item file; returns its SHA-256. Refuses to overwrite unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (item files are frozen; pass overwrite=True only before G5)")
    path.parent.mkdir(parents=True, exist_ok=True)
    members = items_to_members(items, cell_meta)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            zf.writestr(zi, members[name])
    tmp.replace(path)
    return sha256_file(path)


def load_items(path: str | Path) -> tuple[list[dict], dict]:
    """Read an item file -> (items, cell_meta). Token arrays come back as int32."""
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        def arr(name: str) -> np.ndarray:  # noqa: E306
            return np.lib.format.read_array(io.BytesIO(zf.read(f"{name}.npy")), allow_pickle=False)
        ctx, ctx_off = arr("ctx").astype(np.int32), arr("ctx_off")
        q_item = arr("q_item")
        qp, qpo = arr("q_prompt").astype(np.int32), arr("q_prompt_off")
        qa, qao = arr("q_answer").astype(np.int32), arr("q_answer_off")
        qco = arr("q_cand_off")
        cand, cando = arr("cand").astype(np.int64), arr("cand_off")
        arrays = {nm: (arr(f"arr.{nm}"), arr(f"arr.{nm}.off")) for nm in meta["arrays"]}
    items = []
    for i in range(meta["n_items"]):
        items.append(dict(context_ids=ctx[ctx_off[i]:ctx_off[i + 1]], queries=[], meta=meta["items"][i],
                          arrays={nm: a[o[i]:o[i + 1]] for nm, (a, o) in arrays.items()}))
    for j in range(meta["n_queries"]):
        cs = [cand[cando[c]:cando[c + 1]].tolist() for c in range(qco[j], qco[j + 1])]
        items[int(q_item[j])]["queries"].append(dict(prompt_ids=qp[qpo[j]:qpo[j + 1]], answer_ids=qa[qao[j]:qao[j + 1]],
                                                     candidates=cs, meta=meta["queries"][j]))
    return items, meta["cell"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- manifest
def eval_code_sha256() -> str:
    """SHA-256 over the evaluation package sources (sorted relative path + bytes)."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


#: Cell-meta keys that describe the *code version* that wrote a file, not its content. They are
#: excluded from :func:`item_content_sha256` (verification round 2, ITEM-EVAL-CODE-SHA-STAMPS).
PROVENANCE_META_KEYS: tuple[str, ...] = ("eval_code_sha256",)


def item_content_sha256(path: str | Path) -> str:
    """SHA-256 of an item file's *content*: every zip member (sorted by name), with the cell meta's
    code-version stamp (``PROVENANCE_META_KEYS``) removed from ``meta.json``.

    Regenerating an unchanged cell with any later eval-code version gives the same content SHA (the
    file SHA differs only through the embedded stamp). This is the identity G5 checks.
    """
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            b = zf.read(name)
            if name == "meta.json":
                meta = json.loads(b)
                for k in PROVENANCE_META_KEYS:
                    meta.get("cell", {}).pop(k, None)
                b = canonical_json(meta)
            h.update(name.encode() + b"\0" + hashlib.sha256(b).digest())
    return h.hexdigest()


def write_manifest(ns_root: str | Path, extra: dict | None = None) -> Path:
    """(Re)write ``<ns_root>/MANIFEST.json`` listing every item file with its SHA-256 and counts.

    Per file: ``sha256`` (bytes), ``content_sha256`` (:func:`item_content_sha256`) and the cell meta
    (which carries the generating code's ``eval_code_sha256``). Top level: ``eval_code_sha256`` is the
    code at manifest-write time and ``item_eval_code_sha256`` counts the stamps embedded in the files
    (which code version generated the frozen items).
    """
    ns_root = Path(ns_root)
    files = {}
    stamps: dict[str, int] = {}
    for p in sorted(ns_root.rglob("*.npz")):
        with zipfile.ZipFile(p) as zf:
            meta = json.loads(zf.read("meta.json"))
        files[str(p.relative_to(ns_root))] = dict(sha256=sha256_file(p), content_sha256=item_content_sha256(p),
                                                  bytes=p.stat().st_size, n_items=meta["n_items"],
                                                  n_queries=meta["n_queries"], cell=meta["cell"])
        st = str(meta["cell"].get("eval_code_sha256"))
        stamps[st] = stamps.get(st, 0) + 1
    for p in sorted(ns_root.rglob("*.npy")):  # auxiliary arrays (e.g. LM-3 banned 4-grams)
        files[str(p.relative_to(ns_root))] = dict(sha256=sha256_file(p), bytes=p.stat().st_size)
    man = dict(format=FORMAT, namespace=ns_root.name, files=files, item_eval_code_sha256=stamps, **(extra or {}))
    out = ns_root / "MANIFEST.json"
    out.write_text(json.dumps(_jsonable(man), indent=1, sort_keys=True) + "\n")
    return out


def verify_manifest(ns_root: str | Path, content: bool = False) -> list[str]:
    """Paths whose SHA-256 no longer matches the manifest (empty = all good). ``content=True`` also
    checks ``content_sha256`` where the manifest records it."""
    ns_root = Path(ns_root)
    man = json.loads((ns_root / "MANIFEST.json").read_text())
    bad = []
    for rel, info in man["files"].items():
        p = ns_root / rel
        if not p.is_file() or sha256_file(p) != info["sha256"]:
            bad.append(rel)
        elif content and "content_sha256" in info and item_content_sha256(p) != info["content_sha256"]:
            bad.append(rel)
    return bad


def refresh_manifest(ns_root: str | Path) -> Path:
    """Rewrite an existing manifest without touching any item file: keeps its extra fields
    (tokenizer, word lists, filler, cap), stamps the current ``eval_code_sha256`` and adds
    ``content_sha256`` / ``item_eval_code_sha256``. Refuses if a file no longer matches."""
    ns_root = Path(ns_root)
    bad = verify_manifest(ns_root)
    if bad:
        raise ValueError(f"{ns_root}: files differ from the manifest: {bad[:5]}")
    old = json.loads((ns_root / "MANIFEST.json").read_text())
    extra = {k: v for k, v in old.items() if k not in ("format", "namespace", "files", "item_eval_code_sha256")}
    extra["eval_code_sha256"] = eval_code_sha256()
    return write_manifest(ns_root, extra=extra)
