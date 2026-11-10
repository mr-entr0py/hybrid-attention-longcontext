#!/usr/bin/env python
"""Measured-cost benchmark driver (protocol §10.2; secondary, scoped to "fla Triton + PyTorch SDPA on
sm_120, this implementation").  Built on the VT-15-validated primitives of ``hybridlc.bench.harness``.

What one session measures (per scale, D384 first):

* ``vt14``    -- inference-state ``nbytes`` after a bf16-autocast prefill (B = 1, T in {1K,4K,16K,32K})
                 and of the static decode buffers, against ``analytic.state_bytes`` (to the byte);
* ``check``   -- the benchmark's static-buffer one-step decoder vs ``HybridLM.decode_step`` (logits);
* ``prefill`` -- whole model, B in {1, 8} x T in {1K,4K,16K,32K}; exact-shape CUDA graph and eager;
                 SDPA pinned to FLASH (cuDNN also); last-token logits only; cold and hot L2;
* ``decode``  -- one-step exact-shape graph per (L, B), no masks, cuDNN (primary) and FLASH, GDN via
                 explicit ``fused_recurrent`` with its state ``copy_``-ed into static buffers;
                 B in {1, 8, B_max(L)}, B_max from an 8 GiB state+activation budget (analytic, then
                 verified by allocation);
* ``kernel``  -- per-kernel achieved FLOP/s / bytes/s and roofline fraction (attention core prefill and
                 decode, GDN chunk core and fused_recurrent step, projection GEMMs);
* ``flex``    -- optional FlexAttention attention-core comparator ("equal toolchain, not equal maturity").

Statistics: 5 warm-up replays, then N timed replays (CUDA events, synchronise before reading); min-of-N
primary, median / IQR secondary; cold L2 = zero a 256 MiB buffer before every replay (untimed).  After
each replay the NVML clocks-event reasons and the power/thermal violation counters are read; a replay
with any reason other than GpuIdle, or with a violation-counter increase, is discarded and replaced
(up to 3N attempts).  Configs run in randomised interleaved blocks (order from the ``bench`` stream).

Idle criterion G-cost: a 5 s poller (nvidia-smi query-compute-apps + pmon) logs every foreign process;
the session waits for >= 60 s without foreign SM activity and starts only with utilisation < 5 %.
Compliance is recorded in ``session.jsonl`` and in every row's time window (``t0``/``t1``).

Resumable: every finished (or skipped) measurement is one JSONL row with a unique ``key``; a restart
skips keys already present.  Weights are random-init, seed index drawn from the ``bench`` stream.

    python scripts/benchmark_inference.py run  --out experiments/bench/cost_session1 [--scales D384 D512]
    python scripts/benchmark_inference.py warm --out ...            # Triton pre-warm (separate process)
    python scripts/benchmark_inference.py summarise --out ... > tables.md
"""
from __future__ import annotations

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[1] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---

import argparse
import ctypes
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO / ".cache" / "triton_bench"))
os.environ["FLA_FLASH_QLA"] = "0"
os.environ.setdefault("FLA_CACHE_RESULTS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.attention import SDPBackend  # noqa: E402

from hybridlc.bench import harness as H  # noqa: E402
from hybridlc.models import attention as A  # noqa: E402
from hybridlc.models import linear as LIN  # noqa: E402
from hybridlc.models.analytic import flops_per_token, linear_core_flops, linear_dims, param_counts, state_bytes  # noqa: E402
from hybridlc.models.build import build_model  # noqa: E402
from hybridlc.models.config import ModelConfig  # noqa: E402
from hybridlc.models.layers import logits_fp32  # noqa: E402
from hybridlc.training import gpumon  # noqa: E402
from hybridlc.utils.seeds import BENCH_ENTROPY, crc32, entropy_sequence  # noqa: E402

CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C6", "C12", "X-gc12", "X-E0"]
TS = [1024, 4096, 16384, 32768]
PREFILL_BS = [1, 8]
GiB = 2**30
BUDGET = 8 * GiB                 # §10.2: state + activation budget (decode B_max; also used for prefill skips)
PEAK_FLOPS = 112.6e12            # §10.1 dense BF16
PEAK_BW = 960e9                  # §10.1
BE = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
GPU_IDLE_BIT = 0x1
SESSION = 1


# ============================================================================ NVML (ctypes; no new packages)
class NVML:
    POWER, THERMAL, SYNC_BOOST, BOARD_LIMIT, LOW_UTIL, RELIABILITY = range(6)

    class _VS(ctypes.Structure):
        _fields_ = [("referenceTime", ctypes.c_ulonglong), ("violationTime", ctypes.c_ulonglong)]

    class _UT(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self, idx: int = 0) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.h = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(self.h)) == 0
        self._reasons = getattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons

    def reasons(self) -> int | None:
        v = ctypes.c_ulonglong()
        return int(v.value) if self._reasons(self.h, ctypes.byref(v)) == 0 else None

    def violations(self) -> tuple:
        out = []
        for p in (self.POWER, self.THERMAL, self.RELIABILITY, self.BOARD_LIMIT):
            s = self._VS()
            out.append(int(s.violationTime) if self.lib.nvmlDeviceGetViolationStatus(self.h, p, ctypes.byref(s)) == 0 else None)
        return tuple(out)

    def _uint(self, fn, *args) -> int | None:
        v = ctypes.c_uint()
        return int(v.value) if fn(self.h, *args, ctypes.byref(v)) == 0 else None

    def snap(self) -> dict:
        u = self._UT()
        ok = self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0
        return dict(sm_mhz=self._uint(self.lib.nvmlDeviceGetClockInfo, 1),
                    mem_mhz=self._uint(self.lib.nvmlDeviceGetClockInfo, 2),
                    temp_c=self._uint(self.lib.nvmlDeviceGetTemperature, 0),
                    power_w=(lambda x: None if x is None else x / 1000)(self._uint(self.lib.nvmlDeviceGetPowerUsage)),
                    util=u.gpu if ok else None, reasons=self.reasons(), violations_ns=self.violations())


# ============================================================================ idle monitor (G-cost)
def compute_apps() -> list[dict]:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    out = []
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[0].isdigit():
            out.append(dict(pid=int(p[0]), name=p[1], mib=p[2]))
    return out


class Monitor(threading.Thread):
    """5 s poller. A *foreign compute process* is any compute context not owned by this process tree;
    *foreign activity* is foreign SM utilisation > 0 in ``nvidia-smi pmon`` (graphics included)."""

    def __init__(self, path: Path, own: int) -> None:
        super().__init__(daemon=True)
        self.path, self.own, self.stop_ev = path, own, threading.Event()
        self.last: dict = {}
        self.lock = threading.Lock()

    def poll(self) -> dict:
        rec: dict = {"ts": time.time()}
        try:
            own = gpumon.descendants(self.own)
            rec["gpu"] = gpumon.query_gpu(0)
            apps = compute_apps()
            pm = gpumon.query_pmon(0)
            rec["foreign_compute"] = [a for a in apps if a["pid"] not in own]
            rec["foreign_pmon"] = [p for p in pm if p["pid"] not in own and (p["sm"] or 0) > 0]
            rec["foreign_sm_pct"] = float(sum(p["sm"] or 0 for p in pm if p["pid"] not in own))
            rec["own_sm_pct"] = float(sum(p["sm"] or 0 for p in pm if p["pid"] in own))
        except Exception as e:  # pragma: no cover
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    def run(self) -> None:
        with open(self.path, "a") as f:
            while not self.stop_ev.is_set():
                rec = self.poll()
                with self.lock:
                    self.last = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self.stop_ev.wait(5.0)


def wait_idle(mon: Monitor, log, need_s: float = 60.0, max_wait_s: float = 3600.0) -> dict:
    """Wait until ``need_s`` seconds of polls without foreign SM activity, util < 5 % at the end."""
    t0 = time.time()
    clean_since = None
    polls = []
    while True:
        rec = mon.poll()
        polls.append(rec)
        busy = rec.get("error") or rec.get("foreign_sm_pct", 0) > 0 or (rec.get("gpu", {}).get("utilization.gpu") or 0) >= 5
        now = time.time()
        if busy:
            clean_since = None
        elif clean_since is None:
            clean_since = now
        if clean_since is not None and now - clean_since >= need_s:
            break
        if now - t0 > max_wait_s:
            log(f"idle wait timed out after {max_wait_s:.0f} s; proceeding as CONTENDED")
            break
        time.sleep(5.0)
    window = [p for p in polls if p["ts"] >= (clean_since or polls[0]["ts"])]
    fc = sorted({(a["pid"], a["name"]) for p in window for a in p.get("foreign_compute", [])})
    return dict(waited_s=time.time() - t0, clean_window_s=(time.time() - clean_since) if clean_since else 0.0,
                polls=len(window), start_util=polls[-1].get("gpu", {}).get("utilization.gpu"),
                activity_ok=clean_since is not None and time.time() - clean_since >= need_s,
                foreign_compute_contexts=[dict(pid=p, name=n) for p, n in fc],
                strict_ok=not fc)


# ============================================================================ timing
class Timer:
    """Replay timing with the harness's 256 MiB L2 flush and NVML replay screening."""

    def __init__(self, nv: NVML, n: int, warmup: int) -> None:
        self.nv, self.n, self.warmup = nv, n, warmup

    def time(self, fn, cold: bool) -> dict:
        flush = H._flush_buf(H.L2_FLUSH_BYTES) if cold else None  # noqa: SLF001
        for _ in range(self.warmup):
            if flush is not None:
                flush.zero_()
            fn()
        torch.cuda.synchronize()
        kept, disc, reasons_seen = [], [], {}
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = self.nv.snap()
        attempts = 0
        while len(kept) < self.n and attempts < 3 * self.n:
            attempts += 1
            v0 = self.nv.violations()
            if flush is not None:
                flush.zero_()                     # untimed: before the start event
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            t = s.elapsed_time(e) * 1e3
            r = self.nv.reasons()
            v1 = self.nv.violations()
            reasons_seen[r] = reasons_seen.get(r, 0) + 1
            viol = any(a is not None and b is not None and b > a for a, b in zip(v0, v1))
            if r is None or (r & ~GPU_IDLE_BIT) or viol:
                disc.append(dict(us=t, reasons=r, viol=viol))
            else:
                kept.append(t)
        after = self.nv.snap()
        out = dict(n_target=self.n, n_kept=len(kept), n_discarded=len(disc), attempts=attempts,
                   reasons_seen={str(k): v for k, v in reasons_seen.items()}, discarded=disc[:20],
                   nvml_before=before, nvml_after=after, times_us=[round(x, 3) for x in kept])
        if len(kept) >= 4:
            q = statistics.quantiles(kept, n=4)
            out.update(min_us=min(kept), median_us=statistics.median(kept), iqr_us=q[2] - q[0])
        elif kept:
            out.update(min_us=min(kept), median_us=statistics.median(kept), iqr_us=None)
        return out


# ============================================================================ models
def bench_seed_index() -> int:
    ss = entropy_sequence(BENCH_ENTROPY, crc32("cost_benchmark_weights"))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def bench_ids(B: int, T: int, V: int, tag: str) -> torch.Tensor:
    rng = np.random.default_rng(entropy_sequence(BENCH_ENTROPY, crc32(f"ids/{tag}/{B}/{T}")))
    return torch.as_tensor(rng.integers(8, V, (B, T)), dtype=torch.long, device="cuda")


def prepare(model: nn.Module) -> nn.Module:
    """Hoist the bf16 casts autocast would do on every call: nn.Linear weights stored in bf16 and a bf16
    copy of the tied embedding for the logits GEMM.  Numerically identical to fp32 weights under bf16
    autocast (the cast is the same round-to-nearest); norms, A_log, dt_bias, conv kernels and the
    embedding lookup stay fp32 as in the production path."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = m.weight.data.to(torch.bfloat16)
    model._w_logits = model.embed.weight.detach().to(torch.bfloat16)
    model.logits = types.MethodType(lambda self, h: logits_fp32(h, self._w_logits), model)
    return model


def build(scale: str, name: str) -> nn.Module:
    cfg = ModelConfig.for_scale(scale, name)
    return prepare(build_model(cfg, bench_seed_index()).to("cuda").eval())


def ac():
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)


def has_attn(cfg) -> bool:
    return any(c in "AC" for c in cfg.pattern)


# ============================================================================ static-buffer decoder
class Decoder:
    """One decode step at exact cache length L for B rows, all state in static buffers.

    Attention layers: KV buffers [B, L+1, H, hd] bf16 (the production cache layout); the step writes the
    new k/v into slot L (``copy_``) and runs SDPA over all L+1 keys with no mask (``attn_inference``
    Tq == 1 path, backend pinned).  F_gc conv state and GDN state / conv caches are fp32 static buffers;
    GDN layers call ``GDNMixer._core(..., recurrent=True)`` (explicit ``fused_recurrent_gated_delta_rule``)
    and the returned state is ``copy_``-ed back.  Mirrors ``HybridLM.extend`` except that the KV append is
    an in-place slot write instead of ``torch.cat``.
    """

    def __init__(self, model, L: int, B: int, templates: list[dict], fill_seed: int) -> None:
        self.m, self.L, self.B = model, L, B
        g = torch.Generator(device="cuda").manual_seed(fill_seed)
        self.bufs: list[dict[str, torch.Tensor]] = []
        for tpl in templates:
            st = {}
            for key, (shape, dtype) in tpl.items():
                if key in ("k", "v"):
                    t = torch.empty((B, L + 1) + tuple(shape[2:]), dtype=dtype, device="cuda")
                    t.normal_(0, 1, generator=g)
                else:
                    t = torch.empty((B,) + tuple(shape[1:]), dtype=dtype, device="cuda")
                    t.normal_(0, 0.02, generator=g)
                st[key] = t
            self.bufs.append(st)
        self.tok = torch.randint(8, model.cfg.vocab_size, (B, 1), device="cuda", generator=g)
        self.pos = torch.full((B, 1), L, dtype=torch.long, device="cuda")
        self.out = torch.empty(B, model.cfg.vocab_size, dtype=torch.float32, device="cuda")

    def nbytes(self) -> dict:
        out = {"kv": 0, "gdn_state": 0, "conv": 0}
        for st in self.bufs:
            for key, t in st.items():
                cat = "kv" if key in ("k", "v") else "gdn_state" if key == "state" else "conv"
                out[cat] += t.numel() * t.element_size()
        out["total"] = sum(out.values())
        return out

    @torch.no_grad()
    def step(self) -> None:
        m, L = self.m, self.L
        h = m.embed(self.tok)
        for blk, st in zip(m.layers, self.bufs):
            x = blk.norm1(h)
            mix = blk.mixer
            if isinstance(mix, A.AttnMixer):
                q, k, v, conv = mix._qkv(x, None, st.get("conv"), mix.use_conv)
                q, k = mix._norm_rope(q, k, self.pos, None)
                st["k"][:, L:L + 1].copy_(k)
                st["v"][:, L:L + 1].copy_(v)
                o = A.attn_inference(q.transpose(1, 2).to(st["k"].dtype), st["k"].transpose(1, 2),
                                     st["v"].transpose(1, 2), mix.scale)
                y = mix._gate_out(o.transpose(1, 2).to(v.dtype), x)
                if mix.use_conv:
                    st["conv"].copy_(conv)
            else:
                y, new = mix._core(x, cu_seqlens=None, state=st, want_state=True, recurrent=True)
                for key, t in new.items():
                    st[key].copy_(t)
            u = h + y
            h = u + blk.mlp(blk.norm2(u))
        self.out.copy_(m.logits(m.norm_f(h[:, -1])))


def templates_for(model) -> list[dict]:
    ids = bench_ids(1, 64, model.cfg.vocab_size, "tpl")
    with torch.no_grad(), ac():
        _, cache = model.prefill(ids)
    return [{k: (tuple(t.shape), t.dtype) for k, t in st.items()} for st in cache.layers]


# ============================================================================ driver
class Session:
    def __init__(self, args) -> None:
        self.a = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.out / "rows.jsonl"
        self.done: set[str] = set()
        if self.rows_path.exists():
            for line in self.rows_path.read_text().splitlines():
                try:
                    self.done.add(json.loads(line)["key"])
                except Exception:
                    pass
        if getattr(args, "retry_errors", False) and self.rows_path.exists():
            for line in self.rows_path.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") == "error":
                    self.done.discard(r["key"])
        self.logf = open(self.out / "driver.log", "a")
        self.nv = NVML()
        self.timer = Timer(self.nv, args.n, args.warmup)
        self.t_start = time.time()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def write(self, row: dict) -> None:
        row.setdefault("session", SESSION)
        with open(self.rows_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.done.add(row["key"])

    def meta(self, rec: dict) -> None:
        with open(self.out / "session.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def order(self, tag: str, items: list) -> list:
        rng = np.random.default_rng(entropy_sequence(BENCH_ENTROPY, crc32(f"order/s{SESSION}/{tag}")))
        idx = rng.permutation(len(items))
        return [items[i] for i in idx]

    def over_time(self) -> bool:
        return (time.time() - self.t_start) / 3600 > self.a.max_hours

    # ------------------------------------------------------------------ phases
    def phase_vt14(self, scale: str, models: dict) -> None:
        for name, model in models.items():
            cfg = model.cfg
            for T in TS:
                key = f"vt14|{scale}|{name}|1|{T}"
                if key in self.done:
                    continue
                ids = bench_ids(1, T, cfg.vocab_size, "vt14")
                with torch.no_grad(), ac():
                    _, cache = model.prefill(ids)
                meas = cache.state_nbytes()
                dtypes_ok = all(t.dtype == (torch.bfloat16 if k in ("k", "v") else torch.float32)
                                for st in cache.layers for k, t in st.items())
                ana = state_bytes(cfg, T)
                self.write(dict(key=key, phase="vt14", scale=scale, config=name, pattern=cfg.pattern, B=1, T=T,
                                measured=meas, analytic=ana, equal=meas == ana, dtypes_ok=dtypes_ok))
                del cache
            torch.cuda.empty_cache()

    def phase_check(self, scale: str, models: dict) -> None:
        """(1) bf16-hoisted weights vs the production fp32-weights + autocast model (prefill last logits);
        (2) static-buffer decoder vs HybridLM.decode_step on a real prefill (B = 2, L = 512)."""
        for name in ("C0", "C3", "C12", "X-gc12", "X-E0"):
            key = f"check_weights|{scale}|{name}"
            if key in self.done or name not in models:
                continue
            ref_m = build_model(ModelConfig.for_scale(scale, name), bench_seed_index()).to("cuda").eval()
            ids = bench_ids(2, 512, ref_m.cfg.vocab_size, "check")
            with torch.no_grad(), ac():
                lr, _ = ref_m.prefill(ids)
                lb, _ = models[name].prefill(ids)
            res = dict(max_abs=(lr - lb).abs().max().item(), ref_absmax=lr.abs().max().item(),
                       bit_identical=bool(torch.equal(lr, lb)))
            self.write(dict(key=key, phase="check_weights", scale=scale, config=name, result=res))
            self.log(f"check_weights {scale} {name}: {res}")
            del ref_m
            torch.cuda.empty_cache()
        for name in ("C0", "C3", "C12", "X-gc12", "X-E0"):
            key = f"check|{scale}|{name}"
            if key in self.done or name not in models:
                continue
            m = models[name]
            L, B = 512, 2
            ids = bench_ids(B, L, m.cfg.vocab_size, "check")
            tok = bench_ids(B, 1, m.cfg.vocab_size, "checktok")
            with torch.no_grad(), ac():
                _, cache = m.prefill(ids)
                ref, _ = m.decode_step(tok, cache.clone())
                tpl = [{k: (tuple(t.shape), t.dtype) for k, t in st.items()} for st in cache.layers]
                dec = Decoder(m, L, B, tpl, 0)
                for st, cst in zip(dec.bufs, cache.layers):
                    for k, t in cst.items():
                        if k in ("k", "v"):
                            st[k][:, :L].copy_(t)
                        else:
                            st[k].copy_(t)
                dec.tok.copy_(tok)
                res = {}
                for be in (["cudnn", "flash"] if has_attn(m.cfg) else ["none"]):
                    A.INFERENCE_HALF_BACKEND = BE.get(be, SDPBackend.FLASH_ATTENTION)
                    d2 = Decoder.__new__(Decoder)
                    d2.__dict__.update(dec.__dict__)
                    d2.bufs = [{k: t.clone() for k, t in st.items()} for st in dec.bufs]
                    d2.out = torch.empty_like(dec.out)
                    d2.step()
                    diff = (d2.out - ref.float()).abs().max().item()
                    res[be] = dict(max_abs=diff, argmax_agree=float((d2.out.argmax(-1) == ref.argmax(-1)).float().mean()),
                                   ref_absmax=ref.abs().max().item())
                A.INFERENCE_HALF_BACKEND = SDPBackend.FLASH_ATTENTION
            self.write(dict(key=key, phase="check", scale=scale, config=name, L=L, B=B, result=res))
            self.log(f"check {scale} {name}: {res}")
            del cache, dec
            torch.cuda.empty_cache()

    def _spy_assert(self, spy, backend: str) -> dict:
        calls = spy.calls
        bad = [c for c in calls if c.mask_kind != "none" or c.efficient or c.math
               or (backend == "flash" and (not c.flash or c.cudnn)) or (backend == "cudnn" and (not c.cudnn or c.flash))]
        if bad:
            raise AssertionError(f"SDPA backend/mask assertion failed: {bad[:2]}")
        return dict(n_sdpa_calls=len(calls), backend_ok=True, masks="none")

    def phase_prefill(self, scale: str, models: dict) -> None:
        a = self.a
        for B in PREFILL_BS:
            for T in TS:
                for name in self.order(f"prefill/{scale}/{B}/{T}", list(models)):
                    if self.over_time():
                        self.log("time budget reached; stopping prefill")
                        return
                    m = models[name]
                    bes = a.prefill_backends if has_attn(m.cfg) else ["none"]
                    variants = [(mode, be, l2) for be in bes for mode in a.prefill_modes for l2 in ("cold", "hot")
                                if not (mode == "eager" and be == "cudnn" and not a.eager_cudnn)]
                    keys = {v: f"prefill|{scale}|{name}|{B}|{T}|{v[0]}|{v[1]}|{v[2]}" for v in variants}
                    skipkey = f"prefill|{scale}|{name}|{B}|{T}|skip"
                    if skipkey in self.done or all(k in self.done for k in keys.values()):
                        continue
                    self._prefill_unit(scale, name, m, B, T, variants, keys, skipkey)

    def _prefill_unit(self, scale, name, m, B, T, variants, keys, skipkey) -> None:
        cfg = m.cfg
        ids = bench_ids(B, T, cfg.vocab_size, "prefill")
        # analytic pre-filter: cache alone over budget -> skip without trying
        sb = state_bytes(cfg, T, batch=B)["total"]
        base = torch.cuda.memory_allocated()
        if sb > BUDGET:
            self.write(dict(key=skipkey, phase="prefill", status="skip", scale=scale, config=name, B=B, T=T,
                            reason=f"analytic state {sb / GiB:.2f} GiB > budget"))
            self.log(f"prefill {scale} {name} B={B} T={T}: SKIP (analytic state {sb / GiB:.2f} GiB)")
            return
        holder = {}

        def fn():
            holder.pop("o", None)       # free the previous output (KV cache) before the next call allocates one
            with torch.no_grad(), ac():
                holder["o"] = m.prefill(ids)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            A.INFERENCE_HALF_BACKEND = SDPBackend.FLASH_ATTENTION
            fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
        except torch.OutOfMemoryError:
            holder.clear()
            torch.cuda.empty_cache()
            self.write(dict(key=skipkey, phase="prefill", status="skip", scale=scale, config=name, B=B, T=T,
                            reason="OOM in eager probe"))
            self.log(f"prefill {scale} {name} B={B} T={T}: SKIP (OOM)")
            return
        holder.clear()
        if peak > BUDGET:
            self.write(dict(key=skipkey, phase="prefill", status="skip", scale=scale, config=name, B=B, T=T,
                            reason=f"measured eager peak {peak / GiB:.2f} GiB > budget", peak_bytes=peak))
            self.log(f"prefill {scale} {name} B={B} T={T}: SKIP (peak {peak / GiB:.2f} GiB)")
            torch.cuda.empty_cache()
            return
        by_be: dict = {}
        for v in variants:
            by_be.setdefault((v[0], v[1]), []).append(v[2])
        for (mode, be), l2s in by_be.items():
            if all(keys[(mode, be, l2)] in self.done for l2 in l2s):
                continue
            A.INFERENCE_HALF_BACKEND = BE.get(be, SDPBackend.FLASH_ATTENTION)
            t0 = time.time()
            try:
                with H.sdpa_spy() as spy:
                    if mode == "graph":
                        f = H.graphed(fn)
                    else:
                        fn()
                        f = fn
                spyinfo = self._spy_assert(spy, be) if has_attn(cfg) else dict(n_sdpa_calls=len(spy.calls))
                for l2 in l2s:
                    k = keys[(mode, be, l2)]
                    if k in self.done:
                        continue
                    t1 = time.time()
                    res = self.timer.time(f, cold=(l2 == "cold"))
                    self.write(dict(key=k, phase="prefill", status="ok", scale=scale, config=name, pattern=cfg.pattern,
                                    k=cfg.k, B=B, T=T, mode=mode, backend=be, l2=l2, eager_peak_bytes=peak,
                                    t0=t1, t1=time.time(), **spyinfo, **res))
                    self.log(f"prefill {scale} {name:6s} B={B} T={T:5d} {mode:5s} {be:5s} {l2}: "
                             f"min {res.get('min_us', float('nan')) / 1e3:.3f} ms med {res.get('median_us', float('nan')) / 1e3:.3f} "
                             f"(kept {res['n_kept']}, disc {res['n_discarded']})")
            except Exception as e:
                for l2 in l2s:
                    self.write(dict(key=keys[(mode, be, l2)], phase="prefill", status="error", scale=scale, config=name,
                                    B=B, T=T, mode=mode, backend=be, l2=l2, error=f"{type(e).__name__}: {e}",
                                    tb=traceback.format_exc()[-2000:]))
                self.log(f"prefill {scale} {name} B={B} T={T} {mode} {be}: ERROR {type(e).__name__}: {e}")
            finally:
                f = None
                holder.clear()
                A.INFERENCE_HALF_BACKEND = SDPBackend.FLASH_ATTENTION
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            _ = t0

    # ---------------------------------------------------------------- decode
    def bmax_analytic(self, cfg, L: int) -> int:
        per = state_bytes(cfg, L + 1)["total"] + 4 * cfg.vocab_size   # state after the step + fp32 logits row
        return int(BUDGET // per)

    def phase_decode(self, scale: str, models: dict, tpls: dict) -> None:
        a = self.a
        bmax_cache = {}
        for L in TS:
            for bclass in ("1", "8", "max"):
                for name in self.order(f"decode/{scale}/{L}/{bclass}", list(models)):
                    if self.over_time():
                        self.log("time budget reached; stopping decode")
                        return
                    m = models[name]
                    bes = a.decode_backends if has_attn(m.cfg) else ["none"]
                    unitkey = f"decode|{scale}|{name}|{L}|{bclass}"
                    variants = [(be, l2) for be in bes for l2 in ("cold", "hot")]
                    keys = {v: f"{unitkey}|{v[0]}|{v[1]}" for v in variants}
                    if unitkey + "|skip" in self.done or all(k in self.done for k in keys.values()):
                        continue
                    if bclass == "max":
                        if (name, L) not in bmax_cache:
                            bmax_cache[(name, L)] = self.find_bmax(scale, name, m, L, tpls[name])
                        B = bmax_cache[(name, L)]["B_max_verified"]
                        if B is None:
                            self.write(dict(key=unitkey + "|skip", phase="decode", status="skip", scale=scale, config=name,
                                            L=L, bclass=bclass, reason="B_max verification failed"))
                            continue
                    else:
                        B = int(bclass)
                    self._decode_unit(scale, name, m, L, B, bclass, tpls[name], variants, keys)

    def find_bmax(self, scale, name, m, L, tpl) -> dict:
        key = f"bmax|{scale}|{name}|{L}"
        for line in self.rows_path.read_text().splitlines() if self.rows_path.exists() else []:
            r = json.loads(line)
            if r["key"] == key:
                return r
        cfg = m.cfg
        Ba = self.bmax_analytic(cfg, L)
        B, tries = Ba, []
        verified = None
        for _ in range(4):
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            ok, used = False, None
            try:
                A.INFERENCE_HALF_BACKEND = BE["cudnn"] if has_attn(cfg) else SDPBackend.FLASH_ATTENTION
                dec = Decoder(m, L, B, tpl, 1)
                with ac():
                    g = H.graphed(dec.step)
                g()
                torch.cuda.synchronize()
                used = torch.cuda.max_memory_allocated() - base
                ok = used <= BUDGET
            except torch.OutOfMemoryError as e:
                used = None
                tries.append(dict(B=B, ok=False, error=str(e)[:200]))
            finally:
                dec = g = None
                A.INFERENCE_HALF_BACKEND = SDPBackend.FLASH_ATTENTION
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if used is not None:
                tries.append(dict(B=B, ok=ok, used_bytes=used))
            if ok:
                verified = B
                break
            B = int(B * (BUDGET / used) * 0.98) if used else int(B * 0.8)
        row = dict(key=key, phase="bmax", scale=scale, config=name, L=L, budget_bytes=BUDGET, B_max_analytic=Ba,
                   state_per_seq=state_bytes(cfg, L + 1)["total"], B_max_verified=verified, tries=tries)
        self.write(row)
        self.log(f"B_max {scale} {name} L={L}: analytic {Ba}, verified {verified} ({tries[-1]})")
        return row

    def _decode_unit(self, scale, name, m, L, B, bclass, tpl, variants, keys) -> None:
        cfg = m.cfg
        dec = None
        try:
            torch.cuda.empty_cache()
            dec = Decoder(m, L, B, tpl, 2)
            nb = dec.nbytes()
            ana = state_bytes(cfg, L + 1, batch=B)
            for be in dict.fromkeys(v[0] for v in variants):
                l2s = [v[1] for v in variants if v[0] == be]
                if all(keys[(be, l2)] in self.done for l2 in l2s):
                    continue
                A.INFERENCE_HALF_BACKEND = BE.get(be, SDPBackend.FLASH_ATTENTION)
                g = None
                try:
                    with H.sdpa_spy() as spy, ac():
                        g = H.graphed(dec.step)
                    spyinfo = self._spy_assert(spy, be) if has_attn(cfg) else dict(n_sdpa_calls=len(spy.calls))
                    for l2 in l2s:
                        t1 = time.time()
                        res = self.timer.time(g, cold=(l2 == "cold"))
                        self.write(dict(key=keys[(be, l2)], phase="decode", status="ok", scale=scale, config=name,
                                        pattern=cfg.pattern, k=cfg.k, L=L, B=B, bclass=bclass, backend=be, l2=l2,
                                        buffers_nbytes=nb, analytic_nbytes=ana, nbytes_equal=nb == ana,
                                        t0=t1, t1=time.time(), **spyinfo, **res))
                        self.log(f"decode {scale} {name:6s} L={L:5d} B={B:5d} {be:5s} {l2}: min {res.get('min_us', float('nan')):.1f} us "
                                 f"med {res.get('median_us', float('nan')):.1f} (kept {res['n_kept']}, disc {res['n_discarded']})")
                except Exception as e:
                    for l2 in l2s:
                        self.write(dict(key=keys[(be, l2)], phase="decode", status="error", scale=scale, config=name,
                                        L=L, B=B, bclass=bclass, backend=be, l2=l2, error=f"{type(e).__name__}: {e}",
                                        tb=traceback.format_exc()[-2000:]))
                    self.log(f"decode {scale} {name} L={L} B={B} {be}: ERROR {type(e).__name__}: {e}")
                finally:
                    g = None
                    A.INFERENCE_HALF_BACKEND = SDPBackend.FLASH_ATTENTION
                    torch.cuda.synchronize()
        finally:
            dec = None
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- kernels
    def _kernel(self, key: str, fn_build, flops: float, nbytes: float, meta: dict) -> None:
        if key in self.done:
            return
        try:
            torch.cuda.empty_cache()
            f = fn_build()
            res = self.timer.time(f, cold=True)
            t = res.get("min_us")
            roof_us = max(flops / PEAK_FLOPS, nbytes / PEAK_BW) * 1e6
            row = dict(key=key, phase="kernel", status="ok", flops=flops, bytes=nbytes, roofline_us=roof_us,
                       achieved_tflops=(flops / (t * 1e-6) / 1e12) if t else None,
                       achieved_gbs=(nbytes / (t * 1e-6) / 1e9) if t else None,
                       roofline_fraction=(roof_us / t) if t else None, l2="cold", **meta, **res)
            self.write(row)
            self.log(f"kernel {key}: min {t:.1f} us, roofline frac {row['roofline_fraction']:.3f}")
        except Exception as e:
            self.write(dict(key=key, phase="kernel", status="error", error=f"{type(e).__name__}: {e}", **meta))
            self.log(f"kernel {key}: ERROR {type(e).__name__}: {e}")
        finally:
            f = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def phase_kernel(self, scale: str, models: dict) -> None:
        cfg = models["C3"].cfg if "C3" in models else next(iter(models.values())).cfg
        d, dff, V = cfg.d_model, cfg.d_ff, cfg.vocab_size
        Hh, Dh = cfg.n_attn_heads, cfg.attn_head_dim
        bf = torch.bfloat16
        # attention core, prefill (causal)
        for be in ("flash", "cudnn"):
            for B in PREFILL_BS:
                for T in TS:
                    def mk(B=B, T=T, be=be):
                        q = torch.randn(B, Hh, T, Dh, device="cuda", dtype=bf)
                        k_, v_ = torch.randn_like(q), torch.randn_like(q)

                        def f():
                            with torch.nn.attention.sdpa_kernel([BE[be]]):
                                torch.nn.functional.scaled_dot_product_attention(q, k_, v_, is_causal=True)
                        f()
                        return H.graphed(f)
                    self._kernel(f"kernel|{scale}|attn_prefill|{be}|{B}|{T}", mk, 2 * B * Hh * T * T * Dh,
                                 4 * B * Hh * T * Dh * 2, dict(scale=scale, kernel="attn_prefill", backend=be, B=B, T=T))
        # attention core, decode (harness case, exact shape, no mask)
        for be in ("cudnn", "flash"):
            for B in (1, 8):
                for L in TS:
                    self._kernel(f"kernel|{scale}|attn_decode|{be}|{B}|{L}",
                                 lambda B=B, L=L, be=be: H.decode_attention_case(B, L, BE[be], H=Hh, Dh=Dh),
                                 4 * B * Hh * L * Dh, 2 * B * Hh * L * Dh * 2 + 2 * B * Hh * Dh * 2,
                                 dict(scale=scale, kernel="attn_decode", backend=be, B=B, L=L))
        # GDN chunk core and fused_recurrent step (GDN-A and GDN-E geometries)
        for ch, src in (("-", "C0"), ("E", "X-E0")):
            mcfg = models[src].cfg
            mix = next(b.mixer for b in models[src].layers if b.kind == ch)
            g_ = linear_dims(mcfg, ch)
            Hq, HV, dk, dv = g_["H"], g_["HV"], g_["dk"], g_["dv"]
            for B in PREFILL_BS:
                for T in TS:
                    def mkc(B=B, T=T):
                        q = torch.randn(B, T, Hq, dk, device="cuda", dtype=bf)
                        k_ = torch.randn_like(q)
                        v_ = torch.randn(B, T, HV, dv, device="cuda", dtype=bf)
                        gg = torch.randn(B, T, HV, device="cuda", dtype=bf)
                        bb = torch.randn(B, T, HV, device="cuda", dtype=bf)

                        def f():
                            LIN.chunk_gated_delta_rule(q=q, k=k_, v=v_, g=gg, beta=bb, initial_state=None,
                                                       output_final_state=True, use_qk_l2norm_in_kernel=True,
                                                       use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
                                                       state_v_first=True, cu_seqlens=None, use_gate_in_kernel=True,
                                                       A_log=mix.A_log, dt_bias=mix.dt_bias)
                        f()
                        return H.graphed(f)
                    by = B * T * (2 * Hq * dk * 2 + 2 * HV * dv * 2 + 2 * HV * 2) + B * HV * dk * dv * 4
                    self._kernel(f"kernel|{scale}|gdn_chunk|{ch}|{B}|{T}", mkc, B * T * linear_core_flops(mcfg, ch), by,
                                 dict(scale=scale, kernel="gdn_chunk", geometry="GDN-A" if ch == "-" else "GDN-E", B=B, T=T))
            for B in (1, 8, 256):
                def mkr(B=B):
                    q = torch.randn(B, 1, Hq, dk, device="cuda", dtype=bf)
                    k_ = torch.randn_like(q)
                    v_ = torch.randn(B, 1, HV, dv, device="cuda", dtype=bf)
                    gg = torch.randn(B, 1, HV, device="cuda", dtype=bf)
                    bb = torch.randn(B, 1, HV, device="cuda", dtype=bf)
                    s0 = torch.zeros(B, HV, dv, dk, device="cuda", dtype=torch.float32)

                    def f():
                        _, s1 = LIN.fused_recurrent_gated_delta_rule(
                            q=q, k=k_, v=v_, g=gg, beta=bb, initial_state=s0, output_final_state=True,
                            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=mix.A_log, dt_bias=mix.dt_bias,
                            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False, state_v_first=True, cu_seqlens=None)
                        s0.copy_(s1)
                    f()
                    return H.graphed(f)
                st = B * HV * dk * dv * 4
                # compulsory bytes (§10.1): state read + state write = 2 x state.  The timed region also
                # contains the static-buffer copy_ back (non-compulsory; L2-resident up to ~64 MiB).
                self._kernel(f"kernel|{scale}|gdn_recurrent|{ch}|{B}", mkr, B * HV * 6 * dk * dv, 2 * st,
                             dict(scale=scale, kernel="gdn_recurrent_step", geometry="GDN-A" if ch == "-" else "GDN-E", B=B))
        # projection GEMMs (bf16, fp32 accumulate)
        for (nm, K, N) in (("d_to_d", d, d), ("d_to_dff", d, dff), ("dff_to_d", dff, d), ("logits_d_to_V", d, V)):
            for M in (1, 8, 4096, 65536, 262144):
                if nm == "logits_d_to_V" and M > 8:
                    continue        # only last-token logits are computed in this study
                def mkg(M=M, K=K, N=N):
                    a_ = torch.randn(M, K, device="cuda", dtype=bf)
                    w = torch.randn(K, N, device="cuda", dtype=bf)
                    o = torch.empty(M, N, device="cuda", dtype=bf)

                    def f():
                        torch.mm(a_, w, out=o)
                    f()
                    return H.graphed(f)
                self._kernel(f"kernel|{scale}|gemm|{nm}|{M}", mkg, 2 * M * K * N, 2 * (M * K + K * N + M * N),
                             dict(scale=scale, kernel="gemm", gemm=nm, M=M, K=K, N=N))

    def phase_flex(self, scale: str, models: dict) -> None:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        cfg = next(iter(models.values())).cfg
        Hh, Dh = cfg.n_attn_heads, cfg.attn_head_dim
        # Session-1 defect (fixed after the run): one compiled object reused across > 8 shapes hit Dynamo's
        # recompile_limit (8) and silently fell back to eager flex_attention (dense scores -> OOM; default
        # block mask built on the CPU inside graph capture).  Now: fresh compile per shape, and the decode
        # block mask is built outside capture.
        def fresh():
            torch._dynamo.reset()
            return torch.compile(flex_attention, dynamic=False)
        for B in PREFILL_BS:
            for T in TS:
                def mk(B=B, T=T):
                    fx = fresh()
                    q = torch.randn(B, Hh, T, Dh, device="cuda", dtype=torch.bfloat16)
                    k_, v_ = torch.randn_like(q), torch.randn_like(q)
                    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, T, T, device="cuda")
                    out = torch.empty_like(q)

                    def f():
                        out.copy_(fx(q, k_, v_, block_mask=bm))
                    f()
                    return H.graphed(f)
                self._kernel(f"kernel|{scale}|attn_prefill|flex|{B}|{T}", mk, 2 * B * Hh * T * T * Dh,
                             4 * B * Hh * T * Dh * 2, dict(scale=scale, kernel="attn_prefill", backend="flex", B=B, T=T))
        for B in (1, 8):
            for L in TS:
                def mkd(B=B, L=L):
                    fx = fresh()
                    q = torch.randn(B, Hh, 1, Dh, device="cuda", dtype=torch.bfloat16)
                    k_ = torch.randn(B, Hh, L, Dh, device="cuda", dtype=torch.bfloat16)
                    v_ = torch.randn_like(k_)
                    out = torch.empty_like(q)
                    bm = create_block_mask(lambda b, h, qi, ki: ki >= 0, None, None, 1, L, device="cuda")  # full, no masking

                    def f():
                        out.copy_(fx(q, k_, v_, block_mask=bm))
                    f()
                    return H.graphed(f)
                self._kernel(f"kernel|{scale}|attn_decode|flex|{B}|{L}", mkd, 4 * B * Hh * L * Dh,
                             2 * B * Hh * L * Dh * 2 + 2 * B * Hh * Dh * 2,
                             dict(scale=scale, kernel="attn_decode", backend="flex", B=B, L=L))


def env_record() -> dict:
    import fla
    import triton
    from hybridlc.utils.provenance import env_info
    e = {}
    try:
        e = env_info()
    except Exception as ex:  # pragma: no cover
        e = {"env_info_error": str(ex)}
    e.update(torch=torch.__version__, fla=fla.__version__, triton=triton.__version__,
             cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0),
             TRITON_CACHE_DIR=os.environ.get("TRITON_CACHE_DIR"), FLA_FLASH_QLA=os.environ.get("FLA_FLASH_QLA"))
    return e


def cmd_run(a) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    S = Session(a)
    S.log(f"session {SESSION} start; out={S.out}; phases={a.phases}; scales={a.scales}; done rows={len(S.done)}")
    mon = Monitor(S.out / "gpu_monitor.jsonl", os.getpid())
    idle = wait_idle(mon, S.log, need_s=a.idle_s) if not a.no_idle_wait else {"skipped": True}
    S.meta(dict(event="start", ts=time.time(), pid=os.getpid(), args=vars(a), idle_pre=idle, env=env_record(),
                bench_seed_index=bench_seed_index(), nvml=S.nv.snap()))
    S.log(f"idle pre-window: {json.dumps(idle)}")
    mon.start()
    try:
        for scale in a.scales:
            S.log(f"=== scale {scale}: building models")
            models = {n: build(scale, n) for n in a.configs}
            tpls = {n: templates_for(m) for n, m in models.items()}
            torch.cuda.synchronize()
            S.log(f"models built; allocated {torch.cuda.memory_allocated() / GiB:.2f} GiB")
            if "vt14" in a.phases:
                S.phase_vt14(scale, models)
            if "check" in a.phases:
                S.phase_check(scale, models)
            if "prefill" in a.phases:
                S.phase_prefill(scale, models)
            if "decode" in a.phases:
                S.phase_decode(scale, models, tpls)
            if "kernel" in a.phases:
                S.phase_kernel(scale, models)
            del models, tpls
            torch.cuda.empty_cache()
        if "flex" in a.phases:          # optional comparator, last so it never displaces a primary phase
            for scale in a.scales:
                if S.over_time():
                    S.log("time budget reached; skipping flex")
                    break
                S.phase_flex(scale, {"C12": build(scale, "C12")})
    finally:
        mon.stop_ev.set()
        S.meta(dict(event="end", ts=time.time(), elapsed_h=(time.time() - S.t_start) / 3600, nvml=S.nv.snap()))
        S.log("session end")
    return 0


def cmd_warm(a) -> int:
    """Separate-process Triton pre-warm: every prefill/decode/kernel shape once (no timing)."""
    a2 = argparse.Namespace(**vars(a))
    a2.out = str(Path(a.out) / "_warm")
    a2.n, a2.warmup = 1, 1
    S = Session(a2)
    for scale in a.scales:
        models = {n: build(scale, n) for n in a.configs}
        tpls = {n: templates_for(m) for n, m in models.items()}
        for n, m in models.items():
            for B in PREFILL_BS:
                for T in TS:
                    if state_bytes(m.cfg, T, batch=B)["total"] > BUDGET:
                        continue
                    try:
                        with torch.no_grad(), ac():
                            m.prefill(bench_ids(B, T, m.cfg.vocab_size, "warm"))
                    except torch.OutOfMemoryError:
                        pass
                    torch.cuda.empty_cache()
            for L in TS:
                for B in (1, 8, 64):
                    try:
                        dec = Decoder(m, L, B, tpls[n], 0)
                        with ac():
                            dec.step()
                    except torch.OutOfMemoryError:
                        pass
                    dec = None
                    torch.cuda.empty_cache()
            S.log(f"warm {scale} {n} done")
    return 0


# ============================================================================ summarise
def load_rows(out: Path) -> list[dict]:
    rows = {}
    for line in (out / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["key"]] = r          # last write wins
    return list(rows.values())


def compliance(out: Path, rows: list[dict]) -> dict:
    """Per-row G-cost tier.  ``contended`` = any monitor poll with foreign SM > 0 in [t0 - 60 s, t1 + 5 s]
    (protocol: no foreign activity >= 60 s before and during).  Kernel rows carry no t0/t1, so their
    window is taken from the driver log (previous log line .. own log line)."""
    recs = [json.loads(x) for x in (out / "gpu_monitor.jsonl").read_text().splitlines()]
    act = [r["ts"] for r in recs if "error" not in r and r.get("foreign_sm_pct", 0) > 0]
    day = time.strftime("%Y-%m-%d", time.localtime(recs[0]["ts"]))
    logt, prev = {}, None
    for line in (out / "run.log").read_text().splitlines():
        if len(line) > 10 and line[0] == "[" and line[9] == "]" and line[3] == ":":
            t = time.mktime(time.strptime(f"{day} {line[1:9]}", "%Y-%m-%d %H:%M:%S"))
            if line[11:].startswith("kernel "):
                logt[line[11:].split(":")[0].split()[1]] = (prev if prev is not None else t, t + 1)
            prev = t
    tier = {}
    for r in rows:
        if r.get("status") != "ok" or "n_kept" not in r:
            continue
        t0, t1 = (r["t0"], r["t1"]) if "t0" in r else logt.get(r["key"], (None, None))
        tier[r["key"]] = None if t0 is None else any(t0 - 60 <= a <= t1 + 5 for a in act)
    return dict(tier=tier, act=act, recs=recs)


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}"


def _tsk(T):
    return f"{T // 1024}K"


def cmd_summarise(a) -> int:
    """Markdown tables, every entry read from rows.jsonl / session.jsonl / gpu_monitor.jsonl."""
    out = Path(a.out)
    rows = load_rows(out)
    P = print
    ok = [r for r in rows if r.get("status") == "ok"]
    scales = [s for s in ("D384", "D512") if any(r.get("scale") == s for r in rows)]
    cfg_k = {n: ModelConfig.for_scale("D512", n).k for n in CONFIGS}

    C = compliance(out, rows)
    TIER = C["tier"]

    def mk(r) -> str:
        return " †" if TIER.get(r["key"]) else ""

    def get(phase, **kw):
        return [r for r in ok if r["phase"] == phase and all(r.get(k) == v for k, v in kw.items())]

    def one(phase, **kw):
        g = get(phase, **kw)
        return g[0] if g else None

    # ---------------------------------------------------------------- compliance
    P("## Session record\n")
    for line in (out / "session.jsonl").read_text().splitlines():
        m = json.loads(line)
        if m["event"] == "start":
            ip = m.get("idle_pre", {})
            P(f"- start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['ts']))}; pid {m['pid']}; torch {m['env'].get('torch')}, "
              f"fla {m['env'].get('fla')}, triton {m['env'].get('triton')}, cuDNN {m['env'].get('cudnn')}; "
              f"TRITON_CACHE_DIR `{m['env'].get('TRITON_CACHE_DIR')}`; bench seed index {m['bench_seed_index']}")
            P(f"- idle pre-window: waited {_fmt(ip.get('waited_s'), 0)} s, clean window {_fmt(ip.get('clean_window_s'), 0)} s over "
              f"{ip.get('polls')} polls, start util {ip.get('start_util')} %, activity_ok={ip.get('activity_ok')}, "
              f"strict_ok={ip.get('strict_ok')}, foreign compute contexts {ip.get('foreign_compute_contexts')}")
        else:
            P(f"- end {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['ts']))}; elapsed {m['elapsed_h']:.2f} h")
    mp = out / "gpu_monitor.jsonl"
    if mp.exists():
        recs = [json.loads(x) for x in mp.read_text().splitlines()]
        good = [r for r in recs if "error" not in r]
        act = [r for r in good if r.get("foreign_sm_pct", 0) > 0]
        fc = sorted({(c["pid"], c["name"]) for r in good for c in r.get("foreign_compute", [])})
        P(f"- during-session monitor: {len(recs)} polls (5 s), {len(recs) - len(good)} errors; polls with foreign SM > 0: "
          f"{len(act)} (max {max([r['foreign_sm_pct'] for r in act], default=0):.0f} %); foreign compute contexts seen: "
          f"{[dict(pid=p, name=n) for p, n in fc]}")
        if act:
            P(f"  - foreign-activity poll times: {[time.strftime('%H:%M:%S', time.localtime(r['ts'])) for r in act][:30]}")
        g = [r["gpu"] for r in good if "gpu" in r]
        for f, lab in (("temperature.gpu", "temp °C"), ("power.draw", "power W"), ("clocks.sm", "SM MHz"), ("clocks.mem", "mem MHz")):
            v = [x.get(f) for x in g if isinstance(x.get(f), (int, float))]
            if v:
                P(f"  - {lab}: min {min(v):.0f}, median {statistics.median(v):.0f}, max {max(v):.0f}")
        thr = [x for x in g if any(str(x.get(k, "")).strip().lower() == "active" for k in gpumon.THROTTLE_FIELDS)]
        P(f"  - polls with a throttle flag active: {len(thr)}")
    timed = [r for r in ok if "n_kept" in r]
    P(f"- timed series: {len(timed)}; replays kept {sum(r['n_kept'] for r in timed)}, discarded "
      f"{sum(r['n_discarded'] for r in timed)}; series with < N kept: {sum(r['n_kept'] < r['n_target'] for r in timed)}")
    rs: dict = {}
    for r in timed:
        for k_, v in r["reasons_seen"].items():
            rs[k_] = rs.get(k_, 0) + v
    P(f"- clocks_event_reasons after replays (bitmask: count): {rs}")
    sm = [x for r in timed for x in (r["nvml_before"].get("sm_mhz"), r["nvml_after"].get("sm_mhz")) if x]
    tc = [x for r in timed for x in (r["nvml_before"].get("temp_c"), r["nvml_after"].get("temp_c")) if x]
    pw = [x for r in timed for x in (r["nvml_after"].get("power_w"),) if x]
    if sm:
        P(f"- NVML around series: SM clock {min(sm)}–{max(sm)} MHz (median {statistics.median(sm):.0f}); temp {min(tc)}–{max(tc)} °C; "
          f"power after series {min(pw):.0f}–{max(pw):.0f} W")
    errs = [r for r in rows if r.get("status") == "error"]
    P(f"- error rows: {len(errs)}" + (f" e.g. {errs[0]['key']}: {errs[0]['error'][:200]}" if errs else ""))
    P("")

    P("## G-cost compliance per timed series\n")
    P("Activity reading: a series is **compliant** iff no 5 s monitor poll in [t0 − 60 s, t1 + 5 s] shows foreign SM > 0; "
      "otherwise **contended** (marked † in every table; appendix-only).\n")
    P("| Phase | Scale | compliant | contended | no window |")
    P("|---|---|---|---|---|")
    grp: dict = {}
    for r in timed:
        g = grp.setdefault((r["phase"], r["scale"]), [0, 0, 0])
        t = TIER.get(r["key"])
        g[0 if t is False else 1 if t else 2] += 1
    for (ph, sc), g in sorted(grp.items()):
        P(f"| {ph} | {sc} | {g[0]} | {g[1]} | {g[2]} |")
    if C["act"]:
        P(f"\nFirst foreign-activity poll: {time.strftime('%H:%M:%S', time.localtime(C['act'][0]))}; "
          f"{len(C['act'])} of {len(C['recs'])} polls show foreign SM > 0.\n")
    errs_ = [r for r in rows if r.get("status") in ("error", "skip")]
    P("## Missing / failed cells\n")
    for r in errs_:
        P(f"- `{r['key']}`: {r['status']}: {(r.get('error') or r.get('reason'))[:140]}")
    P("")

    # ---------------------------------------------------------------- vt14
    P("## Inference-state bytes: measured nbytes vs analytic (§3.4 / VT-14)\n")
    P("| Scale | Config | T | measured total B | analytic total B | equal | dtypes as stored |")
    P("|---|---|---|---|---|---|---|")
    for r in sorted([r for r in rows if r["phase"] == "vt14"], key=lambda r: (r["scale"], CONFIGS.index(r["config"]), r["T"])):
        P(f"| {r['scale']} | {r['config']} | {_tsk(r['T'])} | {r['measured']['total']:,} | {r['analytic']['total']:,} | "
          f"{'yes' if r['equal'] else '**NO**'} | {'yes' if r['dtypes_ok'] else '**NO**'} |")
    dec = [r for r in ok if r["phase"] == "decode"]
    n_eq = sum(r["nbytes_equal"] for r in dec)
    P(f"\nDecode static buffers (state after the step, L + 1 tokens, all B): {n_eq} of {len(dec)} timed decode series have "
      f"buffer nbytes equal to `state_bytes(cfg, L + 1, batch=B)` to the byte.\n")

    # ---------------------------------------------------------------- checks
    P("## Implementation checks\n")
    P("| Scale | Config | check | result |")
    P("|---|---|---|---|")
    for r in rows:
        if r["phase"] in ("check_weights", "check"):
            P(f"| {r['scale']} | {r['config']} | {r['phase']} | `{json.dumps(r['result'])}` |")
    P("")

    # ---------------------------------------------------------------- prefill
    def prefill_table(scale, B, mode, be, l2, stat="min_us"):
        P(f"| Config | k | " + " | ".join(_tsk(T) for T in TS_ALL) + " |")
        P("|---|---|" + "---|" * len(TS_ALL))
        for n in CONFIGS:
            cells = []
            for T in TS_ALL:
                b = be if n not in ("C0", "X-E0") else "none"
                r = one("prefill", scale=scale, config=n, B=B, T=T, mode=mode, backend=b, l2=l2)
                if r is None:
                    sk = [x for x in rows if x["key"] == f"prefill|{scale}|{n}|{B}|{T}|skip"]
                    cells.append("skip" if sk else "–")
                else:
                    cells.append((_fmt(r[stat] / 1e3, 2) if stat != "iqr" else f"{_fmt(r['median_us'] / 1e3, 2)} ({_fmt((r['iqr_us'] or 0) / 1e3, 3)})") + mk(r))
            P(f"| {n} | {cfg_k[n]} | " + " | ".join(cells) + " |")
        P("")

    for scale in scales:
        P(f"## Prefill latency, {scale} (ms; whole model, last-token logits; min-of-N unless noted)\n")
        for B in PREFILL_BS_ALL:
            P(f"### {scale}, B = {B}: CUDA graph, FLASH, cold L2 (primary)\n")
            prefill_table(scale, B, "graph", "flash", "cold")
            P(f"{scale}, B = {B}: graph, FLASH, cold L2: median (IQR)\n")
            prefill_table(scale, B, "graph", "flash", "cold", stat="iqr")
            P(f"{scale}, B = {B}: graph, FLASH, hot L2\n")
            prefill_table(scale, B, "graph", "flash", "hot")
            P(f"{scale}, B = {B}: graph, cuDNN, cold L2\n")
            prefill_table(scale, B, "graph", "cudnn", "cold")
            P(f"{scale}, B = {B}: eager, FLASH, cold L2\n")
            prefill_table(scale, B, "eager", "flash", "cold")
        skips = [r for r in rows if r["phase"] == "prefill" and r.get("status") == "skip" and r["scale"] == scale]
        if skips:
            P(f"Skipped prefill shapes at {scale} (budget {BUDGET / GiB:.0f} GiB state + activation, excluding weights):\n")
            for r in skips:
                P(f"- {r['config']} B={r['B']} T={_tsk(r['T'])}: {r['reason']}")
            P("")
        P(f"### {scale}: prefill achieved throughput (graph, FLASH, cold; tokens/s and model TFLOP/s)\n")
        P("Model FLOPs = B·T·(fwd FLOP/token at T − 2dV) + B·2dV (§3.4 formula; last-token logits only).\n")
        P("| Config | B | T | min ms | Mtok/s | TFLOP/s | fraction of 112.6 |")
        P("|---|---|---|---|---|---|---|")
        for n in CONFIGS:
            cfg = ModelConfig.for_scale(scale, n)
            for B in PREFILL_BS_ALL:
                for T in TS_ALL:
                    r = one("prefill", scale=scale, config=n, B=B, T=T, mode="graph",
                            backend="flash" if has_attn(cfg) else "none", l2="cold")
                    if not r:
                        continue
                    fl = B * T * (flops_per_token(cfg, T) - 2 * cfg.d_model * cfg.vocab_size) + B * 2 * cfg.d_model * cfg.vocab_size
                    t = r["min_us"] * 1e-6
                    P(f"| {n} | {B} | {_tsk(T)} | {_fmt(t * 1e3)}{mk(r)} | {_fmt(B * T / t / 1e6)} | {_fmt(fl / t / 1e12)} | {_fmt(fl / t / PEAK_FLOPS, 3)} |")
        P("")

    # ---------------------------------------------------------------- decode
    for scale in scales:
        P(f"## Decode, {scale} (one-step exact-shape graph, no masks)\n")
        for bclass in ("1", "8", "max"):
            for be, l2 in (("cudnn", "cold"), ("cudnn", "hot"), ("flash", "cold")):
                P(f"### {scale}, B = {bclass if bclass != 'max' else 'B_max(L)'}: {be}, {l2} L2 — per-step latency µs (min-of-N)"
                  + (" [primary]" if (be, l2) == ("cudnn", "cold") else "") + "\n")
                P("| Config | k | " + " | ".join(f"L={_tsk(L)}" for L in TS_ALL) + " |")
                P("|---|---|" + "---|" * len(TS_ALL))
                for n in CONFIGS:
                    cells = []
                    for L in TS_ALL:
                        b = be if n not in ("C0", "X-E0") else "none"
                        r = one("decode", scale=scale, config=n, L=L, bclass=bclass, backend=b, l2=l2)
                        if r is None:
                            cells.append("–")
                        elif bclass == "max":
                            cells.append(f"{_fmt(r['min_us'], 1)} (B={r['B']})" + mk(r))
                        else:
                            cells.append(_fmt(r["min_us"], 1) + mk(r))
                    P(f"| {n} | {cfg_k[n]} | " + " | ".join(cells) + " |")
                P("")
        P(f"### {scale}: decode throughput, tokens/s = B / min latency (cuDNN, cold L2)\n")
        P("| Config | k | B | " + " | ".join(f"L={_tsk(L)}" for L in TS_ALL) + " |")
        P("|---|---|---|" + "---|" * len(TS_ALL))
        for n in CONFIGS:
            for bclass in ("1", "8", "max"):
                cells = []
                for L in TS_ALL:
                    b = "cudnn" if n not in ("C0", "X-E0") else "none"
                    r = one("decode", scale=scale, config=n, L=L, bclass=bclass, backend=b, l2="cold")
                    cells.append("–" if r is None else f"{r['B'] / (r['min_us'] * 1e-6):,.0f}" + mk(r))
                P(f"| {n} | {cfg_k[n]} | {bclass if bclass != 'max' else 'B_max'} | " + " | ".join(cells) + " |")
        P("")
        P(f"### {scale}: decode median (IQR) µs, cuDNN cold, B = 1\n")
        P("| Config | " + " | ".join(f"L={_tsk(L)}" for L in TS_ALL) + " |")
        P("|---|" + "---|" * len(TS_ALL))
        for n in CONFIGS:
            cells = []
            for L in TS_ALL:
                b = "cudnn" if n not in ("C0", "X-E0") else "none"
                r = one("decode", scale=scale, config=n, L=L, bclass="1", backend=b, l2="cold")
                cells.append("–" if r is None else f"{_fmt(r['median_us'], 1)} ({_fmt(r['iqr_us'], 1)})" + mk(r))
            P(f"| {n} | " + " | ".join(cells) + " |")
        P("")
        P(f"### {scale}: whole-step decode roofline fraction (cuDNN, cold L2)\n")
        P("Bytes per step = 2·N_total (bf16 weights) + B·[k·(L+1)·4d (KV read) + 2·S_lin (state read+write)] (§10.1 generalised to B); "
          "roofline time = bytes / 960 GB/s; fraction = roofline / measured min.\n")
        P("| Config | B class | " + " | ".join(f"L={_tsk(L)}" for L in TS_ALL) + " |")
        P("|---|---|" + "---|" * len(TS_ALL))
        for n in CONFIGS:
            cfg = ModelConfig.for_scale(scale, n)
            Nt = param_counts(cfg)["total"]
            for bclass in ("1", "8", "max"):
                cells = []
                for L in TS_ALL:
                    b = "cudnn" if has_attn(cfg) else "none"
                    r = one("decode", scale=scale, config=n, L=L, bclass=bclass, backend=b, l2="cold")
                    if r is None:
                        cells.append("–")
                        continue
                    sb = state_bytes(cfg, L + 1)
                    by = 2 * Nt + r["B"] * (sb["kv"] + 2 * (sb["gdn_state"] + sb["conv"]))
                    cells.append(_fmt(by / PEAK_BW / (r["min_us"] * 1e-6), 3) + mk(r))
                P(f"| {n} | {bclass} | " + " | ".join(cells) + " |")
        P("")

    # ---------------------------------------------------------------- bmax
    P("## B_max(L): 8 GiB state + activation budget\n")
    P("Analytic: floor(8 GiB / (state_bytes(L+1) + 4·V fp32 logits row)). Verified: static buffers allocated and the one-step graph "
      "captured and replayed; `used` = peak allocated minus the pre-allocation baseline (resident weights excluded); B shrunk "
      "proportionally until used ≤ 8 GiB.\n")
    P("| Scale | Config | " + " | ".join(f"L={_tsk(L)} analytic / verified (used GiB)" for L in TS_ALL) + " |")
    P("|---|---|" + "---|" * len(TS_ALL))
    for scale in scales:
        for n in CONFIGS:
            cells = []
            for L in TS_ALL:
                r = [x for x in rows if x["key"] == f"bmax|{scale}|{n}|{L}"]
                if not r:
                    cells.append("–")
                    continue
                r = r[0]
                used = [t for t in r["tries"] if t.get("ok")]
                cells.append(f"{r['B_max_analytic']:,} / {r['B_max_verified']:,} ({used[-1]['used_bytes'] / GiB:.2f})"
                             if r["B_max_verified"] else f"{r['B_max_analytic']:,} / failed")
            P(f"| {scale} | {n} | " + " | ".join(cells) + " |")
    P("")

    # ---------------------------------------------------------------- kernels
    P("## Kernel roofline fractions (CUDA graph, cold L2, min-of-N)\n")
    P("Roofline time = max(FLOPs / 112.6 TFLOP/s, compulsory bytes / 960 GB/s); fraction = roofline / measured.\n")
    for scale in scales:
        P(f"### {scale}\n")
        P("| Kernel | variant | shape | min µs | TFLOP/s | GB/s | roofline frac |")
        P("|---|---|---|---|---|---|---|")
        for r in [x for x in ok if x["phase"] == "kernel" and x["scale"] == scale]:
            var = r.get("backend") or r.get("geometry") or r.get("gemm")
            shape = ", ".join(f"{k}={r[k]}" for k in ("B", "T", "L", "M", "K", "N") if k in r)
            P(f"| {r['kernel']} | {var} | {shape} | {_fmt(r['min_us'], 1)}{mk(r)} | {_fmt(r['achieved_tflops'], 2)} | "
              f"{_fmt(r['achieved_gbs'], 0)} | {_fmt(r['roofline_fraction'], 3)} |")
        P("")
    return 0


TS_ALL = [1024, 4096, 16384, 32768]
PREFILL_BS_ALL = [1, 8]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "warm", "summarise"])
    ap.add_argument("--out", default=str(REPO / "experiments" / "bench" / "cost_session1"))
    ap.add_argument("--scales", nargs="+", default=["D384", "D512"])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--phases", nargs="+", default=["vt14", "check", "prefill", "decode", "kernel", "flex"])
    ap.add_argument("--prefill-modes", nargs="+", default=["graph", "eager"])
    ap.add_argument("--prefill-backends", nargs="+", default=["flash", "cudnn"])
    ap.add_argument("--eager-cudnn", action="store_true", help="also time eager prefill under cuDNN")
    ap.add_argument("--decode-backends", nargs="+", default=["cudnn", "flash"])
    ap.add_argument("--n", type=int, default=H.N_TIMED)
    ap.add_argument("--warmup", type=int, default=H.N_WARMUP)
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--no-idle-wait", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="re-run keys whose latest row is an error")
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--ts", nargs="+", type=int, default=None, help="restrict T / L (smoke tests)")
    ap.add_argument("--prefill-bs", nargs="+", type=int, default=None, help="restrict prefill B (smoke tests)")
    a = ap.parse_args(argv)
    global TS, PREFILL_BS
    if a.ts:
        TS = a.ts
    if a.prefill_bs:
        PREFILL_BS = a.prefill_bs
    return {"run": cmd_run, "warm": cmd_warm, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
