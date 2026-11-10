# Measured-cost benchmarks: session 1 of 2 (protocol §10.2)

**Scope.** Secondary costs (§10.2), scoped to "fla Triton + PyTorch SDPA on sm_120, this implementation". RTX 5080, torch 2.11.0+cu128, fla 0.6.0, Triton 3.6.0, cuDNN 9.19.
- **Configs:** C0, C1, C2, C3, C4, C6, C12, X-gc12 and X-E0.
- **Scales:** both, D384 first. The final scale is not chosen yet.
- **Weights:** random-init from the `bench` stream (seed index 3656060305 = `entropy_sequence(BENCH_ENTROPY, crc32("cost_benchmark_weights"))`).
- **When:** 2026-11-10, 13:41:01–14:51:58 (1.20 GPU-h of measurement). Triton pre-warm ran as a separate process from 13:38.
- **This is session 1 only.** §10.2 requires 2 sessions on different days. Session 2 is still owed, and no between-session variability is available.

**Raw data:** `experiments/bench/cost_session1/`. Every table below is generated from these files by `cmd_summarise` in `scripts/benchmark_inference.py`. No number was typed by hand.
- `rows.jsonl`: one row per measurement or skip, with all kept replay times and NVML snapshots.
- `session.jsonl`: start/end records, the idle pre-window and the environment.
- `gpu_monitor.jsonl`: 5 s G-cost polls.
- `run.log` / `driver.log`.

**Regenerating the tables.** The script now carries the R2-LIVE lock guard, which another workflow added after the run. Running it as `__main__` takes the GPU lock, so import it instead:
`CUDA_VISIBLE_DEVICES= python -c "import importlib.util as u,argparse;s=u.spec_from_file_location('b','scripts/benchmark_inference.py');m=u.module_from_spec(s);s.loader.exec_module(m);m.cmd_summarise(argparse.Namespace(out='experiments/bench/cost_session1'))"`.

## Verdicts at a glance

| Item | Result |
|---|---|
| VT-14-style state bytes | **72/72** prefill caches (B = 1, T ∈ {1K, 4K, 16K, 32K}, 9 configs × 2 scales) equal `analytic.state_bytes` to the byte, with dtypes as stored (bf16 KV, fp32 GDN/conv). **768/768** decode static-buffer sets equal `state_bytes(L+1, batch=B)` to the byte |
| Hoisted bf16 weights vs the production fp32-weights + autocast path | Prefill last-token logits **bit-identical** for C0, C3, C12, X-gc12 and X-E0 at both scales |
| Benchmark decoder vs `HybridLM.decode_step` | FLASH: max \|Δlogit\| **0.0** (bit-identical). cuDNN: ≤ 0.0256 (kernel numerics). Argmax agreement 1.000 everywhere |
| Backend / mask assertion | Every captured prefill/decode graph passed the `sdpa_spy` assertion: only the pinned backend enabled, no mask, mem-efficient never enabled |
| Replay screening | 1,678 timed series; 50,340 replays kept and **0 discarded**. NVML clocks-event reasons were 0 after every replay; no power/thermal violation-counter increase |
| Clocks / thermals | SM clock 2,700–2,797 MHz around series (median 2,767). Temperature 41–69 °C. No throttle flag in any monitor poll. Clocks unlocked (`-lgc` needs root) |
| G-cost, activity reading | Clean 61.6 s pre-window, start utilisation 0 %. Foreign SM activity first appears at **14:33:49** (58 of 835 polls; gnome-shell / Xorg / a graphics process "exe", ≤ 42 % SM). **All D384 primary series are compliant**, and so are 334 of 372 D512 prefill series. **All D512 decode and D512 kernel series**, 38 D512 prefill series (B = 8 at 16K/32K) and the 8 D384 Flex rows are contended (marked †) |
| G-cost, strict reading | **Not met by any row.** Two foreign compute contexts were resident for the whole session at 0 % SM: the CARLA client `carla_env/bin/python` (pid 124395, 454 MiB) and `gnome-remote-desktop-daemon` (pid 293356, 305 MiB). They were not killed (user processes). The CARLA UE4 servers now running (pids 2083373 and 2108186, about 11 GB, high utilisation) started at **15:16:27 and 15:18:17, after the session ended**, and appear in no session poll |

**Compliance consequence (per the coordinator's instruction and U-14).** Rows marked † are **non-compliant and appendix-only**. Under the strict "no foreign compute process" reading, the whole session is appendix-grade unless the PI accepts the activity reading (idle resident contexts at 0 % SM). **This is a PI decision.** Until then, only the analytic costs (§10.1) are primary for RQ-D/H6.

## Key numbers (D384, activity-compliant, min-of-N, CUDA graph, cold L2)

**Prefill, B = 1, T = 32K (FLASH):**

| C0 | C3 | C6 | C12 | X-gc12 | X-E0 |
|---|---|---|---|---|---|
| 51.95 ms | 77.37 ms | 102.97 ms | 154.31 ms | 158.74 ms | 63.29 ms |

At 1K the configs are within 2.28–2.72 ms of one another. At B = 8, T = 32K, C0 takes 408.88 ms and C6 831.08 ms. C12/X-gc12 at that shape are missing (driver defect, below).

**Decode, B = 1 (cuDNN):**
- C0 is flat with L: 870.4–872.5 µs from 1K to 32K.
- C3 rises from 950.3 to 1,126.4 µs and C12 from 1,173.5 to 1,835.0 µs.
- At B = 1 the step is launch/latency-bound: the whole-step roofline fraction is about 0.07 for C0, and 0.37 for C12 at 32K.

**Decode, B = 8, L = 32K:** C0 1,089.5 µs, C3 2,514.9 µs, C12 6,680.6 µs.

**B_max(L), D384, verified:**

| L | C0 | C3 | C12 |
|---|---|---|---|
| 1K | 5,185 | 1,397 | 442 |
| 32K | 5,185 | 56 | 14 |

**Throughput at B_max (cuDNN):**

| L | C0 | C3 | C12 |
|---|---|---|---|
| 1K | 92,634 tok/s | 70,181 tok/s | 39,948 tok/s |
| 32K | 92,590 tok/s | 5,226 tok/s | 1,314 tok/s |

**Kernel roofline fractions (D384, compliant):**

| Kernel | Roofline fraction |
|---|---|
| FLASH causal prefill attention | 0.28 (B1, 1K) → 0.97 (B8, 32K) |
| cuDNN decode attention | 0.21 (B1, 1K) → 0.89 (B8, 32K) |
| GDN-A chunk core | 0.056–0.110 at B = 1; 0.20–0.26 at B = 8 |
| GDN-A fused_recurrent step | 0.05 (B1) / 0.20 (B8) / 0.61 (B256) |
| GEMMs at M ≥ 64K | 0.68–0.95 |

These figures confirm the §10.2 "unequal kernel maturity" hazard.

## Defects found (none in `src/hybridlc/bench/harness.py`)

No harness defect was found, and the harness was not modified. The driver uses `harness.graphed`, `harness.sdpa_spy`, `harness.decode_attention_case`, and the harness's 256 MiB flush buffer and N/warm-up constants. Its own replay loop mirrors `harness.time_replays`: events, synchronise, flush before the start event, min/median/IQR. It adds per-replay NVML screening and replacement, which the harness lacks. That is a feature gap, not a defect.

Two defects were in **the driver** (`scripts/benchmark_inference.py`). Both are fixed in code and **not re-measured**, because the GPU is no longer idle.
1. **Prefill output retained across calls (fixed during the run, after D384 prefill).** The prefill closure kept the previous call's output (the whole KV cache) alive while the next call allocated, so graph warm-up/eager replays needed about 2× cache memory.
   - **Errors:** 12 D384 rows (C12 and X-gc12 at B = 8, 32K) and 16 D512 rows (X-gc12 B = 8 16K graph-FLASH; C3/C4 B = 8 32K graph; C6 B = 8 32K all variants).
   - **Retry:** `--retry-errors` reruns exactly these keys in the next idle window.
   - **Timings unaffected:** the eager-probe peak used for the budget skip was measured with no output retained, and the extra live memory did not change any kernel's work.
2. **Flex comparator (known error class; classified as a driver defect, out of harness scope).** One `torch.compile(flex_attention, dynamic=False)` object was reused across 32 shapes, and Dynamo's `recompile_limit` is 8. The first 8 compiles (D384 prefill, B ∈ {1, 8} × T) are valid; their rows are present. Every later shape silently fell back to **eager** `flex_attention`:
   - prefill materialised dense scores, giving OOMs of exactly B·H·T²·4 bytes (8, 32, 64 and 256 GiB requests);
   - decode with `block_mask=None` built the default mask on the CPU inside graph capture, giving "Cannot copy between CPU and CUDA tensors during CUDA graph capture".

   This affects all 8 D384 Flex decode cells and all 16 D512 Flex cells. The harness's own VT-15 Flex test (single shape, mask built outside capture) is not implicated.
   - **Fixed in code:** `torch._dynamo.reset()` plus a fresh compile per shape, and the decode block mask is built outside capture.
   - **Not re-measured.** The surviving D384 Flex prefill rows are contended (†) because they ran at the end.

## Missing cells versus the plan

| Planned | Status |
|---|---|
| Prefill: 2 scales × 9 configs × 8 shapes | Present except the table's "–" cells. Driver defect 1: D384 C12/X-gc12 B8·32K; D512 C3/C4/C6 B8·32K graph (C6 also eager); X-gc12 B8·16K graph-FLASH. **Skip (budget):** D512 C12 and X-gc12 at B8·32K (OOM in the eager probe, i.e. peak above the free ~13 GiB and hence above the 8 GiB budget) |
| Decode: 2 × 9 × 4 L × {1, 8, B_max} × {cuDNN, FLASH (attention configs)} × {cold, hot} | **Complete:** 768 series |
| Kernel table | Complete, apart from the Flex cells above |
| Eager cuDNN prefill | Not planned (time budget); eager FLASH only |

## Caveats

1. **Session 1 of 2.** Session 2 must run on a different day, ideally after the PI rules on the strict vs activity G-cost reading. It should include the `--retry-errors` cells and the re-measured Flex comparator.
2. **D512 decode and kernel numbers are contended (†)** by intermittent desktop graphics load (up to 42 % SM in single polls), and so are the D512 B = 8 16K/32K prefill rows. They are appendix-only. At D512 C6 B8·16K (491.46 ms †) the k-trend is visibly broken (C4 401.60 → C12 615.55), so contention inflation is real there. Do not compare D512 † rows with D384 rows.
3. **Weights.** `nn.Linear` weights are stored in bf16 and the tied embedding has a bf16 copy for the logits GEMM. This hoists the casts autocast would perform on every call; the logits are bit-identical to the production path. Norms, `A_log`, `dt_bias`, conv kernels and the embedding lookup stay fp32. The production eval path (fp32 weights under autocast, recasting every call) would add k-independent cast traffic. That path was not timed.
4. **The decoder is a benchmark implementation.** Attention KV lives in static `[B, L+1, H, 64]` buffers (production layout), and each step writes slot L in place. `HybridLM.decode_step` instead `torch.cat`s the whole KV every step, which would add an O(L) copy. That path was not timed. KV/GDN states are random-filled, which is valid because there are no masks and the kernels do content-independent work. The step is validated against `decode_step` (bit-identical under FLASH).
5. **Memory budgets** (my definitions, not specified in the protocol beyond "8 GiB state + activation"):
   - **Prefill skip:** analytic state > 8 GiB, or measured eager peak minus the pre-call baseline > 8 GiB, or OOM.
   - **B_max:** analytic = ⌊8 GiB / (state(L+1) + 4V)⌋. Verified = allocation plus graph capture and replay, with peak minus baseline ≤ 8 GiB.
   - **Results:** verified B_max is 84–100 % of analytic. The gap is largest for the attention-free C0/X-E0 (activations dominate).
   - **Residency:** all 9 models of a scale are resident (≈ 0.3 GiB at D384), excluded from the budget.
6. **Cold L2** = harness 256 MiB zero before every replay (untimed). For prefill above ≈ 4K, cold and hot are indistinguishable, as expected. For whole-step decode at D384 B = 1, cold exceeds hot by 1.6 % (C12, 32K: 1,835.0 vs 1,806.2 µs) to 7.4 % (C0: 870.4 vs 810.5 µs).
7. **Replay screening is approximate.** NVML reasons are sampled right after each replay's synchronise, and violation counters are differenced around the replay. Zero discards means no detected event, not proof of none.
8. **Roofline model.**
   - **Peaks:** whitepaper values (112.6 TFLOP/s, 960 GB/s), not measured.
   - **Bytes:** compulsory only.
   - **GDN recurrent step:** the timed region includes the static-buffer `copy_` (L2-resident up to ≈ 64 MiB) but bytes count 2 × state.
   - **Whole-step decode:** uses §10.1 generalised to B.
   - **Prefill model FLOPs:** use the nominal §3.4 formula (causal average; GDN core formula), minus the full-logit term.
9. **Eager prefill is host/launch-bound at small shapes.** D384 B = 1 at 1K–4K: eager 5.25–13.27 ms against graph 2.28–8.06 ms. The overhead grows with the number of GDN layers (C0 13.15 ms, C12 5.25 ms at 1K). Graph numbers are primary.
10. **Triton cache.** A separate cache (`.cache/triton_bench`) was seeded from the archived tarball and warmed in a separate process. The archived production cache (`.cache/triton`, `runs/_artifacts/triton_cache.tar`) was not touched.
11. **FlexAttention** is attention-core only ("equal toolchain, not equal maturity"). It is not a whole-model comparator.

---

# Generated tables
## Session record

- start 2026-11-10 13:41:01; pid 2013291; torch 2.11.0+cu128, fla 0.6.0, triton 3.6.0, cuDNN 91900; TRITON_CACHE_DIR `/home/sedrica-lynx/Downloads/outeropt/.cache/triton_bench`; bench seed index 3656060305
- idle pre-window: waited 62 s, clean window 62 s over 12 polls, start util 0.0 %, activity_ok=True, strict_ok=False, foreign compute contexts [{'pid': 124395, 'name': '/home/sedrica-lynx/CARLA_0.9.15/carla_env/bin/python'}, {'pid': 293356, 'name': '/usr/libexec/gnome-remote-desktop-daemon'}]
- end 2026-11-10 14:51:58; elapsed 1.20 h
- during-session monitor: 835 polls (5 s), 0 errors; polls with foreign SM > 0: 58 (max 42 %); foreign compute contexts seen: [{'pid': 124395, 'name': '/home/sedrica-lynx/CARLA_0.9.15/carla_env/bin/python'}, {'pid': 293356, 'name': '/usr/libexec/gnome-remote-desktop-daemon'}]
  - foreign-activity poll times: ['14:33:49', '14:33:54', '14:34:04', '14:34:20', '14:34:40', '14:35:00', '14:35:06', '14:35:26', '14:35:31', '14:35:36', '14:36:12', '14:36:17', '14:36:22', '14:36:27', '14:36:48', '14:36:53', '14:37:18', '14:37:23', '14:37:28', '14:37:33', '14:38:04', '14:38:40', '14:39:16', '14:39:21', '14:39:46', '14:39:51', '14:40:22', '14:40:32', '14:40:42', '14:40:47']
  - temp °C: min 38, median 63, max 70
  - power W: min 15, median 242, max 261
  - SM MHz: min 195, median 2760, max 2797
  - mem MHz: min 405, median 14801, max 14801
  - polls with a throttle flag active: 0
- timed series: 1678; replays kept 50340, discarded 0; series with < N kept: 0
- clocks_event_reasons after replays (bitmask: count): {'0': 50340}
- NVML around series: SM clock 2700–2797 MHz (median 2767); temp 41–69 °C; power after series 60–263 W
- error rows: 52 e.g. prefill|D384|C12|8|32768|graph|flash|cold: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 326.81 MiB is free. Process 124395 has 454.00 MiB memory in use. Process 293356 has

## G-cost compliance per timed series

Activity reading: a series is **compliant** iff no 5 s monitor poll in [t0 − 60 s, t1 + 5 s] shows foreign SM > 0; otherwise **contended** (marked † in every table; appendix-only).

| Phase | Scale | compliant | contended | no window |
|---|---|---|---|---|
| decode | D384 | 384 | 0 | 0 |
| decode | D512 | 0 | 384 | 0 |
| kernel | D384 | 71 | 8 | 0 |
| kernel | D512 | 0 | 71 | 0 |
| prefill | D384 | 388 | 0 | 0 |
| prefill | D512 | 334 | 38 | 0 |

First foreign-activity poll: 14:33:49; 58 of 835 polls show foreign SM > 0.

## Missing / failed cells

- `prefill|D384|C12|8|32768|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 326.81 MiB is free. Pro
- `prefill|D384|C12|8|32768|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 326.81 MiB is free. Pro
- `prefill|D384|C12|8|32768|eager|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 584.81 MiB is free. Pro
- `prefill|D384|C12|8|32768|eager|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 584.81 MiB is free. Pro
- `prefill|D384|C12|8|32768|graph|cudnn|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 582.81 MiB is free. Pro
- `prefill|D384|C12|8|32768|graph|cudnn|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 582.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 708.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 708.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|eager|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 328.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|eager|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 328.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|graph|cudnn|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 324.81 MiB is free. Pro
- `prefill|D384|X-gc12|8|32768|graph|cudnn|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 324.81 MiB is free. Pro
- `prefill|D512|X-gc12|8|16384|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 322.81 MiB is free. Pro
- `prefill|D512|X-gc12|8|16384|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 322.81 MiB is free. Pro
- `prefill|D512|C12|8|32768|skip`: skip: OOM in eager probe
- `prefill|D512|X-gc12|8|32768|skip`: skip: OOM in eager probe
- `prefill|D512|C3|8|32768|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 677.94 MiB is free. Pro
- `prefill|D512|C3|8|32768|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 677.94 MiB is free. Pro
- `prefill|D512|C3|8|32768|graph|cudnn|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 946.75 MiB is free. Pro
- `prefill|D512|C3|8|32768|graph|cudnn|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 946.75 MiB is free. Pro
- `prefill|D512|C4|8|32768|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 697.44 MiB is free. Pro
- `prefill|D512|C4|8|32768|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 697.44 MiB is free. Pro
- `prefill|D512|C4|8|32768|graph|cudnn|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 862.75 MiB is free. Pro
- `prefill|D512|C4|8|32768|graph|cudnn|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 862.75 MiB is free. Pro
- `prefill|D512|C6|8|32768|graph|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 383.00 MiB is free. Pro
- `prefill|D512|C6|8|32768|graph|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 383.00 MiB is free. Pro
- `prefill|D512|C6|8|32768|eager|flash|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 904.38 MiB is free. Pro
- `prefill|D512|C6|8|32768|eager|flash|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 904.38 MiB is free. Pro
- `prefill|D512|C6|8|32768|graph|cudnn|cold`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 916.44 MiB is free. Pro
- `prefill|D512|C6|8|32768|graph|cudnn|hot`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 916.44 MiB is free. Pro
- `kernel|D384|attn_decode|flex|1|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|1|4096`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|1|16384`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|1|32768`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|8|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|8|4096`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|8|16384`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D384|attn_decode|flex|8|32768`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_prefill|flex|1|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_prefill|flex|1|4096`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_prefill|flex|1|16384`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 15.46 GiB of which 4.20 GiB is free. Process
- `kernel|D512|attn_prefill|flex|1|32768`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 15.46 GiB of which 11.40 GiB is free. Proce
- `kernel|D512|attn_prefill|flex|8|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_prefill|flex|8|4096`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of 15.46 GiB of which 280.38 MiB is free. Pr
- `kernel|D512|attn_prefill|flex|8|16384`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB. GPU 0 has a total capacity of 15.46 GiB of which 9.62 GiB is free. Proces
- `kernel|D512|attn_prefill|flex|8|32768`: error: OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 GiB. GPU 0 has a total capacity of 15.46 GiB of which 9.74 GiB is free. Proce
- `kernel|D512|attn_decode|flex|1|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|1|4096`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|1|16384`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|1|32768`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|8|1024`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|8|4096`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|8|16384`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo
- `kernel|D512|attn_decode|flex|8|32768`: error: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memo

## Inference-state bytes: measured nbytes vs analytic (§3.4 / VT-14)

| Scale | Config | T | measured total B | analytic total B | equal | dtypes as stored |
|---|---|---|---|---|---|---|
| D384 | C0 | 1K | 1,327,104 | 1,327,104 | yes | yes |
| D384 | C0 | 4K | 1,327,104 | 1,327,104 | yes | yes |
| D384 | C0 | 16K | 1,327,104 | 1,327,104 | yes | yes |
| D384 | C0 | 32K | 1,327,104 | 1,327,104 | yes | yes |
| D384 | C1 | 1K | 2,789,376 | 2,789,376 | yes | yes |
| D384 | C1 | 4K | 7,507,968 | 7,507,968 | yes | yes |
| D384 | C1 | 16K | 26,382,336 | 26,382,336 | yes | yes |
| D384 | C1 | 32K | 51,548,160 | 51,548,160 | yes | yes |
| D384 | C2 | 1K | 4,251,648 | 4,251,648 | yes | yes |
| D384 | C2 | 4K | 13,688,832 | 13,688,832 | yes | yes |
| D384 | C2 | 16K | 51,437,568 | 51,437,568 | yes | yes |
| D384 | C2 | 32K | 101,769,216 | 101,769,216 | yes | yes |
| D384 | C3 | 1K | 5,713,920 | 5,713,920 | yes | yes |
| D384 | C3 | 4K | 19,869,696 | 19,869,696 | yes | yes |
| D384 | C3 | 16K | 76,492,800 | 76,492,800 | yes | yes |
| D384 | C3 | 32K | 151,990,272 | 151,990,272 | yes | yes |
| D384 | C4 | 1K | 7,176,192 | 7,176,192 | yes | yes |
| D384 | C4 | 4K | 26,050,560 | 26,050,560 | yes | yes |
| D384 | C4 | 16K | 101,548,032 | 101,548,032 | yes | yes |
| D384 | C4 | 32K | 202,211,328 | 202,211,328 | yes | yes |
| D384 | C6 | 1K | 10,100,736 | 10,100,736 | yes | yes |
| D384 | C6 | 4K | 38,412,288 | 38,412,288 | yes | yes |
| D384 | C6 | 16K | 151,658,496 | 151,658,496 | yes | yes |
| D384 | C6 | 32K | 302,653,440 | 302,653,440 | yes | yes |
| D384 | C12 | 1K | 18,874,368 | 18,874,368 | yes | yes |
| D384 | C12 | 4K | 75,497,472 | 75,497,472 | yes | yes |
| D384 | C12 | 16K | 301,989,888 | 301,989,888 | yes | yes |
| D384 | C12 | 32K | 603,979,776 | 603,979,776 | yes | yes |
| D384 | X-gc12 | 1K | 19,095,552 | 19,095,552 | yes | yes |
| D384 | X-gc12 | 4K | 75,718,656 | 75,718,656 | yes | yes |
| D384 | X-gc12 | 16K | 302,211,072 | 302,211,072 | yes | yes |
| D384 | X-gc12 | 32K | 604,200,960 | 604,200,960 | yes | yes |
| D384 | X-E0 | 1K | 3,686,400 | 3,686,400 | yes | yes |
| D384 | X-E0 | 4K | 3,686,400 | 3,686,400 | yes | yes |
| D384 | X-E0 | 16K | 3,686,400 | 3,686,400 | yes | yes |
| D384 | X-E0 | 32K | 3,686,400 | 3,686,400 | yes | yes |
| D512 | C0 | 1K | 1,769,472 | 1,769,472 | yes | yes |
| D512 | C0 | 4K | 1,769,472 | 1,769,472 | yes | yes |
| D512 | C0 | 16K | 1,769,472 | 1,769,472 | yes | yes |
| D512 | C0 | 32K | 1,769,472 | 1,769,472 | yes | yes |
| D512 | C1 | 1K | 3,719,168 | 3,719,168 | yes | yes |
| D512 | C1 | 4K | 10,010,624 | 10,010,624 | yes | yes |
| D512 | C1 | 16K | 35,176,448 | 35,176,448 | yes | yes |
| D512 | C1 | 32K | 68,730,880 | 68,730,880 | yes | yes |
| D512 | C2 | 1K | 5,668,864 | 5,668,864 | yes | yes |
| D512 | C2 | 4K | 18,251,776 | 18,251,776 | yes | yes |
| D512 | C2 | 16K | 68,583,424 | 68,583,424 | yes | yes |
| D512 | C2 | 32K | 135,692,288 | 135,692,288 | yes | yes |
| D512 | C3 | 1K | 7,618,560 | 7,618,560 | yes | yes |
| D512 | C3 | 4K | 26,492,928 | 26,492,928 | yes | yes |
| D512 | C3 | 16K | 101,990,400 | 101,990,400 | yes | yes |
| D512 | C3 | 32K | 202,653,696 | 202,653,696 | yes | yes |
| D512 | C4 | 1K | 9,568,256 | 9,568,256 | yes | yes |
| D512 | C4 | 4K | 34,734,080 | 34,734,080 | yes | yes |
| D512 | C4 | 16K | 135,397,376 | 135,397,376 | yes | yes |
| D512 | C4 | 32K | 269,615,104 | 269,615,104 | yes | yes |
| D512 | C6 | 1K | 13,467,648 | 13,467,648 | yes | yes |
| D512 | C6 | 4K | 51,216,384 | 51,216,384 | yes | yes |
| D512 | C6 | 16K | 202,211,328 | 202,211,328 | yes | yes |
| D512 | C6 | 32K | 403,537,920 | 403,537,920 | yes | yes |
| D512 | C12 | 1K | 25,165,824 | 25,165,824 | yes | yes |
| D512 | C12 | 4K | 100,663,296 | 100,663,296 | yes | yes |
| D512 | C12 | 16K | 402,653,184 | 402,653,184 | yes | yes |
| D512 | C12 | 32K | 805,306,368 | 805,306,368 | yes | yes |
| D512 | X-gc12 | 1K | 25,460,736 | 25,460,736 | yes | yes |
| D512 | X-gc12 | 4K | 100,958,208 | 100,958,208 | yes | yes |
| D512 | X-gc12 | 16K | 402,948,096 | 402,948,096 | yes | yes |
| D512 | X-gc12 | 32K | 805,601,280 | 805,601,280 | yes | yes |
| D512 | X-E0 | 1K | 6,488,064 | 6,488,064 | yes | yes |
| D512 | X-E0 | 4K | 6,488,064 | 6,488,064 | yes | yes |
| D512 | X-E0 | 16K | 6,488,064 | 6,488,064 | yes | yes |
| D512 | X-E0 | 32K | 6,488,064 | 6,488,064 | yes | yes |

Decode static buffers (state after the step, L + 1 tokens, all B): 768 of 768 timed decode series have buffer nbytes equal to `state_bytes(cfg, L + 1, batch=B)` to the byte.

## Implementation checks

| Scale | Config | check | result |
|---|---|---|---|
| D384 | C0 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.545766830444336, "bit_identical": true}` |
| D384 | C3 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.535963296890259, "bit_identical": true}` |
| D384 | C12 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.856234550476074, "bit_identical": true}` |
| D384 | X-gc12 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.629166603088379, "bit_identical": true}` |
| D384 | X-E0 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.6514694690704346, "bit_identical": true}` |
| D384 | C0 | check | `{"none": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.658205270767212}}` |
| D384 | C3 | check | `{"cudnn": {"max_abs": 0.010676980018615723, "argmax_agree": 1.0, "ref_absmax": 3.6922383308410645}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.6922383308410645}}` |
| D384 | C12 | check | `{"cudnn": {"max_abs": 0.014057956635951996, "argmax_agree": 1.0, "ref_absmax": 3.5095443725585938}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.5095443725585938}}` |
| D384 | X-gc12 | check | `{"cudnn": {"max_abs": 0.011827826499938965, "argmax_agree": 1.0, "ref_absmax": 3.6222262382507324}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.6222262382507324}}` |
| D384 | X-E0 | check | `{"none": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.6228506565093994}}` |
| D512 | C0 | check_weights | `{"max_abs": 0.0, "ref_absmax": 4.156803131103516, "bit_identical": true}` |
| D512 | C3 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.973829746246338, "bit_identical": true}` |
| D512 | C12 | check_weights | `{"max_abs": 0.0, "ref_absmax": 3.838602066040039, "bit_identical": true}` |
| D512 | X-gc12 | check_weights | `{"max_abs": 0.0, "ref_absmax": 4.138885974884033, "bit_identical": true}` |
| D512 | X-E0 | check_weights | `{"max_abs": 0.0, "ref_absmax": 4.15678071975708, "bit_identical": true}` |
| D512 | C0 | check | `{"none": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.5088772773742676}}` |
| D512 | C3 | check | `{"cudnn": {"max_abs": 0.017324626445770264, "argmax_agree": 1.0, "ref_absmax": 3.540727138519287}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.540727138519287}}` |
| D512 | C12 | check | `{"cudnn": {"max_abs": 0.024090975522994995, "argmax_agree": 1.0, "ref_absmax": 3.575352191925049}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.575352191925049}}` |
| D512 | X-gc12 | check | `{"cudnn": {"max_abs": 0.025572657585144043, "argmax_agree": 1.0, "ref_absmax": 3.6341042518615723}, "flash": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.6341042518615723}}` |
| D512 | X-E0 | check | `{"none": {"max_abs": 0.0, "argmax_agree": 1.0, "ref_absmax": 3.611854314804077}}` |

## Prefill latency, D384 (ms; whole model, last-token logits; min-of-N unless noted)

### D384, B = 1: CUDA graph, FLASH, cold L2 (primary)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.28 | 5.36 | 19.28 | 51.95 |
| C1 | 1 | 2.29 | 5.56 | 21.50 | 60.37 |
| C2 | 2 | 2.32 | 5.75 | 23.89 | 68.82 |
| C3 | 3 | 2.32 | 5.96 | 26.14 | 77.37 |
| C4 | 4 | 2.33 | 6.15 | 28.41 | 85.98 |
| C6 | 6 | 2.36 | 6.55 | 33.07 | 102.97 |
| C12 | 12 | 2.44 | 7.74 | 46.62 | 154.31 |
| X-gc12 | 12 | 2.58 | 8.06 | 48.58 | 158.74 |
| X-E0 | 0 | 2.72 | 6.92 | 25.48 | 63.29 |

D384, B = 1: graph, FLASH, cold L2: median (IQR)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.29 (0.002) | 5.37 (0.005) | 19.31 (0.016) | 52.00 (0.037) |
| C1 | 1 | 2.30 (0.008) | 5.56 (0.006) | 21.61 (0.078) | 60.51 (0.128) |
| C2 | 2 | 2.32 (0.003) | 5.76 (0.006) | 24.05 (0.092) | 69.01 (0.125) |
| C3 | 3 | 2.33 (0.003) | 5.96 (0.004) | 26.32 (0.133) | 77.64 (0.141) |
| C4 | 4 | 2.34 (0.002) | 6.16 (0.005) | 28.58 (0.150) | 86.16 (0.104) |
| C6 | 6 | 2.36 (0.008) | 6.56 (0.005) | 33.21 (0.177) | 103.23 (0.142) |
| C12 | 12 | 2.44 (0.002) | 7.74 (0.006) | 46.91 (0.192) | 154.63 (0.288) |
| X-gc12 | 12 | 2.58 (0.004) | 8.07 (0.005) | 48.86 (0.177) | 159.50 (0.403) |
| X-E0 | 0 | 2.72 (0.003) | 6.92 (0.006) | 25.57 (0.077) | 63.32 (0.024) |

D384, B = 1: graph, FLASH, hot L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.27 | 5.35 | 19.26 | 51.92 |
| C1 | 1 | 2.28 | 5.55 | 21.49 | 60.38 |
| C2 | 2 | 2.30 | 5.74 | 23.87 | 68.79 |
| C3 | 3 | 2.32 | 5.95 | 26.11 | 77.29 |
| C4 | 4 | 2.33 | 6.14 | 28.39 | 85.81 |
| C6 | 6 | 2.36 | 6.54 | 32.99 | 102.94 |
| C12 | 12 | 2.44 | 7.73 | 46.47 | 154.16 |
| X-gc12 | 12 | 2.58 | 8.05 | 48.68 | 158.87 |
| X-E0 | 0 | 2.71 | 6.91 | 25.46 | 63.24 |

D384, B = 1: graph, cuDNN, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.28 | 5.36 | 19.28 | 51.95 |
| C1 | 1 | 2.32 | 5.52 | 21.49 | 60.24 |
| C2 | 2 | 2.37 | 5.69 | 23.67 | 68.36 |
| C3 | 3 | 2.42 | 5.86 | 25.76 | 76.59 |
| C4 | 4 | 2.46 | 6.03 | 27.82 | 84.79 |
| C6 | 6 | 2.55 | 6.37 | 32.05 | 101.15 |
| C12 | 12 | 2.82 | 7.39 | 44.79 | 150.51 |
| X-gc12 | 12 | 2.96 | 7.70 | 46.69 | 155.36 |
| X-E0 | 0 | 2.72 | 6.92 | 25.48 | 63.29 |

D384, B = 1: eager, FLASH, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 13.15 | 13.09 | 19.91 | 52.20 |
| C1 | 1 | 12.71 | 12.58 | 22.04 | 60.59 |
| C2 | 2 | 12.18 | 12.02 | 24.31 | 69.14 |
| C3 | 3 | 11.60 | 11.60 | 26.54 | 77.65 |
| C4 | 4 | 11.10 | 11.22 | 28.84 | 86.22 |
| C6 | 6 | 9.85 | 9.97 | 33.39 | 103.22 |
| C12 | 12 | 5.25 | 7.87 | 46.75 | 154.42 |
| X-gc12 | 12 | 8.21 | 8.30 | 48.87 | 159.27 |
| X-E0 | 0 | 13.24 | 13.27 | 26.02 | 63.50 |

### D384, B = 8: CUDA graph, FLASH, cold L2 (primary)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 8.51 | 44.59 | 202.98 | 408.88 |
| C1 | 1 | 8.74 | 46.61 | 223.02 | 479.00 |
| C2 | 2 | 8.97 | 48.58 | 243.13 | 548.99 |
| C3 | 3 | 9.20 | 50.73 | 263.24 | 619.58 |
| C4 | 4 | 9.42 | 52.62 | 283.37 | 690.03 |
| C6 | 6 | 9.90 | 56.71 | 323.58 | 831.08 |
| C12 | 12 | 11.25 | 68.82 | 444.65 | – |
| X-gc12 | 12 | 11.82 | 73.74 | 465.79 | – |
| X-E0 | 0 | 9.69 | 48.30 | 219.07 | 441.62 |

D384, B = 8: graph, FLASH, cold L2: median (IQR)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 8.51 (0.005) | 44.66 (0.027) | 203.03 (0.039) | 408.99 (0.062) |
| C1 | 1 | 8.76 (0.006) | 46.68 (0.052) | 223.12 (0.082) | 479.21 (0.317) |
| C2 | 2 | 8.98 (0.007) | 48.67 (0.049) | 243.25 (0.107) | 549.20 (0.156) |
| C3 | 3 | 9.22 (0.006) | 50.78 (0.030) | 263.38 (0.102) | 619.83 (0.348) |
| C4 | 4 | 9.43 (0.008) | 52.69 (0.039) | 283.59 (0.148) | 690.59 (0.230) |
| C6 | 6 | 9.91 (0.011) | 56.80 (0.064) | 323.84 (0.108) | 831.69 (0.344) |
| C12 | 12 | 11.27 (0.015) | 68.91 (0.060) | 444.92 (0.179) | – |
| X-gc12 | 12 | 11.86 (0.029) | 73.85 (0.093) | 466.16 (0.276) | – |
| X-E0 | 0 | 9.70 (0.005) | 48.38 (0.031) | 219.26 (0.085) | 442.00 (0.146) |

D384, B = 8: graph, FLASH, hot L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 8.48 | 44.60 | 202.96 | 408.88 |
| C1 | 1 | 8.73 | 46.63 | 223.00 | 478.90 |
| C2 | 2 | 8.95 | 48.57 | 243.11 | 549.01 |
| C3 | 3 | 9.19 | 50.67 | 263.15 | 619.56 |
| C4 | 4 | 9.41 | 52.61 | 283.39 | 689.68 |
| C6 | 6 | 9.88 | 56.73 | 323.70 | 829.77 |
| C12 | 12 | 11.24 | 68.79 | 444.69 | – |
| X-gc12 | 12 | 11.83 | 73.74 | 465.88 | – |
| X-E0 | 0 | 9.68 | 48.29 | 219.09 | 441.71 |

D384, B = 8: graph, cuDNN, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 8.51 | 44.59 | 202.98 | 408.88 |
| C1 | 1 | 8.75 | 46.67 | 222.91 | 478.60 |
| C2 | 2 | 9.00 | 48.66 | 242.95 | 548.87 |
| C3 | 3 | 9.27 | 50.67 | 262.93 | 618.94 |
| C4 | 4 | 9.51 | 52.62 | 283.02 | 689.47 |
| C6 | 6 | 10.01 | 56.71 | 323.28 | 829.41 |
| C12 | 12 | 11.48 | 68.61 | 443.81 | – |
| X-gc12 | 12 | 12.07 | 73.56 | 464.72 | – |
| X-E0 | 0 | 9.69 | 48.30 | 219.07 | 441.62 |

D384, B = 8: eager, FLASH, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 13.38 | 44.84 | 203.01 | 408.91 |
| C1 | 1 | 12.86 | 46.81 | 223.05 | 478.97 |
| C2 | 2 | 12.21 | 48.86 | 243.08 | 549.27 |
| C3 | 3 | 12.04 | 50.96 | 263.37 | 620.07 |
| C4 | 4 | 11.21 | 52.95 | 283.57 | 689.74 |
| C6 | 6 | 10.58 | 56.96 | 323.79 | 831.17 |
| C12 | 12 | 11.49 | 69.11 | 444.87 | – |
| X-gc12 | 12 | 12.03 | 74.00 | 466.10 | – |
| X-E0 | 0 | 13.46 | 48.62 | 219.12 | 441.77 |

### D384: prefill achieved throughput (graph, FLASH, cold; tokens/s and model TFLOP/s)

Model FLOPs = B·T·(fwd FLOP/token at T − 2dV) + B·2dV (§3.4 formula; last-token logits only).

| Config | B | T | min ms | Mtok/s | TFLOP/s | fraction of 112.6 |
|---|---|---|---|---|---|---|
| C0 | 1 | 1K | 2.28 | 0.45 | 20.86 | 0.185 |
| C0 | 1 | 4K | 5.36 | 0.76 | 35.52 | 0.315 |
| C0 | 1 | 16K | 19.28 | 0.85 | 39.51 | 0.351 |
| C0 | 1 | 32K | 51.95 | 0.63 | 29.33 | 0.260 |
| C0 | 8 | 1K | 8.51 | 0.96 | 44.79 | 0.398 |
| C0 | 8 | 4K | 44.59 | 0.73 | 34.17 | 0.304 |
| C0 | 8 | 16K | 202.98 | 0.65 | 30.03 | 0.267 |
| C0 | 8 | 32K | 408.88 | 0.64 | 29.81 | 0.265 |
| C1 | 1 | 1K | 2.29 | 0.45 | 21.01 | 0.187 |
| C1 | 1 | 4K | 5.56 | 0.74 | 36.35 | 0.323 |
| C1 | 1 | 16K | 21.50 | 0.76 | 44.78 | 0.398 |
| C1 | 1 | 32K | 60.37 | 0.54 | 38.72 | 0.344 |
| C1 | 8 | 1K | 8.74 | 0.94 | 44.01 | 0.391 |
| C1 | 8 | 4K | 46.61 | 0.70 | 34.68 | 0.308 |
| C1 | 8 | 16K | 223.02 | 0.59 | 34.53 | 0.307 |
| C1 | 8 | 32K | 479.00 | 0.55 | 39.04 | 0.347 |
| C2 | 1 | 1K | 2.32 | 0.44 | 20.97 | 0.186 |
| C2 | 1 | 4K | 5.75 | 0.71 | 37.16 | 0.330 |
| C2 | 1 | 16K | 23.89 | 0.69 | 48.69 | 0.432 |
| C2 | 1 | 32K | 68.82 | 0.48 | 45.80 | 0.407 |
| C2 | 8 | 1K | 8.97 | 0.91 | 43.32 | 0.385 |
| C2 | 8 | 4K | 48.58 | 0.67 | 35.17 | 0.312 |
| C2 | 8 | 16K | 243.13 | 0.54 | 38.28 | 0.340 |
| C2 | 8 | 32K | 548.99 | 0.48 | 45.92 | 0.408 |
| C3 | 1 | 1K | 2.32 | 0.44 | 21.11 | 0.188 |
| C3 | 1 | 4K | 5.96 | 0.69 | 37.78 | 0.336 |
| C3 | 1 | 16K | 26.14 | 0.63 | 52.19 | 0.463 |
| C3 | 1 | 32K | 77.37 | 0.42 | 51.25 | 0.455 |
| C3 | 8 | 1K | 9.20 | 0.89 | 42.63 | 0.379 |
| C3 | 8 | 4K | 50.73 | 0.65 | 35.50 | 0.315 |
| C3 | 8 | 16K | 263.24 | 0.50 | 41.46 | 0.368 |
| C3 | 8 | 32K | 619.58 | 0.42 | 51.20 | 0.455 |
| C4 | 1 | 1K | 2.33 | 0.44 | 21.21 | 0.188 |
| C4 | 1 | 4K | 6.15 | 0.67 | 38.47 | 0.342 |
| C4 | 1 | 16K | 28.41 | 0.58 | 55.08 | 0.489 |
| C4 | 1 | 32K | 85.98 | 0.38 | 55.59 | 0.494 |
| C4 | 8 | 1K | 9.42 | 0.87 | 42.06 | 0.374 |
| C4 | 8 | 4K | 52.62 | 0.62 | 35.98 | 0.319 |
| C4 | 8 | 16K | 283.37 | 0.46 | 44.18 | 0.392 |
| C4 | 8 | 32K | 690.03 | 0.38 | 55.41 | 0.492 |
| C6 | 1 | 1K | 2.36 | 0.43 | 21.40 | 0.190 |
| C6 | 1 | 4K | 6.55 | 0.63 | 39.67 | 0.352 |
| C6 | 1 | 16K | 33.07 | 0.50 | 59.47 | 0.528 |
| C6 | 1 | 32K | 102.97 | 0.32 | 62.22 | 0.553 |
| C6 | 8 | 1K | 9.90 | 0.83 | 40.78 | 0.362 |
| C6 | 8 | 4K | 56.71 | 0.58 | 36.64 | 0.325 |
| C6 | 8 | 16K | 323.58 | 0.41 | 48.62 | 0.432 |
| C6 | 8 | 32K | 831.08 | 0.32 | 61.67 | 0.548 |
| C12 | 1 | 1K | 2.44 | 0.42 | 21.83 | 0.194 |
| C12 | 1 | 4K | 7.74 | 0.53 | 42.53 | 0.378 |
| C12 | 1 | 16K | 46.62 | 0.35 | 68.03 | 0.604 |
| C12 | 1 | 32K | 154.31 | 0.21 | 73.17 | 0.650 |
| C12 | 8 | 1K | 11.25 | 0.73 | 37.87 | 0.336 |
| C12 | 8 | 4K | 68.82 | 0.48 | 38.24 | 0.340 |
| C12 | 8 | 16K | 444.65 | 0.29 | 57.06 | 0.507 |
| X-gc12 | 1 | 1K | 2.58 | 0.40 | 20.72 | 0.184 |
| X-gc12 | 1 | 4K | 8.06 | 0.51 | 40.87 | 0.363 |
| X-gc12 | 1 | 16K | 48.58 | 0.34 | 65.32 | 0.580 |
| X-gc12 | 1 | 32K | 158.74 | 0.21 | 71.15 | 0.632 |
| X-gc12 | 8 | 1K | 11.82 | 0.69 | 36.13 | 0.321 |
| X-gc12 | 8 | 4K | 73.74 | 0.44 | 35.74 | 0.317 |
| X-gc12 | 8 | 16K | 465.79 | 0.28 | 54.50 | 0.484 |
| X-E0 | 1 | 1K | 2.72 | 0.38 | 18.84 | 0.167 |
| X-E0 | 1 | 4K | 6.92 | 0.59 | 29.59 | 0.263 |
| X-E0 | 1 | 16K | 25.48 | 0.64 | 32.13 | 0.285 |
| X-E0 | 1 | 32K | 63.29 | 0.52 | 25.88 | 0.230 |
| X-E0 | 8 | 1K | 9.69 | 0.85 | 42.26 | 0.375 |
| X-E0 | 8 | 4K | 48.30 | 0.68 | 33.91 | 0.301 |
| X-E0 | 8 | 16K | 219.07 | 0.60 | 29.90 | 0.266 |
| X-E0 | 8 | 32K | 441.62 | 0.59 | 29.67 | 0.263 |

## Prefill latency, D512 (ms; whole model, last-token logits; min-of-N unless noted)

### D512, B = 1: CUDA graph, FLASH, cold L2 (primary)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.86 | 7.99 | 31.71 | 77.03 |
| C1 | 1 | 2.90 | 8.30 | 34.84 | 88.51 |
| C2 | 2 | 2.93 | 8.62 | 38.10 | 99.95 |
| C3 | 3 | 2.98 | 8.90 | 40.93 | 111.27 |
| C4 | 4 | 3.01 | 9.25 | 44.01 | 122.79 |
| C6 | 6 | 3.07 | 9.79 | 49.83 | 145.61 |
| C12 | 12 | 3.27 | 11.62 | 68.16 | 214.79 |
| X-gc12 | 12 | 3.45 | 12.11 | 71.00 | 221.42 |
| X-E0 | 0 | 3.87 | 11.31 | 44.32 | 101.33 |

D512, B = 1: graph, FLASH, cold L2: median (IQR)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.87 (0.004) | 8.02 (0.022) | 31.73 (0.019) | 77.07 (0.024) |
| C1 | 1 | 2.91 (0.002) | 8.34 (0.013) | 34.93 (0.074) | 88.63 (0.074) |
| C2 | 2 | 2.94 (0.003) | 8.64 (0.020) | 38.20 (0.080) | 100.07 (0.102) |
| C3 | 3 | 2.98 (0.004) | 8.95 (0.026) | 41.07 (0.095) | 111.56 (0.169) |
| C4 | 4 | 3.01 (0.004) | 9.28 (0.021) | 44.17 (0.132) | 123.12 (0.150) |
| C6 | 6 | 3.08 (0.003) | 9.85 (0.031) | 50.01 (0.117) | 146.07 (0.191) |
| C12 | 12 | 3.28 (0.003) | 11.68 (0.023) | 68.47 (0.167) | 215.12 (0.307) |
| X-gc12 | 12 | 3.46 (0.003) | 12.17 (0.037) | 71.23 (0.193) | 222.07 (0.513) |
| X-E0 | 0 | 3.88 (0.002) | 11.34 (0.031) | 44.36 (0.036) | 101.43 (0.070) |

D512, B = 1: graph, FLASH, hot L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.86 | 7.98 | 31.68 | 77.01 |
| C1 | 1 | 2.89 | 8.29 | 34.80 | 88.52 |
| C2 | 2 | 2.93 | 8.60 | 38.11 | 99.95 |
| C3 | 3 | 2.97 | 8.86 | 40.94 | 111.36 |
| C4 | 4 | 3.00 | 9.24 | 43.99 | 122.82 |
| C6 | 6 | 3.06 | 9.80 | 49.84 | 145.81 |
| C12 | 12 | 3.27 | 11.63 | 67.99 | 214.81 |
| X-gc12 | 12 | 3.45 | 12.10 | 70.84 | 221.74 |
| X-E0 | 0 | 3.87 | 11.30 | 44.30 | 101.31 |

D512, B = 1: graph, cuDNN, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 2.86 | 7.99 | 31.71 | 77.03 |
| C1 | 1 | 2.93 | 8.26 | 34.36 | 88.20 |
| C2 | 2 | 2.98 | 8.56 | 37.18 | 99.35 |
| C3 | 3 | 3.05 | 8.80 | 40.26 | 110.49 |
| C4 | 4 | 3.10 | 9.08 | 42.82 | 121.64 |
| C6 | 6 | 3.22 | 9.60 | 48.76 | 144.10 |
| C12 | 12 | 3.57 | 11.18 | 65.92 | 211.16 |
| X-gc12 | 12 | 3.75 | 11.65 | 69.41 | 218.23 |
| X-E0 | 0 | 3.87 | 11.31 | 44.32 | 101.33 |

D512, B = 1: eager, FLASH, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 13.29 | 13.38 | 31.76 | 77.18 |
| C1 | 1 | 12.72 | 13.11 | 34.96 | 88.61 |
| C2 | 2 | 12.29 | 12.30 | 37.64 | 100.12 |
| C3 | 3 | 11.91 | 12.04 | 40.74 | 111.42 |
| C4 | 4 | 11.31 | 11.24 | 43.72 | 123.09 |
| C6 | 6 | 9.88 | 10.51 | 49.96 | 146.09 |
| C12 | 12 | 5.27 | 11.83 | 67.98 | 214.79 |
| X-gc12 | 12 | 8.08 | 12.39 | 71.07 | 221.96 |
| X-E0 | 0 | 13.43 | 13.67 | 44.78 | 101.37 |

### D512, B = 8: CUDA graph, FLASH, cold L2 (primary)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 12.84 | 70.23 | 294.71 | 629.04 † |
| C1 | 1 | 13.11 | 73.00 | 321.41 | 727.48 † |
| C2 | 2 | 13.42 | 75.75 | 348.13 | 826.42 † |
| C3 | 3 | 13.74 | 78.60 | 374.83 | – |
| C4 | 4 | 14.07 | 81.31 | 401.60 | – |
| C6 | 6 | 14.69 | 86.99 | 491.46 † | – |
| C12 | 12 | 16.52 | 103.79 | 615.55 | skip |
| X-gc12 | 12 | 17.33 | 110.68 | – | skip |
| X-E0 | 0 | 14.93 | 77.40 | 344.46 † | 689.70 † |

D512, B = 8: graph, FLASH, cold L2: median (IQR)

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 12.86 (0.022) | 70.26 (0.030) | 294.79 (0.070) | 639.56 (9.577) † |
| C1 | 1 | 13.13 (0.028) | 73.05 (0.030) | 321.50 (0.087) | 738.92 (13.384) † |
| C2 | 2 | 13.45 (0.028) | 75.81 (0.037) | 348.25 (0.092) | 830.31 (12.906) † |
| C3 | 3 | 13.76 (0.016) | 78.66 (0.050) | 375.00 (0.097) | – |
| C4 | 4 | 14.08 (0.031) | 81.45 (0.053) | 401.79 (0.421) | – |
| C6 | 6 | 14.71 (0.025) | 87.04 (0.038) | 492.30 (0.811) † | – |
| C12 | 12 | 16.55 (0.017) | 103.94 (0.058) | 615.79 (0.270) | skip |
| X-gc12 | 12 | 17.38 (0.030) | 110.79 (0.066) | – | skip |
| X-E0 | 0 | 14.95 (0.011) | 77.45 (0.030) | 349.38 (7.101) † | 693.01 (2.657) † |

D512, B = 8: graph, FLASH, hot L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 12.82 | 70.21 | 294.70 | 628.29 † |
| C1 | 1 | 13.09 | 72.98 | 321.41 | 727.44 † |
| C2 | 2 | 13.42 | 75.75 | 348.09 | 826.62 † |
| C3 | 3 | 13.73 | 78.58 | 374.84 | – |
| C4 | 4 | 14.04 | 81.35 | 401.61 | – |
| C6 | 6 | 14.67 | 86.95 | 490.97 † | – |
| C12 | 12 | 16.50 | 103.82 | 615.65 | skip |
| X-gc12 | 12 | 17.32 | 110.69 | – | skip |
| X-E0 | 0 | 14.92 | 77.39 | 345.04 † | 688.97 † |

D512, B = 8: graph, cuDNN, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 12.84 | 70.23 | 294.71 | 629.04 † |
| C1 | 1 | 13.15 | 72.99 | 321.44 † | 684.33 † |
| C2 | 2 | 13.49 | 75.76 | 348.02 | 825.15 † |
| C3 | 3 | 13.83 | 78.56 | 374.77 | – |
| C4 | 4 | 14.16 | 81.32 | 401.25 | – |
| C6 | 6 | 14.83 | 86.95 | 482.61 † | – |
| C12 | 12 | 16.84 | 103.71 | 614.69 | skip |
| X-gc12 | 12 | 17.59 | 110.59 | 642.72 | skip |
| X-E0 | 0 | 14.93 | 77.40 | 344.46 † | 689.70 † |

D512, B = 8: eager, FLASH, cold L2

| Config | k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|---|
| C0 | 0 | 13.67 | 70.34 | 294.79 | 629.00 † |
| C1 | 1 | 13.95 | 73.11 | 321.55 † | 727.73 † |
| C2 | 2 | 14.30 | 75.88 | 348.31 | 827.06 † |
| C3 | 3 | 14.37 | 78.68 | 374.94 | 941.57 † |
| C4 | 4 | 14.69 | 81.53 | 401.74 | 1,024.73 † |
| C6 | 6 | 15.38 | 87.16 | 484.03 † | – |
| C12 | 12 | 16.76 | 104.08 | 616.12 | skip |
| X-gc12 | 12 | 17.57 | 110.96 | 643.67 | skip |
| X-E0 | 0 | 15.50 | 77.43 | 345.06 † | 691.88 † |

Skipped prefill shapes at D512 (budget 8 GiB state + activation, excluding weights):

- C12 B=8 T=32K: OOM in eager probe
- X-gc12 B=8 T=32K: OOM in eager probe

### D512: prefill achieved throughput (graph, FLASH, cold; tokens/s and model TFLOP/s)

Model FLOPs = B·T·(fwd FLOP/token at T − 2dV) + B·2dV (§3.4 formula; last-token logits only).

| Config | B | T | min ms | Mtok/s | TFLOP/s | fraction of 112.6 |
|---|---|---|---|---|---|---|
| C0 | 1 | 1K | 2.86 | 0.36 | 31.20 | 0.277 |
| C0 | 1 | 4K | 7.99 | 0.51 | 44.68 | 0.397 |
| C0 | 1 | 16K | 31.71 | 0.52 | 45.05 | 0.400 |
| C0 | 1 | 32K | 77.03 | 0.43 | 37.09 | 0.329 |
| C0 | 8 | 1K | 12.84 | 0.64 | 55.62 | 0.494 |
| C0 | 8 | 4K | 70.23 | 0.47 | 40.68 | 0.361 |
| C0 | 8 | 16K | 294.71 | 0.44 | 38.78 | 0.344 |
| C0 | 8 | 32K | 629.04 † | 0.42 | 36.34 | 0.323 |
| C1 | 1 | 1K | 2.90 | 0.35 | 30.99 | 0.275 |
| C1 | 1 | 4K | 8.30 | 0.49 | 44.90 | 0.399 |
| C1 | 1 | 16K | 34.84 | 0.47 | 48.69 | 0.432 |
| C1 | 1 | 32K | 88.51 | 0.37 | 44.54 | 0.396 |
| C1 | 8 | 1K | 13.11 | 0.62 | 54.87 | 0.487 |
| C1 | 8 | 4K | 73.00 | 0.45 | 40.83 | 0.363 |
| C1 | 8 | 16K | 321.41 | 0.41 | 42.22 | 0.375 |
| C1 | 8 | 32K | 727.48 † | 0.36 | 43.35 | 0.385 |
| C2 | 1 | 1K | 2.93 | 0.35 | 30.85 | 0.274 |
| C2 | 1 | 4K | 8.62 | 0.48 | 45.02 | 0.400 |
| C2 | 1 | 16K | 38.10 | 0.43 | 51.55 | 0.458 |
| C2 | 1 | 32K | 99.95 | 0.33 | 50.30 | 0.447 |
| C2 | 8 | 1K | 13.42 | 0.61 | 53.97 | 0.479 |
| C2 | 8 | 4K | 75.75 | 0.43 | 40.97 | 0.364 |
| C2 | 8 | 16K | 348.13 | 0.38 | 45.13 | 0.401 |
| C2 | 8 | 32K | 826.42 † | 0.32 | 48.67 | 0.432 |
| C3 | 1 | 1K | 2.98 | 0.34 | 30.64 | 0.272 |
| C3 | 1 | 4K | 8.90 | 0.46 | 45.32 | 0.402 |
| C3 | 1 | 16K | 40.93 | 0.40 | 54.53 | 0.484 |
| C3 | 1 | 32K | 111.27 | 0.29 | 54.94 | 0.488 |
| C3 | 8 | 1K | 13.74 | 0.60 | 53.08 | 0.471 |
| C3 | 8 | 4K | 78.60 | 0.42 | 41.05 | 0.365 |
| C3 | 8 | 16K | 374.83 | 0.35 | 47.63 | 0.423 |
| C4 | 1 | 1K | 3.01 | 0.34 | 30.51 | 0.271 |
| C4 | 1 | 4K | 9.25 | 0.44 | 45.25 | 0.402 |
| C4 | 1 | 16K | 44.01 | 0.37 | 56.80 | 0.504 |
| C4 | 1 | 32K | 122.79 | 0.27 | 58.62 | 0.521 |
| C4 | 8 | 1K | 14.07 | 0.58 | 52.21 | 0.464 |
| C4 | 8 | 4K | 81.31 | 0.40 | 41.19 | 0.366 |
| C4 | 8 | 16K | 401.60 | 0.33 | 49.79 | 0.442 |
| C6 | 1 | 1K | 3.07 | 0.33 | 30.30 | 0.269 |
| C6 | 1 | 4K | 9.79 | 0.42 | 45.93 | 0.408 |
| C6 | 1 | 16K | 49.83 | 0.33 | 60.91 | 0.541 |
| C6 | 1 | 32K | 145.61 | 0.23 | 64.34 | 0.571 |
| C6 | 8 | 1K | 14.69 | 0.56 | 50.67 | 0.450 |
| C6 | 8 | 4K | 86.99 | 0.38 | 41.34 | 0.367 |
| C6 | 8 | 16K | 491.46 † | 0.27 | 49.40 | 0.439 |
| C12 | 1 | 1K | 3.27 | 0.31 | 29.58 | 0.263 |
| C12 | 1 | 4K | 11.62 | 0.35 | 46.64 | 0.414 |
| C12 | 1 | 16K | 68.16 | 0.24 | 68.09 | 0.605 |
| C12 | 1 | 32K | 214.79 | 0.15 | 73.93 | 0.657 |
| C12 | 8 | 1K | 16.52 | 0.50 | 46.87 | 0.416 |
| C12 | 8 | 4K | 103.79 | 0.32 | 41.76 | 0.371 |
| C12 | 8 | 16K | 615.55 | 0.21 | 60.32 | 0.536 |
| X-gc12 | 1 | 1K | 3.45 | 0.30 | 28.07 | 0.249 |
| X-gc12 | 1 | 4K | 12.11 | 0.34 | 44.77 | 0.398 |
| X-gc12 | 1 | 16K | 71.00 | 0.23 | 65.40 | 0.581 |
| X-gc12 | 1 | 32K | 221.42 | 0.15 | 71.74 | 0.637 |
| X-gc12 | 8 | 1K | 17.33 | 0.47 | 44.77 | 0.398 |
| X-gc12 | 8 | 4K | 110.68 | 0.30 | 39.20 | 0.348 |
| X-E0 | 1 | 1K | 3.87 | 0.26 | 24.90 | 0.221 |
| X-E0 | 1 | 4K | 11.31 | 0.36 | 34.10 | 0.303 |
| X-E0 | 1 | 16K | 44.32 | 0.37 | 34.80 | 0.309 |
| X-E0 | 1 | 32K | 101.33 | 0.32 | 30.45 | 0.270 |
| X-E0 | 8 | 1K | 14.93 | 0.55 | 51.66 | 0.459 |
| X-E0 | 8 | 4K | 77.40 | 0.42 | 39.87 | 0.354 |
| X-E0 | 8 | 16K | 344.46 † | 0.38 | 35.83 | 0.318 |
| X-E0 | 8 | 32K | 689.70 † | 0.38 | 35.79 | 0.318 |

## Decode, D384 (one-step exact-shape graph, no masks)

### D384, B = 1: cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 870.4 | 872.5 | 872.4 | 872.4 |
| C1 | 1 | 897.0 | 901.2 | 921.6 | 958.4 |
| C2 | 2 | 919.5 | 931.8 | 980.5 | 1,048.6 |
| C3 | 3 | 950.3 | 964.6 | 1,040.4 | 1,126.4 |
| C4 | 4 | 972.2 | 989.2 | 1,093.6 | 1,204.2 |
| C6 | 6 | 1,024.0 | 1,056.8 | 1,200.1 | 1,364.0 |
| C12 | 12 | 1,173.5 | 1,263.6 | 1,505.3 | 1,835.0 |
| X-gc12 | 12 | 1,282.0 | 1,366.0 | 1,611.8 | 1,941.5 |
| X-E0 | 0 | 952.3 | 950.3 | 951.8 | 950.2 |

### D384, B = 1: cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 810.5 | 804.9 | 805.3 | 805.1 |
| C1 | 1 | 832.2 | 834.2 | 838.2 | 884.9 |
| C2 | 2 | 858.3 | 861.0 | 913.7 | 1,015.5 |
| C3 | 3 | 884.9 | 890.1 | 1,014.7 | 1,097.9 |
| C4 | 4 | 910.6 | 914.8 | 1,063.7 | 1,173.2 |
| C6 | 6 | 963.7 | 971.9 | 1,169.3 | 1,331.7 |
| C12 | 12 | 1,116.1 | 1,235.1 | 1,479.1 | 1,806.2 |
| X-gc12 | 12 | 1,213.1 | 1,331.6 | 1,582.4 | 1,906.3 |
| X-E0 | 0 | 885.9 | 885.7 | 884.5 | 882.9 |

### D384, B = 1: flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 870.4 | 872.5 | 872.4 | 872.4 |
| C1 | 1 | 899.0 | 907.3 | 931.8 | 966.6 |
| C2 | 2 | 931.8 | 944.1 | 995.3 | 1,067.0 |
| C3 | 3 | 960.5 | 983.0 | 1,062.9 | 1,153.0 |
| C4 | 4 | 986.6 | 1,013.8 | 1,122.3 | 1,235.0 |
| C6 | 6 | 1,046.5 | 1,095.6 | 1,239.1 | 1,411.1 |
| C12 | 12 | 1,218.6 | 1,335.3 | 1,586.7 | 1,925.2 |
| X-gc12 | 12 | 1,323.0 | 1,437.7 | 1,693.7 | 2,031.6 |
| X-E0 | 0 | 952.3 | 950.3 | 951.8 | 950.2 |

### D384, B = 8: cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,091.6 | 1,087.5 | 1,087.5 | 1,089.5 |
| C1 | 1 | 1,132.5 | 1,183.7 | 1,355.7 | 1,577.0 |
| C2 | 2 | 1,175.6 | 1,273.9 | 1,611.7 | 2,048.0 |
| C3 | 3 | 1,210.3 | 1,349.6 | 1,851.4 | 2,514.9 |
| C4 | 4 | 1,253.4 | 1,427.5 | 2,099.2 | 2,979.8 |
| C6 | 6 | 1,325.1 | 1,585.1 | 2,588.6 | 3,911.7 |
| C12 | 12 | 1,536.0 | 2,037.8 | 4,041.7 | 6,680.6 |
| X-gc12 | 12 | 1,675.3 | 2,172.9 | 4,191.2 | 6,836.2 |
| X-E0 | 0 | 1,110.0 | 1,105.9 | 1,103.4 | 1,105.5 |

### D384, B = 8: cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,025.2 | 1,023.7 | 1,024.1 | 1,023.6 |
| C1 | 1 | 1,075.9 | 1,144.9 | 1,314.6 | 1,536.8 |
| C2 | 2 | 1,141.9 | 1,224.3 | 1,562.5 | 2,003.7 |
| C3 | 3 | 1,179.4 | 1,311.4 | 1,811.1 | 2,470.1 |
| C4 | 4 | 1,215.9 | 1,393.4 | 2,053.5 | 2,935.2 |
| C6 | 6 | 1,288.9 | 1,542.9 | 2,541.9 | 3,861.0 |
| C12 | 12 | 1,502.8 | 2,001.3 | 3,992.4 | 6,634.8 |
| X-gc12 | 12 | 1,636.8 | 2,134.4 | 4,142.8 | 6,778.2 |
| X-E0 | 0 | 1,099.0 | 1,096.8 | 1,094.9 | 1,097.0 |

### D384, B = 8: flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,091.6 | 1,087.5 | 1,087.5 | 1,089.5 |
| C1 | 1 | 1,138.7 | 1,191.9 | 1,363.9 | 1,583.1 |
| C2 | 2 | 1,194.0 | 1,280.0 | 1,622.0 | 2,062.4 |
| C3 | 3 | 1,232.9 | 1,368.0 | 1,875.9 | 2,533.3 |
| C4 | 4 | 1,282.0 | 1,452.0 | 2,130.0 | 3,004.4 |
| C6 | 6 | 1,366.0 | 1,619.9 | 2,634.8 | 3,955.7 |
| C12 | 12 | 1,611.8 | 2,115.6 | 4,144.1 | 6,774.8 |
| X-gc12 | 12 | 1,755.1 | 2,251.8 | 4,291.6 | 6,928.4 |
| X-E0 | 0 | 1,110.0 | 1,105.9 | 1,103.4 | 1,105.5 |

### D384, B = B_max(L): cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 55,972.9 (B=5185) | 55,972.8 (B=5185) | 55,968.7 (B=5185) | 55,999.5 (B=5185) |
| C1 | 1 | 32,696.3 (B=2721) | 18,745.3 (B=1077) | 11,984.9 (B=315) | 11,036.6 (B=162) |
| C2 | 2 | 24,451.1 (B=1847) | 14,426.1 (B=601) | 11,002.8 (B=162) | 10,844.2 (B=84) |
| C3 | 3 | 19,905.5 (B=1397) | 12,554.2 (B=416) | 10,750.0 (B=109) | 10,715.1 (B=56) |
| C4 | 4 | 17,099.8 (B=1123) | 11,726.8 (B=319) | 10,872.8 (B=84) | 10,688.5 (B=42) |
| C6 | 6 | 14,292.0 (B=807) | 11,145.2 (B=217) | 10,731.5 (B=56) | 10,602.5 (B=28) |
| C12 | 12 | 11,064.3 (B=442) | 11,008.0 (B=113) | 10,641.4 (B=28) | 10,656.8 (B=14) |
| X-gc12 | 12 | 12,189.7 (B=436) | 10,967.0 (B=110) | 10,840.1 (B=28) | 10,809.3 (B=14) |
| X-E0 | 0 | 43,684.9 (B=1921) | 43,680.7 (B=1921) | 43,678.8 (B=1921) | 43,674.6 (B=1921) |

### D384, B = B_max(L): cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 55,973.7 (B=5185) | 55,975.4 (B=5185) | 55,957.6 (B=5185) | 56,007.5 (B=5185) |
| C1 | 1 | 32,701.9 (B=2721) | 18,740.7 (B=1077) | 11,973.7 (B=315) | 11,022.3 (B=162) |
| C2 | 2 | 24,459.1 (B=1847) | 14,432.6 (B=601) | 10,985.1 (B=162) | 10,828.2 (B=84) |
| C3 | 3 | 19,898.6 (B=1397) | 12,563.4 (B=416) | 10,728.7 (B=109) | 10,699.6 (B=56) |
| C4 | 4 | 17,103.7 (B=1123) | 11,710.6 (B=319) | 10,863.3 (B=84) | 10,665.8 (B=42) |
| C6 | 6 | 14,290.5 (B=807) | 11,136.7 (B=217) | 10,711.3 (B=56) | 10,561.3 (B=28) |
| C12 | 12 | 11,056.9 (B=442) | 10,971.6 (B=113) | 10,589.8 (B=28) | 10,593.6 (B=14) |
| X-gc12 | 12 | 12,183.2 (B=436) | 10,957.1 (B=110) | 10,790.7 (B=28) | 10,758.2 (B=14) |
| X-E0 | 0 | 43,687.9 (B=1921) | 43,674.8 (B=1921) | 43,683.2 (B=1921) | 43,673.1 (B=1921) |

### D384, B = B_max(L): flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 55,972.9 (B=5185) | 55,972.8 (B=5185) | 55,968.7 (B=5185) | 55,999.5 (B=5185) |
| C1 | 1 | 33,503.2 (B=2721) | 19,330.1 (B=1077) | 12,657.7 (B=315) | 11,863.0 (B=162) |
| C2 | 2 | 25,552.9 (B=1847) | 15,079.4 (B=601) | 11,855.8 (B=162) | 11,388.9 (B=84) |
| C3 | 3 | 21,156.9 (B=1397) | 13,251.6 (B=416) | 11,498.5 (B=109) | 11,284.5 (B=56) |
| C4 | 4 | 18,480.1 (B=1123) | 12,446.7 (B=319) | 11,403.3 (B=84) | 11,309.1 (B=42) |
| C6 | 6 | 15,694.8 (B=807) | 12,040.2 (B=217) | 11,327.5 (B=56) | 11,335.6 (B=28) |
| C12 | 12 | 12,723.2 (B=442) | 12,715.0 (B=113) | 11,476.0 (B=28) | 10,780.6 (B=14) |
| X-gc12 | 12 | 13,809.6 (B=436) | 11,833.3 (B=110) | 11,664.4 (B=28) | 10,950.7 (B=14) |
| X-E0 | 0 | 43,684.9 (B=1921) | 43,680.7 (B=1921) | 43,678.8 (B=1921) | 43,674.6 (B=1921) |

### D384: decode throughput, tokens/s = B / min latency (cuDNN, cold L2)

| Config | k | B | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|---|
| C0 | 0 | 1 | 1,149 | 1,146 | 1,146 | 1,146 |
| C0 | 0 | 8 | 7,329 | 7,356 | 7,356 | 7,343 |
| C0 | 0 | B_max | 92,634 | 92,634 | 92,641 | 92,590 |
| C1 | 1 | 1 | 1,115 | 1,110 | 1,085 | 1,043 |
| C1 | 1 | 8 | 7,064 | 6,758 | 5,901 | 5,073 |
| C1 | 1 | B_max | 83,220 | 57,454 | 26,283 | 14,678 |
| C2 | 2 | 1 | 1,088 | 1,073 | 1,020 | 954 |
| C2 | 2 | 8 | 6,805 | 6,280 | 4,964 | 3,906 |
| C2 | 2 | B_max | 75,539 | 41,661 | 14,723 | 7,746 |
| C3 | 3 | 1 | 1,052 | 1,037 | 961 | 888 |
| C3 | 3 | 8 | 6,610 | 5,928 | 4,321 | 3,181 |
| C3 | 3 | B_max | 70,181 | 33,136 | 10,140 | 5,226 |
| C4 | 4 | 1 | 1,029 | 1,011 | 914 | 830 |
| C4 | 4 | 8 | 6,383 | 5,604 | 3,811 | 2,685 |
| C4 | 4 | B_max | 65,673 | 27,203 | 7,726 | 3,929 |
| C6 | 6 | 1 | 977 | 946 | 833 | 733 |
| C6 | 6 | 8 | 6,037 | 5,047 | 3,090 | 2,045 |
| C6 | 6 | B_max | 56,465 | 19,470 | 5,218 | 2,641 |
| C12 | 12 | 1 | 852 | 791 | 664 | 545 |
| C12 | 12 | 8 | 5,208 | 3,926 | 1,979 | 1,198 |
| C12 | 12 | B_max | 39,948 | 10,265 | 2,631 | 1,314 |
| X-gc12 | 12 | 1 | 780 | 732 | 620 | 515 |
| X-gc12 | 12 | 8 | 4,775 | 3,682 | 1,909 | 1,170 |
| X-gc12 | 12 | B_max | 35,768 | 10,030 | 2,583 | 1,295 |
| X-E0 | 0 | 1 | 1,050 | 1,052 | 1,051 | 1,052 |
| X-E0 | 0 | 8 | 7,207 | 7,234 | 7,250 | 7,237 |
| X-E0 | 0 | B_max | 43,974 | 43,978 | 43,980 | 43,984 |

### D384: decode median (IQR) µs, cuDNN cold, B = 1

| Config | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|
| C0 | 878.6 (2.0) | 874.5 (2.0) | 874.5 (2.0) | 874.5 (2.0) |
| C1 | 901.2 (2.0) | 903.2 (2.0) | 923.7 (2.1) | 962.6 (2.1) |
| C2 | 925.7 (0.6) | 933.9 (0.0) | 983.0 (2.1) | 1,050.6 (2.1) |
| C3 | 952.4 (2.1) | 966.7 (2.0) | 1,042.4 (0.0) | 1,130.5 (2.1) |
| C4 | 974.9 (2.0) | 993.3 (0.6) | 1,097.7 (2.1) | 1,208.3 (4.1) |
| C6 | 1,028.1 (2.0) | 1,060.9 (2.1) | 1,202.2 (2.0) | 1,366.0 (2.0) |
| C12 | 1,177.6 (2.6) | 1,265.7 (2.0) | 1,507.3 (2.0) | 1,839.1 (2.1) |
| X-gc12 | 1,286.1 (2.1) | 1,368.1 (2.1) | 1,613.9 (2.1) | 1,947.6 (3.4) |
| X-E0 | 956.4 (2.0) | 954.4 (4.0) | 954.4 (2.1) | 952.4 (2.0) |

### D384: whole-step decode roofline fraction (cuDNN, cold L2)

Bytes per step = 2·N_total (bf16 weights) + B·[k·(L+1)·4d (KV read) + 2·S_lin (state read+write)] (§10.1 generalised to B); roofline time = bytes / 960 GB/s; fraction = roofline / measured min.

| Config | B class | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 1 | 0.069 | 0.069 | 0.069 | 0.069 |
| C0 | 8 | 0.073 | 0.073 | 0.073 | 0.073 |
| C0 | max | 0.257 | 0.257 | 0.257 | 0.257 |
| C1 | 1 | 0.069 | 0.074 | 0.094 | 0.117 |
| C1 | 8 | 0.080 | 0.110 | 0.212 | 0.315 |
| C1 | max | 0.349 | 0.525 | 0.760 | 0.812 |
| C2 | 1 | 0.069 | 0.078 | 0.114 | 0.157 |
| C2 | 8 | 0.087 | 0.142 | 0.307 | 0.447 |
| C2 | max | 0.424 | 0.646 | 0.811 | 0.835 |
| C3 | 1 | 0.068 | 0.082 | 0.133 | 0.193 |
| C3 | 8 | 0.094 | 0.171 | 0.380 | 0.530 |
| C3 | max | 0.494 | 0.725 | 0.824 | 0.838 |
| C4 | 1 | 0.068 | 0.086 | 0.150 | 0.223 |
| C4 | 8 | 0.099 | 0.198 | 0.434 | 0.587 |
| C4 | max | 0.555 | 0.768 | 0.830 | 0.837 |
| C6 | 1 | 0.067 | 0.093 | 0.180 | 0.274 |
| C6 | 8 | 0.111 | 0.242 | 0.513 | 0.661 |
| C6 | max | 0.638 | 0.798 | 0.833 | 0.840 |
| C12 | 1 | 0.066 | 0.108 | 0.247 | 0.374 |
| C12 | 8 | 0.140 | 0.337 | 0.637 | 0.762 |
| C12 | max | 0.791 | 0.813 | 0.833 | 0.832 |
| X-gc12 | 1 | 0.061 | 0.100 | 0.231 | 0.354 |
| X-gc12 | 8 | 0.131 | 0.318 | 0.615 | 0.745 |
| X-gc12 | max | 0.725 | 0.799 | 0.819 | 0.821 |
| X-E0 | 1 | 0.068 | 0.069 | 0.068 | 0.069 |
| X-E0 | 8 | 0.107 | 0.108 | 0.108 | 0.108 |
| X-E0 | max | 0.339 | 0.339 | 0.339 | 0.339 |

## Decode, D512 (one-step exact-shape graph, no masks)

### D512, B = 1: cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 956.4 † | 954.3 † | 954.4 † | 954.0 † |
| C1 | 1 | 987.1 † | 993.2 † | 1,023.6 † | 1,065.0 † |
| C2 | 2 | 1,015.8 † | 1,028.1 † | 1,095.6 † | 1,173.5 † |
| C3 | 3 | 1,046.5 † | 1,067.0 † | 1,159.2 † | 1,280.0 † |
| C4 | 4 | 1,074.8 † | 1,105.9 † | 1,224.7 † | 1,380.4 † |
| C6 | 6 | 1,136.6 † | 1,185.8 † | 1,355.8 † | 1,586.8 † |
| C12 | 12 | 1,304.2 † | 1,402.4 † | 1,740.8 † | 2,195.4 † |
| X-gc12 | 12 | 1,409.0 † | 1,509.3 † | 1,849.3 † | 2,304.0 † |
| X-E0 | 0 | 1,062.9 † | 1,062.9 † | 1,058.8 † | 1,060.9 † |

### D512, B = 1: cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 901.5 † | 898.8 † | 898.2 † | 899.7 † |
| C1 | 1 | 929.2 † | 929.2 † | 938.0 † | 1,048.8 † |
| C2 | 2 | 960.4 † | 960.0 † | 1,078.6 † | 1,150.0 † |
| C3 | 3 | 989.4 † | 993.4 † | 1,140.7 † | 1,252.2 † |
| C4 | 4 | 1,017.7 † | 1,029.2 † | 1,204.0 † | 1,356.8 † |
| C6 | 6 | 1,076.8 † | 1,139.3 † | 1,336.6 † | 1,560.3 † |
| C12 | 12 | 1,240.5 † | 1,383.3 † | 1,720.1 † | 2,170.3 † |
| X-gc12 | 12 | 1,339.1 † | 1,486.8 † | 1,822.6 † | 2,272.2 † |
| X-E0 | 0 | 993.3 † | 993.0 † | 990.3 † | 990.7 † |

### D512, B = 1: flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 956.4 † | 954.3 † | 954.4 † | 954.0 † |
| C1 | 1 | 989.2 † | 997.3 † | 1,030.1 † | 1,075.2 † |
| C2 | 2 | 1,024.0 † | 1,040.4 † | 1,105.9 † | 1,187.8 † |
| C3 | 3 | 1,056.8 † | 1,087.1 † | 1,179.6 † | 1,298.4 † |
| C4 | 4 | 1,089.5 † | 1,126.4 † | 1,250.9 † | 1,411.0 † |
| C6 | 6 | 1,155.1 † | 1,220.6 † | 1,392.6 † | 1,634.3 † |
| C12 | 12 | 1,345.5 † | 1,472.5 † | 1,808.4 † | 2,291.7 † |
| X-gc12 | 12 | 1,447.9 † | 1,579.0 † | 1,914.8 † | 2,404.4 † |
| X-E0 | 0 | 1,062.9 † | 1,062.9 † | 1,058.8 † | 1,060.9 † |

### D512, B = 8: cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,161.2 † | 1,161.2 † | 1,160.5 † | 1,161.2 † |
| C1 | 1 | 1,206.2 † | 1,259.5 † | 1,482.7 † | 1,779.3 † |
| C2 | 2 | 1,242.7 † | 1,355.8 † | 1,800.2 † | 2,390.0 † |
| C3 | 3 | 1,290.2 † | 1,458.1 † | 2,123.4 † | 3,006.5 † |
| C4 | 4 | 1,324.6 † | 1,556.5 † | 2,443.2 † | 3,619.8 † |
| C6 | 6 | 1,409.0 † | 1,744.9 † | 3,078.8 † | 4,836.4 † |
| C12 | 12 | 1,638.0 † | 2,311.2 † | 4,957.8 † | 8,477.7 † |
| X-gc12 | 12 | 1,779.7 † | 2,451.4 † | 5,110.8 † | 8,622.1 † |
| X-E0 | 0 | 1,202.2 † | 1,200.1 † | 1,199.7 † | 1,201.7 † |

### D512, B = 8: cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,139.9 † | 1,142.8 † | 1,141.6 † | 1,139.0 † |
| C1 | 1 | 1,182.8 † | 1,235.6 † | 1,455.7 † | 1,753.6 † |
| C2 | 2 | 1,212.8 † | 1,326.3 † | 1,772.3 † | 2,361.1 † |
| C3 | 3 | 1,256.8 † | 1,424.5 † | 2,095.0 † | 2,973.8 † |
| C4 | 4 | 1,300.0 † | 1,516.6 † | 2,410.6 † | 3,580.7 † |
| C6 | 6 | 1,376.4 † | 1,709.5 † | 3,040.4 † | 4,793.8 † |
| C12 | 12 | 1,609.0 † | 2,276.5 † | 4,911.9 † | 8,423.7 † |
| X-gc12 | 12 | 1,744.5 † | 2,414.5 † | 5,065.7 † | 8,569.5 † |
| X-E0 | 0 | 1,189.3 † | 1,190.0 † | 1,187.9 † | 1,187.1 † |

### D512, B = 8: flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 1,161.2 † | 1,161.2 † | 1,160.5 † | 1,161.2 † |
| C1 | 1 | 1,212.4 † | 1,269.7 † | 1,488.9 † | 1,787.5 † |
| C2 | 2 | 1,259.5 † | 1,374.2 † | 1,816.2 † | 2,408.4 † |
| C3 | 3 | 1,308.3 † | 1,482.7 † | 2,146.3 † | 3,034.1 † |
| C4 | 4 | 1,351.3 † | 1,597.4 † | 2,478.0 † | 3,657.7 † |
| C6 | 6 | 1,445.3 † | 1,808.4 † | 3,125.2 † | 4,894.7 † |
| C12 | 12 | 1,712.1 † | 2,453.5 † | 5,065.3 † | 8,600.1 † |
| X-gc12 | 12 | 1,855.5 † | 2,594.8 † | 5,222.4 † | 8,752.1 † |
| X-E0 | 0 | 1,202.2 † | 1,200.1 † | 1,199.7 † | 1,201.7 † |

### D512, B = B_max(L): cudnn, cold L2 — per-step latency µs (min-of-N) [primary]

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 51,805.2 (B=3924) † | 51,830.8 (B=3924) † | 51,826.7 (B=3924) † | 51,823.6 (B=3924) † |
| C1 | 1 | 30,126.1 (B=2051) † | 17,644.5 (B=810) † | 11,683.8 (B=236) † | 11,085.8 (B=124) † |
| C2 | 2 | 23,119.9 (B=1388) † | 14,075.9 (B=451) † | 10,834.9 (B=121) † | 10,822.7 (B=63) † |
| C3 | 3 | 19,007.5 (B=1050) † | 12,269.6 (B=312) † | 10,722.3 (B=82) † | 10,731.5 (B=42) † |
| C4 | 4 | 16,375.8 (B=844) † | 11,553.8 (B=239) † | 10,881.0 (B=63) † | 10,571.8 (B=31) † |
| C6 | 6 | 13,921.3 (B=606) † | 11,101.2 (B=163) † | 10,784.8 (B=42) † | 10,671.1 (B=21) † |
| C12 | 12 | 11,219.9 (B=332) † | 11,130.9 (B=85) † | 10,739.7 (B=21) † | 10,245.2 (B=10) † |
| X-gc12 | 12 | 12,353.6 (B=328) † | 11,125.7 (B=83) † | 10,938.4 (B=21) † | 10,397.7 (B=10) † |
| X-E0 | 0 | 40,134.7 (B=1098) † | 40,126.4 (B=1098) † | 40,124.4 (B=1098) † | 40,127.5 (B=1098) † |

### D512, B = B_max(L): cudnn, hot L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 51,807.8 (B=3924) † | 51,813.6 (B=3924) † | 51,825.3 (B=3924) † | 51,828.5 (B=3924) † |
| C1 | 1 | 30,120.7 (B=2051) † | 17,650.3 (B=810) † | 11,678.0 (B=236) † | 11,081.6 (B=124) † |
| C2 | 2 | 23,105.4 (B=1388) † | 14,077.2 (B=451) † | 10,827.2 (B=121) † | 10,811.4 (B=63) † |
| C3 | 3 | 18,997.2 (B=1050) † | 12,269.4 (B=312) † | 10,714.4 (B=82) † | 10,721.8 (B=42) † |
| C4 | 4 | 16,365.5 (B=844) † | 11,553.4 (B=239) † | 10,869.3 (B=63) † | 10,557.5 (B=31) † |
| C6 | 6 | 13,921.4 (B=606) † | 11,097.9 (B=163) † | 10,761.7 (B=42) † | 10,627.7 (B=21) † |
| C12 | 12 | 11,207.6 (B=332) † | 11,110.2 (B=85) † | 10,688.4 (B=21) † | 10,196.3 (B=10) † |
| X-gc12 | 12 | 12,349.5 (B=328) † | 11,108.0 (B=83) † | 10,889.1 (B=21) † | 10,352.6 (B=10) † |
| X-E0 | 0 | 40,140.1 (B=1098) † | 40,128.6 (B=1098) † | 40,135.3 (B=1098) † | 40,124.4 (B=1098) † |

### D512, B = B_max(L): flash, cold L2 — per-step latency µs (min-of-N)

| Config | k | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 0 | 51,805.2 (B=3924) † | 51,830.8 (B=3924) † | 51,826.7 (B=3924) † | 51,823.6 (B=3924) † |
| C1 | 1 | 30,955.5 (B=2051) † | 18,293.8 (B=810) † | 12,366.8 (B=236) † | 11,747.3 (B=124) † |
| C2 | 2 | 24,243.2 (B=1388) † | 14,696.5 (B=451) † | 11,726.9 (B=121) † | 11,370.5 (B=63) † |
| C3 | 3 | 20,260.9 (B=1050) † | 12,971.0 (B=312) † | 11,450.3 (B=82) † | 11,304.9 (B=42) † |
| C4 | 4 | 17,726.5 (B=844) † | 12,257.3 (B=239) † | 11,399.8 (B=63) † | 11,345.9 (B=31) † |
| C6 | 6 | 15,290.4 (B=606) † | 11,982.4 (B=163) † | 11,378.7 (B=42) † | 11,392.0 (B=21) † |
| C12 | 12 | 12,842.0 (B=332) † | 12,833.8 (B=85) † | 11,565.0 (B=21) † | 10,362.8 (B=10) † |
| X-gc12 | 12 | 13,927.4 (B=328) † | 11,943.9 (B=83) † | 11,741.2 (B=21) † | 10,521.6 (B=10) † |
| X-E0 | 0 | 40,134.7 (B=1098) † | 40,126.4 (B=1098) † | 40,124.4 (B=1098) † | 40,127.5 (B=1098) † |

### D512: decode throughput, tokens/s = B / min latency (cuDNN, cold L2)

| Config | k | B | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|---|
| C0 | 0 | 1 | 1,046 † | 1,048 † | 1,048 † | 1,048 † |
| C0 | 0 | 8 | 6,889 † | 6,890 † | 6,894 † | 6,890 † |
| C0 | 0 | B_max | 75,745 † | 75,708 † | 75,714 † | 75,718 † |
| C1 | 1 | 1 | 1,013 † | 1,007 † | 977 † | 939 † |
| C1 | 1 | 8 | 6,632 † | 6,352 † | 5,395 † | 4,496 † |
| C1 | 1 | B_max | 68,081 † | 45,907 † | 20,199 † | 11,185 † |
| C2 | 2 | 1 | 984 † | 973 † | 913 † | 852 † |
| C2 | 2 | 8 | 6,437 † | 5,901 † | 4,444 † | 3,347 † |
| C2 | 2 | B_max | 60,035 † | 32,041 † | 11,168 † | 5,821 † |
| C3 | 3 | 1 | 956 † | 937 † | 863 † | 781 † |
| C3 | 3 | 8 | 6,200 † | 5,486 † | 3,768 † | 2,661 † |
| C3 | 3 | B_max | 55,241 † | 25,429 † | 7,648 † | 3,914 † |
| C4 | 4 | 1 | 930 † | 904 † | 817 † | 724 † |
| C4 | 4 | 8 | 6,039 † | 5,140 † | 3,274 † | 2,210 † |
| C4 | 4 | B_max | 51,539 † | 20,686 † | 5,790 † | 2,932 † |
| C6 | 6 | 1 | 880 † | 843 † | 738 † | 630 † |
| C6 | 6 | 8 | 5,678 † | 4,585 † | 2,598 † | 1,654 † |
| C6 | 6 | B_max | 43,530 † | 14,683 † | 3,894 † | 1,968 † |
| C12 | 12 | 1 | 767 † | 713 † | 574 † | 455 † |
| C12 | 12 | 8 | 4,884 † | 3,461 † | 1,614 † | 944 † |
| C12 | 12 | B_max | 29,590 † | 7,636 † | 1,955 † | 976 † |
| X-gc12 | 12 | 1 | 710 † | 663 † | 541 † | 434 † |
| X-gc12 | 12 | 8 | 4,495 † | 3,263 † | 1,565 † | 928 † |
| X-gc12 | 12 | B_max | 26,551 † | 7,460 † | 1,920 † | 962 † |
| X-E0 | 0 | 1 | 941 † | 941 † | 944 † | 943 † |
| X-E0 | 0 | 8 | 6,655 † | 6,666 † | 6,668 † | 6,657 † |
| X-E0 | 0 | B_max | 27,358 † | 27,364 † | 27,365 † | 27,363 † |

### D512: decode median (IQR) µs, cuDNN cold, B = 1

| Config | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|
| C0 | 958.5 (0.0) † | 956.4 (4.0) † | 956.4 (0.6) † | 956.4 (4.1) † |
| C1 | 991.2 (2.0) † | 995.3 (2.0) † | 1,026.0 (2.0) † | 1,069.1 (2.6) † |
| C2 | 1,019.9 (0.1) † | 1,030.2 (2.4) † | 1,095.7 (2.0) † | 1,175.6 (0.5) † |
| C3 | 1,048.6 (0.0) † | 1,071.1 (2.0) † | 1,163.2 (0.0) † | 1,282.0 (2.0) † |
| C4 | 1,077.2 (2.0) † | 1,108.0 (2.1) † | 1,226.7 (2.4) † | 1,383.2 (2.0) † |
| C6 | 1,138.7 (2.0) † | 1,187.9 (2.0) † | 1,361.9 (2.1) † | 1,590.3 (2.0) † |
| C12 | 1,306.6 (2.1) † | 1,404.9 (2.6) † | 1,744.9 (0.1) † | 2,198.5 (3.1) † |
| X-gc12 | 1,411.1 (2.1) † | 1,511.5 (2.1) † | 1,852.4 (2.4) † | 2,308.1 (4.1) † |
| X-E0 | 1,065.0 (2.0) † | 1,067.0 (2.1) † | 1,064.9 (2.0) † | 1,064.9 (2.1) † |

### D512: whole-step decode roofline fraction (cuDNN, cold L2)

Bytes per step = 2·N_total (bf16 weights) + B·[k·(L+1)·4d (KV read) + 2·S_lin (state read+write)] (§10.1 generalised to B); roofline time = bytes / 960 GB/s; fraction = roofline / measured min.

| Config | B class | L=1K | L=4K | L=16K | L=32K |
|---|---|---|---|---|---|
| C0 | 1 | 0.111 † | 0.112 † | 0.112 † | 0.112 † |
| C0 | 8 | 0.114 † | 0.114 † | 0.114 † | 0.114 † |
| C0 | max | 0.281 † | 0.281 † | 0.281 † | 0.281 † |
| C1 | 1 | 0.110 † | 0.116 † | 0.138 † | 0.165 † |
| C1 | 8 | 0.122 † | 0.159 † | 0.276 † | 0.387 † |
| C1 | max | 0.382 † | 0.562 † | 0.783 † | 0.829 † |
| C2 | 1 | 0.109 † | 0.120 † | 0.161 † | 0.209 † |
| C2 | 8 | 0.131 † | 0.197 † | 0.381 † | 0.521 † |
| C2 | max | 0.451 † | 0.666 † | 0.825 † | 0.841 † |
| C3 | 1 | 0.107 † | 0.124 † | 0.182 † | 0.246 † |
| C3 | 8 | 0.138 † | 0.230 † | 0.454 † | 0.600 † |
| C3 | max | 0.521 † | 0.745 † | 0.833 † | 0.841 † |
| C4 | 1 | 0.106 † | 0.127 † | 0.200 † | 0.279 † |
| C4 | 8 | 0.145 † | 0.258 † | 0.508 † | 0.652 † |
| C4 | max | 0.584 † | 0.783 † | 0.833 † | 0.837 † |
| C6 | 1 | 0.104 † | 0.133 † | 0.232 † | 0.330 † |
| C6 | 8 | 0.158 † | 0.308 † | 0.583 † | 0.718 † |
| C6 | max | 0.659 † | 0.806 † | 0.833 † | 0.839 † |
| C12 | 1 | 0.099 † | 0.148 † | 0.300 † | 0.429 † |
| C12 | 8 | 0.191 † | 0.408 † | 0.698 † | 0.804 † |
| C12 | max | 0.786 † | 0.810 † | 0.830 † | 0.829 † |
| X-gc12 | 1 | 0.092 † | 0.138 † | 0.283 † | 0.409 † |
| X-gc12 | 8 | 0.179 † | 0.386 † | 0.678 † | 0.791 † |
| X-gc12 | max | 0.721 † | 0.796 † | 0.816 † | 0.817 † |
| X-E0 | 1 | 0.109 † | 0.109 † | 0.110 † | 0.110 † |
| X-E0 | 8 | 0.175 † | 0.176 † | 0.176 † | 0.176 † |
| X-E0 | max | 0.372 † | 0.372 † | 0.372 † | 0.372 † |

## B_max(L): 8 GiB state + activation budget

Analytic: floor(8 GiB / (state_bytes(L+1) + 4·V fp32 logits row)). Verified: static buffers allocated and the one-step graph captured and replayed; `used` = peak allocated minus the pre-allocation baseline (resident weights excluded); B shrunk proportionally until used ≤ 8 GiB.

| Scale | Config | L=1K analytic / verified (used GiB) | L=4K analytic / verified (used GiB) | L=16K analytic / verified (used GiB) | L=32K analytic / verified (used GiB) |
|---|---|---|---|---|---|
| D384 | C0 | 6,168 / 5,185 (7.86) | 6,168 / 5,185 (7.86) | 6,168 / 5,185 (7.86) | 6,168 / 5,185 (7.86) |
| D384 | C1 | 3,007 / 2,721 (7.84) | 1,133 / 1,077 (7.83) | 324 / 315 (7.84) | 166 / 162 (7.83) |
| D384 | C2 | 1,988 / 1,847 (7.84) | 624 / 601 (7.84) | 166 / 162 (7.82) | 84 / 84 (7.99) |
| D384 | C3 | 1,485 / 1,397 (7.83) | 430 / 416 (7.83) | 112 / 109 (7.80) | 56 / 56 (7.94) |
| D384 | C4 | 1,185 / 1,123 (7.84) | 328 / 319 (7.84) | 84 / 84 (7.97) | 42 / 42 (7.92) |
| D384 | C6 | 844 / 807 (7.84) | 223 / 217 (7.84) | 56 / 56 (7.93) | 28 / 28 (7.90) |
| D384 | C12 | 453 / 442 (7.84) | 113 / 113 (7.98) | 28 / 28 (7.88) | 14 / 14 (7.88) |
| D384 | X-gc12 | 447 / 436 (7.84) | 113 / 110 (7.78) | 28 / 28 (7.89) | 14 / 14 (7.88) |
| D384 | X-E0 | 2,289 / 1,921 (7.84) | 2,289 / 1,921 (7.84) | 2,289 / 1,921 (7.84) | 2,289 / 1,921 (7.84) |
| D512 | C0 | 4,681 / 3,924 (7.85) | 4,681 / 3,924 (7.85) | 4,681 / 3,924 (7.85) | 4,681 / 3,924 (7.85) |
| D512 | C1 | 2,268 / 2,051 (7.83) | 852 / 810 (7.85) | 243 / 236 (7.82) | 124 / 124 (7.99) |
| D512 | C2 | 1,496 / 1,388 (7.83) | 468 / 451 (7.83) | 125 / 121 (7.78) | 63 / 63 (7.98) |
| D512 | C3 | 1,116 / 1,050 (7.84) | 323 / 312 (7.82) | 84 / 82 (7.82) | 42 / 42 (7.94) |
| D512 | C4 | 890 / 844 (7.83) | 246 / 239 (7.82) | 63 / 63 (7.97) | 31 / 31 (7.80) |
| D512 | C6 | 634 / 606 (7.83) | 167 / 163 (7.84) | 42 / 42 (7.92) | 21 / 21 (7.90) |
| D512 | C12 | 340 / 332 (7.83) | 85 / 85 (7.98) | 21 / 21 (7.88) | 10 / 10 (7.50) |
| D512 | X-gc12 | 336 / 328 (7.83) | 85 / 83 (7.82) | 21 / 21 (7.88) | 10 / 10 (7.50) |
| D512 | X-E0 | 1,310 / 1,098 (7.84) | 1,310 / 1,098 (7.84) | 1,310 / 1,098 (7.84) | 1,310 / 1,098 (7.84) |

## Kernel roofline fractions (CUDA graph, cold L2, min-of-N)

Roofline time = max(FLOPs / 112.6 TFLOP/s, compulsory bytes / 960 GB/s); fraction = roofline / measured.

### D384

| Kernel | variant | shape | min µs | TFLOP/s | GB/s | roofline frac |
|---|---|---|---|---|---|---|
| attn_prefill | flash | B=1, T=1024 | 25.9 | 31.07 | 121 | 0.276 |
| attn_prefill | flash | B=1, T=4096 | 208.9 | 61.68 | 60 | 0.548 |
| attn_prefill | flash | B=1, T=16384 | 2,205.7 | 93.47 | 23 | 0.830 |
| attn_prefill | flash | B=1, T=32768 | 8,056.8 | 102.35 | 12 | 0.909 |
| attn_prefill | flash | B=8, T=1024 | 97.8 | 65.90 | 257 | 0.585 |
| attn_prefill | flash | B=8, T=4096 | 1,042.4 | 98.89 | 97 | 0.878 |
| attn_prefill | flash | B=8, T=16384 | 15,214.6 | 108.40 | 26 | 0.963 |
| attn_prefill | flash | B=8, T=32768 | 60,484.6 | 109.07 | 13 | 0.969 |
| attn_prefill | cudnn | B=1, T=1024 | 57.3 | 14.05 | 55 | 0.125 |
| attn_prefill | cudnn | B=1, T=4096 | 192.0 | 67.12 | 66 | 0.596 |
| attn_prefill | cudnn | B=1, T=16384 | 2,090.0 | 98.64 | 24 | 0.876 |
| attn_prefill | cudnn | B=1, T=32768 | 7,819.2 | 105.46 | 13 | 0.937 |
| attn_prefill | cudnn | B=8, T=1024 | 114.7 | 56.19 | 219 | 0.499 |
| attn_prefill | cudnn | B=8, T=4096 | 1,044.4 | 98.69 | 96 | 0.876 |
| attn_prefill | cudnn | B=8, T=16384 | 15,189.0 | 108.58 | 27 | 0.964 |
| attn_prefill | cudnn | B=8, T=32768 | 60,115.9 | 109.74 | 13 | 0.975 |
| attn_decode | cudnn | B=1, L=1024 | 7.8 | 0.20 | 202 | 0.211 |
| attn_decode | cudnn | B=1, L=4096 | 13.9 | 0.45 | 453 | 0.472 |
| attn_decode | cudnn | B=1, L=16384 | 34.8 | 0.72 | 723 | 0.753 |
| attn_decode | cudnn | B=1, L=32768 | 64.8 | 0.78 | 777 | 0.809 |
| attn_decode | cudnn | B=8, L=1024 | 18.4 | 0.68 | 685 | 0.713 |
| attn_decode | cudnn | B=8, L=4096 | 63.5 | 0.79 | 793 | 0.826 |
| attn_decode | cudnn | B=8, L=16384 | 233.4 | 0.86 | 862 | 0.898 |
| attn_decode | cudnn | B=8, L=32768 | 471.0 | 0.85 | 855 | 0.891 |
| attn_decode | flash | B=1, L=1024 | 12.3 | 0.13 | 128 | 0.134 |
| attn_decode | flash | B=1, L=4096 | 19.9 | 0.32 | 316 | 0.329 |
| attn_decode | flash | B=1, L=16384 | 42.6 | 0.59 | 591 | 0.616 |
| attn_decode | flash | B=1, L=32768 | 73.7 | 0.68 | 683 | 0.711 |
| attn_decode | flash | B=8, L=1024 | 28.2 | 0.45 | 446 | 0.465 |
| attn_decode | flash | B=8, L=4096 | 71.6 | 0.70 | 703 | 0.732 |
| attn_decode | flash | B=8, L=16384 | 243.7 | 0.83 | 826 | 0.861 |
| attn_decode | flash | B=8, L=32768 | 473.1 | 0.85 | 851 | 0.887 |
| gdn_chunk | GDN-A | B=1, T=1024 | 52.8 | 6.27 | 47 | 0.056 |
| gdn_chunk | GDN-A | B=1, T=4096 | 122.8 | 10.79 | 78 | 0.096 |
| gdn_chunk | GDN-A | B=1, T=16384 | 429.7 | 12.34 | 89 | 0.110 |
| gdn_chunk | GDN-A | B=1, T=32768 | 1,011.7 | 10.48 | 75 | 0.093 |
| gdn_chunk | GDN-A | B=8, T=1024 | 90.1 | 29.42 | 219 | 0.261 |
| gdn_chunk | GDN-A | B=8, T=4096 | 396.4 | 26.75 | 193 | 0.238 |
| gdn_chunk | GDN-A | B=8, T=16384 | 1,849.3 | 22.93 | 165 | 0.204 |
| gdn_chunk | GDN-A | B=8, T=32768 | 3,784.7 | 22.41 | 161 | 0.199 |
| gdn_recurrent_step | GDN-A | B=1 | 4.1 | 0.04 | 48 | 0.050 |
| gdn_recurrent_step | GDN-A | B=8 | 8.2 | 0.14 | 193 | 0.201 |
| gdn_recurrent_step | GDN-A | B=256 | 86.0 | 0.44 | 585 | 0.610 |
| gdn_chunk | GDN-E | B=1, T=1024 | 94.2 | 6.70 | 28 | 0.059 |
| gdn_chunk | GDN-E | B=1, T=4096 | 258.0 | 9.78 | 38 | 0.087 |
| gdn_chunk | GDN-E | B=1, T=16384 | 966.3 | 10.44 | 39 | 0.093 |
| gdn_chunk | GDN-E | B=1, T=32768 | 2,003.9 | 10.07 | 38 | 0.089 |
| gdn_chunk | GDN-E | B=8, T=1024 | 190.4 | 26.49 | 112 | 0.235 |
| gdn_chunk | GDN-E | B=8, T=4096 | 759.8 | 26.56 | 103 | 0.236 |
| gdn_chunk | GDN-E | B=8, T=16384 | 3,219.5 | 25.07 | 95 | 0.223 |
| gdn_chunk | GDN-E | B=8, T=32768 | 6,610.9 | 24.42 | 92 | 0.217 |
| gdn_recurrent_step | GDN-E | B=1 | 6.1 | 0.07 | 97 | 0.101 |
| gdn_recurrent_step | GDN-E | B=8 | 9.8 | 0.36 | 482 | 0.502 |
| gdn_recurrent_step | GDN-E | B=256 | 368.6 | 0.31 | 410 | 0.427 |
| gemm | d_to_d | M=1, K=384, N=384 | 6.1 | 0.05 | 49 | 0.051 |
| gemm | d_to_d | M=8, K=384, N=384 | 8.2 | 0.29 | 38 | 0.039 |
| gemm | d_to_d | M=4096, K=384, N=384 | 18.4 | 65.54 | 357 | 0.582 |
| gemm | d_to_d | M=65536, K=384, N=384 | 221.2 | 87.39 | 457 | 0.776 |
| gemm | d_to_d | M=262144, K=384, N=384 | 1,015.8 | 76.11 | 397 | 0.676 |
| gemm | d_to_dff | M=1, K=384, N=1024 | 7.6 | 0.10 | 103 | 0.107 |
| gemm | d_to_dff | M=8, K=384, N=1024 | 7.5 | 0.84 | 108 | 0.112 |
| gemm | d_to_dff | M=4096, K=384, N=1024 | 47.1 | 68.43 | 262 | 0.608 |
| gemm | d_to_dff | M=65536, K=384, N=1024 | 525.9 | 98.00 | 352 | 0.870 |
| gemm | d_to_dff | M=262144, K=384, N=1024 | 2,035.7 | 101.27 | 363 | 0.899 |
| gemm | dff_to_d | M=1, K=1024, N=384 | 6.1 | 0.13 | 129 | 0.135 |
| gemm | dff_to_d | M=8, K=1024, N=384 | 7.6 | 0.82 | 106 | 0.110 |
| gemm | dff_to_d | M=4096, K=1024, N=384 | 54.9 | 58.70 | 225 | 0.521 |
| gemm | dff_to_d | M=65536, K=1024, N=384 | 501.3 | 102.80 | 370 | 0.913 |
| gemm | dff_to_d | M=262144, K=1024, N=384 | 1,934.3 | 106.58 | 382 | 0.947 |
| gemm | logits_d_to_V | M=1, K=384, N=16384 | 26.6 | 0.47 | 474 | 0.494 |
| gemm | logits_d_to_V | M=8, K=384, N=16384 | 20.4 | 4.92 | 628 | 0.655 |
| attn_prefill | flex | B=1, T=1024 | 38.2 † | 21.08 | 82 | 0.187 |
| attn_prefill | flex | B=1, T=4096 | 219.1 † | 58.81 | 57 | 0.522 |
| attn_prefill | flex | B=1, T=16384 | 2,263.6 † | 91.07 | 22 | 0.809 |
| attn_prefill | flex | B=1, T=32768 | 8,187.9 † | 100.71 | 12 | 0.894 |
| attn_prefill | flex | B=8, T=1024 | 102.4 † | 62.93 | 246 | 0.559 |
| attn_prefill | flex | B=8, T=4096 | 1,065.0 † | 96.79 | 95 | 0.860 |
| attn_prefill | flex | B=8, T=16384 | 15,482.4 † | 106.53 | 26 | 0.946 |
| attn_prefill | flex | B=8, T=32768 | 61,093.9 † | 107.98 | 13 | 0.959 |

### D512

| Kernel | variant | shape | min µs | TFLOP/s | GB/s | roofline frac |
|---|---|---|---|---|---|---|
| attn_prefill | flash | B=1, T=1024 | 33.9 † | 31.69 | 124 | 0.281 |
| attn_prefill | flash | B=1, T=4096 | 256.0 † | 67.12 | 66 | 0.596 |
| attn_prefill | flash | B=1, T=16384 | 2,809.8 † | 97.83 | 24 | 0.869 |
| attn_prefill | flash | B=1, T=32768 | 10,488.8 † | 104.83 | 13 | 0.931 |
| attn_prefill | flash | B=8, T=1024 | 118.8 † | 72.32 | 282 | 0.642 |
| attn_prefill | flash | B=8, T=4096 | 1,361.5 † | 100.95 | 99 | 0.897 |
| attn_prefill | flash | B=8, T=16384 | 20,281.3 † | 108.43 | 26 | 0.963 |
| attn_prefill | flash | B=8, T=32768 | 80,395.2 † | 109.41 | 13 | 0.972 |
| attn_prefill | cudnn | B=1, T=1024 | 57.3 † | 18.74 | 73 | 0.166 |
| attn_prefill | cudnn | B=1, T=4096 | 229.3 † | 74.91 | 73 | 0.665 |
| attn_prefill | cudnn | B=1, T=16384 | 2,717.3 † | 101.16 | 25 | 0.898 |
| attn_prefill | cudnn | B=1, T=32768 | 10,282.6 † | 106.93 | 13 | 0.950 |
| attn_prefill | cudnn | B=8, T=1024 | 145.0 † | 59.26 | 231 | 0.526 |
| attn_prefill | cudnn | B=8, T=4096 | 1,369.6 † | 100.35 | 98 | 0.891 |
| attn_prefill | cudnn | B=8, T=16384 | 20,279.3 † | 108.44 | 26 | 0.963 |
| attn_prefill | cudnn | B=8, T=32768 | 80,141.3 † | 109.76 | 13 | 0.975 |
| attn_decode | cudnn | B=1, L=1024 | 7.8 † | 0.27 | 270 | 0.281 |
| attn_decode | cudnn | B=1, L=4096 | 15.9 † | 0.53 | 527 | 0.548 |
| attn_decode | cudnn | B=1, L=16384 | 45.0 † | 0.75 | 745 | 0.776 |
| attn_decode | cudnn | B=1, L=32768 | 83.9 † | 0.80 | 800 | 0.833 |
| attn_decode | cudnn | B=8, L=1024 | 24.1 † | 0.70 | 696 | 0.725 |
| attn_decode | cudnn | B=8, L=4096 | 81.9 † | 0.82 | 820 | 0.854 |
| attn_decode | cudnn | B=8, L=16384 | 310.8 † | 0.86 | 864 | 0.900 |
| attn_decode | cudnn | B=8, L=32768 | 620.5 † | 0.87 | 865 | 0.901 |
| attn_decode | flash | B=1, L=1024 | 11.6 † | 0.18 | 180 | 0.188 |
| attn_decode | flash | B=1, L=4096 | 22.0 † | 0.38 | 382 | 0.398 |
| attn_decode | flash | B=1, L=16384 | 51.2 † | 0.66 | 656 | 0.683 |
| attn_decode | flash | B=1, L=32768 | 90.1 † | 0.74 | 745 | 0.776 |
| attn_decode | flash | B=8, L=1024 | 29.7 † | 0.56 | 565 | 0.588 |
| attn_decode | flash | B=8, L=4096 | 88.1 † | 0.76 | 762 | 0.794 |
| attn_decode | flash | B=8, L=16384 | 323.2 † | 0.83 | 831 | 0.865 |
| attn_decode | flash | B=8, L=32768 | 624.6 † | 0.86 | 860 | 0.895 |
| gdn_chunk | GDN-A | B=1, T=1024 | 54.4 † | 8.12 | 61 | 0.072 |
| gdn_chunk | GDN-A | B=1, T=4096 | 129.0 † | 13.70 | 99 | 0.122 |
| gdn_chunk | GDN-A | B=1, T=16384 | 501.7 † | 14.09 | 101 | 0.125 |
| gdn_chunk | GDN-A | B=1, T=32768 | 1,126.4 † | 12.55 | 90 | 0.111 |
| gdn_chunk | GDN-A | B=8, T=1024 | 106.5 † | 33.20 | 247 | 0.295 |
| gdn_chunk | GDN-A | B=8, T=4096 | 558.5 † | 25.31 | 183 | 0.225 |
| gdn_chunk | GDN-A | B=8, T=16384 | 2,463.7 † | 22.95 | 165 | 0.204 |
| gdn_chunk | GDN-A | B=8, T=32768 | 4,971.5 † | 22.75 | 163 | 0.202 |
| gdn_recurrent_step | GDN-A | B=1 | 5.7 † | 0.03 | 46 | 0.048 |
| gdn_recurrent_step | GDN-A | B=8 | 6.1 † | 0.26 | 343 | 0.357 |
| gdn_recurrent_step | GDN-A | B=256 | 120.4 † | 0.42 | 557 | 0.581 |
| gdn_chunk | GDN-E | B=1, T=1024 | 139.3 † | 7.48 | 26 | 0.066 |
| gdn_chunk | GDN-E | B=1, T=4096 | 403.5 † | 10.33 | 33 | 0.092 |
| gdn_chunk | GDN-E | B=1, T=16384 | 1,531.9 † | 10.88 | 33 | 0.097 |
| gdn_chunk | GDN-E | B=1, T=32768 | 3,160.0 † | 10.55 | 32 | 0.094 |
| gdn_chunk | GDN-E | B=8, T=1024 | 282.6 † | 29.49 | 104 | 0.262 |
| gdn_chunk | GDN-E | B=8, T=4096 | 1,185.4 † | 28.12 | 89 | 0.250 |
| gdn_chunk | GDN-E | B=8, T=16384 | 4,877.3 † | 27.34 | 84 | 0.243 |
| gdn_chunk | GDN-E | B=8, T=32768 | 9,775.1 † | 27.28 | 83 | 0.242 |
| gdn_recurrent_step | GDN-E | B=1 | 6.1 † | 0.13 | 172 | 0.179 |
| gdn_recurrent_step | GDN-E | B=8 | 14.3 † | 0.44 | 586 | 0.611 |
| gdn_recurrent_step | GDN-E | B=256 | 655.3 † | 0.31 | 410 | 0.427 |
| gemm | d_to_d | M=1, K=512, N=512 | 7.6 † | 0.07 | 69 | 0.072 |
| gemm | d_to_d | M=8, K=512, N=512 | 7.6 † | 0.55 | 71 | 0.074 |
| gemm | d_to_d | M=4096, K=512, N=512 | 57.3 † | 37.47 | 156 | 0.333 |
| gemm | d_to_d | M=65536, K=512, N=512 | 351.8 † | 97.66 | 383 | 0.867 |
| gemm | d_to_d | M=262144, K=512, N=512 | 1,316.8 † | 104.37 | 408 | 0.927 |
| gemm | d_to_dff | M=1, K=512, N=1536 | 10.2 † | 0.15 | 154 | 0.160 |
| gemm | d_to_dff | M=8, K=512, N=1536 | 9.8 † | 1.29 | 165 | 0.171 |
| gemm | d_to_dff | M=4096, K=512, N=1536 | 83.6 † | 77.08 | 220 | 0.685 |
| gemm | d_to_dff | M=65536, K=512, N=1536 | 994.9 † | 103.61 | 271 | 0.920 |
| gemm | d_to_dff | M=262144, K=512, N=1536 | 3,935.2 † | 104.78 | 273 | 0.931 |
| gemm | dff_to_d | M=1, K=1536, N=512 | 14.3 † | 0.11 | 110 | 0.115 |
| gemm | dff_to_d | M=8, K=1536, N=512 | 10.2 † | 1.23 | 157 | 0.164 |
| gemm | dff_to_d | M=4096, K=1536, N=512 | 79.3 † | 81.21 | 231 | 0.721 |
| gemm | dff_to_d | M=65536, K=1536, N=512 | 966.2 † | 106.68 | 279 | 0.947 |
| gemm | dff_to_d | M=262144, K=1536, N=512 | 3,643.4 † | 113.17 | 295 | 1.005 |
| gemm | logits_d_to_V | M=1, K=512, N=16384 | 34.3 † | 0.49 | 490 | 0.510 |
| gemm | logits_d_to_V | M=8, K=512, N=16384 | 24.2 † | 5.56 | 706 | 0.735 |

