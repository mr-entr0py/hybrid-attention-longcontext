# Plan to the 27th-evening presentation (drafted 2026-11-24)

Numbers: `scripts/planning/budget.py`, D384, idle-RTX-5080 hours; estimates, not measurements. Per run: k0 1.76 h, k1 1.80,
k3 1.90, k12 2.21, so one seed across k in {0,1,3,12} is ~7.7 h (~28 h on an RTX 4060 laptop GPU, measured ~3.7x slower).
Remaining Tier-A work is ~80 h (pilots 9.8, LR sweep 4.2, core 5 seeds 38.4, controls 8.8, branches 5.8, evaluation 9.8,
cost benchmarks 3.0) plus 4-9 h of gate checks. The 5080 is free from ~Tue 16:30 (after the O4 soak) to ~Fri 14:00: ~69 h.
**Full Tier A does not fit; prioritise.**

| When | What |
|---|---|
| Tue 16:30-22:00 | Gate checks VT-22b, VT-23b, VT-13r, R6, R2e-X; G-RV audit in parallel (CPU) |
| Tue night - Wed 10:00 | S2 FEAS pilots (9.8 h): scale, Stage-B budget, SSMax, TEF mix |
| Wed 10:00-14:00 | LR sweep (4.2 h), then the freeze (items, code, configs) |
| Wed 14:00 - Fri 06:00 | Core runs in seed blocks (all four k per seed); 3 seeds ~23 h (the target); a 4th seed (~8 h more) only if time allows |
| in parallel | Optional second worker (RTX 4060 laptop or a teammate's NVIDIA GPU): one whole seed block, only after the gates are re-run on that hardware and a decision row allows mixed hardware |
| as blocks land | Frozen evaluations on the 5080; controls/branches/D768 scale check only if time remains |

**Seed decision (team, 2026-11-24):** 3 seeds is the target; a 4th is welcome but there is no hard cap. Protocol text says 4-5 seeds, so record 3 as a logged deviation and report the reduced power honestly.
**Minimum for the talk:** 3 complete seed blocks, frozen evaluations, the cost frontier and the reproductions.
**Hardware notes:** Apple-silicon machines cannot run the Triton/fla kernels (use for analysis and slides only). Mixed hardware
changes kernels, so each machine takes whole seed blocks (paired contrasts stay within a machine) and the gates are re-checked there.
**Risks:** O4 soak fails (Xid faults); the all-attention reference is weak at 16K/32K (protocol: pause and re-scope to <= 8K);
the shared GPU's owner needs it back; pre-freeze statistics changes (multiplicity family, seed budget) undecided; H1 may be inconclusive.
