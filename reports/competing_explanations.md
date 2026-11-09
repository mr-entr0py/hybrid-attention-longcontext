# Competing-explanations register

**Companion to:** `reports/experimental_protocol.md` v1.0 (frozen 2026-11-09). §14 of the protocol summarises this file.

**Built from:**
- `reports/initial_proposal.md` §14;
- `reports/reviews/review_A_round1.md` §4 and `review_B_round1.md` §4;
- `reports/reviews/debate_record.md` (I-01…I-29).

**Owner:** PI. New rows may be appended after G5; existing rows change only through a logged deviation.

**How to read a row.**
- Each likely outcome lists the rival explanations that could produce it.
- For each rival, the row gives:
  - the experiment or analysis that discriminates it;
  - the **signature** expected if that rival is true;
  - the tier, and whether the check runs unconditionally;
  - the pre-registered consequence for the claim.
- An outcome is reported as "genuine" only after every rival row for it has been checked and its signature is absent.

**Tags.**
- Access: [read-full], [read-section], [abstract-only], [code-inspected]. "per X" means the access was X's and was not re-checked by the PI.
- Epistemic: [EST] established in the cited source; [HYP] hypothesis; [INF] inference.
- Measurements: [MEAS-B]/[MEAS-B2] are Reviewer B's contended-GPU measurements.
- Nemotron-H ([2504.03624](https://arxiv.org/abs/2504.03624)) is not a controlled full-vs-linear ratio experiment and is not used as evidence here.

**Notation.**
- CS, MS, DG, DC, FD are the evaluation suites of protocol §9.
- C{k} are composition configs; X-* are controls (protocol §3.2).
- S_f is a family score; L_k = S_k − 0.2·S₀ − 0.8·S₁₂; R is recovery; Δ₀(T) = S₁₂(T) − S₀(T).

---

## O1. H1 or H1-min supported (L₃ or L₁ > 0 at the primary cell)

| ID | Rival explanation | Discriminating experiment / analysis | Signature if the rival is true | Tier / when | Consequence for the claim |
|---|---|---|---|---|---|
| E1.1 | **Genuine.** A few attention layers carry long-range retrieval; [2402.18510](https://arxiv.org/abs/2402.18510) [abstract-only per PROP]; Qiao et al. report that long-range information is "mainly introduced and processed by full attention" ([2606.15378v1](https://arxiv.org/html/2606.15378v1) [read-section per A1]) | — | Survives every row below | — | "Supported (package claim)" |
| E1.2 | **Weak long-length reference** (thin Stage B, RoPE at 32K, QK-norm softmax dilution at d_h = 64) [MEAS-B: needle weight 0.64 → 0.05 from 1K to 32K at unit gains] | G-REF (S₁₂(T) ≥ 0.8·S₁₂(C0)); 2× Stage-B branch at {0, 3, 12}; attention diagnostics at 32K (needle mass, entropy, maximum logit); SSMax rule in FEAS; NoPE arm | S₁₂(T)/S₁₂(C0) < 0.8; **or** the 2× branch raises S₁₂ by > 5 pp; **or** low needle mass with high entropy | A, unconditional | A length failing G-REF leaves the primary cell. If the 2× branch moves S₁₂ by > 5 pp, the verdict reads "conditional on long-stage budget" and R is recomputed |
| E1.3 | **The task fits inside the GDN state**, so attention is not needed at the tested load | R-load ladder up to N = 1,024 (16K, 32K) and N = 2,048 (32K, N50 only); levels reported as N/N50₀; per-level R | S₀ ≈ S₁₂ at N ≤ N50₀, and the endpoints diverge only above it. H1 then holds trivially at low load | A (CS) | H1-robust (R ≥ 0.8 per informative level) is reported. Recovery is reported only for levels above N50₀ |
| E1.4 | **Shortcuts:** lexical anomaly, copy-the-only-number, position heuristics | D ∈ {3, 15} same-format distractors; binding-error rate; C2 oracle-position control; depth analysis | Collapse at D = 15; a high binding-error rate; C2 ≈ main condition; accuracy tied to depth extremes | A (CS, MS) | The affected level is reported as shortcut-prone and excluded from the per-level interpretation. The confirmatory statistic is unchanged |
| E1.5 | **Ceiling compression:** every k ≥ 1 is near 100% | Ladder levels; N50; per-level R | R ≈ 1 only at easy levels; differences only at top levels | A | Reported with per-level R; H1-robust is required for "robust" wording |
| E1.6 | **Conv/token-shift asymmetry.** GDN layers supply previous-token features that let one attention layer act as an induction head ("smeared keys": [2209.11895](https://arxiv.org/abs/2209.11895) [read-section per A1, B2]), while the all-full reference has no cheap local mixing | X-gc12 (F_g + ShortConv(4) on q/k/v; SiLU on v only) × 2 seeds; X-gc1 and X-gc3 if δ_gc > 3 pp; co-primary if > 6 pp | δ_gc = S₁₂(X-gc12) − S₁₂(C12) > 3 pp, and R₃ under F_gc below R₃ under F_g | A unconditional; B conditional | Operator-level wording only if δ_gc ≤ 3 pp or both families agree. H1 is otherwise reported in both families |
| E1.7 | **Learning speed / exposure:** hybrids learn the TEF formats faster ([Jelassi 2402.01032](https://arxiv.org/html/2402.01032): transformers need "100x less samples" than GSSMs to copy [read-section per B1]) | TEF dose ladder 250 / 500 / 1,000 / 2,000 / 4,000 (seed-0 anchors); still-closing flag; zero-shot Tier 0 (LM-3, LM-4) | The S₁₂ − S_k gap changes by > 3 pp between 1,000 and 4,000 steps; elicited and zero-shot rankings disagree | A | The verdict is "budget-conditional"; the disagreement is reported (I-27) |
| E1.8 | **LR favours hybrids** (gated layers tolerate higher LR; [2505.06708](https://arxiv.org/abs/2505.06708) [abstract-only per A2]) | Minimax-regret LR rule; per-config fallback; gate on both mixers by design; Tier C η*/√2 | The ranking flips at η*/√2 | A rule; C check | Reported as LR-sensitive if Tier C runs and flips |
| E1.9 | **Seed luck / bimodal seeds** (abrupt circuit formation) | n₀ = 5 / 4 seeds; effect-blind +2-seed trigger; success counts with Clopper–Pearson CIs; every seed plotted | Seed range > 12 pp; success counts < n with a two-mode spread | A | The bimodal pattern is reported as a finding. No few-seed test is claimed to be valid under bimodality [PI-SIM] |
| E1.10 | **Metric artefact** (EM_tf thresholding) | L_k on gold log-probability | Sign disagreement | A | The "metric-robust" label is withheld |
| E1.11 | **Item-selection bias in the gate** (winner's curse) | G-REF judged on independent dev items (generator seed 77); test items untouched | — (removed by design) | A | — |

## O2. H1 falsified (L₃ < 0 with all guards passing)

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E2.1 | **Genuine:** 3 of 12 layers do not recover 80% | — | Survives every row below | — | "Falsified, for this scale, GDN-A, gated MHA and recipe" |
| E2.2 | **Learnability under fixed exposure** | Dose guard: TEF-4,000 keeps L₃ < 0, and k = 3 plateaued (\|S₃(4,000) − S₃(2,000)\| < 3 pp) | The gap shrinks with steps; k = 3 still rising | A | Falsification is blocked; the verdict is "inconclusive: learning speed" |
| E2.3 | **Under-trained long stage or base LM** | 2× Stage-B branch at {0, 3, 12}; LM-1 tail slope; Tier C 4× Stage-A tokens | The 2× branch changes S₁₂ by > 5 pp, or hybrids gain more from budget | A (2×); C (4×) | "Conditional on the long-stage budget"; L₃ is recomputed on the 2× branches |
| E2.4 | **Poor placement under the centred rule** | H4 placement at k* (k* = 3 includes the end-of-group pattern {3, 7, 11}) | The best placement recovers far more than the centred one | B | Reported. H1 stays defined on centred placement as pre-registered |
| E2.5 | **LR unfairness against hybrids** | As E1.8 | As E1.8 | A / C | As E1.8 |
| E2.6 | **RoPE large distances under-trained in the few attention layers** | NoPE arm X-nope3 / X-nope12 (seed 0, + seed 1 on trigger); R-dist × filler | Under NoPE the gap shrinks, mainly at large distance d | A / B | Claims are conditional on RoPE; the PE × k interaction is reported |
| E2.7 | **QK-norm dilution hurts a few long-range layers more** | SSMax rule (FEAS); Tier C non-chosen variant at k ∈ {3, 12}; attention diagnostics | Low needle mass at 32K in C3's attention layers; SSMax closes the gap | A / C | Reported as a parameterisation dependence |
| E2.8 | **The reference has more compute per token**, not just more memory | Not separable within budget (I-28) | — | Infeasible | Stated as a limitation |

## O3. H5(a) supported (Δ₀(T) grows with log₂T)

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E3.1 | **Linear-state interference grows with the amount of filler** | Load × length factorial (R-load at fixed N across 8K, 16K, 32K); R-dist with natural vs repetitive filler, analysed per architecture | S₀ falls with T at fixed N and fixed distance; repetitive filler restores S₀ | A (CS, MS) | The mechanism is labelled "load/interference" |
| E3.2 | **Decay over distance** (α_t) | R-dist: fixed T = 32K, d ∈ {256, 2K, 8K, 30K} | S₀ depends on d, not on T or filler | A (MS) | The mechanism is labelled "distance/decay" |
| E3.3 | **Ladder composition** (harder levels entering at larger T) | Fixed level set {R1 D ∈ {3, 15}; R-load N ∈ {8, 32}} at every T | — (removed by design; the debate's re-derivation showed 3.55 / 1.96 pp per doubling for B1's rule) | A | — |
| E3.4 | **Exposure per length** | TEF with equal item counts per log₂ bucket (by design); zero-shot LM-4 copy-after-gap slope | The slope is present after the TEF but absent or reversed in LM-4 | A | The "exposure-robust" label is withheld |
| E3.5 | **Reference collapse or non-monotonicity** at long T | G-REF per length; S₁₂(T)/S₁₂(C0) curve | A non-monotone gap; S₁₂ falls relative to C0 | A | "Conditional on reference adequacy" label; the slope over passing lengths is reported as a sensitivity analysis |
| E3.6 | **Softmax dilution in the reference** at long T | Attention diagnostics; SSMax | S₁₂ falls with T at fixed d; low needle mass | A | Reported jointly with E3.5 |

## O3b. H5(a) not supported (flat or negative slope)

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E3b.1 | **Genuine:** at in-distribution lengths with balanced exposure, the benefit does not grow with T | TOST ±2 pp per doubling | Both one-sided tests pass | A | "No length effect" |
| E3b.2 | **The reference degrades as fast as the all-linear model** | G-REF; per-architecture S(T) curves | S₁₂(T) and S₀(T) fall in parallel | A | Reported as reference degradation, not as an absence of benefit |
| E3b.3 | **Floor for k = 0 at every length** | Per-level S₀ vs chance; log-probability metric | S₀ ≈ floor at every T | A | The slope is uninformative; reported as such |

## O4. H3 supported (R_ret(3) > R_hop(3))

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E4.1 | **Genuine:** composition needs more global layers (≥ h operations; theory prior [2610.25250](https://arxiv.org/abs/2610.25250) [abstract-only per A1]) | Hop ladder h ∈ {1, 2, 3, 4} at fixed k; Tier C k = 2 adjacent {5, 6} vs spread {3, 9} | The required k tracks h; spread beats adjacent at h = 2 | A; C | "Task-dependent requirement (literal multi-hop)" |
| E4.2 | **Multi-hop is simply harder** (difficulty calibration) | Difficulty-equated level pairs (\|ΔS₁₂\| ≤ 10 pp on DG, U-11); C0 controls; log-probability | D₃ vanishes or flips at equated levels | A | "Supported" needs sign agreement at equated levels where they exist |
| E4.3 | **Link order interacts with recurrence** (JRT, [2407.05483](https://arxiv.org/abs/2407.05483) [read-section per PROP]) | M-order forward vs reversed (MS, 16K) | Large order effect for k = 0, small for k = 12 | A (MS) | Reported as a modulator |
| E4.4 | **Specific to the delta rule** | GLA arm (Tier C; promoted under U-13) | The GLA curve shape differs | C | Claims are scoped "for GDN-A" |
| E4.5 | **Scale artefact** of the probability-scale ratio | Fieller intervals; logit GLMM family × k interaction | Interaction present on one scale only, with no crossover | A | Reported with both scales (I-14) |

## O5. Aggregation insensitive to k

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E5.1 | **Not attention-limited at this scale** | A-freq ρ ladder with C0 | C0 is high and every k is equal at every ρ | A | Reported |
| E5.2 | **Floor or ceiling** | ρ ladder down to 1.25 (and 1.1 diagnostic); chance level 0.5 | Every k near 0.5 or near 1.0 | A | The family is uninformative; reported |
| E5.3 | **Delta-rule overwrite** makes counting mechanism-specific | A-count is a diagnostic only; A-freq (relative) is primary; GLA (C) | GDN weak at absolute counts only | A / C | Counting is never used for H3 (I-14) |

## O6. A hybrid beats all-full (reference inversion; "strong H6")

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E6.1 | **Genuine complementarity** (local GDN + global attention) | — | Survives every row below | — | Reported as the reference-inversion outcome with R_best |
| E6.2 | **Conv asymmetry** | X-gc12 | The advantage vanishes once attention has the conv | A | As E1.6 |
| E6.3 | **Temperature parameterisation** (QK-norm dilution) | SSMax (FEAS); Tier C non-chosen variant | The advantage vanishes with log-length scaling | A / C | Reported |
| E6.4 | **Output-gate asymmetry / massive activations** | Gate on both mixers by design; activation-spike diagnostics are exploratory | — (gate asymmetry removed by design) | A | — |
| E6.5 | **LR favours hybrids** | As E1.8 | As E1.8 | A / C | As E1.8 |
| E6.6 | **Long relative positions rarely trained in the all-full model** (STRING, [2410.18745](https://arxiv.org/abs/2410.18745) [read-section per PROP]) | NoPE arm; 2× Stage-B branch; LM-1 per-position loss | NoPE or 2× closes the gap; the k = 12 LM-1 tail is worse | A | Reported |
| E6.7 | **The all-full model learns the TEF formats more slowly** | Dose ladder | The gap closes with TEF steps | A | Budget-conditional |
| E6.8 | **Small-scale all-full deficit.** At 0.22B, all-full NIAH@16K was 35.95 vs 49.58 and 67.81 for sliding-window hybrids ([Qiao 2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2 [read-section, PI-VER in debate]) | Reference inversion with R_best; Tier C scale extension | It persists after E6.2–E6.7 are excluded | A; C | Reported as "at this scale" |

## O7. Placement effect (H4)

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E7.1 | **Depth role** (middle-layer retrieval) | c₁ = middle − mean(ends) | c₁ > 0 | B | Reported |
| E7.2 | **Feature availability at early layers.** Layer 0 has no preceding token mixing; the early arm uses layer 1 to avoid this | Patterns avoid layer 0; Tier C k = 1 at layer {0} | Early is worst only when the arm uses layer 0 | B / C | Reported |
| E7.3 | **The last layer cannot feed later processing** | The late arm includes layer 11; binding-error rate | Late is worst, with a high binding-error rate | B | Reported |
| E7.4 | **Spacing confounded with depth** | Patterns preserve spacing across arms (protocol §3.2) | — (controlled by design) | B | — |
| E7.5 | **Seed noise** | n₀ seeds; TOST ±5 pp | Contrasts within the margins | B | "No placement effect" only via TOST |

## O8. Measured speed-ups differ from analytic ones

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E8.1 | **Kernel maturity:** fla's GDN chunk core runs at ≈ 10% of peak vs 81% for FA2 [MEAS-B, INF] | Roofline-fraction table; FlexAttention comparator ("equal toolchain, not equal maturity") | The Flex vs cuDNN gap measures maturity; the GDN fraction is low | A | Claims are scoped to "this implementation"; analytic costs stay primary |
| E8.2 | **L2 residency** (64 MiB L2: 33 vs 84 µs [MEAS-B]) | Cold vs hot L2 | The cold–hot gap is proportional to L2-resident bytes | A | Both are reported; cold is primary |
| E8.3 | **Launch overhead** (GDN at batch 1) | Per-shape CUDA graphs vs eager; batch sweep | The graph–eager gap is large at small batch and T | A | Graph numbers are primary |
| E8.4 | **Masked-cache SDPA dispatch** (1.80 ms vs 85–90 µs per layer at 32K [MEAS-B]) | Exact-shape graphs without masks; backend pinned and asserted; CI test | Mem-efficient kernel in a trace | A | The run is invalid and must be repeated |
| E8.5 | **Contention** (shared CARLA GPU; up to 28% run-to-run variation [MEAS-B2]) | G-cost idle criteria; min-of-N; 2 sessions | Session-to-session differences > 10% | A | No idle window → appendix only (U-14) |

## O9. LM loss flat while retrieval differs

| ID | Note | Check | Consequence |
|---|---|---|---|
| E9.1 | Not a rival explanation. This is the expected validity pattern (Wang et al.; Qiao et al.) | Report LM-2 and Tier 0 next to the task scores; never select on loss | — |

## O10. Arm differences caused by infrastructure

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E10.1 | **FA2 backward nondeterminism** (k ≥ 1 only [MEAS-B]) | Treated as part of seed variance ([2103.04514](https://arxiv.org/abs/2103.04514) [abstract-only per A2, B2]); deterministic G0 audits; standing bitwise replicate | The replicate differs in audit mode | A | Infrastructure bug; fix and rerun the affected runs |
| E10.2 | **Autotune choices under contention** | Triton cache pre-warmed on an idle GPU and archived (SHA-256) | Cache hash differs between runs | A | Rerun with the archived cache |
| E10.3 | **Scoring-path artefacts** (fla's automatic `fused_recurrent` switch at q_len ≤ 64, `gated_deltanet.py` L227–230 [code-inspected, PI-VER]) | Explicit fla op calls; 5% single-pass re-score | EM flip rate ≥ 1% | A | Rescore |
| E10.4 | **Implementation bugs** in either mixer | Weight-transplant logit tests (Qwen3ForCausalLM; fla GatedDeltaNetForCausalLM); VT-01…VT-20 | Logit mismatch > 1e-3 relative | A (G0) | No training until fixed |

## O11. Elicited (post-TEF) and zero-shot results disagree

| ID | Note | Check | Consequence |
|---|---|---|---|
| E11.1 | Open conceptual question (I-27/U-16). No experiment in budget decides which is "the" answer | Tier 0 (LM-1, LM-3, LM-4) on the pre-TEF checkpoint vs CS after the TEF | Both reported; claims scoped to "capability elicited after identical exposure" |

## O12. H6′: GDN-E (4× per-head d_k at equal parameters) closes the gap, or does not

| ID | Rival explanation | Discriminating experiment / analysis | Signature | Tier | Consequence |
|---|---|---|---|---|---|
| E12.1 | **State capacity, not softmax, limits retrieval** at the tested load | X-E0 vs C{k*} (Welch; TOST ±5 pp); state bytes alongside (6.19 vs 65.55 MiB at D512, 32K) | Δ_E within ±5 pp | B | "A larger recurrent state can substitute at this load" |
| E12.2 | **Head-count change confounds the arm** (1 head vs 4) | Not separable in budget; per-head d_k wording (I-16, B2-W5a) | — | B | Stated as a limitation |
| E12.3 | **GDN-E kernel or numerics differ** | VT-16 (K = 256 / 192 numerics as GDN-A) | VT-16 fails | A (S0) | Fallback: config D at D512 |

## O13. The conv control (F_gc) changes the verdict

| ID | Interpretation | Check | Consequence |
|---|---|---|---|
| E13.1 | The H1 answer depends on token-shift features in the full layers | δ_gc > 3 pp, plus X-gc1 and X-gc3 | H1 and H1-min are reported in both families. Package wording only. Co-primary if δ_gc > 6 pp |

## O14. The NoPE control changes R₃

| ID | Interpretation | Check | Consequence |
|---|---|---|---|
| E14.1 | A position-encoding × k interaction (NoPE helped a 0.22B sliding-window hybrid: 49.58 → 67.81, Qiao Table 2) | \|ΔR₃\| > 0.1 or \|ΔS₁₂\| > 3 pp → NoPE seed 1 | Claims stated as "conditional on RoPE θ = 5e5"; the interaction is reported descriptively |

---

## Status table

Filled in as results arrive. One row per checked rival:

| Row | Checked on (run IDs) | Signature present? | Date | Note |
|---|---|---|---|---|
| — | — | — | — | — |
