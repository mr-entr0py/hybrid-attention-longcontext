# Experimental-design audit (Agent 1): can the frozen protocol answer RQ-A to RQ-D?

**Date:** 2026-11-10, about 20:50–21:40 IST. **Role:** experimental-design auditor (workflow `wf_f382f689-3e6`, resumed after the 17:00 usage limit).
**Mode:** report-only. I wrote this file and scratch files under `.audit/agent1-design/` only. I did not edit anything in `src/`, `scripts/`, `tests/`, `configs/` or `experiments/`, and I committed nothing.
**GPU:** none used. Every number below comes from CPU-only numpy/scipy runs, read-only loads of the frozen item files, or arithmetic on project files.

**Scope audited:** frozen protocol v1.0 (`reports/experimental_protocol.md`), DEV-01 (`reports/reproduction/protocol_amendment.md`), the PI rulings (`reports/pi_rulings_2026-11-10.md`, FINAL; D058–D091), and the analysis code that implements them (`src/hybridlc/analysis/`).

**Read:**
- `reports/STATUS.md`, `continuation_audit.md`, `experimental_protocol.md` (all 20 sections), `decisions.md` (D000–D091, DEV-01…04);
- `pi_rulings_2026-11-10.md` (§0–§10, the residual register), `competing_explanations.md`;
- `reviews/debate_record.md` (§6.2–6.4 in full; I-03, I-04, I-15 and the PI-SIM summary);
- `literature/literature_review.md` §2–§13;
- `reproduction/protocol_amendment.md` and `results_R5.md`;
- `s0_integration.md` §12 and verification rounds 1–3;
- `src/hybridlc/INTERFACES.md`;
- code: `analysis/pipeline.py`, `verdicts.py`, `protocol.py`; `data/tef.py`; `eval/generators.py`; `experiments/prereg/DTS10_acceptance.v2.yaml`; `configs/experiments/plan.yaml`.

**Provenance of the computations.**
- An earlier instance of this audit was interrupted at 17:00. It left five simulation scripts in `.audit/agent1-design/` (15:30–15:41).
- I reviewed their code line by line and re-ran all five. Every output reproduced **byte-for-byte**, because the seeds are fixed (`diff` against `.audit/agent1-design/r2/*.out`: identical).
- My new checks are `.audit/agent1-design/r2/design_checks_r2.py`, `…r2b.py` and `…r2c.py`, with their `.out` files.
- An independent peer review written today, `reports/reviews/continuation_adversarial_review.md` (Agent 5, 15:43; I cite the version committed in `0c54713`, because the file is being revised by a concurrent re-run and its IDs may change), overlaps with several findings here. Where we converge I cite it, and where I give numbers they are from my own runs. §6 maps the overlap.

**Severity scale:**
- **FATAL**: the named claim cannot follow from the design as registered, even if every run succeeds.
- **MAJOR**: a central claim follows only with a serious caveat that is currently unaddressed, or a RQ may go unanswered.
- **MODERATE**: it weakens a control, a decision rule or the stated power.
- **MINOR**: hygiene, wording or a small bias.

Tags: [EVID] evidence (file:line, command output); [INF] my arithmetic or inference; [SPEC] speculation.

---

## 0. Bottom line

The core design is careful:
- matched non-embedding parameters (spread 0.094% / 0.137%);
- identical data, data order and TEF stream;
- common random numbers for the shared modules;
- a paired per-seed contrast in which the seed effect cancels exactly;
- reference-adequacy gating on independent dev items;
- every evaluation length inside the training length;
- exact analytic costs.

**I found no flaw that is fatal to the narrow confirmatory package claim.** That claim is "k F_g layers of 12, the rest GDN-A as built, recover ≥ 80% of the k = 0 → 12 gap on literal key–value retrieval at 16–32K, after an identical task-exposure fine-tune, token- and parameter-matched, at one scale."

But **as registered, the design is unlikely to deliver definitive answers to most of RQ-A to RQ-D**:

| RQ | Answerable as registered? | Main reasons (finding IDs) |
|---|---|---|
| **RQ-A, composition** | **Only in strong-effect worlds.** Otherwise "inconclusive" is likely | The stated power is for unadjusted α, but the rule is Holm-4. Planning σ = 2–3 pp is unevidenced, and the only in-repo seed SD on a below-ceiling recall cell is 4.6–5.6 pp (**D-1**). The 8K fallback can return "supported" when the hybrid is *worse* than k = 0 (**D-2**). The support direction has no guard for a handicapped reference or for slow k = 0 elicitation (**D-6**) |
| **RQ-B, placement** | **No under the default budget.** It is Tier B, and the D043(3) default is "Tier A only, then ask". If funded, only large effects can be resolved | At k* ∈ {2, 3} the arms are ±2- or ±1-layer shifts. The MDE is 8.6 pp at the registered Holm-4 step, not the stated 7.3 pp. "No effect" is reachable only at σ ≈ 2 (**D-4**) |
| **RQ-C, task dependence** | **Retrieval vs multi-hop: often not evaluable as registered** (fixable before G5). **Aggregation: not as "integration"** | The per-seed evaluability rule gets *less* likely to pass as seeds are added (**D-3**). The H3 null is unreachable (Agent 5 F-2, confirmed). A-freq at 32K is largely solvable by recency or last-4K counting (**D-5**) |
| **RQ-D, cost** | **Yes for analytic state, FLOPs and bytes.** Measured costs are implementation-scoped | H6 is an identity. Training compute is not reported (**D-11**). H6′ (GDN-E) is underpowered and confounded (**D-13**) |

**DEV-01 and the PI rulings** greatly strengthen explanation (6), implementation and kernel artefacts, through the reproduction ladder, FEAS-PREC v2 and the k-differential residual register. DEV-04 (TEF `answer_norm: block`) restores the length balance that H5(a) relies on. None of them changes statistical power, the placement design or benchmark validity.

**Changes that must be made before FEAS:** a decision on the noisy FEAS-384 rule, and on the single-seed FEAS-512 rule if that path runs (**D-7**, **D-8**). Everything else can wait until G5, at zero GPU-h for most items (§4).

---

## 1. Severity-ranked findings

| ID | Sev. | Area | Finding | Evidence | Consequence if unaddressed | Recommendation · class · cost |
|---|---|---|---|---|---|---|
| **D-1** | **MAJOR** | Seeds and power (RQ-A) | **(a)** The protocol's operating characteristics (power 0.88 at n = 5, 0.967 for the 5→7 rule, σ = 3 pp, G = 30 pp) are from PI-SIM at **unadjusted one-sided α = 0.05**. The support rule is Holm over {H1, H1-min, H3, H5a}.<br>**(b)** Under the registered rule (H5a rejected first, H3 not evaluable, adaptive 5→7), P(H1 supported \| R₃ = 1.0) is **0.85 / 0.50 / 0.44** at (G, σ) = (30, 3) / (20, 3) / (30, 5), against 0.97 / 0.76 / 0.70 unadjusted. At fixed n = 5 and the typical Holm step α/3 it is **0.63** at (30, 3). Type I at R₃ = 0.8 is 0.016–0.019, so the rule is conservative and part of α is wasted.<br>**(c)** Under the PI's own prediction R₃ ∈ [0.6, 1.1], P(no H1 verdict) is **≥ 0.41** (G 30, σ 3), ≥ 0.60 (G 20, σ 3) and ≥ 0.63 (G 30, σ 5).<br>**(d)** σ = 2–3 pp is [INF]. The only in-repo seed SD on a below-ceiling recall cell is **4.55 / 5.57 pp** (R5, 1024/256 cell, n = 3; 95% CI 2.4–29 pp). That is a different regime, so this is weak evidence, but it does not support σ ≤ 3.<br>**(e)** The effect is L₃ = G·(R₃ − 0.8), so it shrinks with G. At G = 15 pp (the G-REF minimum) and σ = 3, d = 0.77 at R₃ = 1.0 | Protocol L67, L675–679; debate L999 ("All tests are one-sided at α = 0.05"); D020 (L27); `r2/sim_h1_holm.out`, `r2/design_checks_r2.out` §D, `r2b.out` §H; `results_R5.md` L29, L49; `r2c.out` §J | The headline H1 is reported "inconclusive" much more often than the protocol advertises. The n₀ budget rests on overstated power | Restate the operating characteristics under the registered rule over G ∈ {15, 20, 30} and σ ∈ {2, 3, 5}, and extend VT-18 to reproduce them.<br>Decide the family structure by PI row:<br>• weighted Holm (H1 .5, H5a .3, H1-min .1, H3 .1): 0.945 vs 0.853 at (30, 3);<br>• fixed sequence: 0.967;<br>• or move H1-min and H3 out of the support family.<br>Budget the U-05 seeds as the expected case (D-14).<br>**before-G5 · 0 GPU-h** |
| **D-2** | **MAJOR** (conditional on the fallback) | R(k) / L_k interpretation | If neither 16K nor 32K passes G-REF, the primary cell is 8K **with no G-REF at 8K**. L_k ≥ 0 ⇔ R ≥ 0.8 holds only for G > 0. For G < 0 the sign meaning reverses.<br>Example: S₀ = 55, S₁₂ = 50, S₃ = 52 gives L₃ = +1 although R₃ = 0.6, and the hybrid is below the all-linear model.<br>Simulated (n = 5, σ = 3, Holm α/3): P("H1 supported") = **0.20** at G = −10 pp, R₃ = 0.5 (hybrid worse than k = 0), and 0.10 at G = +5, R₃ = 1.2.<br>The code blocks *falsification* at the fallback but not *support* | Protocol L59–62, L642; `analysis/pipeline.py` L311–316 (`select_primary`), L802 (`gref_ok` requires no fallback, so only falsification is blocked); `verdicts.py` (no G-sign check); `r2/design_checks_r2.out` §E | A "supported" verdict whose meaning is the opposite of H1, labelled only "not long-context" | At 8K apply G-REF (a) and (b). If 8K fails, H1 and H1-min are "not evaluable" (p = 1). Add a unit test with G < 0.<br>**before-G5 · 0 GPU-h** |
| **D-3** | **MAJOR** | H3 (RQ-C), multiplicity | H3 is evaluable only if **every** seed has G ≥ 15 pp in both families. At σ = 3 and G_hop = 20 pp, P(evaluable) is **0.53 at n = 5 and 0.41 at n = 7**: adding seeds makes H3 *less* evaluable, and the U-05 trigger adds seeds about 80% of the time at σ = 3. A non-evaluable H3 still takes a Holm slot (p = 1), which lowers H1's threshold from α/2 to α/3.<br>A pooled-denominator alternative (seed-pooled G ≥ 15 pp in both families, every seed G > 5 pp, per-seed numerators over pooled G) gives evaluability 0.995 / 0.998. Support at D₃ = 0.3 rises to **0.67 / 0.90** (from 0.41 / 0.39). Type I is 0.013 (vs 0.011).<br>Convergent with Agent 5: the H3 null (TOST ±0.10) is unreachable (P ≤ 0.07–0.31 under a true null at σ = 3), and the copy-vs-compose and load confounds remain | Protocol L70, L709; `pipeline.py` L544; `r2/sim_h3_alt.out`, `r2/sim_secondary.out` (H3 block); Agent 5 F-2, M-4 | The confirmatory test of RQ-C is likely to be skipped. Its non-evaluability also costs H1 power | Adopt the pooled-denominator evaluability rule, or drop a non-evaluable H3 from the Holm family by a pre-registered rule. Declare the H3 null unreachable; report an estimate with CI instead.<br>**before-G5 · 0 GPU-h** |
| **D-4** | **MAJOR** | Placement (RQ-B) | **(a)** RQ-B is Tier B only. Under the default budget (Tier A, then ask; user unanswered) it is not run.<br>**(b)** With spacing preserved, the arms are **{1,5,9}/{2,6,10}/{3,7,11}** at k* = 3 (shifts of ±1 layer) and {1,7}/{3,9}/{5,11} at k* = 2 (±2). Only k* = 1 ({1}/{6}/{11}) is a real early/middle/late contrast. k* = 3 is also the default when no k qualifies.<br>**(c)** The stated MDE of 7.3 / 4.9 pp (n = 5) was computed at α = 0.025 (Holm over 2). The registered secondary family has 4 members, so the first step is α = 0.0125. The MDE becomes **8.6 pp** (n = 5, σ = 3) and 11.7 pp (n = 4). At a true 7.3 pp the power is 0.60.<br>**(d)** "No placement effect" (TOST ±5 pp, both contrasts) passes with P = 0.29 at σ = 3 and 0.01 at σ = 5 under a true zero | Protocol L131–137, L716, L724; debate L1072–1076; D043(3); `r2/sim_secondary.out` (H4 block) | RQ-B is likely unanswered. A k* = 3 null would be over-generalised into "placement doesn't matter" | (i) Restrict placement wording to the shift tested ("±1 layer at k* = 3").<br>(ii) Restate the MDE at Holm-4.<br>(iii) Estimation-first reporting of c₁ and c₂.<br>(iv) Optional: clustered arms at k* = 3 ({1,2,3} vs {9,10,11}), 2 arms × n₀, ≈ 19 GPU-h at D384.<br>Ask the user for the Tier-B decision early.<br>**before-G5 · 0 h (optional 19 h)** |
| **D-5** | **MAJOR** (FATAL for an "aggregation is insensitive to k ⇒ integration needs little attention" reading) | Benchmark validity (RQ-C aggregation) | On the frozen CS items at **32K**:<br>• recency heuristic: 0.791 / 0.667 / 0.613 / 0.534 (ρ = 4 / 2 / 1.5 / 1.25);<br>• last-4,096-token counter: **0.926 / 0.729 / 0.641 / 0.537**;<br>• last-8,192: 0.987 / 0.832 / 0.694 / 0.595.<br>At ≤ 4K, a last-1,024 counter solves ρ = 4 at 0.99–1.00. The chance level is 0.5. Only ρ ≤ 1.5 requires counting over more than 8K tokens | `r2/afreq_floors.out` (re-run of the project scorer's heuristics on `experiments/items/test/CS/A-freq_*`); `s0_integration.md` L359, L395; Agent 5 F-3 | A k-insensitive A-freq result cannot be separated from "every model uses a short-window heuristic". RQ-C's "information integration" is weakly measured | Pre-register floor-relative wording: integration claims only above the last-8K floor at ρ ≤ 1.5. Pre-register a heuristic-adversarial subset (queries where recency and last-4K counting are wrong).<br>Add recency- and window-balanced A-freq cells **in the VT-24r regeneration that already has to happen before FEAS** (one regeneration, not two).<br>**before-G5 (cheapest before FEAS) · eval ≈ 0.05–0.1 GPU-h per model** |
| **D-6** | **MAJOR** | Explanations (1) vs (3)/(4) | The support direction has no guard comparable to the falsification guards:<br>• no flag for an adequate-but-beaten k = 12 reference;<br>• R_best ≡ R by default, because DG is scored on C0/C12 only;<br>• the still-closing flag tracks S₁₂ − S_k but not k = 0;<br>• every confound check is 1-seed (2× branch, dose, SSMax, NoPE) or 2-seed (X-gc12) | Protocol L383, L651–653, L693–703; `pipeline.py` L223–247, L861–889; Agent 5 M-1, M-3, M-7 (I concur) | "Supported" can partly reflect a handicapped reference or a slowly elicited k = 0 | Score DG on all core configurations; add a "reference exceeded" label (a hybrid > S₁₂ by ≥ 5 pp, paired p < 0.05); extend the still-closing flag to L_k and R(k) between TEF 1,000 and 4,000 (data already planned).<br>**before-G5 · ≤ 1 GPU-h (DG on C1/C3)** |
| **D-7** | **MODERATE** | FEAS decision noise (scale) | The rule is D384 iff **both** seeds have S₁₂ − S₀ ≥ 25 pp at 32K, on FD cells of 100 × 4 (gap item SE ≈ 1.9 pp).<br>P(adopt D384) at σ = 3 is **0.25 / 0.55 / 0.74 / 0.92** at a true gap of 25 / 28 / 30 / 33 pp. A seed-mean rule gives 0.50 / 0.82 / 0.94 / 0.99. The price is P(adopt \| true 20 pp) of 0.06 instead of 0.02.<br>A noise-driven escalation costs ≈ +58 GPU-h base (Tier A 147.3 vs 89.1 h). The adequacy clause (S₁₂ ≥ 0.85·S₁₂(C0) in both seeds) can only lower these figures | Protocol L833; `DTS10_acceptance.v2.yaml` L33; DEV-01 part C; `r2/design_checks_r2.out` §A–B, `r2c.out` §I | A 26–45% chance of a D512 path driven by noise when D384 is adequate (true gap 28–30 pp) | PI row **before FEAS**: keep the conjunction or switch to the seed mean (a new prereg version, since FEAS has not run). Either way, state the rule's operating characteristic in the FEAS report so that an escalation is not read as "D384 too small".<br>**must-before-FEAS · 0 GPU-h** |
| **D-8** | **MODERATE** | FEAS-512 path | The D512 path decides 32K vs 16K vs STOP on **seed 101 only**. P(gap clause passes, ≥ 20 pp) at σ = 3 is 0.50 / 0.67 / 0.86 at a true 20 / 22 / 25 pp | `DTS10_acceptance.v2.yaml` L34; plan.yaml L224–237; `r2b.out` §G | A STOP or an L_ext = 16K re-scope driven by one noisy seed | If FEAS-512 runs, add seed 102 for k ∈ {0, 12} on the selected branch (≈ 7 GPU-h [INF from §13.2 rates]), or pre-state the noise.<br>**before FEAS-512 (conditional) · ≈ 7 GPU-h** |
| **D-9** | **MODERATE** | PE controls (explanation 3) | NoPE runs only at seed 0. The U-06 trigger (\|ΔR₃\| > 0.1 or \|ΔS₁₂\| > 3 pp) fires **53 / 74 / 89%** of the time under *no* effect (run-level noise 2 / 3 / 5 pp). So it adds a seed on noise and cannot exclude a PE × k interaction ≲ 8 pp. GDN's implicit position source (conv and decay) has no control; GDN-nc is Tier C | Protocol L319, L733; `r2/sim_secondary.out` (trigger block); Agent 5 m-1 | "Conditional on RoPE" is the only honest wording; the PE rival is excluded only if it is large | Make NoPE seed 1 unconditional at k ∈ {3, 12} and report the difference-in-differences with CI (X-nope3 1.90 h + X-nope12 2.21 h ≈ 4.4 GPU-h at D384 with evaluation), **or** keep the trigger and pre-state the detection limit.<br>**before-G5 · ≈ 4.4 GPU-h (optional)** |
| **D-10** | **MODERATE** | Conv control and multiplicity | **U-02** (δ_gc > 3 pp, mean of 2 seeds): false-fire 0.16 and miss 0.25 at a true 5 pp (σ = 3).<br>**Co-primary** (δ_gc > 6 pp) has **no defined combination or multiplicity rule**: the code sets only a flag, and the F_gc family has its own α = 0.05 Holm. If "supported in either family" were read as support, the FWER would approach 2α | Protocol L730–733; `verdicts.py` L160–180; `pipeline.py` L1154–1159; `r2/sim_secondary.out`; Agent 5 M-8 (2-seed F_gc agreement) | Ambiguous confirmatory claim if U-02 fires high | Pre-register: the package claim stays F_g. If co-primary, H1 is "supported" only if both families support (or Holm over 8). Operator-level wording requires an informative F_gc family (Agent 5 M-8).<br>**before-G5 · 0 GPU-h** |
| **D-11** | **MODERATE** | Token and compute fairness | Tokens are matched exactly per stage (VT-11). Compute is not:<br>• training PFLOP/run is **136 (C0) vs 189 (C3) vs 347 (C12)** at D384 (2.55×), and 360 vs 890 at D512;<br>• inference at 32K is 59.1 vs 357.1 MFLOP/token (6.0×) at D384.<br>No arm separates "unbounded memory" from "more compute"; the pathway diagnostics are exploratory and post hoc | Protocol L184–201 (Train PFLOP/run), L258, L503; I-28 | Claims answer a token- and parameter-matched question, not a compute-matched one. RQ-D omits training cost | Word every claim "token- and parameter-matched, not compute-matched". Report training PFLOP and measured GPU-h per config beside the inference costs in RQ-D/H6.<br>**before-G5 · 0 GPU-h** |
| **D-12** | **MODERATE** | Explanation (6): training precision | RR-R5-power: only a bf16 GDN *training* handicap larger than 0.115 is excluded. It is k-differential, and nothing before S4 covers it. FEAS-PREC v2 covers scoring precision only | `pi_rulings` §7 L684; D079; `results_R5.md` L25–30 | An unknown downward bias on S₀ and an inflated G, which also affects the H5(a) slope if the handicap grows with T [SPEC] | Optional control: S3 recipe (0.15B tokens, 4K, seed 100), k ∈ {0, 12}, bf16 vs no-autocast fp32. Compare LM-2 and LM-3 (Δ ∈ [1K, 4K)) against a pre-stated 0.01-nat margin.<br>**optional · ≈ 2–3.5 GPU-h [INF: fp32 throughput unmeasured]** |
| **D-13** | **MODERATE** | Explanation (2): state capacity | GDN-E (H6′) has 2 seeds. TOST ±5 pp declares "equivalent" with P = **0.24** under a true Δ = 0 (σ = 3; 0.50 at σ = 2). The arm changes per-head d_k, head count and state bytes together (≈ 2.8× at D384, 3.7× at D512) | Protocol L75, L160, L718–721; `r2/sim_secondary.out` (H6′ block); Agent 5 F-1 | The capacity rival is not separable from a genuine benefit | Estimation-only H6′, **or** +3 X-E0 seeds (≈ 6.2 GPU-h at D384). Word it as "≈ 3× recurrent state, one head".<br>**before-G5 · 0 or ≈ 6 GPU-h** |
| **D-14** | **MODERATE** (cost) | Seed trigger U-05 | P(fire) is **0.15 / 0.51 / 0.80 / 0.98** at σ = 2 / 2.5 / 3 / 4 (D384, n₀ = 5), so the expected cost is +2.3 / 7.8 / 12.2 / 14.9 GPU-h. The range clause uses raw scores, which include the shared seed effect τ. τ cancels in the paired test but still fires the trigger: at σ = 2 and τ = 4.5, P(fire) = **0.69** (vs 0.15 at τ = 2) | Protocol L681–685, L817; `r2/design_checks_r2.out` §C, `r2b.out` §F | Tier A is ≈ 101 h base rather than 89 h at σ = 3. Seeds may be bought for variance that does not affect the test | Compute the range clause on seed-centred scores. This keeps bimodality detection, because a k-specific mode survives centring. Budget n₀ + 2 as the expected case.<br>**before-G5 · 0 GPU-h** |
| D-15 | MINOR | Placement selection bias | k* is chosen from the same CS test scores later reused as the middle arm (`select_kstar(ret)`). When k* = 1 is selected, the c₁ bias is +2.3 pp at a true R(1) = 0.2 and +0.9 pp at 0.3. The conditional type I is 0.047 vs a nominal 0.0125. Unconditional inflation is small (P(k* = 1) = 0.04 / 0.50) | `pipeline.py` L487, L916–925; `r2/sim_secondary.out` (last block) | Small positive bias in c₁ | Choose k* on DG (dev) items; disclose the residual seed-level bias.<br>**before-G5 · 0 GPU-h** |
| D-16 | MINOR | G-REF selection on the same seeds | G-REF uses dev items but the same trained models. Near the threshold the falsification test becomes slightly anti-conservative: 0.073 at true G = 15 and 0.083 at G = 14, against 0.05 (0.052 at G = 20) | `r2/sim_secondary.out` (G-REF block); protocol L640 | Rarely matters, because falsification also needs the guards | Sensitivity analysis: falsification also computed with the primary length fixed a priori (both 16K and 32K).<br>**before-G5 · 0 GPU-h** |
| D-17 | MINOR | Train/eval format | TEF queries are **sequential** (queries 2–4 follow earlier Q/A pairs); evaluation queries are **branched** from the context snapshot. The difference is identical across k, but its effect could differ by k [SPEC] | Protocol L377, L450; `data/tef.py` L29–30; `eval/generators.py` L16; Agent 5 m-8 | Unknown-sign k-differential shift | Report EM by TEF query index on a TEF-format dev cell (exploratory).<br>**optional · ≈ 0.05 GPU-h per model** |
| D-18 | MINOR | Text consistency | §19 gives U-07's default as "k* = 1"; §3.2 and the code (`KSTAR_DEFAULT = 3`) say k* = 3 when none qualifies | Protocol L1029 vs L131; `analysis/protocol.py` L132–134 | Confusion at the Tier-B decision | Erratum: "k* = 3 if none qualifies; the Tier-B budget is priced at k* = 1".<br>**before-G5 · 0** |
| D-19 | MINOR | FEAS-5 (SSMax) | The SSMax adoption is decided on one seed (101), with thresholds of +3 / −2 pp | Protocol L327; plan.yaml L215–218, L246–247 | A near-random adoption decision (identical across arms, so k-contrasts are unbiased) | Optional SSMax arm on seed 102 (≈ 1.3 GPU-h: plan.yaml rows 0.69 + 0.20 + 0.43).<br>**optional, before FEAS** |
| D-20 | MINOR | U-15 | Doubling contexts barely helps: item SE falls from 0.54 / 0.76 to 0.38 / 0.54 pp against σ ≥ 2–3 pp | `r2/design_checks_r2.out` §A; protocol L486 | ≈ +2 h of evaluation for negligible power | Keep (cheap), but do not count it as power |

**Checked and found sound (no finding):**
- **Parameter counts.** VT-01 is MET (`pi_rulings` L158). I re-derived the GDN-A count 1,056,904 (D512) / 595,334 (D384) from fla's projections: q/k d×256, v/g/o d×512, a/b d×4, A_log/dt_bias 4 each, conv (256 + 256 + 512)×4, o_norm 128.
- **Item SE claim.** The protocol states 0.5–0.7 pp; I get 0.54 pp for two lengths and 0.76 pp for one, at p = 0.5 (`r2` §A).
- **PI-SIM.** It is reproduced within Monte Carlo error: for example, PAIR n = 5, σ = 3 gives 0.878 vs 0.880, and ADAPT 5→7 gives 0.967 vs 0.967 (`r2/sim_h1_holm.out` sanity block).
- **TEF length balance.** It is restored by DEV-04 (1/7 of the answer weight per bucket; D063).
- **Shared modules.** They are bit-identical across k (VT-10).
- **Evaluation lengths.** All are within the training length.

---

## 2. The six explanations: can the design separate each from a genuine full-attention benefit?

Explanation (2) is a *mechanism* by which full attention helps, not an artefact. The rival reading is "a larger recurrent state would do as well".

| # | Explanation | Separable from (1)? | Arm / diagnostic that separates it | Missing or weak |
|---|---|---|---|---|
| 1 | **Genuine full-attention benefit** (package claim: gated MHA as built vs GDN-A as built) | It is the target. "Genuine" is licensed only after rows 2–6 | H1 and H1-min (L_k, paired per seed, at the G-REF-selected 16K/32K primary cell); H2 curve (descriptive); H5(a); reference inversion; R_best | Power under Holm-4 at realistic σ and G (D-1); the 8K fallback (D-2); no support-direction guard for a handicapped reference (D-6). Several of rows 2–6 are excluded only for large effects |
| 2 | **Recurrent-state capacity limit** | **Partly** | • Load dependence: R-load N ∈ {8…1,024} at 16K/32K plus N = 2,048 at 32K, with N50 and N/N50₀ (CS, every model).<br>• Load vs length: fixed N across 8K/16K/32K (CS).<br>• Distance vs interference: R-dist × filler (MS; C0/C12 seeds 0–1, C1/C3 seed 0).<br>• Larger state: GDN-E X-E0 (Tier B) | GDN-E has 2 seeds (TOST power 0.24 at σ = 3), is confounded with head count, and is unfunded by default (D-13). No query-first diagnostic separates "cannot store all N" from "does not know what to store": an evaluation-only R-load-query-first cell at 32K, ≈ 0.05 GPU-h per model, though out of format for the TEF [SPEC] |
| 3 | **Positional-encoding effect** | **Partly (large effects only)** | X-nope3 and X-nope12 at seed 0 (+ seed 1 on trigger); RoPE θ = 5e5, justified to 32K (Llama 3 §3.2 per protocol L316); all lengths in range | One seed, and the trigger fires on noise 53–89% of the time (D-9). PE is structurally confounded with k: k = 0 has no explicit PE, while k = 12 has only RoPE and no conv. GDN's implicit position (conv, decay) is uncontrolled outside Tier C GDN-nc |
| 4 | **Training data / budget** (TEF exposure, Stage-B budget) | **Partly** | • Identical TEF stream and DEV-04 length-balanced answer weight.<br>• TEF dose ladder: 250/500/1,000 for all runs (DC 100 × 4) and 2,000/4,000 for seed-0 anchors, with the still-closing flag.<br>• 2× Stage-B branch at {0, 3, 12}, seed 0 (saturation guard).<br>• FEAS-3 budget rule; LM-4 zero-shot corroboration for H5(a); Tier C 4× Stage-A | Exposure-matched is not learning-speed-matched. All guards are single-seed. The still-closing flag ignores k = 0 (D-6). FEAS-3 is decided on k = 12 only, with 2 seeds. There is no TEF-LR sensitivity arm. Stage B is thin: 30/60/90M tokens = 916 / 1,831 / 2,747 windows of 32K, ≈ 60% PG19 (protocol L294–298) |
| 5 | **Evaluation shortcut** | **Yes for retrieval; partly for multi-hop; no for aggregation** | • Same-format distractors D ∈ {3, 15}; binding / absent / stale error types; rank accuracy.<br>• C0 (no filler) and C2 (oracle position); depth and distance recorded.<br>• T4 template slice.<br>• VT-24/VT-24r InductionLM: retrieval is literal induction copy by design, with M-hop as the negative control.<br>• The M-hop "nearest value fact" heuristic is at chance after the round-1 fix (`s0_integration.md` L390) | A-freq heuristic floors (D-5). Retrieval is literal-prefix copy, so the claim is about associative recall, not semantic retrieval (Agent 5 M-2). Sequential-vs-branched query shift (D-17). The M-hop chance floor is 25% (4 chains), which compresses G_hop and feeds D-3 |
| 6 | **Implementation / kernel artefact** | **Yes for model equivalence and scoring precision; partly for training precision** | • VT-01…VT-25 with named residuals (D090).<br>• VT-09/22/22b transplants: C12 ↔ `Qwen3ForCausalLM`, C0 ↔ fla.<br>• R4/R5 (Zoology); R6 known-answer pipeline.<br>• FEAS-PREC v2 Q1–Q4: a between-k rule with a P4 fp32 cache path, plus R-ATT.<br>• VT-13r and F-infra (resume exposure by k); TC-PF | RR-R5-power: a training-precision handicap < 0.115 is not excluded (D-12). Measured costs are implementation-specific (analytic primary, which is correct) |

---

## 3. Topic-by-topic audit

### 3.1 Isolation of attention composition

- **Isolated.** Only `Mix_{a_l}` differs (protocol L106). Shared modules are bit-identical per seed (VT-10), and data, order, TEF, optimiser and schedule are identical. The paired contrast cancels the shared seed effect exactly, since 1 − 0.2 − 0.8 = 0 (`r2b.out` §H).
- **Not isolated, and inherent to the README question:**
  - Adding a full layer removes a GDN layer (recurrence, conv, decay).
  - The two mixers differ in four ways as built:
    - conv on q/k/v in GDN only, controlled by X-gc12 (2 seeds);
    - L2-normalised q/k in GDN vs QK-RMSNorm with RoPE in F_g, with no control;
    - implicit position/recency in GDN, with no control;
    - per-token compute, with no control (D-11).
  - This is why D040 restricts claims to package wording. That restriction is correct and should be kept.

### 3.2 Parameter matching and allocation

- **Matching.** Non-embedding N(k) = 41,007,200 − 3,208k (D512) and 21,309,384 − 2,438k (D384) (protocol L203–206). These are verified (VT-01). The embedding is 17.0% (D512) / 22.8% (D384) of total parameters, identical across k.
- **Allocation [INF].** At equal parameters the mixers spend them differently.
  - F_g has full-width Q and K (d each).
  - GDN-A has half-width Q/K (256 at D512) but a full-width output gate W_g (d×d, ≈ 25% of the mixer) plus conv and decay.
  - This is the "as built" comparison. It is not a confound for package claims, but it is another reason operator-level wording ("softmax memory vs delta-rule memory") is unsupported.
- **Tokens per non-embedding parameter.** About 31 (D384) and 24 (D512) before the TEF (protocol L208), so neither model is starved relative to a Chinchilla-style ratio.

### 3.3 Token and compute fairness

See D-11. The consequences:
1. **Wording.** The answer is to "how much full attention at equal data and parameters". A compute-matched comparison would give k = 0 about 2.5× the tokens.
   - [SPEC] For high-load retrieval, where k = 0 is plausibly state-limited (PI-SIM delta-rule cosine 0.84 → 0.23 from N = 32 to 512 for 4 × 64 heads, protocol L233–236), more tokens are unlikely to close the gap.
   - For low-load levels and LM loss they might.
2. **RQ-D.** The H6 hull margin is an identity for any per-layer-additive cost. Training cost is per-layer additive at fixed T, so it should be reported beside inference cost.
3. **Exposure to infrastructure faults differs by k.** k = 12 Stage B runs 2.2× longer (protocol L258), while the hangs concentrate in GDN workloads (RR-Xid). Both are handled by F-infra and VT-13r.

### 3.4 Position-encoding controls

See D-9. Additional points:
- The literature leans towards NoPE on the full layers of hybrids (Qiao: NoPE raised NIAH@16K from 49.58 to 67.81 at 0.22B; Kimi; per `literature_review.md` §7). So the PE × k interaction is *a priori* plausible. A one-seed arm detects it only if it is large (≈ ≥ 8 pp, per Agent 5's estimate and the trigger noise above).
- RoPE θ = 5e5 is well justified for 32K. A θ sweep is not needed.
- Every evaluation length is inside the training length, which removes the extrapolation confound that dominates prior hybrid-vs-Transformer comparisons (Bae: Transformer NIAH ≈ 0 beyond 8K, per literature review §3).

### 3.5 Placement methodology (k* rule)

See D-4 and D-15.
- The rule (the smallest k ∈ {1, 2, 3} with seed-pooled R̂_ret(k) ≥ 0.3) sensibly avoids a floor triad.
- R̂(2) comes from Tier-B C2 with 2 seeds, which is noisy. When k* = 2, the "centred top-up" adds fresh seeds, which also reduces selection bias.
- Spacing preservation removes the spacing-vs-depth confound (E7.4) but compresses the depth manipulation at k ≥ 2.
- Placement at a single k cannot answer "independently of how many" in general. Word it as "at k*".

### 3.6 Context-length exposure

**Stage-B length** (D384, from the protocol §3.1 and §6.3 rows):
- 229 / 458 / 687 steps × 131,072 tokens = 30 / 60 / 90M tokens;
- that is 916 / 1,831 / 2,747 sequences of 32K, of which 60% are PG19 32K windows (≈ 550 / 1,100 / 1,650);
- these, plus the TEF, are the only exposure of attention to relative positions above 4K. The literature on left-skewed position frequencies (STRING, per literature review §3) makes a handicapped long-range reference plausible.
- G-REF adequacy (S₁₂(T) ≥ 0.8·S₁₂(C0)) excludes *collapse*, not *handicap* (D-6).

**TEF length balance:**
- equal item counts per bucket (907 per bucket per 1,000 steps; `tef.py` L13–14);
- with DEV-04, equal answer weight per bucket (1/7);
- pretraining exposure is length-skewed (Stage A at 4K) but identical across k.

[SPEC] This may still interact with architecture, because attention must learn long RoPE distances while GDN is length-agnostic. That would bias the H5(a) slope *downwards* at 32K if the k = 12 reference is exposure-limited. The LM-4 corroboration and G-REF per length are the existing checks.

### 3.7 Seeds and power (PI-SIM check)

**PI-SIM is reproduced** within Monte Carlo error under its own assumptions: τ = 2, σ ∈ {2, 3}, G = 30, unadjusted α (`r2/sim_h1_holm.out` sanity block).

**The issues** are its use at unadjusted α (D-1) and its σ and G assumptions:
- σ: R5 gives 4.6–5.6 pp on a recall cell, with n = 3 and a wide CI.
- G: G-REF admits G down to 15 pp, FEAS-384 targets ≥ 25 pp and G-FULL ≥ 20 pp.

**Per-seed SD of L₃** is 1.296σ. The standardised effect at R₃ = 1.0 is d = 0.2G/(1.296σ): 1.54 at (G 30, σ 3), 1.03 at (G 20, σ 3) and 0.93 at (G 30, σ 5) (`r2b.out` §H).

**Bimodality.** No few-seed test is valid under bimodal seeds (protocol L679). Success counts are the right supplement.

### 3.8 Recovery metric R(k) and L_k

- L_k = G·(R − 0.8), so the test's effect size scales with G (D-1e), and the sign meaning reverses for G < 0 (D-2).
- R on the ladder mean is the G-weighted average of per-level R. Levels where k ≥ 1 is at ceiling and k = 0 fails push R towards 1. That makes H1-robust (R ≥ 0.8 per informative level) the right secondary check (E1.5).
- Floors enter S₀ and are differenced out only when k = 0 sits at the floor:
  - R1 D = 3 copy-any-in-context-value floor: 25%;
  - M-hop chain floor: 25%;
  - A-freq: recency floor, not 0.5.
- H3's per-seed ratios are noisy when per-seed G is near 15 pp: SD(R) ≈ 0.36 per seed at G = 15, σ = 3 [INF, from Var(R̂) ≈ (σ²/G²)(…), per review A1 L707–709]. Hence D-3's pooled-denominator recommendation.

### 3.9 Multiple comparisons

Families as registered:
- support Holm-4, one-sided;
- falsification Holm-2;
- secondary Holm-4, two-sided;
- exploratory BH q = 0.10;
- per-level Holm.

Issues:
1. Power is not restated under Holm (D-1).
2. A non-evaluable H3 consumes a slot (D-3).
3. The co-primary F_gc combination is undefined (D-10).
4. The placement MDE was computed for Holm-2 although the family is Holm-4 (D-4c).

The sensitivity "downgrade to suggestive" and the H3 equated-level conjunction only reduce power. They do not inflate type I.

### 3.10 Benchmark validity

- **Retrieval** (R1, R-load) is a valid literal associative-recall benchmark:
  - same-format distractors;
  - an exact token-space construction;
  - no separator inside contexts;
  - answerability verified by an order-6 induction oracle (VT-24/24r).
- **Multi-hop** is valid as templated pointer-chasing (see Agent 5 M-4 for copy-vs-compose).
- **Aggregation** fails as long-range integration at ρ ≥ 2 (D-5).
- **Elicited capability.** The TEF trains on the evaluation generators: template T1 is shared, while keys, values and words are disjoint (VT-12). Evaluation therefore measures "capability elicited after identical exposure" (D037). The scope wording already reflects this.

---

## 4. Recommendations, cost and classification

GPU-h are D384-path figures from protocol §13.2 and `plan.yaml` rates unless noted. "0" means analysis or wording only.

| # | Recommendation | Addresses | Class | GPU-h | Consequence |
|---|---|---|---|---|---|
| R-1 | PI row on the FEAS-384 U-01 rule: keep the both-seeds conjunction or switch to the seed mean. State the operating characteristic either way (new prereg version before FEAS) | D-7 | **must-before-FEAS** | 0 | Avoids a 26–45% chance of a noise-driven +58 GPU-h D512 path when D384 is adequate; P(adopt \| true 20 pp) rises from 0.02 to 0.06 under the seed mean |
| R-2 | If FEAS-512 runs, add seed 102 (k ∈ {0, 12}, selected branch) before deciding 32K / 16K / STOP | D-8 | **must-before-FEAS-512** (conditional) | ≈ 7 [INF] | A STOP or re-scope rests on 2 seeds, not 1 |
| R-3 | In the VT-24r regeneration that is already required before FEAS, add recency- and window-balanced A-freq cells, plus window-4K/8K floor columns | D-5 | before-G5 (cheapest before FEAS) | eval ≈ 0.05–0.1 per model | Makes "integration" claims testable; one regeneration instead of two |
| R-4 | Restate H1/H1-min/H3/H5a operating characteristics under the registered Holm-4 rule (G ∈ {15, 20, 30}, σ ∈ {2, 3, 5}); extend VT-18; decide the family structure (weighted Holm, fixed sequence, or moving H1-min/H3 to secondary) | D-1 | before-G5 | 0 | Honest power; P(H1) at (30, 3) adaptive: 0.85 (registered), 0.945 (weighted), 0.967 (fixed sequence), at type I 0.018 / 0.037 / 0.051 |
| R-5 | 8K fallback: G-REF (a) and (b) at 8K, else H1/H1-min not evaluable | D-2 | before-G5 | 0 | Removes wrong-direction "supported" verdicts |
| R-6 | H3 pooled-denominator evaluability; declare the H3 null unreachable (estimate with CI) | D-3 | before-G5 | 0 | Evaluability 0.53 → 0.995 and support 0.41 → 0.67 at D₃ = 0.3 (n = 5, σ = 3) |
| R-7 | Placement: restrict wording to the shift tested; MDE at Holm-4; select k* on DG; fix the U-07 text; ask the user about Tier B now | D-4, D-15, D-18 | before-G5 | 0 (optional clustered arms ≈ 19) | RQ-B claims match what is tested |
| R-8 | U-05 range clause on seed-centred scores; budget n₀ + 2 as expected at σ ≥ 3 | D-14 | before-G5 | 0 (planning: +12.2 h expected at σ = 3) | Seeds are bought only for variance that matters; the budget is realistic |
| R-9 | Define the co-primary F_gc decision and multiplicity rule; informative-F_gc requirement for operator wording | D-10 | before-G5 | 0 | No ambiguous or doubled-α confirmatory claim |
| R-10 | Score DG on all core configurations; "reference exceeded" label; still-closing flag on L_k and R(k), including k = 0 | D-6 | before-G5 | ≤ 1 | A support-direction guard against a handicapped reference and slow k = 0 elicitation |
| R-11 | Word claims "token- and parameter-matched, not compute-matched"; report training PFLOP and GPU-h per config in RQ-D | D-11 | before-G5 | 0 | Correct scope; a complete cost axis |
| R-12 | NoPE seed 1 unconditional at k ∈ {3, 12} with a difference-in-differences CI (or pre-state the detection limit) | D-9 | before-G5 (optional) | ≈ 4.4 | PE × k is resolvable at a smaller effect size |
| R-13 | H6′ estimation-only, or +3 X-E0 seeds | D-13 | before-G5 | 0 or ≈ 6.2 | Capacity rival estimable |
| R-14 | Second seed for the S5b branches (TEF-4,000 at 4 anchors, 2× Stage B at {0, 3, 12}) | D-6, row 4 of §2 | optional | ≈ 5.8 | Dose and saturation guards no longer single-seed |
| R-15 | Training-precision control (S3 recipe, bf16 vs fp32, k ∈ {0, 12}) | D-12 | optional | ≈ 2–3.5 [INF] | Bounds RR-R5-power before S4 |
| R-16 | Query-first R-load cell at 32K (evaluation only, exploratory; out of TEF format) | §2 row 2 | optional | ≈ 0.05 per model | Separates capacity from selection [SPEC] |
| R-17 | SSMax arm on seed 102 | D-19 | optional, before FEAS | ≈ 1.3 | Less random SSMax adoption |
| R-18 | Report EM by TEF query index on a TEF-format dev cell | D-17 | optional | ≈ 0.05 per model | Sizes the sequential-vs-branched shift |

**Totals.**
- Must-before-FEAS: 0 GPU-h, plus ≈ 7 conditional on FEAS-512.
- Before-G5, required items: ≈ 1 GPU-h, plus evaluation of the new A-freq cells.
- Optional: ≈ 4.4 + 6.2 + 5.8 + 3.5 + 1.3 + 19 ≈ 40 GPU-h if all are taken. The highest value per GPU-h is R-12 (NoPE seed 1) and R-14 (second-seed branches).

---

## 5. What I verified, and how

| Claim | Command / source | Result |
|---|---|---|
| The prior instance's simulations are correct and reproducible | Code review of `.audit/agent1-design/{sim_h1_holm,sim_secondary,sim_alt_multiplicity,sim_h3_alt,afreq_floors}.py`; re-run with `python <script>.py > r2/<script>.out` in `.venv`; `diff` | All 5 identical (seeded). Run times 1.3–22.8 s |
| PI-SIM numbers | `r2/sim_h1_holm.out`, sanity block | Matches debate §6.2 within MC error |
| Item SE, FEAS rule noise, U-05 rate, H1 by Holm position, 8K fallback | `python r2/design_checks_r2.py` → `r2/design_checks_r2.out` | As in D-1, D-2, D-7, D-14, D-20 |
| U-05 vs τ, FEAS-512, SD(L₃) | `python r2/design_checks_r2b.py` → `.out` | As in D-8, D-14, §3.7 |
| Seed-mean FEAS alternative; R5 SD CI | `python r2/design_checks_r2c.py` → `.out` | As in D-7, D-1(d) |
| A-freq floors on the frozen items | `r2/afreq_floors.out` (project helper `eval.scorer._afreq_heuristics`, plus the same rule at 8,192) | As in D-5; matches `s0_integration.md` L359 at the cells it lists |
| Fallback, k*, co-primary and evaluability behaviour of the code | `sed`/`grep` of `analysis/pipeline.py` L311–316, L487, L544, L798–806, L916–925, L1154–1159; `verdicts.py` L160–180 | As cited |
| FEAS rules as hashed | `experiments/prereg/DTS10_acceptance.v2.yaml` L31–39 (rules), L40–42 (readings) | "BOTH seeds" (U-01); seed 101 only (FEAS-512); SSMax seed 101 |

**Not verified:**
- Throughput-based GPU-h for the new arms. These use the protocol's contended per-run rates and `plan.yaml` rows, not new measurements. The fp32 training throughput for R-15 is unknown.
- Whether σ in our LM regime resembles R5's MQAR regime.
- Literature claims, which are cited per the project's literature review. I fetched no URL.
- I ran no GPU job.

---

## 6. Convergence with Agent 5's continuation adversarial review (independent; 15:43)

| This audit | Agent 5 | Status |
|---|---|---|
| D-1 (Holm power, σ evidence) | m-0 (MODERATE) | Converge on the Holm effect. I add the R5 σ evidence, the G-dependence and the Holm-position range, and rate it MAJOR because RQ-A's answerability and the n₀ budget rest on it |
| D-3 (H3 evaluability) | F-2 and M-4 | Converge. I add that evaluability *falls* with n, plus the pooled-denominator fix with numbers |
| D-5 (A-freq floors) | F-3 | Converge (identical floor numbers from independent scripts) |
| D-6 (reference handicap, elicitation) | M-1, M-3, M-7 | Concur. Not re-derived |
| D-9 (NoPE trigger) | m-1 | Converge |
| D-10 (conv / co-primary) | M-8 | Complementary: M-8 is 2-seed F_gc agreement; D-10 is the undefined co-primary rule |
| D-11 (compute) | M-5 | Converge. I add the training PFLOP ratio |
| D-12 (RR-R5-power) | m-12 | Converge |
| D-13 (GDN-E) | F-1 | Converge |
| D-17 (query format) | m-8 | Converge |
| **New in this audit** | — | D-2 (8K fallback sign reversal), D-4 (±1-layer arms; Holm-4 MDE; RQ-B unfunded by default), D-7 and D-8 (FEAS decision noise), D-14 (U-05 rate and τ sensitivity), D-15 (k* selection bias), D-16 (G-REF selection near threshold), D-18, D-19, D-20 |
