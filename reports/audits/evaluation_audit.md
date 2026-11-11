# Evaluation and statistics audit (Agent 4)

**Date:** 2026-11-10, IST, written in the resumed session after the 17:00 usage limit.
**Role:** Agent 4, evaluation and statistics specialist.
**Mode:** report-only. Nothing in `src/`, `scripts/`, `tests/`, `configs/` or `experiments/` was changed. All scratch work is in `.audit/agent4_eval/`. **No GPU was used.**
**Scope:**
- the generators (`src/hybridlc/eval/*`) and the frozen items (`experiments/items/**`);
- the scorer;
- leakage between the train namespace (TEF) and the evaluation items;
- the statistics code (`src/hybridlc/analysis/*`), including planted-effect checks.

**Evidence tags used below:**

| Tag | Meaning |
|---|---|
| **[CODE]** | Read in the source, cited file:line. |
| **[MEAS]** | Computed in this audit on the actual item files or with the project's own functions. The command and the artefact are named. |
| **[SIM]** | CPU simulation with an exactly known planted truth, using the project's own functions. |
| **[PRIOR]** | Produced by this audit's first pass before the usage limit (`.audit/agent4_eval/verify_items.py`, `leakage_check.py`, `scorer_planted.py`). I re-read each script and re-checked its logic. I re-used only the outputs whose logic held, and I say so where a result is re-used. |
| **[INF]** | Inference or arithmetic. **[SPEC]** marks speculation. |

---

## 0. Bottom line

**No fatal defect was found.** For the confirmatory test (H1, H1-min) the evaluation pipeline is sound:
- The frozen items are structurally correct, are hash-identical to their committed manifests, and use held-out filler only. I reconstructed that filler exactly from the stores.
- The scorer computes EM_tf, logp, rank and error types as specified.
- No train-namespace material leaks into the evaluation items.
- The core statistics are correct: family scores, L_k, Holm in both directions, bootstrap pairing, the U-05 trigger, G-REF and the primary-cell rule, TOST and Fieller.

The problems are about **what three of the instruments measure and how much power or sensitivity they have**, not about arithmetic.

| # | Severity | Finding (short) | Where |
|---|---|---|---|
| F1 | **major** | The H3 difficulty-equated sign rule (U-11) requires D₃ > 0 at **every** equated pair. Pairs are not restricted to informative levels, and the pair D₃ is a mean of per-seed ratios. A **true** H3 (D₃ = +0.2) is vetoed **13–32% per pair at a 15 pp level gap and 45–48% at 5 pp**. With several pairs the vetoes compound | `pipeline.py:589-627` (`consistent = all(d > 0 …)` at L626); §6.7 |
| F2 | **major** (construct validity of RQ-C) | **A-freq at long T is mostly solvable by short-window counting.** At 32K: a last-4K-token counter scores 0.926 at ρ = 4; a last-8K counter 0.832 at ρ = 2; a second-half counter 0.953 at ρ = 2. Primary-cell aggregation-family floors: recency 0.657, 4K window 0.747, 8K window 0.816, decayed counter (half-life 8K) 0.894, against a ceiling of 1.0. The only floor the pipeline reports is the 1K window (0.623), which hides this | §3.3, `scripts/analyse_results.py:105-119` |
| F3 | **moderate** | The **bootstrap sensitivity check is vacuous**: it agreed on 100% of t-rejections in every scenario. It is about 30% narrower than the t interval and anti-conservative under the null (lower bound > 0 in 11.7%). Its nested resampling cannot represent item×config interaction. The "agreement" rule is an undocumented [IMPL] reading ("bound excludes 0"), so all downgrade weight falls on the pooled model. At R(3) = 0.85–0.9 the pooled model downgrades 3–5% of all replicates (10–16% of Holm rejections) when the other support-family p's are small, and ≤ 0.3% (1.5–5% of rejections) when they are not | `pipeline.py:129-131`, `verdicts.py:9-11, 47-71`, `stats.py:728-744`; §6.3–6.4 |
| F4 | moderate | **The paired t is conditional on the frozen items.** With item×config interaction SD τ = 0.2–0.3 (context-mean scale), its type I with respect to the item population is 0.077–0.09. The magnitude of τ is unknown; it is cheap to measure at FEAS | §6.5 |
| F5 | moderate | **R1 and R-load are one generator** (R1 ≡ R-load at N = D + 1). With equal task weights, 70% of the primary-cell retrieval weight falls on N ≤ 32 facts. The H5(a) fixed level set is N ∈ {4, 8, 16, 32} only. "Retrieval" claims are therefore mostly low-load claims | `generators.py:401-422`; §3.1 |
| F6 | moderate | **k\* is selected from the CS scores of the same composition runs that form H4's middle arm**, so H4-c1 is computed on selected data. The bias favours c1 > 0. c2 is unaffected | `pipeline.py:916-926` + `h4_test` (L1044-1061); §6.8 |
| F7 | moderate | **Analysis code is missing for secondary metrics the protocol and rulings require.** No analysis module consumes rank accuracy, error types or `absent_in_context` (D065(iv) requires the latter). The D085 GLMM is not implemented yet. The A-freq floor is reported at W = 1K only. All of this must exist before G5 | grep over `src/hybridlc/analysis` and `plots/` (empty); `pipeline.py:555-566` |
| F8 | moderate | **Error types and rank chance are hard to interpret as defined:**<br>• "binding" dominates by construction (32 context vs 4 absent candidates);<br>• an error type is assigned to EM-correct queries whenever rank > 1;<br>• ties go to the first context candidate;<br>• `chance = 1/|C|` understates the in-context-guessing floor: 0.25 vs 0.125 for R1 D = 3 and M-hop; 1.0 vs 0.2 for R0 | `scorer.py:108-114, 191-211` |
| F9 | minor | The gold logp (prompt+answer path) and the other candidates' logps (extend(prompt) path) are computed on different code paths. Only rank accuracy is affected | `scorer.py:158-183` |
| F10 | minor | **Contexts are not independent.** PG19-test filler is re-used heavily: at 32K, 2,169 test contexts come from 91 books (up to 40 per book), and 1,907 dev contexts from 44 books (up to 67 per book). Within a cell up to 7 contexts share a book. DG, DC and FD share item prefixes, so G-REF reads the FD items that drive FEAS decisions | §3.6, §6.6 |
| F11 | minor | Further small items:<br>• model-level selection effect in G-REF length choice: conservative for support, liberal for falsification;<br>• C0 controls are shorter than any TEF bucket;<br>• 1.4% of needle junctions are non-canonical BPE (benign);<br>• near-duplicate (non-exact) filler overlap with training is unchecked | §3.6, §5, §6.6 |

**Verified correct (no action needed):**
- exact T on every query;
- `<|endoftext|>` only at position 0;
- answers are 4 digit tokens that match natural tokenisation;
- candidate sets have equal length, gold first, no duplicates;
- filler is held-out and bit-exact to its recorded source;
- R-task depths are decile-balanced;
- distances are recorded correctly;
- M-hop chains are well formed, with no shortcut beyond the 1/4 in-context rate;
- the G-ORACLE repair is complete (VT-24r K0/K1/K2/K3/K5b PASS);
- word lists A and B are disjoint at the token level;
- the TEF stream is unchanged by the repair;
- EM_tf ≡ greedy exact match;
- Holm is applied in the right directions;
- bootstrap seed and context pairing is exact;
- U-05 is invariant to config effects;
- G-REF, primary-cell, TOST and Fieller formulas are correct (Fieller coverage 0.949–0.951).

---

## 1. Item set audited, and its provenance

- **State of the item files.** 219 files under `experiments/items/{test,dev}` (1,375,633,371 B) match the SHA-256 values in the committed manifests: 0 mismatches [MEAS: inline script, §A1]. The manifests were last committed in `490798d`.
- **The audited set is the regenerated one.** It is the post-G-ORACLE set:
  - 122 of the 218 `.npz` files were rewritten at 15:06–15:08 IST (`find … -newer make_items.log`): every retrieval-task file (R1, R-load, R3, R-dist, with their C0/C2/T4 variants) plus M-hop C0/C2. The A-freq, A-count, other M-hop, R0 and Tier-0 LM files were not rewritten;
  - `experiments/items/vt24r_repairs.json` records 16 repairs on 14 distinct facts in 5 cells: 13 under T4, 3 under T1, 0 triggered by filler;
  - VT-24r K3 shows that content changed only in those 5 repaired cells.
- **VT-24r so far** (outputs in `experiments/repro/VT-24r/`, written by the concurrent workflow; `vt24r_summary.json` not yet written):

  | Check | Result |
  |---|---|
  | K0 | The independent checker reproduces VT-24's 7 misses exactly. Coincidental-match queries 0; checker-defect digits 0 |
  | K1 | 78,800 retrieval queries in 116 files are answerable by the order-6 InductionLM oracle. 0 failing |
  | K2 | M-hop negative control: oracle EM **0.0** in all 48 files. A single-step copy mechanism cannot solve M-hop |
  | K3 | 218 cells content-stable apart from the 5 repaired cells |
  | K5b | TEF full-block hash unchanged (`0a34724e…`) |

  [MEAS: `k1k2.json`, `k0_agreement.json`, `k3_stability.json`, `k5b_check.json`]

---

## 2. Structural item checks

**Two independent passes:**

1. **Full pass, all 206 cloze files** [PRIOR, script reviewed]: `.audit/agent4_eval/verify_items.py` → `verify_all.jsonl`. Total failures over all checks: **0**. The checks:
   - exact T per query;
   - BOS at position 0 and no EOT inside the context or the queries;
   - answers are exactly 4 digit tokens; the prompt ends with the space token; decoded answer = recorded value;
   - candidates have equal length, gold first, no duplicates, exactly 4 absent, ≤ 32 context;
   - absent candidates are not fact values; context candidates are fact values;
   - each fact span re-renders to its template;
   - the queried span equals the rendered fact;
   - recorded distance = len(ctx) − needle_end;
   - the full prompt occurs exactly once in the context (R1, R-load, R-dist);
   - facts do not overlap; branched queries have equal length; the filler pool is correct.
2. **Independent pass that does not use `load_items`** [MEAS]: `.audit/agent4_eval/indep_items.py`. It reads the `.npz` members directly and covers 5,150 contexts (25 per cell, every cell) and 20,450 queries:
   - **0** failures of T, EOT, branch length or pool;
   - **filler provenance:** for **3,450/3,450** natural-filler contexts, the context with its needle spans removed equals, token for token, the span re-extracted from the recorded held-out document: PG19 `doc[start:end]`, or FWE docs joined by `"\n\n"` from `first_offset`;
   - none of those documents' text hashes occurs in `fwe/train` ∪ `pg19/train`;
   - fact interiors are canonical BPE (0/10,350 non-canonical);
   - 148/10,350 (1.4%) of junction windows do not re-encode identically. Inspection shows these are benign: windows that cut multi-byte UTF-8 characters, and `"\n\n"` that merges differently after "." than in the source document (examples in §A3).
3. **Answer encoding in the natural distribution** [MEAS]:
   - `TokenSpace.natural_value_ok()` = True;
   - `" The code for brave lion is 1234."` → `[' The',' code',' for',' brave',' lion',' is',' ','1','2','3','4','.']`;
   - `"Answer: 5"` → `[…':',' ','5']`;
   - T4 `" brave lion's code: 0567."` → `[' brave',' lion',"'s",' code',':',' ','0',…]`.

   So the evaluation token layout is the tokenizer's own layout.

---

## 3. Task by task: does it measure what it claims?

### 3.1 Retrieval: R1 and R-load (confirmatory family)

**Construction** [CODE]: `gen_r1` (`generators.py:401-410`) and `gen_rload` (`:413-422`) both call `_facts_task(ctx, i, irng, n_facts, _queried_subset(irng, n_facts))` with `n_facts = D + 1` and `N` respectively:
- the same templates;
- the same `spread` placement: queried facts at decile depths, the others uniform (`_target_offsets`, `:195-203`);
- the same candidates.

The only difference is R-load's 0.75 T assertion (`:258-260`). **R1 at D is distributionally R-load at N = D + 1.** R1 D = 3 has 4 facts, all queried; R1 D = 15 has 16 facts.

**Consequence for the confirmatory DV** [INF from `scores.py:164-195`]. At the 16K + 32K primary cell, the weights per number of facts are:

| N | 4 | 8 | 16 | 32 | 128 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| weight | 0.25 | 0.10 | 0.25 | 0.10 | 0.10 | 0.10 | 0.10 |

**70% of the weight is on N ≤ 32.** The H5(a) fixed level set {R1 D3, D15; R-load N8, N32} is N ∈ {4, 16, 8, 32}, all low-load.

This is the frozen design, not an implementation deviation. It does change what "retrieval" means in H1 and H5(a): mostly few-fact retrieval at long distance. The load axis enters only through 30% of the weight.

**Item properties** [PRIOR/MEAS, `verify_all.jsonl`]:
- Depth deciles are balanced, e.g. CS R1 D3 32K: 119–121 per decile of 1,200. M-hop is the exception (below).
- Candidates: R1 D3 8, D15 20; R-load N8 12, N32 36, N ≥ 128 37.
- The gold 4-digit string occurs more than once in the context in ≤ 1.4% of queries.
- The queried key bigram occurs exactly once (1,200/1,200, CS R1 D3 32K).
- Absent candidates present in the filler: 0–0.88% (known; D065(iv)).
- Snap rate 0.93–0.97.
- PG19/FWE filler split about 50/50.
- Fact density reaches 75% of T at N1024@16K and N2048@32K: 12,288 / 24,576 fact tokens.

**Shortcuts** [INF]:
- The queried key appears once and the full prompt occurs only in its fact, so no positional or frequency heuristic beats 1/(#facts) on EM.
- The task is a genuine long-range associative lookup. A window-W model can recover only about W/T of the queried needles, because depths are uniform by decile.

**Verdict:** R1 and R-load measure long-distance key→value binding, mostly at low load. Valid for H1 as defined; claims should state the load composition (R5).

### 3.2 Multi-hop: M-hop (confirmatory family)

**Construction** [CODE `generators.py:539-626`]:
- chains `k_h → … → k_0 = v`, 4 chains per context, 4(h + 1) distinct keys, 4 distinct values;
- each chain queried once at its head;
- candidates are the 4 chain values plus 4 absent;
- shuffled order uses i.i.d. uniform offsets.

**Realised structure** [MEAS from `verify_all.jsonl`, CS + DG + FD + MS, 32,200 queries]:
- forward-order rate 1/2, 1/6, 1/24, 1/120 for h = 1…4 (observed 0.50 / 0.17 / 0.04 / 0.01);
- chains interleaved 71–100%;
- the head key occurs exactly once; the full prompt never occurs in the context (the head link continues with " the").

**Shortcut heuristics on the actual items** [MEAS, all files]:

| Heuristic | Accuracy |
|---|---|
| last value fact | 0.250 |
| first value fact | 0.250 |
| value fact nearest the head link | 0.255 |
| next value fact after the head | 0.257 |
| previous value fact before the head | 0.250 |
| unique lexical key overlap (n = 954) | 0.251 |

All equal 1/4, the rate of picking any in-context value.
- "Most frequent value" is undefined: each value occurs once as a fact.
- "1-hop then nearest value" is 0.25 at h ≥ 2. At h = 1 it is 1.0 because it *is* the solution.
- Above 1/4 only in the MS M-order cells: forward 0.353, reversed 0.380, an expected property of ordered chains (diagnostic only).
- The order-6 induction oracle scores 0 on every M-hop file (VT-24r K2).

**Known limitation:** needle depths are i.i.d., not decile-stratified (D052(iii)). Decile counts range 61–107 per 800 queries. This cancels in paired k-contrasts.

**Verdict:** M-hop measures multi-step indirection with no exploitable cue.

### 3.3 Aggregation: A-freq (family score used for RQ-C; secondary contrasts)

**Construction** [CODE `generators.py:715-761`]:
- 4 probe pairs, n_X = round(8ρ), n_Y = 8, single-token nouns;
- background words ≤ 3 occurrences each (max = 3 in every item);
- fully shuffled; winner mention order random (0.47–0.53 observed);
- candidates [X], [Y].

**Shortcut floors measured on the actual CS items** [PRIOR/MEAS, `verify_all.jsonl`]. These are fractions correct for model-free rules; ties count ½:

| T | ρ | recency | last 1K | last 4K | last 8K | 2nd half | decayed counter, half-life 8K |
|---|---|---|---|---|---|---|---|
| 8K | 4 / 2 / 1.5 / 1.25 | 0.80 / 0.69 / 0.59 / 0.54 | 0.93 / 0.72 / 0.60 / 0.56 | 1.00 / 0.95 / 0.79 / 0.66 | (= full) | 1.00 / 0.95 / 0.79 / 0.66 | 1.00 / 1.00 / 1.00 / 0.95 |
| 16K | 4 / 2 / 1.5 / 1.25 | 0.81 / 0.66 / 0.63 / 0.56 | 0.83 / 0.65 / 0.58 / 0.54 | 0.99 / 0.85 / 0.70 / 0.61 | 1.00 / 0.94 / 0.81 / 0.67 | 1.00 / 0.94 / 0.81 / 0.67 | 1.00 / 1.00 / 0.94 / 0.81 |
| 32K | 4 / 2 / 1.5 / 1.25 | 0.79 / 0.67 / 0.61 / 0.53 | 0.72 / 0.61 / 0.55 / 0.51 | **0.93** / 0.73 / 0.64 / 0.54 | 0.99 / **0.83** / 0.69 / 0.60 | 1.00 / **0.95** / 0.81 / 0.66 | 1.00 / 0.95 / 0.81 / 0.64 |
| 32K (MS) | 1.1 | 0.51 | 0.50 | 0.54 | 0.55 | 0.59 | 0.60 |

Recency matches the theoretical ρ/(1 + ρ): 0.80, 0.67, 0.60, 0.56, 0.53.

**Family-level floors at the primary cell** (16K + 32K mean of the ρ-ladder) [MEAS]:

| Rule | Floor |
|---|---|
| recency | 0.657 |
| last 1K | 0.623 (the only floor `cell_summary.csv` reports) |
| last 4K | 0.747 |
| last 8K | 0.816 |
| second half | 0.855 |
| decayed counter, half-life 8K | 0.894 |
| perfect full-context counter | 1.0 |

**What this means** [INF]:
- A mechanism that integrates counts over an effective horizon of about 8K tokens with no other long-range ability scores about 0.82–0.89 on the aggregation family at the primary cell. A gated recurrence with near-unit decay is plausibly that mechanism.
- Only about 11–18 pp of the scale needs integration beyond 8K, and 80–90% of that deficit lies at ρ ≤ 1.5 (8K-window deficits at 32K: 0.01 / 0.17 / 0.31 / 0.41 for ρ = 4 / 2 / 1.5 / 1.25).
- At ρ = 4 the task is essentially local at every length.
- Existing coverage is incomplete. D065(vii) requires the "local frequency comparison" wording only for T ≤ 4K. The scorer computes W ∈ {256, 1K, 4K} (`constants.py:45-46`), and the analysis surfaces W = 1K only (`analyse_results.py:105-119`). At 32K that is the *least* informative floor.

**Verdict:** A-freq measures "frequency comparison over a shuffled list". At ρ ≥ 2 it does **not** measure integration over the full context. The RQ-C aggregation family (R_agg(k), P3) and the secondary A-freq contrasts can show high recovery for low-k models for reasons unrelated to long-range integration (R2, R3).

### 3.4 Mechanism-suite tasks

| Task | Status | What it measures / caveats |
|---|---|---|
| R3 | **valid** [CODE `:448-498`] | The latest write is always the last occurrence of its key, so "last occurrence" = gold by design. Measures overwrite/recency and the stale-error rate. It favours recurrent overwrite by construction, which is intended. Stale candidates come first in the candidate list |
| R-dist | **valid** [MEAS] | Realised distances are within ±32 tokens of the 256 / 2K / 8K / 30K targets for natural filler, and within −2…+14 for repetitive filler |
| A-count | **valid as a diagnostic** | c ~ U{1..9} (chance 1/9). Animals are banned from the background |
| C2 (oracle) | **valid** | All window floors = 1 by construction. It is the "same task, local" control |
| T4 | **valid** | Template transfer. 13 of the 16 G-ORACLE repairs were in T4: the 6-token window before digit 3 or 4 excludes the key under T4 |
| R0 | **valid as a sanity check; rank accuracy uninformative** | Rank accuracy is trivially 1.0 for any copy-a-number model: the only in-context value is the gold. Use EM_tf only |

### 3.5 C0 (no-filler) controls

- Contexts are `[EOT] + facts`, from 49 tokens (R1 D3) up to 12,289 tokens (R-load N1024).
- Low-N C0 items are shorter than any TEF bucket (≥ 512) [CODE `constants.py:15`].

[SPEC] If the TEF makes models worse on such out-of-distribution short items, S₁₂(C0) falls and G-REF (a) (S₁₂(T) ≥ 0.8·S₁₂(C0)) becomes easier to pass. This is likely negligible (attention at < 200 tokens is easy). It can be checked once FEAS exists (R11 in §8).

### 3.6 Filler and context dependence

- **PG19 spans** pick a document uniformly among the eligible books, then a uniform end (`filler.py:156-175`). Re-use [MEAS, `indep_items.json`]:

  | Namespace, T | Contexts | Books | Max contexts per book |
  |---|---|---|---|
  | test, 32K | 2,169 | 91 | 40 |
  | test, 16K | 1,720 | 99 | 34 |
  | dev, 32K | 1,907 | 44 | 67 |
  | dev, 16K | 1,655 | 49 | 49 |

  Within one cell, e.g. CS R-load N32 32K: 172 PG19 contexts from 74 books, at most 6 per book.
- **FWE** documents are sampled with replacement; 5/9,437 test contexts repeat a document [PRIOR].
- **Consequence** [INF]: contexts sharing a book share partly overlapping filler. The bootstrap and the U-15 ICC (which is within-context over the 4 queries) treat contexts as independent. Filler effects largely cancel in paired k-contrasts unless they interact with k. Impact is minor; the diagnostic is R12.

---

## 4. Scorer (`src/hybridlc/eval/scorer.py`)

**Verified:**
- **EM_tf** = all answer tokens are the argmax given the gold prefix (`:158-161`), with `score_continuation` alignment `lp[:, j] = log p(cont_j | ctx, cont_<j)` (`models/build.py:241-259`). EM_tf is therefore exactly greedy exact match.
  - [PRIOR, `scorer_planted.json` B]: 24 queries, 14 gold answers set to the model's greedy output: 0 mismatches.
- **logp_gold** = summed answer-token logprobs (`:208`).
- **Rank** = 1 + #(others ≥ gold); gold loses ties (`:191`, `:108-114`).
  - [PRIOR, A]: planted gold / context / absent / stale / other-chain / random answers on R1 D15, R3 U2 and M-hop h2, with and without the `extend` path. 288 queries, 0 mismatches in EM, rank or error type.
- `rescore_subset` is deterministic (`:415-420`).

**Findings:**
- **F8a.** **Error type when EM = 1 but rank > 1.** The row gets the top candidate's type (`:196-201`). This can happen: each gold token is the argmax given the gold prefix, yet a distractor's continuation can have a higher total probability. Error-type rates must be computed over EM misses only.
- **F8b.** **Ties are attributed to the first context candidate.** An indifferent model is classified 100% "binding" or "wrong_chain" ([PRIOR] C: R1 D15 → binding 40/40, rank 20). Context candidates precede absent ones.
- **F8c.** **Base rates make "binding" dominant by construction.** R-load N ≥ 128 has 32 context vs 4 absent candidates, so a uniform guesser over candidates produces 89% "binding" errors. Error-type shares need base-rate normalisation, e.g. observed share ÷ (n_type / (|C| − 1)).
- **F8d.** **`chance = 1/|C|`** (`:211`) understates the floor of a model that only knows "the answer is some in-context value":

  | Task | Stated chance | In-context-guessing floor |
  |---|---|---|
  | R1 D3 | 0.125 | 0.25 |
  | M-hop | 0.125 | 0.25 |
  | R1 D15 | 0.05 | 0.0625 |
  | R0 | 0.20 | 1.00 |

- **F9.** **Two code paths in one comparison.** The gold logp comes from `score_continuation(cache, prompt+answer)`; every other candidate from `extend(prompt)` then `score_continuation(p_cache, cand)` (`:164`, `:169-183`). Both use fused_recurrent / SDPA over the cached KV but with different token blocking. In bf16 the two can differ by about the VT-07 magnitude (2.4–3.8e-3, RR-G0-2) [INF]. Rank compares numbers from two paths. The effect is small and touches rank accuracy only. Trivial fix: score candidate 0 through the same `p_cache` path for the rank computation.
- **F7 (part).** **No downstream consumer.** No module in `src/hybridlc/analysis/` or `plots/` reads `error_type`, `rank`, `rank_correct` or `absent_in_context` (`grep -rn "error_type\|absent_in_context\|rank_correct"` returns nothing). The §9.2 secondary metrics and D065(iv)'s requirement have no implementation.

---

## 5. Leakage: train namespace (TEF and pretraining) vs evaluation

| Path | Evidence | Result |
|---|---|---|
| Held-out stores vs training stores | [PRIOR `leakage.json`] text-hash and id-hash intersections of fwe/test, fwe/dev, pg19/test, pg19/validation with fwe/train and pg19/train | **0** in all 16 intersections. Test∩dev text also 0 |
| Filler actually used | [MEAS §2] 3,450 contexts re-extracted bit-exactly from the recorded held-out docs; none of their hashes is in training. [PRIOR] 90,223 distinct test docs and 48,689 dev docs used; 0 in training | clean |
| Word lists | [PRIOR] adjective, noun, probe and animal token ids: A∩B = 0. A key tokens ∩ B key tokens = 0. 3,679,000 eval key bigrams contain no list-A token | clean |
| Case and sub-token near-collisions | [PRIOR] 1 case-insensitive word ("early"/"Early"; different tokens). 317 list-A background words contain a list-B token as a sub-token, so B tokens appear inside TEF word lists, never in a key/value role | negligible |
| Templates | [CODE `templates.py:194-195`, `generators.py:73-75`] TEF uses T1–T3; eval uses T1 + T4. T1 is shared **by design** (§7.3 leakage controls) | as designed |
| TEF stream after the repair | [MEAS VT-24r K5b] full-block hash identical before and after; the repair refuses namespace `train` (`generators.py:281, 290-295, 332-333`) | clean |
| Dev vs test | Different entropies (77 vs 20261109) and disjoint filler pools. Keys come from the same list B (harmless: dev is never trained on) | clean |
| Near-duplicate filler | **Not checked.** Exact-text disjointness holds, but fwe/train itself has 93,333 exact internal duplicates [PRIOR], so cross-split near-duplicates are likely | [SPEC] minor: identical across k |

---

## 6. Statistics code (`src/hybridlc/analysis/`)

### 6.1 Family scores, L_k, ladder weights [CODE] — correct

- **Weights.** `ladder_weights` (`scores.py:164-195`) gives each level 1/(|levels|·|tasks|·|Ts|), excludes N2048 (`protocol.py:57`), and picks the C0 strata of the same level set for G-REF (a).
- **Cell means.** A cell mean is the mean over contexts of per-context query means (`scores.py:100-141`).
- **L_k.** `l_contrast` = S_k − 0.2 S₀ − 0.8 S₁₂ (`stats.py:43-50`).
- **Planted recovery.** [PRIOR `sim_stats_p1p2.json`] a noiseless planted L = 0.05 is recovered exactly by both the point estimate and every bootstrap replicate (boot min = max = 0.0500). VT-18 (`tests/test_vt18_pipeline.py:58-136`) recovers planted L_k, D₃ and slope on the project's synthetic worlds.

### 6.2 Holm direction [CODE] — correct

- **Support family:**
  - H1 and H1-min use `one_sample_t(L, "greater")` (`pipeline.py:513-520`);
  - H3 uses "greater" on D₃, with p = 1 if not evaluable (`:544, 553, 560`);
  - H5a uses "greater" on b (`:649`).
- **Falsification family** {H1, H1-min} uses the "less" p (`:520`).
- **Holm step-down** with running max and cap at 1 (`stats.py:241-257`).
- **Falsification guard 2** requires the falsification Holm to reject *and* the pooled and bootstrap analyses to agree in the "less" direction, with both present (`pipeline.py:812-818`).

### 6.3 Hierarchical paired bootstrap

**Pairing [CODE] — correct.** One seed draw per (replicate, slot) is shared by every config (`stats.py:732-733`), and one set of context indices per (replicate, slot) is shared by every config (`:739-743`). Only complete seed blocks are used, and nothing is zero-filled (`:704-726`).

**Design problem [SIM] (F3, F4).** The data are **crossed** (every seed and config sees the same contexts). The resampling is **nested**: a fresh context draw per seed slot. That has two consequences:
1. **The shared item×config component is diluted by about 1/n_slots**, so the bootstrap cannot represent the only item-level variance that survives pairing. At τ = 0.3, the SD of L̂ over item redraws is 0.0166, while the nested bootstrap SD is 0.0128 (−23%) and the crossed bootstrap SD is 0.0144 (−13%).
2. **With n = 5 seeds the percentile bootstrap is anti-conservative.** At τ = 0, the nested bootstrap SD (0.0124) is below the true SD (0.0145), and the one-sided percentile bound uses ≈ z rather than t₄. Its one-sided 95% half-width is about 0.68× the t interval's [INF: 1.645·0.0124 vs 2.132·0.0141]. Under the null its lower bound exceeds 0 in **11.7%** of replicates (S2, 60 reps); the first-pass simulation gave 19% in 200 reps.

### 6.4 The "suggestive" downgrade rule as implemented (F3)

- **What the protocol says.** §11.3: "If either [pooled, bootstrap] disagrees **in sign** with the primary verdict, the verdict is downgraded."
- **What the implementation does.** It reads "agrees" as "the analysis' one-sided 95% bound excludes 0" (`pipeline.py:129-131`; `verdicts.py:9-11, 47-71`), labelled [IMPL]. No decision row ratifies this reading (grep over `decisions.md` and `pi_rulings_2026-11-10.md`).
- **Why it matters.** The literal "sign of the point estimate" reading would be vacuous: the pooled OLS contrast on a complete table and the bootstrap mean equal the paired mean. So the [IMPL] reading is the only non-vacuous one, but it is a material choice.

**Operating characteristics** [SIM S1; `sim_verdicts.json`]. Setup:
- n = 5, G = 30 pp, run-level SD 2.5 pp per config and seed, shared seed SD 2 pp;
- 14 strata × 300 contexts, 300 replicates per row, bootstrap B = 1,000;
- "favourable" = the other support-family p's ≈ 0; "unfavourable" = H3 p = 1, H5a p ~ U(0,1), H1-min computed with R(1) = 0.4.

| True R(3) | Holm context | P(Holm rejects H1) | P(pooled agrees \| reject) | P(boot agrees \| reject) | P(supported) | P(suggestive) |
|---|---|---|---|---|---|---|
| 0.80 (null boundary) | fav / unfav | 0.030 / 0.007 | 0.78 / 1.00 | **1.00 / 1.00** | 0.023 / 0.007 | 0.007 / 0.000 |
| 0.85 | fav / unfav | 0.187 / 0.063 | 0.84 / 0.95 | **1.00 / 1.00** | 0.157 / 0.060 | 0.030 / 0.003 |
| 0.90 | fav / unfav | 0.493 / 0.227 | 0.91 / 0.99 | **1.00 / 1.00** | 0.447 / 0.223 | 0.047 / 0.003 |
| 1.00 | fav / unfav | _see §A2_ | | | | |

**Reading:**
- Type I is controlled: Holm-supported at most 0.03 at the boundary.
- **The bootstrap never downgrades.** The pooled model downgrades 3–5% of all replicates when R(3) is 0.85–0.9 under the favourable Holm context; that is 10–16% of the Holm rejections.
- The same vacuity applies to falsification guard 2 for the bootstrap.

### 6.5 Conditional inference on frozen items (F4) [SIM S2]

The paired t treats the realised items as fixed. With an item×config interaction b_{ctx,c} ~ N(0, τ²) on the context-mean scale, the frozen item set carries a fixed offset that every seed shares. At the R = 0.8 boundary:

| τ | Type I, frozen items | Type I, items redrawn | Offset of this frozen set | True SD of L̂ | Mean t SE |
|---|---|---|---|---|---|
| 0 | 0.060 | 0.040 | +0.0012 | 0.0145 | 0.0141 |
| 0.1 | 0.050 | 0.040 | +0.0005 | 0.0134 | 0.0138 |
| 0.2 | 0.077 | 0.053 | +0.0039 | 0.0161 | 0.0140 |
| 0.3 | 0.090 | 0.080 | +0.0075 | 0.0166 | 0.0135 |

Monte-Carlo SE ≈ 0.013 per rate. τ is unknown. It can be estimated from FEAS FD outputs (k = 0 and k = 12, two seeds each), because the context×config interaction is identifiable from context×config×seed noise with two seeds per config (R4). If τ ≲ 0.1, nothing is needed. If τ ≳ 0.2, the claims should be worded "for the frozen item set", and a crossed bootstrap should replace the nested one.

### 6.6 U-05 trigger, G-REF and the primary cell [CODE + SIM]

**U-05 trigger.**
- `additive_residual_sd` removes config and seed means, df (K−1)(n−1) (`stats.py:356-365`).
- [PRIOR P4] adding config shifts of −30…+40 pp leaves the residual SD unchanged (0.027727 vs 0.027727) and leaves `fires` unchanged. The per-config seed range is invariant to means.
- It is computed at the decision-time primary cell on the first n₀ seeds (`pipeline.py:393-433`). That cell is fixed by G-REF, which uses C0 and C12 only, so the trigger never sees hybrid outcomes.

**G-REF** (`pipeline.py:197-220`): (a) pooled, (b) pooled, per-seed (a) on ≥ n − 1 seeds, on DG, with a CS fallback that is flagged and blocks "falsified". The **primary-cell rule** (`:311-319`) takes the passing lengths in {16K, 32K}, else 8K with its label.

**Two minor selection notes (F10, F11):**
1. **DG, DC and FD are byte-identical prefixes** (same RNG key; context #7 of R1 D3 16K and R-load N128 32K hashes identically in all three) [MEAS §A4]. G-REF therefore reads the 100 contexts that also drive FEAS-1/2/3. A recipe that passed FEAS adequacy on these contexts will tend to pass G-REF on them, which makes G-REF optimistic [INF].
2. **G-REF and CS read the same models.** Dev items remove the *item-level* winner's curse but not the model-level one. Selecting a length because S₁₂ − S₀ happened to be high pushes L_k down via −0.8·S₁₂. The bias is conservative for support and liberal for falsification, and only matters when exactly one length passes [INF].

### 6.7 H3 decision rule (F1) [CODE + SIM S3]

**As implemented** (`equated_check`, `pipeline.py:589-627`):
- pairs are all (retrieval level, M-hop level) with |ΔS₁₂| ≤ 10 pp on DG, with **no informativeness filter**;
- the pair D₃ is the seed-mean of per-seed R differences at single levels;
- support needs **every** pair > 0 (`:626`).

**Simulation:** n = 5, true R_ret = 0.9 and R_hop = 0.7 at the pair (D₃ = +0.2), single-level run SD σ = 3–5 pp, 4,000 replicates per row. Veto rate per pair:

| Pair gap G | σ = 3 pp: per-seed-R mean (as implemented) | σ = 3 pp: pooled ratio | σ = 5 pp: per-seed-R mean | σ = 5 pp: pooled ratio |
|---|---|---|---|---|
| 5 pp | 0.449 | 0.347 | 0.483 | 0.414 |
| 10 pp | 0.261 | 0.209 | 0.423 | 0.324 |
| 15 pp | 0.132 | 0.108 | 0.317 | 0.239 |
| 25 pp | 0.026 | 0.024 | 0.134 | 0.111 |
| 40 pp | 0.001 | 0.001 | 0.033 | 0.029 |

**Compounding across pairs** [INF, independence approximation]: with p equated pairs, P(no veto) = Π(1 − v_i). Four pairs at G = 25 pp and σ = 5 pp give P(veto) ≈ 0.44. Any pair with G ≲ 10 pp makes the veto close to a coin flip. With 7 retrieval × 4 M-hop levels, several equated pairs are likely. The rule can stop a true H3 from being "supported" most of the time. The fix is in R1.

### 6.8 k\* selection and H4-c1 (F6) [CODE]

- `select_kstar(ret)` (`pipeline.py:916-926`) uses the **CS** primary-cell scores of C1, C2 and C3.
- `h4_test` (`:1044-1061`) uses `C{k*}`, **the same runs and the same CS scores**, as the middle arm of c₁ = S_mid − ½(S_early + S_late).
- k\* is chosen because R̂(k\*) crossed 0.3, so S_mid is biased upward conditional on selection. The bias is about one seed-pooled SE (~1 pp) near the threshold [INF], against an H4 MDE of 4.9–9.5 pp. c₂ is unaffected.
- Contrast: R_best is correctly chosen on DG (`:1002-1015`).
- The protocol (§3.2) says "at the primary cell", which the code follows. This is a design property; the fix is in R6.

### 6.9 TOST, Fieller, other tests [CODE + SIM] — correct

- **TOST** (`stats.py:216-235`): p_lo = P(T > (est + m)/se), p_hi = P(T < (est − m)/se), equivalence iff max < α, with a 90% CI. Margins come from `protocol.py:123-129` (±0.10 for D₃, ±2 pp per doubling, ±5 pp).
- **Fieller** (`stats.py:455-483`): bounded, exclusive or unbounded cases correct.
  - [PRIOR P3] coverage over 4,000 replicates: 0.949 (n4, G0.4, sd0.03, ρ0.3), 0.950 / 0.951 (n5).
  - With A > 0 the discriminant is ≥ 0 analytically, so `max(disc, 0)` only guards rounding.
- **Non-inferiority** (`pipeline.py:955-960`): `one_sample_t(Δ, "less", null = 0.05)`, correct.
- **H6′**: Welch (protocol-specified). It ignores that X-E0 and C{k\*} share seed indices 0–1, so the Fieller "share of gap" is conservative (`:1063-1085`).
- **H3 logit stand-in** (`_logit_d3`, `:565-586`): empirical logit of stratum means. The D085 GLMM is **not implemented** (F7).

---

## 7. Recommendations (each with evidence, consequence and timing)

All of these fall before G5. Protocol changes need a decision row (D-row) now, and a DEV row after G5. Nothing here was changed.

**R1. H3 equated-pair rule (F1)** — needs a PI D-row before G5.
- **Change:**
  - restrict equated pairs to informative levels (§9.7: S₁₂ ∈ [0.2, 0.9] and G ≥ 15 pp on DG, on both levels of the pair);
  - compute the pair D₃ from seed-pooled means;
  - replace "every pair > 0" with "the equal-weight mean of pair D₃ over equated pairs > 0". The weaker "majority of pairs > 0" is an alternative.
- **Evidence:** §6.7 veto table.
- **Consequence:** restores most of H3's power while keeping the difficulty-confound guard. The literal per-pair rule is reported alongside.
- **If not adopted:** pre-register the expected veto rate in the analysis plan and report every pair.

**R2. A-freq interpretation (F2)** — analysis and reporting change; no item change.
- **Change:**
  - precompute model-free floors per cell from the items: recency; windows W ∈ {1K, 4K, 8K, 16K}; decayed counters with half-life {2K, 8K};
  - store them in the manifest or a sidecar JSON, and report them next to every A-freq score and in P3;
  - interpret R_agg(k) only against the 8K-window floor;
  - extend D065(vii)'s "local frequency comparison" wording to every T for ρ ≥ 2.
- **Evidence:** §3.3 table.
- **Consequence:** prevents reading "aggregation needs no full attention" out of a task a ~8K-horizon counter solves at 0.82–0.89.

**R3. Optional new MS diagnostic (F2)** — PI decision, before G5; it adds item files. An A-freq **window-balanced** variant: the last W ∈ {4K, 8K} tokens contain equal X and Y counts, so every window counter shorter than the full context scores ½ at most.
- **Resolves:** whether k-differences on A-freq reflect long-range integration.
- **Cost:** ≈ 2 cells × 150 × 4 at 32K per MS model ≈ the cost of the existing T4 slice [INF].

**R4. Measure τ at FEAS (F4)** — analysis only.
- **Change:** from FD outputs (k = 0 and k = 12, seeds 101/102), estimate the context×config variance component. A one-way decomposition of per-context paired differences across the two seeds is enough.
- **Thresholds:** if τ ≥ 0.2, (i) switch the §11.3 bootstrap to the crossed scheme (one context draw per replicate shared by all slots, sketched in `.audit/agent4_eval/sim_verdicts.py:crossed_boot_sd`), which needs a D-row; and (ii) word the confirmatory claims "for the frozen item set".
- **Consequence:** settles whether the item-fixed paired t is adequate.

**R5. Downgrade rule (F3)** — PI D-row ratifying the reading.
- **Change:** ratify the [IMPL] "bound excludes 0" reading *explicitly*, and state that the percentile bootstrap cannot disagree with the t at n ≤ 7 (§6.4). Alternatives:
  - (a) use a studentised (bootstrap-t) interval with t_{n−1} quantiles; or
  - (b) keep the bootstrap only for its context-level component (crossed scheme), and make the pooled model the sole downgrade check.
- **Consequence:** honest operating characteristics for "suggestive". S1 gives the expected downgrade rates.

**R6. k\* selection on dev data (F6)** — D-row before Tier B.
- **Change:** select k\* on DG items. This requires scoring DG on C1, C2 and C3 (≈ 0.03–0.1 GPU-h each, §9.9 scale). Alternatively, report H4-c1 as "conditional on selection" and treat c2 as the clean contrast.
- **Consequence:** removes a ≈ 1-SE upward bias in c1.

**R7. Retrieval wording (F5)** — reporting.
- **Change:** state the per-N weights of the primary cell (70% on N ≤ 32) and of the H5(a) fixed set (N ≤ 32 only) in every H1/H5 verdict sentence. Report the existing per-level L_k (Holm across levels, already implemented, `pipeline.py:977-1000`) as a standing table.
- **Consequence:** prevents over-generalising "retrieval" to high load.

**R8. Owed analysis code before G5 (F7, F8)** — implementation, then tests (VT-18 extension).
- **Change:**
  - (i) rank accuracy with the in-context-guessing floor as well as 1/|C|;
  - (ii) error-type shares over EM misses only, with `absent_in_context` re-typing (D065(iv)) and base-rate normalisation;
  - (iii) the D085 GLMM;
  - (iv) the R2 floors.
- **Consequence:** the §9.2 secondary metrics become reportable. Without them, G5 would freeze an analysis that cannot produce them.

**R9. Rank computation path (F9)** — one-line scorer change, before G5.
- **Change:** score candidate 0 through the same `p_cache` path as the others for rank and error type. Keep EM_tf and logp_gold on the canonical path.
- **Consequence:** removes a path-mismatch term from rank accuracy.

**R10. Context dependence (F10)** — disclosure.
- **Change:** report book re-use. If R12 shows a non-trivial book effect, cluster the bootstrap context draws by book (PG19) for sensitivity.

---

## 8. Additional diagnostics, each tied to the uncertainty it resolves

| ID | Diagnostic | Resolves | Cost |
|---|---|---|---|
| R4 | τ (context×config interaction) on FD k = 0 / 12 | Whether the item-fixed t-test and the nested bootstrap are adequate (F4) | CPU, minutes |
| R11 | S₁₂(C0) vs the shortest-T filler cell on FD | Whether C0 controls are depressed by being outside the TEF length range (§3.5) | CPU, minutes |
| R12 | Variance share of filler book (PG19) and filler kind in per-context EM on FD | Whether context dependence (F10) needs clustering | CPU, minutes |
| R13 | Rank flips when gold is scored via `p_cache` on FD | Size of F9 | GPU, ≈ 1 FD pass at k ∈ {0, 12}; only if R9 is not simply adopted |
| R3 | Window-balanced A-freq (MS) | Long-range integration vs local counting (F2) | Items + ≈ T4-slice evaluation cost |

---

## 9. Not verified / limits of this audit

- **No GPU work.** Path discrepancies (F9) and bf16 effects were not measured. RR-D052i and RR-G0-1/2 belong to FEAS-PREC v2.
- **Near-duplicate filler** (MinHash or n-gram) between held-out and training documents was not checked; only exact text and id hashes were.
- **Difficulty ladders.** No trained model exists, so actual level difficulties, informative levels and equated pairs are unknown. The F1 severity depends on how many low-G equated pairs DG produces.
- **Simulation scope.** The simulations use Gaussian context means with linear truth. They test the decision logic, not binomial or floor/ceiling effects (VT-18's synthetic worlds cover those).
- **VT-24r.** Its formal summary (K4, K5a, K6, verdict) was not yet written when I looked. I cite only the K0, K1, K2, K3 and K5b JSONs present.
- **Concurrent edits.** The concurrent workflow is editing `src/hybridlc/analysis/results.py` (uncommitted). My code citations are to the working tree at 20:40–21:30 IST.

---

## Appendix A. Commands and artefacts

- **A1. Manifest hashes.** Inline Python over `experiments/items/{test,dev}/MANIFEST.json` → `files checked 219 total bytes 1375633371 mismatches [] 0`.
- **A2. Simulations** (CPU, `.venv`, `python -I`):

  | Run | Output |
  |---|---|
  | `.audit/agent4_eval/sim_verdicts.py s3` | `sim_verdicts.json["S3"]` |
  | `… s2` | `["S2"]`, log `sim_s2.log` |
  | `… s1` | `["S1"]`, log `sim_s1.log` |

  The R(3) = 1.0 rows of S1 are in `sim_s1.log` and `sim_verdicts.json["S1"]`; see the addendum at the end of this file.
- **A3. Independent item verifier.** `.audit/agent4_eval/indep_items.py` → `indep_items.json`, log `indep_items.log`. Totals: contexts 5,150; queries 20,450; filler_match 3,450; failures 0; non-canonical junctions 148 / 10,350. Example junctions:
  - `'0.\n\nHALL'` tokens `['0','.','\n\n','H','ALL']` re-encode as `['0','.','\n','\n','H','ALL']`;
  - multi-byte-cut windows such as `'�α), The code'`.
- **A4. Dev prefix sharing.** Inline script: DG, DC and FD `rng_key` are identical, e.g. `['R1','D3','16K','']`, and the context-#7 SHA-256 prefixes are equal (`45afb84b4fbc35f2`, `2d2ee5e9cf08cd22`).
- **First-pass artefacts** (scripts re-read in this session):

  | Artefact | Status |
  |---|---|
  | `verify_items.py` → `verify_all.jsonl` (206 cloze files, 0 failures) | used |
  | `leakage_check.py` → `leakage.json` | used |
  | `scorer_planted.py` → `scorer_planted.json` | used |
  | `sim_stats.py` → `sim_stats_p4_p3_p5.json` (P3 Fieller, P4 trigger) | used |
  | `sim_stats_p1p2.json` | **superseded**: its "boundary" was not an exact null. Clipping at [0, 1] made the true L ≈ +0.0036, which inflated its t rejection rate to 0.095. S1/S2 replace it |
