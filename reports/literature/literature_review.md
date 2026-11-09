# Literature review: how much full attention do hybrid LMs need for long context?

**Role:** Principal investigator (PI), consolidating the five Stage-1 specialist reports. **Date:** 2026-11-09.
**Inputs read in full:** `01_nemotron_h.md`, `02_architecture.md`, `03_evaluation.md`, `04_methodology.md`, `05_implementation.md` and their DB files.
**Merged database:** `reports/literature/literature_db.yaml`. It has 117 entries: the agents' 151 entries deduplicated to 114, plus 3 added by the PI. Each entry records which agents reported it (`reported_by`), the highest access level any agent reached, each agent's own access level, a `ratio_evidence` classification, and `pi_notes` where the PI resolved a conflict.
**Checks the PI ran for this review:**
1. Re-read the setup facts of arXiv 2507.06457 in both v1 and v2 via WebFetch [read-section].
2. Inspected `fla/layers/gated_deltanet.py` at commit `98193113` and counted the parameters of four GDN configurations exactly, by instantiating them on the meta device.
3. Inspected the Zoology reproduction configs and README figure at commit `1ad20d1`.
4. Ran two extended web searches on 2026-11-09 for newer controlled ratio sweeps and read three abstracts (§12.4).

**Conventions.**
- Every claim about a paper or repo carries a link and an access tag: `[read-full]`, `[read-section]`, `[abstract-only]` or `[code-inspected]`. The tag is the access level of the reading that supports that particular claim.
- Epistemic tags:
  - **[EST]** established, i.e. reported with data in the cited source;
  - **[HYP]** a hypothesis, from the authors or from us, not yet tested;
  - **[INF]** an inference, interpretation or arithmetic by the PI or an agent;
  - **[MEAS]** measured by an agent on our RTX 5080. The GPU was shared with CARLA during those measurements, so all timings are indicative only.
- **r** is the fraction of token-mixing layers that are full softmax attention, r = k/L. Linear:full ratios are written "a:1".

---

## 0. Bottom line

1. **Production ratios are inherited, not derived from long-context evidence** [EST].
   - Nemotron-H's "~8% attention layers" is copied from Waleffe et al. "As suggested by prior work" ([Nemotron-H](https://arxiv.org/abs/2504.03624) §2.1 [read-full]).
   - MiniMax-01 reports a final 7:1 ratio but not the sweep that chose it ([MiniMax-01](https://arxiv.org/abs/2501.08313) [read-section]).
   - We found no public ablation behind Qwen3-Next's 3:1 ([config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) [code-inspected]).
2. **Controlled sweeps exist and agree on language-modelling loss** [EST]. They are Waleffe 130M, Mamba-2 350M, Bae 350M/1B, Wang 340M/1.3B and Kimi at about 653M activated parameters (§2).
   - Once attention is about 4–10% of layers, loss is flat or U-shaped in r.
   - The first attention layer captures most of the loss gain.
   - An all-attention model is often *worse* than the best hybrid.
3. **Retrieval is not flat in r** [EST]. In the only controlled sweep that reports a recall metric ([Wang et al. 2025](https://arxiv.org/abs/2507.06457), Table 7 [read-section, PI-verified]), averaged RULER recall goes:
   - pure linear 0.256;
   - 24:1 0.338;
   - 12:1 0.340;
   - 6:1 0.390;
   - 3:1 0.397.

   Language-modelling averages stay at 0.472–0.480 across ratios. The 340M models' recall was "insignificant" and is not reported (v1, Table 2).
4. **There is almost no evidence on how the ratio effect changes with length (H5)** [EST/INF].
   - The only ratio × length data is a three-configuration continued-pretraining ablation in Nemotron-Flash-3B. It used sliding-window rather than linear layers. The penalty for keeping only one full-attention layer appears beyond 16K and grows towards 32K.
   - Every other long-context comparison mixes up attention capability with RoPE extrapolation beyond the training length (§3).
5. **Task dependence (H3) is supported only indirectly** (Jet-Nemotron retrofit; Waleffe 8B per-task tables; MiniMax M2 blog), and **placement (H4) is open for training from scratch** (§4–5).
   - From-scratch placement effects were measured only on loss and are tiny (≤ 0.004 nats).
   - Retrofit and conversion studies report large placement effects.
6. **Positional encoding is a first-order confound** [EST]. In several hybrids, removing position encoding from the full-attention layers (NoPE) improves long-context retrieval (§7).
7. **At 20–60M parameters, natural-language and instruction-style long-context benchmarks score at the floor** [EST at 340M, INF for us]. Cloze or log-prob scoring and an identical synthetic-task exposure are needed. Aggregate perplexity is a weak signal (§8).
8. **Novelty, stated honestly** [INF]. The agents' searches and the PI's 2026-11-09 searches found no study that combines all of the following:
   - a full 0→1 sweep with params, data and recipe held fixed;
   - several seeds;
   - per-family (retrieval, multi-hop, aggregation) results at 1K–32K, all within the training length;
   - a from-scratch placement arm;
   - inference cost measured on a single implementation;
   - 20–60M scale.

   Our study is best described as an **extension and partial replication** of Wang et al. (2507.06457) and of Waleffe et al.'s 130M loss sweep (2406.07887). It is not the first ratio study (§12).

---

## 1. Production hybrids: what ratios are used, and how they were chosen

| Model | Cheap mixer | Full attention among mixers | Placement | Position encoding on full layers | How the ratio was chosen | Source [access] |
|---|---|---|---|---|---|---|
| Nemotron-H-8B | Mamba-2 | 4/28 = 14.3% (7.7% of all layers incl. MLP) | 7, 18, 29, 40, hand-placed so each attention layer precedes an MLP | none | inherited from Waleffe et al. | [2504.03624](https://arxiv.org/abs/2504.03624) [read-full]; HF config [code-inspected] |
| Nemotron-H-56B / 47B | Mamba-2 | 10/64 = 15.6% / 5/50 = 10% | every 11 layers from 7 / pruned | none | inherited; 47B pruned with MiniPuzzle (short-context evaluation only) | same [read-full] |
| Nemotron Nano 2 12B / 9B | Mamba-2 | 6/34 = 17.6% / 4/31 = 12.9% | evenly dispersed; the 9B kept the 4 middle layers [INF] | none | "7–8% is reasonable" cited from prior work | [2508.14444](https://arxiv.org/abs/2508.14444) [read-section] |
| Nemotron 3 Nano 30B-A3B | Mamba-2 (+MoE) | 6/29 = 20.7% | 5, 12, 19, 26, 33, 42 | none | not ablated | [2512.20848](https://arxiv.org/abs/2512.20848) [read-section] |
| Nemotron-Flash 1B / 3B | DeltaNet + Mamba2 | 2/12, 3/18 = 16.7% | searched | — | evolutionary search on an LM-loss proxy | [2511.18890](https://arxiv.org/abs/2511.18890) [read-full] |
| Mamba-2-Hybrid 8B (Waleffe) | Mamba-2 | 4/28 = 14.3% | even spacing; first layer Mamba | none | 130M loss sweep | [2406.07887](https://arxiv.org/abs/2406.07887) [read-full] |
| Jamba | Mamba | 1:7 | period 8, offset 4 | none | 1:3 ≈ 1:7 on LM metrics | [2403.19887](https://arxiv.org/abs/2403.19887) [read-section] |
| MiniMax-01 | Lightning (linear) | 7:1 | 1 in 8 | RoPE on half the head dimension | sweep not reported | [2501.08313](https://arxiv.org/abs/2501.08313) [read-section] |
| Qwen3-Next / Qwen3.5 / Olmo Hybrid | GDN | 3:1 | `i % 4 == 3` (end of each group) | partial RoPE (Qwen3-Next: 0.25, θ = 1e7) | no public ablation found | [Qwen3-Next config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) [code-inspected]; transformers 5.22.2 configs [code-inspected] |
| Kimi Linear | KDA (channel-wise gated delta rule) | 3:1 | periodic | none on all MLA layers | perplexity ablation (§2) | [2510.26692](https://arxiv.org/abs/2510.26692) [read-section] |
| Griffin / RecurrentGemma | RG-LRU | 2:1 with *local* MQA (w = 1024), no global attention | periodic | RoPE | not ablated in text read | [2402.19427](https://arxiv.org/abs/2402.19427) [read-section] |
| Samba; GDN-H1 | Mamba; GDN | 1:1 with SWA-2048 (local) | alternating | RoPE | — | [2406.07522](https://arxiv.org/abs/2406.07522) [read-section]; [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) `b53d6d3` [code-inspected] |
| Hymba-1.5B | parallel SSM heads + SWA | full attention in 3 layers | first / middle / last | — | ablation at 300M | [2411.13676](https://arxiv.org/abs/2411.13676) [read-section] |
| Jet-Nemotron-2B | JetBlock (GDN + dynamic conv) | 2 of 28 full (+2 SWA) | 0-indexed 14, 19 | — | PostNAS search on a pretrained Qwen2.5 | [2508.15884](https://arxiv.org/abs/2508.15884) [read-section]; [config](https://github.com/NVlabs/Jet-Nemotron) `94cbd12` [code-inspected] |

**Takeaways** [INF].
- In deployed hybrids, global attention makes up about 10–25% of mixers, i.e. ratios from about 3:1 to 9:1. NVIDIA's own share has drifted up over time: 14.3% of mixers in Nemotron-H-8B, 20.7% in Nemotron 3 Nano.
- None of these choices rests on a published, controlled long-context ratio study.
- Several of these designs use *local* SWA or MLA/GQA as the "cheap" layer. Those are different manipulations from full vs linear attention.
- The "8%" convention counts MLP layers. Over mixers only, the Nemotron-H family sits at 13–21% ([01 §1.2](./01_nemotron_h.md), configs [code-inspected]).

---

## 2. Controlled ratio evidence (the core prior work)

| Study | Scale / data / context | Cheap layer | Ratios tested | Held fixed | Metric(s) | Key numbers | Seeds | Main limitation for us |
|---|---|---|---|---|---|---|---|---|
| [Waleffe et al. 2024](https://arxiv.org/abs/2406.07887) Fig. 4 | 130M, 24 layers; tokens, sequence length and MLP share **not stated** | Mamba-2 | 0, 1, 2, 3, 4, 5, 7, 12, 24 of 24 | per-layer params (head count and MLP expansion) | validation loss only | 2.615, 2.562, 2.554, 2.555, 2.555, 2.560, 2.568, 2.593, 2.631 (digitised, ±0.002) [read-full]. One layer gives about 87% of the gain [INF] | 1 | loss only; possible MLP confound at 100% |
| [Mamba-2 §9.2.3](https://arxiv.org/abs/2405.21060) | 350M, 48 layers | Mamba-2 | 0–24 of 48 attention layers | size, data | perplexity | 8.60 (0), 8.38 (1), 8.26 (6, best), 8.50 (24); Transformer++ 8.68 [read-section] | 1 | perplexity only |
| [Bae et al. 2025 (Meta)](https://arxiv.org/abs/2510.04800) | 350M / 1B, 60B tokens, 8K context | Mamba-2 | Transformer:Mamba 1:0, 1:1, 1:3, 1:5, 1:12, 0:1; front / middle / end placement | params, tokens, recipe | DCLM/PG19 loss, few-shot; NIAH only as a figure | 1B DCLM loss: Transformer 2.750, 1:1 2.725, 1:3 2.733, 1:5 2.735, 1:12 2.741, Mamba 2.758. Placement ≤ 0.004 nats [read-section] | not stated | NIAH figure only; RoPE; Transformer NIAH collapses beyond 8K (extrapolation) |
| [Wang et al. 2025](https://arxiv.org/abs/2507.06457) | 340M / 20B and 1.3B / 100B, FineWeb-Edu, FLA; context "2,048" (v1) or "4,096" (v2) | DeltaNet, GDN, GLA, HGRN, HGRN2, RetNet | pure, 24:1, 12:1, 6:1, 3:1 + Transformer | data, tokens, AdamW cosine; **param matching not stated** | RULER (averaged), commonsense | Table 7 recall average: 0.256, 0.338, 0.340, 0.390, 0.397; Transformer 0.423 (Table 5); GDN pure 0.356 → 3:1 0.436; 340M recall "insignificant" [read-section, PI-verified] | none reported | averaged recall; no length resolution; no placement arm; no seeds; full layer last in each block |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) ablation | ~653M activated, 16 layers, 38.8B tokens, 4K | KDA | KDA:MLA 0:1, 1:1, 3:1, 7:1, 15:1 | FLOPs, recipe | perplexity | val ppl 5.77, 5.66, 5.65, 5.70, 5.82; no conv 5.70; no output gate 5.67 [read-section] | not stated | perplexity only |
| [Jamba](https://arxiv.org/abs/2403.19887) | 1.3B / 250B | Mamba | 1:3 vs 1:7 | size, tokens | LM metrics, ICL | "virtually no difference" (OLLM 16.5 vs 16.0); pure Mamba fails ICL format (IMDB 48.8 vs hybrid 90.9) [read-section] | — | two ratios |
| [Qiao et al. 2026](https://arxiv.org/abs/2606.15378) | **15M–477M non-embedding**, 16K / 32K, Muon | SWA, Lightning, Mamba-2, GDN | mostly 1:1; 1:1 vs 1:3 (SWA-128) | data, recipe | C4 loss, LongPPL, RULER, NIAH | 1:3 ≈ 1:1 on loss; 1:3 worse on LongPPL at small scale. NoPE on full layers lifts NIAH@16K from 49.58 to 67.81 (0.22B) [read-section] | — | not a ratio sweep; closest in scale |
| [Nemotron-Flash](https://arxiv.org/abs/2511.18890) §4.3 | 3B, 25B-token CPT at 29K | SWA (8K window) replaces full attention | 3 FA / 2 FA + 1 SWA / 1 FA + 2 SWA | pretrained base, CPT budget | NIAH by length, short recall | short recall 73.25 / 73.37 / 73.77. NIAH single_3 @32K 0.69 / 0.74 / 0.32; @4–8K all ≥ 0.94. Throughput 2939 / 3737 / 4657 tok/s [read-full] | 1 | CPT not from scratch; SWA, not linear |
| [Mamba in the Llama](https://arxiv.org/abs/2408.15237) | 8B, distillation | Mamba / Mamba-2 | 50 / 25 / 12.5 / 0% attention | teacher, data | LM-eval, MT-Bench | 63.84 / 60.55 / 58.21 / 54.74 [read-section] | — | distillation regime; monotone decline |

### Synthesis
- **[EST] LM loss is flat or non-monotone in r.** This holds in every controlled sweep (Mamba-2, Waleffe, Bae, Kimi), and the best hybrid beats all-full:
  - Mamba-2 8.26 vs 8.68;
  - Bae 2.725 vs 2.750;
  - Kimi 5.65 vs 5.77.
- **[EST] The first attention layer gives the largest loss drop.** Mamba-2: 8.60 → 8.38 for the first layer, against 8.29 → 8.26 for layers 3 to 6. Waleffe: 2.615 → 2.562.
- **[EST] Recall rises steeply and then saturates** (Wang Table 7). One caveat: Wang's text says recall "nearly doubles" from 24:1 to 3:1, but its own table shows 0.338 → 0.397 ([02 §9](./02_architecture.md)).
- **[EST] Nemotron-Flash shows that short-context recall can be blind to the number of full-attention layers while long-context NIAH is not.** This is the strongest prior for H3/H5. It has caveats: SWA, CPT, three configurations, one seed.
- **[INF] Gaps that matter for us:**
  - no from-scratch sweep reports retrieval, multi-hop and aggregation separately;
  - none reports a length-resolved ratio curve with every length inside the training range;
  - none has seeds or CIs;
  - none measures inference cost (memory, prefill, decode) on a single implementation across r. Wang's Fig. 5 is a FLOPs Pareto front ([v1 §4.4](https://arxiv.org/html/2507.06457v1) [read-section]).

---

## 3. Length dependence (H5)

- **[EST] Nemotron-Flash-3B** ([2511.18890](https://arxiv.org/abs/2511.18890) Fig. 9 [read-full]):
  - 1 FA + 2 SWA scores 0.99 / 0.96 / 0.92 at 16K (NIAH s1/s2/s3) and 0.78 / 0.60 / 0.32 at 32K;
  - 3 FA scores 0.96 / 0.91 / 0.69 at 32K;
  - every configuration scores ≥ 0.94 at 4K and 8K.
- **[EST] Waleffe 8B, RULER at each model's trained length** ([2406.07887](https://arxiv.org/abs/2406.07887) Table 10 [read-full]):
  - Transformer 77.62 → 62.33 → 56.49 (4K → 16K → 32K);
  - hybrid 81.99 → 74.19 → 69.93.
  - This is confounded by RoPE + NTK scaling against no position encoding, and by different MLP content.
- **[EST] Beyond the training length both families fail, for different reasons.**
  - Bae: the Transformer's NIAH is ≈ 0 beyond 8K, while hybrids hold to about 1.5× the training length (figure only) ([2510.04800](https://arxiv.org/abs/2510.04800) [read-section]).
  - Zero-shot S-NIAH for recurrent models collapses within 2–4× of a 4K training length at 1.3B. S-NIAH-2: GDN 92.2 at 4K → 29.6 at 8K; Mamba2 56.2 → 17.0 ([GDN](https://arxiv.org/abs/2412.06464) [read-section]).
  - "State collapse" beyond the training length appears in from-scratch Mamba-2 models at 36M–85M ([Stuffed Mamba](https://arxiv.org/abs/2410.07145) [read-section]).
- **[EST] Full attention is underused at long range even inside the trained window.** Relative-position frequencies are left-skewed: with a 2,048 window, positions ≥ 1,536 make up < 5% of occurrences ([STRING](https://arxiv.org/abs/2410.18745) [read-section]).
- **[EST] Long-range information flows through the full layers.** Restricting full attention to a 2,048 window sharply hurts LongPPL, whereas restricting the efficient layers barely matters, even for GDN ([Qiao](https://arxiv.org/abs/2606.15378) [read-section]).
- **[INF] What this means for design.** H5 needs every evaluation length inside the training range: a final training stage at 32K for all r, with cells beyond the trained length tagged as extrapolation. It also needs a **distance-controlled** design (fixed needle–query distance, varying total length) to separate retrieval distance from the amount of irrelevant context that fills a fixed state ([03 §5.0](./03_evaluation.md)).

---

## 4. Task dependence (H3) and theory

- **[EST] Waleffe 8B hybrid vs a same-data Transformer** ([2406.07887](https://arxiv.org/abs/2406.07887) Tables 9–10 [read-full]):
  - The hybrid wins variable tracking (multi-hop) by a wide margin: 83.2 vs 7.25 at 16K. The Transformer's failure is a format failure: it copies the one-shot example's answer.
  - The Transformer wins keyword extraction at every length (32K: 53.5 vs 36.5) and LongBench multi-document QA (16K HotpotQA: 48.61 vs 42.17).
  - Common-word extraction collapses for both.
  - These are single-model comparisons, not ratio curves.
- **[EST] Jet-Nemotron** ([2508.15884](https://arxiv.org/abs/2508.15884) [read-section]):
  - about 2 layers are critical for MMLU and 2–3 for retrieval, and they are *different* layers;
  - math needs roughly their union;
  - this is a retrofit of a pretrained Qwen2.5.
- **[EST] Qualitative scale warning** ([MiniMax M2 blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) [read-full]): small-scale hybrids matched full attention on saturated benchmarks, but "clear deficits in complex, multi-hop reasoning" appeared at scale.
- **[EST] Mechanistic and ablation evidence.**
  - Ablating attention in RecurrentGemma and Jamba-Mini drops retrieval to 0%, while keeping 15% of heads preserves near-perfect retrieval ([2510.19861](https://arxiv.org/abs/2510.19861) [abstract-only]).
  - Retrieval runs through a few "Gather-and-Aggregate" heads, and in hybrids attention takes the Aggregate role ([2504.18574](https://arxiv.org/abs/2504.18574) [abstract-only]).
- **[EST] Theory and controlled synthetics.**
  - In-context retrieval is the RNN bottleneck, and **a single Transformer layer** closes the representational gap ([Wen et al.](https://arxiv.org/abs/2402.18510) [abstract-only]).
  - Fixed-state models cannot copy strings longer than their state allows; Pythia-410M beats Mamba-2.8B on phonebook lookup with ≥ 70 entries ([Jelassi et al.](https://arxiv.org/abs/2402.01032) [read-section]).
  - The order in which information arrives limits recurrent recall; repeating the context gives +11.0 ± 1.3 points across 16 recurrent LMs ([JRT](https://arxiv.org/abs/2407.05483) [read-section]).
  - Associative-recall tokens explain 82% of the perplexity gap between attention and gated convolutions at 70M–360M ([Zoology](https://arxiv.org/abs/2312.04927) [read-section + code-inspected]).
- **[HYP → testable] Our prediction.** Retrieval should need ≥ 1 attention layer, and one may suffice. Multi-hop may need composition depth (≥ 2 attention operations). Aggregation (counting) may not be attention-limited at this scale.
- **[INF] Contrasts that can tell these apart:**
  - hops ∈ {1, 2, 4} at fixed k;
  - adjacent vs spread placement at k = 2;
  - link order (forward / shuffled / reversed) as a factor.

---

## 5. Placement (H4)

| Evidence | Regime | Metric | Finding | Source [access] |
|---|---|---|---|---|
| Waleffe | from scratch | validation loss | "no significantly better configuration than to evenly distribute"; first layer Mamba | [2406.07887](https://arxiv.org/abs/2406.07887) [read-full] |
| Bae Table 11 | from scratch, 1:3 and 1:5 | DCLM loss, few-shot | front 2.742 / middle 2.740 / end 2.742 (1:3); differences ≤ 0.004 nats; text recommends middle | [2510.04800](https://arxiv.org/abs/2510.04800) [read-section] |
| Qiao | from scratch, 1:1 | linear probing | retrieval-probe gains concentrate at middle full-attention layers | [2606.15378](https://arxiv.org/abs/2606.15378) [read-section] |
| Samba | from scratch | attention entropy | low-entropy ("retrieval") attention in middle layers | [2406.07522](https://arxiv.org/abs/2406.07522) [read-section] |
| Hymba | from scratch, 300M ablation | recall average | SWA everywhere 29.78; first / middle / last full layers 48.79; all full 49.90 | [2411.13676](https://arxiv.org/abs/2411.13676) [read-section] |
| Jet-Nemotron | retrofit | MMLU, retrieval | learned placement "significantly outperforms" uniform; task-specific layers | [2508.15884](https://arxiv.org/abs/2508.15884) [read-section] |
| FlashMorph | conversion | recall | learned subset beats heuristics | [2606.30562](https://arxiv.org/abs/2606.30562) [abstract-only] |
| Nemotron-H 56B | pretrained | removal MSE | some attention layers rank among the least important (layer 84), others locally important | [2504.03624](https://arxiv.org/abs/2504.03624) [read-full] |
| Nano 2 9B | pruning | — | kept the 4 middle attention layers of the 12B [INF from config patterns] | configs [code-inspected] |
| Borobia et al. 2026 | ablation of Qwen3.5-0.8B / Falcon-H1-0.5B | likelihood, downstream | importance largest in early/mid layers | [2603.22473](https://arxiv.org/abs/2603.22473) [abstract-only] |

**Conventions in use** [code-inspected]:
- Megatron's `pattern_from_ratios` implements Waleffe's Algorithm 1 (even spacing, starting and ending with Mamba). For L = 12 and k = 6 it puts two attention layers next to each other.
- Qwen3-Next, Qwen3.5, Olmo Hybrid and Wang et al. (v2 App. D) put attention at the end of each group.
- Jamba uses period 8, offset 4.

**[INF] Status.**
- H4 is **open for from-scratch training measured on retrieval**. The only from-scratch data are loss differences ≤ 0.004 nats, and the large effects come from retrofit or conversion studies, where layers had already specialised.
- The prior leans towards middle or spread placement over early, and towards keeping layer 0 linear.

---

## 6. Linear mechanisms, and the case for Gated DeltaNet

| Mixer | Core update | State per layer, per sequence, at d = 512 | Params, library default | Params, matched config | Notes |
|---|---|---|---|---|---|
| Softmax MHA | softmax(qKᵀ/√d_h)V | KV cache 2·d·T·2 B = 2 KiB/token, i.e. 64 MiB at 32K | 4d² | 4d² | position encoding needed (RoPE or NoPE) |
| **Gated DeltaNet (GDN)** | S_t = S_{t−1}·α_t(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ; o_t = S_t q_t | config A (4 heads, d_k 64, d_v 128): 128 KiB fp32 + 16 KiB conv cache as stored by FLA [INF] | 6.05d² | **4.03d²** (1,056,904 params) | short conv k = 4, L2-normalised q and k, swish output gate; [2412.06464](https://arxiv.org/abs/2412.06464) [read-section]; [FLA](https://github.com/fla-org/flash-linear-attention) `98193113` [code-inspected] |
| GLA | S_t = Diag(α_t)S_{t−1} + k_t v_tᵀ | d²/(2H) | 4.05d² | 4.06d² with conv | weak standalone recall; [2312.06635](https://arxiv.org/abs/2312.06635) [read-section] |
| Mamba-2 | h_t = a_t h_{t−1} + B_t x_t (scalar decay) | 512 KiB fp32 (expand 2, N 128) | 6.56d² | ≈ 4d² only at expand ≈ 1.25 | fast path needs `mamba_ssm`, which has no torch-2.11 wheel [code-inspected] |
| DeltaNet / HGRN2 / RetNet | delta rule / gated outer product / fixed decay | ≈ d²/H | 4.04d² / 4.00d² / 8.0d² | — | HGRN2 was the best hybrid partner in Wang et al. (6:1 recall 0.434) |
| SWA (w) | windowed softmax | 2·d·w·2 B (w = 64 gives 128 KiB) | 4d² | 4d² | a local "cheap softmax", not linear |

Parameter counts are measured by instantiation [MEAS by the architecture agent; the GDN rows re-checked by the PI].

**Why GDN is the primary mechanism** [EST evidence, INF choice]:
1. It has the strongest standalone recall among the candidates in independent comparisons:
   - Jet-Nemotron block search, retrieval score: GDN 69.3, Mamba2 68.9, GLA 66.6, RetNet 63.7;
   - Wang et al. pure-linear RULER: GDN 0.356, DeltaNet 0.320, HGRN2 0.203, GLA 0.144;
   - Nemotron-Flash 500M pure Wiki PPL: GDN 23.80, DeltaNet 23.87, Mamba2 24.36, attention 24.44.
2. Industry uses it: Qwen3-Next and Olmo Hybrid use GDN, and Kimi uses KDA, a GDN refinement.
3. It can be matched to MHA's 4d² **without changing the MLP**.
4. Its Triton kernels are verified on sm_120 with no CUDA build [MEAS]:
   - chunk forward/backward error against a naive fp32 recurrence is 2.9e-4 in fp32 and 1.2e-3 in bf16;
   - `cu_seqlens` gives exact document isolation;
   - prefill followed by cached decode matches a full forward (fp32 argmax agreement 1.000).

**Known risks.**
- Standalone strength does not predict hybrid strength: HGRN2 beat GDN at 6:1 in Wang et al.
- The decode kernel `fused_recurrent_gated_delta_rule` silently returns NaN if called with positional arguments in the order (q, k, v, g, beta); always pass keywords [MEAS].
- The recurrent state is fp32 even in bf16 models [MEAS].

---

## 7. Positional encoding and other mechanism confounds

- **NoPE on the full layers of hybrids.** Evidence leans towards it, but is mixed [EST]:
  - Jamba: NoPE ≈ RoPE (HellaSwag 39.6 vs 40.1) ([2403.19887](https://arxiv.org/abs/2403.19887) [read-section]).
  - Waleffe 840M after a 16K extension: NoPE 53.43, RoPE θ = 10K 52.61, RoPE θ = 500K 51.52, on short-context tasks with one run each ([2406.07887](https://arxiv.org/abs/2406.07887) [read-full]).
  - Kimi uses NoPE on all MLA layers, and the RoPE variant was worse at long context ([2510.26692](https://arxiv.org/abs/2510.26692) [read-section]).
  - Qiao: NoPE raises NIAH@16K from 49.58 to 67.81 ([2606.15378](https://arxiv.org/abs/2606.15378) [read-section]).
  - Global NoPE with local RoPE at 1:3 raises RULER retrieval at 256K from 57.1 to 74.8; a higher RoPE θ in the local layers hurts NoPE retrieval ([2501.18795](https://arxiv.org/abs/2501.18795) [read-section]).
  - NoPE length-generalises best on small synthetic tasks ([Kazemnejad et al.](https://arxiv.org/abs/2305.19466) [read-section]).
  - Against: Samba's NoPE variant "blew up" beyond the training length ([2406.07522](https://arxiv.org/abs/2406.07522) [read-section]).
- **[code-inspected] FLA's hybrid config cannot express NoPE**, because `rope_theta` must be greater than 0. FLA's `Attention` layer also hard-requires the `flash_attn` package. We need our own SDPA attention module.
- **[EST] The short conv and output gate exist only in the linear layers.**
  - Kimi's ablation: removing the conv moves validation perplexity from 5.65 to 5.70; removing the output gate gives 5.67 ([2510.26692](https://arxiv.org/abs/2510.26692) [read-section]).
  - FLA warns "ShortConvolution is crucial to the performance" (`fla/layers/gated_deltanet.py` L190–191 [code-inspected, PI]).
  - Full-attention output gates reduce massive activations in GDN hybrids ([2608.12149](https://arxiv.org/abs/2608.12149) [abstract-only]).
- **[code-inspected] Published GDN models match total params by changing depth.** The default GDN layer has about 6d² params against MHA's 4d², so NVlabs GDN-0.4B has 11 layers vs 12 for H1, and flame's 340M configs have 21 GDN layers vs 24 Transformer layers. A fixed-depth ratio sweep cannot do this, so the mixers must be matched instead.
- **[EST] GQA/MLA change the KV axis.** In hybrids GQA costs only +0.04% perplexity ([Waleffe](https://arxiv.org/abs/2406.07887) [read-full]), but it shifts the whole memory frontier. The number of KV heads must stay fixed across r.

---

## 8. Evaluating long context at 20–60M parameters

- **[EST] Floors.**
  - Wang et al.'s 340M models show "insignificant" zero-shot RULER recall (v1, Table 2 [read-section, PI-verified]).
  - LongBench averages are 11–18 even for 1.3B models ([GDN](https://arxiv.org/abs/2412.06464) [read-section]).
  - LongBench v2, InfiniteBench and Michelangelo are far beyond our scale ([2412.15204](https://arxiv.org/abs/2412.15204), [2402.13718](https://arxiv.org/abs/2402.13718) [abstract-only]; [2409.12640](https://arxiv.org/abs/2409.12640) [read-section]).
- **[EST] What can be measured at small scale.**
  - Passkey retrieval works zero-shot in from-scratch Mamba-2 at 36M–85M ([Stuffed Mamba](https://arxiv.org/abs/2410.07145) [read-section]).
  - The associative-recall-hit perplexity slice works at 70M ([Zoology](https://arxiv.org/abs/2312.04927) [read-section]).
  - Fine-tuned GPT-2 137M and Mamba-130M solve BABILong ([2406.10149](https://arxiv.org/abs/2406.10149) [read-section]).
- **[code-inspected, RULER `db43f92`] Problems with RULER as shipped:**
  - lengths are fitted by binary search to at most the target, not exactly;
  - prompts are instructions scored by string match on generated text;
  - `niah_single_1` (one number in a noise haystack) and `vt` (one chain in noise) can be solved by shortcuts [INF];
  - common-word extraction parameters change below 4K.
- **[EST] Perplexity is a weak signal.**
  - LongPPL correlates with LongBench / LongEval / RULER at Pearson −0.96 / −0.90 / −0.90, against −0.18 / 0.24 / 0.27 for standard perplexity ([2410.23771](https://arxiv.org/abs/2410.23771) [read-section]).
  - Making design decisions from perplexity or NIAH "is not optimal" ([ProLong](https://arxiv.org/abs/2410.02660) [read-section]).
  - No correlation between perplexity and long-text understanding ([2405.06105](https://arxiv.org/abs/2405.06105) [abstract-only]).
- **[EST] Synthetic exposure can make tasks measurable, but can also fail.**
  - Samba fine-tuned 100–500 steps before its passkey and phonebook evaluations.
  - ProLong finds clearer signals after SFT.
  - Birdie: fixed pretraining mixtures did *not* give SSMs phonebook retrieval, while an RL-tuned dynamic mixture did ([2411.01030](https://arxiv.org/abs/2411.01030) [read-section]).
- **[EST] Design hazards.**
  - Accuracy is U-shaped over needle position, and putting the query before the context strongly helps recurrent models ([Lost in the Middle](https://arxiv.org/abs/2307.03172) [read-section]; JRT).
  - Literal-match needles overstate ability ([NoLiMa](https://arxiv.org/abs/2502.05167) [read-section]).
  - Synthetic NIAH predicts downstream ability poorly at frontier scale; distractor-dense variants do better ([HELMET](https://arxiv.org/abs/2410.02694) [read-section + code-inspected]).
- **[INF] Sample sizes.**
  - HELMET's and BABILong's 100 items per cell give about ±9.6 pp 95% CIs at p = 0.5.
  - A paired comparison needs n = 500 for a per-cell minimum detectable effect (MDE) of 6.3 pp, or n_eff = 1000 for 4.4 pp (Miller's formula with ω² = 0.25; [2411.00640](https://arxiv.org/abs/2411.00640) [read-section]).
  - Branching 4 queries from one cached prefill gives n_eff ≈ 1000 from 400 contexts at an intraclass correlation (ICC) of 0.2.

---

## 9. Evidence on long-context training recipes

- **[EST] Train at or beyond the target length, mixing long and short data.**
  - Nemotron-H 8B, RULER-128K by continued-pretraining length: 73.68 (128K + synthetic), 70.19 (256K, no synthetic), 79.04 (256K + synthetic), 81.04 (512K + synthetic) ([Nano 2](https://arxiv.org/abs/2508.14444) [read-section]).
  - Continued pretraining on 512K sequences alone hurt short-context scores; mixing in 4K sequences fixed it ([Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) [read-section]).
  - ProLong: long data (books, code) with high-quality short data; train longer than the evaluation length ([2410.02660](https://arxiv.org/abs/2410.02660) [read-section]).
- **[EST] Packing and masking.**
  - Intra-document causal masking beats plain causal masking of packed sequences ([Zhao et al.](https://arxiv.org/abs/2402.13991) [read-section]).
  - Best-fit packing reduces truncation ([Ding et al.](https://arxiv.org/abs/2404.10830) [abstract-only]).
  - [HYP] Packing unrelated documents during continued pretraining may hurt hybrids on distractor-heavy QA (Waleffe [read-full]).
- **[MEAS] Data.**
  - FineWeb-Edu is long-poor. In one shard (726K docs) the median is 628 GPT-2 tokens; docs over 8K are 0.951% of docs and 13.1% of tokens; docs over 32K are 0.037% and 1.79%.
  - Essentially every PG19 book exceeds 32K tokens; the median is about 301K characters ([FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [PG19 card](https://huggingface.co/datasets/deepmind/pg19) [abstract-only] + statistics computed by the implementation agent).
- **[MEAS] Tokenizer.** A 16,384-entry byte-level BPE with single-digit splitting produces 9.8% more tokens than GPT-2 on FineWeb-Edu and 10.1% more on PG19. At d = 512 it cuts tied embeddings from 25.7M to 8.4M parameters.

---

## 10. Evidence on methodology and statistics

- **[EST] How much seeds matter.**
  - Data sampling is the largest variance source. Single-run comparisons give about 10% false positives and 75% false negatives. Bouthillier et al. recommend P(A > B) ≥ 0.75 ([2103.03098](https://arxiv.org/abs/2103.03098) [read-section]).
  - Initialisation and data order contribute comparably ([2002.06305](https://arxiv.org/abs/2002.06305) [read-section]).
  - Outlier seeds are easy to find ([2109.08203](https://arxiv.org/abs/2109.08203) [read-section]).
  - Seed SD on 7B benchmarks is 0.21–2.15 pp, and continuous metrics have far higher signal-to-noise ([2406.10229](https://arxiv.org/abs/2406.10229) [read-section]).
  - In the 124M speedrun, validation-loss SD over 80 reruns is 0.0019 nats ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) `4ea6b937` [code-inspected]).
  - Length generalisation is fragile across seeds ([2402.09371](https://arxiv.org/abs/2402.09371) [abstract-only]).
- **[EST] Statistics with few runs.**
  - With 3 runs, bootstrap CIs undercover ([rliable](https://arxiv.org/abs/2108.13264) [read-section]).
  - Use paired and clustered standard errors and n = (z_{α/2} + z_β)² ω² / δ² ([Miller](https://arxiv.org/abs/2411.00640) [read-section]).
- **[EST] Hyperparameter fairness.**
  - QK-layernorm and z-loss reduce LR sensitivity in small models ([Wortsman et al.](https://arxiv.org/abs/2309.14322) [read-section]).
  - η_opt = 0.3118·C^−0.125 ([DeepSeek LLM](https://arxiv.org/abs/2401.02954) [read-section]).
  - µP transfers across width, not across layer-type composition ([2203.03466](https://arxiv.org/abs/2203.03466) [read-section]).
  - Count non-embedding params ([Kaplan](https://arxiv.org/abs/2001.08361) [read-section]).
- **[EST] Controlled suites keep the data order identical across runs** ([Pythia](https://arxiv.org/abs/2304.01373), [OLMo](https://arxiv.org/abs/2402.00838) [read-section]).
- **[EST/code-inspected] Determinism.**
  - cuBLAS is reproducible with `CUBLAS_WORKSPACE_CONFIG` ([docs](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) [read-section]).
  - FlexAttention's nondeterminism comes from autotuning and is fixed by `torch._inductor.config.deterministic=True` ([note](https://drisspg.github.io/nuggets/FlexAttention-Determinism) [read-full]).
  - FLA kernels autotune, and the GDN and common op paths contain no `tl.atomic_add` [code-inspected]. Autotuning under GPU contention can therefore pick different kernels [INF].

---

## 11. Implementation status on the RTX 5080 (sm_120)

[MEAS] by the implementation and architecture agents; GPU shared during tests.

- **Verified stack:**
  - torch 2.11.0+cu128 (its arch list includes sm_120);
  - triton 3.6.0;
  - flash-linear-attention / fla-core 0.6.0;
  - transformers 5.22.2;
  - SDPA flash, cuDNN and efficient backends;
  - `torch.nn.attention.varlen.varlen_attn` (PyTorch's built-in FA2 kernel) and compiled FlexAttention document masks.
- **Unavailable:**
  - `flash-attn` has no wheel for torch 2.11, and there is no nvcc to build one ([repo](https://github.com/Dao-AILab/flash-attention) [code-inspected]; [issue #1987](https://github.com/Dao-AILab/flash-attention/issues/1987) [read-full]);
  - `mamba_ssm` and `causal-conv1d` have wheels only up to torch 2.10 [code-inspected].
- **FLA carries Blackwell workarounds** [code-inspected]:
  - an inline-asm `tl.dot` fix ([issue #638](https://github.com/fla-org/flash-linear-attention/issues/638) [read-full]);
  - 8-warp configs dropped when Triton is below 3.8.
- **Pitfalls found:**
  - The first Triton call takes about 49 s (JIT and autotune).
  - Eager decode is launch-bound at about 5–7 ms/token, independent of context length, so CUDA graphs are needed.
  - Retained full-sequence logits (512 MB at 16K) distort memory measurements.
  - Whole-model `torch.compile` is slower, because it recompiles per layer on `layer_idx`.
  - Throughput readings varied by about 1.6× between runs because of contention.

---

## 12. Research gap: what is already known, and what is NOT isolated

### 12.1 Already known (confidence in brackets)
- K1 (high): LM loss is nearly insensitive to r once r ≳ 0.05–0.1, and is often lowest for a hybrid (§2).
- K2 (high): pure linear or SSM models fail at associative recall, copying and phonebook-style retrieval beyond their state capacity. A few attention layers fix this at short context (Zoology, Jelassi, Waleffe, Wang).
- K3 (medium): recall rises steeply and then saturates in r (Wang). A ratio of 3:1–6:1 matches or exceeds an all-full Transformer on averaged RULER at 1.3B with 4K context.
- K4 (medium): beyond its training length, full attention with RoPE fails and hybrids degrade more gracefully. Within training lengths, ratio × length data barely exist (one CPT ablation, Nemotron-Flash).
- K5 (medium): NoPE on the full layers of hybrids often helps long-context retrieval.
- K6 (low): different capabilities depend on different attention layers (retrofit evidence).

### 12.2 NOT isolated by any study we located
- **G1. A full 0 → 1 composition sweep in one controlled design** — params matched, tokens and data order identical, one recipe, seeds with CIs — evaluated **per task family** (retrieval vs multi-hop vs aggregation) rather than as a RULER average.
  - Wang et al. is closest. It reports averaged recall only, at 1.3B. Its 340M recall is at floor. It reports no seeds and does not state param matching. Its placement is fixed (full layer last in each block).
- **G2. Length-resolved ratio effects with every length inside the training range** (H5), and a separation of *distance to the needle* from *amount of distracting context*.
- **G3. Task dependence of the required ratio (H3) measured by controlled per-family curves.** Existing evidence is indirect: single hybrid vs Transformer comparisons, or retrofit studies.
- **G4. Placement from scratch measured on retrieval (H4).** From-scratch studies measured loss only, and the large placement effects come from retrofit or conversion.
- **G5. A measured capability–cost frontier** (inference-state bytes, prefill latency, decode throughput) on **one implementation** across r at small scale (H6).
  - NVIDIA's throughput numbers compare different serving stacks ([Nemotron-H §1.6](./01_nemotron_h.md)).
  - Waleffe's 8× decode speedup is *predicted*.
  - Wang reports FLOPs only.
- **G6. Position-encoding × ratio interaction inside a sweep.** Waleffe tested RoPE vs NoPE at a single ratio; Qiao tested NoPE at 1:1 only.

### 12.3 How our design relates to prior controlled sweeps

| Feature | Waleffe 130M | Mamba-2 350M | Bae 350M/1B | Wang 340M/1.3B | Kimi ~653M | Qiao 15–477M | **This study (proposal)** |
|---|---|---|---|---|---|---|---|
| Cheap layer | Mamba-2 (+MLP?) | Mamba-2 | Mamba-2 | 6 linear types | KDA | SWA / GDN / Mamba-2 | GDN (one mechanism) |
| Both endpoints in the same sweep | yes | yes (+Transformer++) | yes | yes (+Transformer) | yes (0:1) | no | yes |
| r grid | 9 points of 24 | 12 points of 48 | 6 | 5 + Transformer | 5 | 1:1, 1:3 | k ∈ {0, 1, 2, 3, 4, 6, 12} of 12 |
| Params matched | per layer | not stated | stated | **not stated** | FLOPs | — | per mixer, model spread ≤ 0.24% |
| Seeds / CIs | 1 / no | 1 / no | — / no | none / no | — / no | — | 3 seeds (confirmatory), 2 (others); hierarchical bootstrap |
| Long-context metric | none | none | NIAH figure | RULER average | none | RULER / NIAH, LongPPL | per-family synthetic suite, 1K–32K, all lengths in training range |
| Length-resolved within training range | — | — | no | no | — | partly | yes, plus a distance-controlled design |
| Placement arm | qualitative (loss) | no | front / middle / end (loss) | no | no | no | 5 patterns at k = 2 (retrieval) |
| Position-encoding control | 840M side test | — | RoPE | — | NoPE | NoPE at 1:1 | one pre-registered scheme for all r + ablation at k ∈ {1, 3, 12} |
| Measured inference cost | predicted | — | figure | FLOPs only | at 1M (final model) | — | memory (analytic + measured), prefill, decode; idle GPU, CUDA graphs |
| Scale | 130M | 350M | 350M / 1B | 340M / 1.3B | ~653M | 15M–477M | ≈ 49M total (41M non-embedding) |

### 12.4 Novelty statement, and threats to it
- **What is new** [INF]: the *combination* of G1–G6 in one controlled, seed-replicated design at 20–60M, with all evaluation lengths inside the training range.
- **What is replication.** We expect to reproduce, not discover:
  - (a) flat LM loss across r and a large drop from k = 0 to k = 1 (Waleffe, Mamba-2, Kimi);
  - (b) recall rising steeply and then saturating (Wang).
- **Threats to novelty and validity:**
  - The search was not exhaustive. It covered the agents' searches plus two PI extended searches on 2026-11-09, which found only non-sweep work: upcycling ([2604.24715](https://arxiv.org/abs/2604.24715) [abstract-only]), post-hoc ablation ([2603.22473](https://arxiv.org/abs/2603.22473) [abstract-only]), and conversion and distillation papers seen only as search snippets (§14).
  - Our scale may be too small for some effects. Wang's 340M recall was at floor, and MiniMax reports that small-scale parity hid multi-hop gaps at scale. Every claim must be scoped to "≈ 50M parameters, 1.1B tokens, GDN, this recipe".

---

## 13. Disagreements between agents and their resolution

| # | Topic | Positions | Evidence used to resolve it | Resolution |
|---|---|---|---|---|
| D1 | GDN configuration and parameter matching | Architecture: gate on, 4 heads, d_k 64, d_v 128, 4.03d², same MLP. Implementation: gate off, 4 heads, d_k = d_v = 128, 4.04d², same MLP; claims FLA calls the gate "crucial". Methodology: gate on, 4 heads × 128, `expand_v = 1` (5.04d²), MLP 1344 for GDN blocks vs 1536 for attention blocks | (i) FLA source: the "crucial" warning is about **ShortConvolution**, not the gate (`gated_deltanet.py` L190–191 @`98193113`) [code-inspected, PI]. (ii) PI meta-device counts at d = 512: A 1,056,904 (+0.79% vs MHA); B 1,058,952 (+0.99%); C 1,321,096 (+25.99%); D (gate on, 2 heads, d_k 128, d_v 256) 1,054,980 (+0.61%). (iii) Published GDNs keep the output gate and use d_v = 2·d_k (FLA defaults `use_gate=True`, `expand_v=2` [code-inspected]). (iv) State-to-width ratio: A's state is 32,768 = d²/8 elements, the same ratio as Qwen3-Next (32 value heads × 128 × 128 = 524,288 = d²/8 at d = 2048 [config-inspected]); FLA's default at d = 2048 is 0.1875·d² | **Config A is primary.** It keeps the published mechanism, swaps only the mixer (identical MLP) and has a realistic state-to-width ratio. C is rejected because MLP width would co-vary with r. B is rejected as primary because it drops the gate; its 2× state is covered by an optional **state-size arm using config D**. Settled. |
| D2 | Default placement rule | Architecture: centred ⌊(j + ½)L/k⌋. Methodology: end of group, round((j + 1)L/k) − 1. Nemotron agent: Megatron Algorithm 1, with r = ½ fixed by hand | Algorithm 1 puts two attention layers side by side at k = 6 and is asymmetric at k = 2 ({3, 8}) [code-inspected by the Nemotron agent]. End of group puts the single k = 1 layer at the last layer (11), which the methodology agent itself flags as a caveat. Middle-layer retrieval evidence (§5) | **Centred-uniform.** k = 1 → {6}; 2 → {3, 9}; 3 → {2, 6, 10}; 4 → {1, 4, 7, 10}; 6 → strict alternation {1, 3, …, 11}. Layer 0 is always linear for k < 12. End of group ({5, 11}) becomes a placement-arm cell. Settled. |
| D3 | k for the placement study | Architecture k = 2; methodology k = 3; Nemotron agent 2 of 12 or 3 of 16 | At 3:1 retrieval may sit near ceiling: Wang's GDN 3:1 scores 0.436 vs the Transformer's 0.423 [read-section]. That would compress placement differences. k = 2 shares a cell with the curve | **k = 2**, with arms centred / early / middle / late / end of group. A judgement call; listed as unresolved U5. |
| D4 | H1 test point | Methodology: r = ¼ only; README: "small proportion" | r = ¼ is the industrial 3:1. r = 1/12 is the smallest nonzero r and the literal "small amount" | **Two pre-registered confirmatory tests:** H1 at r = ¼ and H1-min at r = 1/12, both in one Holm family. Settled. |
| D5 | Length of the extension stage and RoPE θ policy | Methodology: 16K by default (32K if G4 passes), θ constant. Architecture: 32K, with an identical θ increase. Implementation: train the final phase at the maximum evaluation length | Arithmetic: at T = 32K an attention layer costs 2·T·d = 33.5 MFLOP/token in the forward pass, against ≈ 6.8 MFLOP/token/layer for projections and MLP (2·(4d² + 3·d·1536)). Measured single-layer attention at 32K runs at 0.84 Mtok/s [MEAS]. So r = 1 at 32K runs at roughly 34–46K tok/s, and 0.1B tokens take ≈ 0.8 h. In-distribution evaluation lengths are the main fix for H5 (STRING; Stuffed Mamba) | **L_ext = 32K by default**, falling back to 16K only if gate G4 fails. θ is held constant across stages, and the position-encoding scheme is chosen by a baseline-favouring pilot. Settled. |
| D6 | Vocabulary size | Architecture and methodology illustrate with 32K; implementation recommends 16K with digit splitting | [MEAS] +9.8% tokens vs GPT-2; embeddings drop from 34% to 17% of params; the evaluation agent needs single-token digits | **16,384-entry BPE with digit splitting, tied.** Settled. |
| D7 | MLP width | Architecture 1408; methodology 1536 / 1344; implementation 1536 | Matching mixers removes the need to compensate in the MLP; 1536 = FLA's `hidden_ratio=4` rounding [code-inspected] | **1536 for every block.** Settled. |
| D8 | QK-norm in attention | Architecture: no QK-norm. Methodology: QK-RMSNorm | QK-layernorm reduces LR sensitivity ([Wortsman](https://arxiv.org/abs/2309.14322) [read-section]); GDN already L2-normalises q and k | **QK-RMSNorm on** (per head, 64-dim gains). Reopen if the 32K pilot shows capped attention logits hurting retrieval (U4). |
| D9 | Primary position-encoding policy | Architecture: RoPE primary. Methodology: pilot selection among RoPE 1e4 / RoPE 5e5 / NoPE. Nemotron agent: fixed scheme + RoPE/NoPE ablation at r = 1/12 and 1 | The methodology agent's rule includes the architecture agent's candidates and is conservative, since it picks the scheme that favours the r = 1 baseline | **Pilot rule adopted.** The non-chosen family is ablated at k ∈ {1, 3, 12}. Settled. |
| D10 | Evaluation sample size | Evaluation: 400 contexts × 4 branched queries. Methodology: 500 items. Nemotron agent: ≥ 400 | Miller's MDE: n = 500 → 6.26 pp; n_eff = 1000 → 4.43 pp. Design effect 1.6 at ICC 0.2 [INF arithmetic] | **400 contexts**, with 4 queries where the task allows. Raise to 600 if the pilot design effect exceeds 2. Settled. |
| D11 | Wang et al.'s context length | Architecture: 4K. Implementation and evaluation: 2,048 | PI WebFetch: v1 Limitations says "a 2,048-token context window"; v2 says "a 4,096-token context window"; neither states it in the setup | A version difference. **Unresolved by the source (U1).** |
| D12 | What FLA calls "crucial" | Implementation: the output gate. Architecture: the short conv | `gated_deltanet.py` L190–191 | **The short conv.** The implementation agent's statement was wrong. Settled. |
| D13 | Second mechanism | Architecture: GLA (+SWA-64; HGRN2 third). Implementation: gated pair. Others: none | Budget | None in the core. Optional E5 priority list (proposal §12). Settled. |
| D14 | Access level for shared sources (Waleffe, Nemotron-H, Hymba, Jelassi, Zoology, Based, GDN, …) | Different agents read different amounts | — | The merged DB takes the highest level and records each agent's level in `access_detail`. Settled. |
| D15 | Jet-Nemotron layer indices | Nemotron agent: 15, 20 (SWA 21, 22). Architecture / implementation: 14, 19 (SWA 20, 21) | Paper text is 1-indexed; the released config is 0-indexed [code-inspected] | Same layers. Settled. |
| D16 | Default synthetic-exposure option | Evaluation: prefer (i) when both qualify. Methodology: needs a decision. Implementation: either | Birdie: fixed mixtures can fail for SSMs [read-section] | Pre-registered pilot rule (evaluation agent's), with (i) preferred on ties; frozen at G5. Settled as a procedure. |
| D17 | Context length in stage A | 2–4K / 4K / 4–8K | With document masking, attention cost scales with document length. FineWeb-Edu documents are short: 97% are ≤ 4K GPT-2 tokens [MEAS] | **4K.** Settled. |
| D18 | Size of the LR sweep | Architecture: 3 values at r ∈ {0, 1}. Methodology: 5 values at r ∈ {0, ¼, 1} | Cost about 10 GPU-h; including a hybrid guards against the optimum lying between the endpoints | **Methodology agent's 5 × 3 grid.** Settled. |
| D19 | Whether Wang et al. was "controlled" | Architecture: yes (ratio × mechanism). Implementation: parameter matching not stated | PI check: matching not stated in v1 or v2 | Controlled in design; **parameter matching unverified.** Recorded in the DB `pi_notes`. |
| D20 | Hymba placement | Nemotron agent: first / middle / last. Architecture / implementation: [5, 11, 18, 25, 31] | Two different artefacts: the paper's ablation vs the repo's test config | Both correct. Settled. |

### 13.1 Still unresolved after consolidation
- **U1.** Wang et al.'s training context length (2K in v1, 4K in v2), parameter matching and LR. Unstated in the source.
- **U2.** Waleffe's 130M sweep: MLP share, tokens, sequence length and position encoding are unstated. The 840M confirmation shows no figure.
- **U3.** Whether 20–60M models learn the synthetic formats under exposure option (i) or (ii). This is empirical and decided by pilot gate G2.
- **U4.** Whether QK-norm limits how sharply attention can retrieve at 32K. We found no verified evidence either way; there is a reopen criterion.
- **U5.** Whether placement is better tested at k = 2 or k = 3. A judgement call; k = 2 was chosen.
- **U6.** How much the answer depends on GDN's state size (d_k 64 vs 128). Addressed only by the optional state-size arm.
- **U7.** The idle-GPU throughput multiplier, assumed to be 1.6×. Gate G0 measures it.
- **U8.** Whether CARLA can be paused for idle windows. The user decides.
- **U9.** Whether "Nemotron Edge" means Nemotron-Flash. The user should confirm ([01 §3](./01_nemotron_h.md)).

---

## 14. Not accessed, unverified, or unread leads

- **Abstract-only sources used in this review:**
  - Wen et al.; Gather-and-Aggregate; Some Attention; FlashMorph; Massive Activations; Ring-linear; RWKV-7; HGRN2; Mamba; RetNet; DeltaNet (2406.06484);
  - LongBench / LongBench v2 / InfiniteBench; Hu et al. (perplexity);
  - Zhou et al. 2024; Ding et al. 2024;
  - Borobia et al. 2026; HyLo 2026; Pitorro et al. 2024.
- **Seen only as search snippets, not read (no claims made):**
  - MiniCPM-SALA ([2602.11761](https://arxiv.org/abs/2602.11761));
  - KL-guided layer selection for hybrid distillation ([2512.20569](https://arxiv.org/abs/2512.20569));
  - "Hybrid Linear Attention Done Right" ([2601.22156](https://arxiv.org/abs/2601.22156));
  - Neural Attention Search Linear ([2602.03681](https://arxiv.org/abs/2602.03681));
  - Short-window attention for memorisation ([2509.24552](https://arxiv.org/abs/2509.24552));
  - Log-Linear Attention ([2506.04761](https://arxiv.org/abs/2506.04761)).
  - A snippet claimed that in Pitorro et al. 2024 "two attention layers" suffice and placement matters little. **This is not in the abstract and is not verified.**
- **Not accessible:**
  - the Qwen3-Next blog would not render;
  - Long-Data-Collections and SlimPajama-627B returned HTTP 401;
  - the Nemotron 3 white paper and Nemotron 3 Ultra were not read.
- **Numbers that exist only in figures:**
  - Waleffe Fig. 4 (digitised to ±0.002);
  - Bae's NIAH results;
  - the Zoology README recall–state figure (read by eye; state-size units not verified).
