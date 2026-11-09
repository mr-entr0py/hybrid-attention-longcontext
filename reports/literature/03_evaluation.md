# 03 — Long-Context Evaluation for 20–60M-Parameter Hybrid-Attention LMs

**Author:** Agent 2 (Long-Context Evaluation Researcher), Stage 1 literature investigation
**Date:** 2026-11-09
**Companion DB:** `reports/literature/db_evaluation.yaml`, with one entry per source consulted.
**Status:** This is a proposal for Stage 2 (methodology). Nothing in it is frozen yet.

---

## 0. Conventions and access levels

Every claim about external work carries a URL and an access tag:

- **[read-full]**: I read the whole paper. No source in this report reached this level, so the tag is unused.
- **[read-section]**: I read substantial sections of the arXiv HTML or PDF text. In practice I read large portions through a fetch tool, and some long papers were truncated.
- **[abstract-only]**: I saw only the abstract page, a dataset card or search snippets.
- **[code-inspected]**: I cloned the repository and read the source.

Claims are also labelled **(established)** when a cited source reports them, **(hypothesis)** when they are untested here, and **(inference)** when they are my own reasoning. Where I could not access something, I say so.

Repositories cloned (shallow) into `third_party/` (gitignored):

| Repo | Path | Commit |
|---|---|---|
| NVIDIA/RULER | `third_party/ruler` | `db43f92ff634bbce1c33d1fb15858aec4bba509a` |
| HazyResearch/zoology | `third_party/zoology` | `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` |
| booydar/babilong | `third_party/babilong` | `7a6efee29f5cac03c3c410e6799c80fd2ffe3610` |
| princeton-nlp/HELMET | `third_party/helmet` | `aeadacc6593e9a222a84b0660b186a4e23d2efda` |
| NVlabs/GatedDeltaNet | `third_party/gated_deltanet` | `b53d6d3a161267432a79c1c04af69fa52bddc921` |
| EleutherAI/lm-evaluation-harness | `third_party/lm_eval` | `d6de81643928d653435c431bae19945d41d32520` |

---

## 1. Executive summary

1. **Natural-language long-context benchmarks would sit at the floor for our models, so they should not be headline metrics.** This covers LongBench, LongBench v2, InfiniteBench, HELMET's non-synthetic categories and Michelangelo.
   - The most direct evidence comes from *A Systematic Analysis of Hybrid Linear Attention* ([arXiv 2507.06457](https://arxiv.org/html/2507.06457v1), [read-section]). It is a controlled linear:full ratio study (24:1, 12:1, 6:1, 3:1), and it states that its **340M** models "show insignificant recall capability due to their relatively small parameter count" on zero-shot RULER. It therefore reports recall only at 1.3B (established).
   - Our models are 6–17× smaller than that.
   - LongBench v2 is 4-way multiple choice, and even frontier models score near human-expert level (≈50%) ([arXiv 2412.15204](https://arxiv.org/abs/2412.15204), [abstract-only]).
2. **Instruction-style synthetic prompts (RULER as-is) are also likely to sit at the floor zero-shot (inference).** The 2507.06457 evidence above supports this. The task logic is still the right logic.
   - RULER's prompts are instructions such as "What are all the special magic numbers for X mentioned in the provided text?" (`third_party/ruler/scripts/data/synthetic/constants.py`, [code-inspected]).
   - A 50M base LM cannot be expected to follow them.
3. **Some long-range capability is measurable zero-shot even at ~40M parameters, if the task is posed as a completion (cloze) rather than an instruction.**
   - *Stuffed Mamba* trained Mamba-2 models **from scratch at 36M, 47M and 85M** with next-token prediction only. It then measured 5-digit passkey retrieval, defining T_recall as the maximum length with >95% accuracy ([arXiv 2410.07145](https://arxiv.org/html/2410.07145), [read-section]) (established).
   - Zoology's "AR-hit" perplexity slice measures in-context recall on natural text with no prompt at all. It ran at 70M–360M, and it explained 82% of the perplexity gap between attention and gated convolutions ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927), [read-section]) (established).
4. **Recommended design: three tiers of evaluation.** All tiers are scored by **log-probability / teacher-forced exact match**, with no free generation in the headline numbers.
   - **Tier 0 (zero-shot, no task exposure).** Per-position LM loss on held-out long documents. Distance-resolved "long-range repeat-token" loss, a generalisation of Zoology AR hits. Copy-after-gap loss. Cloze-form passkey and KV retrieval.
   - **Tier 1 (primary capability).** A procedurally generated synthetic suite with three families: retrieval, multi-hop and aggregation. The model sees disjoint-surface synthetic data from the same task families in **one fixed, identical training exposure**. Either (i) mix it into the long-context phase of pretraining, or (ii) run a short identical fine-tune. Section 4 sets out a pilot that decides between them.
   - **Tier 2 (optional external anchor).** BABILong qa1–qa3 after fine-tuning on its train split, with PG19 filler, plus RULER via lm-eval-harness on public HF checkpoints. Tier 2 exists only to validate our generators, not as a headline result.
5. **Teacher-forced argmax exact match on the answer span is identical to greedy-decoding exact match** (inference, follows from the definition of greedy decoding). The condition is that the answer has a canonical tokenisation and that "exact match" means exactly the gold token sequence.
   - It costs one forward pass and needs no generation loop.
   - The same pass gives a continuous log-probability and a distractor-aware candidate ranking.
   - The distractor ranking separates "lost the information" from "retrieved the wrong binding".
6. **Sample sizes.**
   - At p≈0.5, a ±5 pp 95% CI needs **n≈385** independent items per cell, and ±3 pp needs **n≈1,068** (Section 6).
   - Comparisons between models on the *same items* should be paired (McNemar / paired bootstrap). Detecting a 10 pp difference with 80% power then needs only ~155–234 items, depending on the discordance rate.
   - HELMET's 100 samples/task (`third_party/helmet/configs/recall.yaml`, [code-inspected]) give ±9.6 pp. BABILong's public 100/length gives the same. Both are too coarse for a ratio sweep.
   - **Recommendation: 400 contexts per (task, length) cell for headline cells, with up to 4 independently branched queries per context.**
7. **Tokens-at-length must be built in token space.** RULER's generators binary-search the haystack size so that the total is ≤ the target length. Lengths are therefore approximate (`niah.py`, `generate_samples`, [code-inspected]). We should splice pre-tokenised segments so that every sample has **exactly** T tokens. We should also never round-trip through text.
8. **Two shortcut traps in RULER as configured (inference from [code-inspected] RULER configs).**
   - `niah_single_1` uses a *noise* haystack with one 7-digit number. "Copy the only number" solves it without matching the key.
   - `vt` uses a noise haystack with **one** chain. Listing every `VAR` token solves it without tracking.
   - Our versions must include same-type distractors.
9. **Reuse vs build.** Write a **custom generator** (~600 LOC, in `src/eval/`) that borrows RULER's task definitions and word lists (Apache-2.0) and Zoology's MQAR logic. Use RULER through lm-eval-harness (which has a `ruler` task group, [code-inspected]) only to cross-validate the generator on public checkpoints.

---

## 2. Literature review: what exists and what transfers to 20–60M models

### 2.1 Synthetic retrieval benchmarks

**Passkey retrieval.** Mohtashami & Jaggi, *Landmark Attention*, NeurIPS 2023 ([arXiv 2305.16300](https://arxiv.org/html/2305.16300v2), [read-section]) (established):
- The prompt opens with "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them."
- The filler "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." repeats.
- The key sentence is "The pass key is <PASS KEY>. Remember it. <PASS KEY> is the pass key." The question is "What is the pass key? The pass key is".
- The passkey is a random integer in 1–50,000 placed at a random position. Results average 50 trials. A response counts as correct if the passkey is the first integer within the first 100 generated tokens.
- *Transfer:* the format is completion-style and works zero-shot at small scale (Stuffed Mamba, 36M–85M; [arXiv 2410.07145](https://arxiv.org/html/2410.07145), [read-section]).
- *Weakness (inference):* there is one salient number in perfectly repetitive filler, so it tests "find the anomaly and copy it", not key–value binding. Samba fine-tuned on it for 500 steps at 4K and reported perfect retrieval to 256K ([arXiv 2406.07522](https://arxiv.org/html/2406.07522), [read-section]), which shows how easily it saturates. **Use it as a sanity check only.**

**Needle-in-a-Haystack (Kamradt).** [github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), README [read-section] (established):
- The haystack is Paul Graham essays.
- The README's example config sweeps 2K–32K tokens in 8 linear steps × 11 depths on a sigmoid scale × seeds {1,2,3}.
- It also offers multi-needle and "UUID-chain" variants. The UUID-chain variant scores hops correct out of chain length.
- *Transfer:* the length × depth grid design transfers. A single needle with a literal lexical match saturates easily.

**RULER.** Hsieh et al. 2024 ([arXiv 2404.06654](https://arxiv.org/html/2404.06654), [read-section]; code [code-inspected] at `third_party/ruler`) (established).

13 tasks, from `scripts/synthetic.yaml`:

| RULER task | Haystack | Key/value types | #keys/#values/#queries |
|---|---|---|---|
| niah_single_1 | noise (repeated "grass is green…") | word → 7-digit number | 1/1/1 |
| niah_single_2 | Paul Graham essay | word → number | 1/1/1 |
| niah_single_3 | essay | word → UUID | 1/1/1 |
| niah_multikey_1 | essay | word → number | 4 keys (3 distractor needles) |
| niah_multikey_2 | "needle" (haystack made entirely of other needles) | word → number | 1/1/1, all filler are distractor KV |
| niah_multikey_3 | needle | UUID → UUID | 1/1/1 |
| niah_multivalue | essay | word → 4 numbers | 1/4/1 |
| niah_multiquery | essay | word → number | 4 keys, 4 queries |
| vt | noise | `VAR X = 12345` chains | 1 chain, 4 hops |
| cwe | numbered word list | — | freq_cw=30, freq_ucw=3, num_cw=10 |
| fwe | Zipf-coded words | — | α=2.0, top-3 |
| qa_1 / qa_2 | SQuAD / HotpotQA + distractor docs | — | — |

Generator details (code-inspected):
- The needle template is `"One of the special magic {type_needle_v} for {key} is: {value}."`.
- Keys come from adjective-noun pairs from `wonderwords`.
- Numbers are 7-digit. UUIDs are v4.
- Depth is drawn from `DEPTHS = np.round(np.linspace(0,100,40))`. In the essay haystack, needles go in at sentence boundaries (NLTK `sent_tokenize`).
- Length is fit by **binary search over the haystack unit count** until `len(tokens)+tokens_to_generate ≤ max_seq_length`. Samples are therefore at or below the target length, not exactly at it.
- VT uses 5-letter uppercase variable names and a 5-digit root value. Chains are inserted at random positions in the noise haystack in order. Several chains can be interleaved with `shuffle_sublists_heap`.
- CWE switches to `freq_cw=6, freq_ucw=1` (and a smaller few-shot example) when `max_seq_length < 4096`. That matters for 1K–2K cells.
- FWE uses 6-letter random "coded" words, Zipf counts `num_words·k^-α/ζ(α)`, vocab = `max_seq_length//50`, and replaces the top-ranked word with "...". The answer is ranks 2–4.
- Metric: `string_match_all`, the fraction of reference strings contained in the prediction, case-insensitive. QA uses `string_match_part`.
- RULER's default is 500 samples/task/length (`config_tasks.sh: NUM_SAMPLES=500`).

Paper findings (established):
- Effective context length is the longest length whose 13-task average beats Llama-2-7B's 4K score of 85.6%.
- Lengths run 4K–128K. Almost all models are near-perfect on vanilla NIAH but degrade with length on the full suite.
- RWKV-v5 and Mamba-2.8B-slimpj "degrade significantly by 8K" and trail Llama-2-7B even at 4K.
- Larger models (Yi-34B vs Yi-6B) degrade less.
- *Transfer:* the task taxonomy, difficulty knobs and depth sampling transfer. The prompts, the ≤-length fitting and two shortcut-prone configs (Section 1, item 8) do not.

**Lost in the Middle.** Liu et al., TACL 2024 ([arXiv 2307.03172](https://arxiv.org/html/2307.03172), [read-section]) (established):
- Multi-document QA uses 10/20/30 documents with the gold document's position swept.
- A synthetic JSON KV retrieval task uses 75/140/300 UUID pairs.
- Accuracy is U-shaped over position, best at the start and end. The U-shape also appears in base MPT-30B, so it is not an artefact of instruction tuning.
- Putting the query both before and after the data ("query-aware contextualization") made KV retrieval near-perfect.
- *Transfer:* (a) a position sweep must cover both ends. (b) **Query-before-context is a manipulation that strongly favours recurrent models** (inference, consistent with JRT below). We should fix query-after-context as the primary condition and treat query-first as a secondary diagnostic.

**Just Read Twice (JRT).** Arora et al. 2024 ([arXiv 2407.05483](https://arxiv.org/html/2407.05483), [read-section]) (established):
- For recurrent LMs, the order in which information arrives determines what a fixed-size state must store. The paper links this to set-disjointness lower bounds.
- Repeating the context gave +11.0 ± 1.3 points across 16 recurrent LMs and 6 recall tasks.
- *Transfer:* in multi-hop chains, **the order of links in the context (forward vs shuffled vs reversed) is a manipulated variable that differs by architecture** (hypothesis). Our generator must control it explicitly.

**NoLiMa.** Modarressi et al. 2025 ([arXiv 2502.05167](https://arxiv.org/html/2502.05167), [read-section]) (established):
- When the needle and the question share minimal lexical overlap, 11 of 13 ≥128K-claimed models fall to ≤50% of their short-context score at 32K.
- Literal matches make the task easier, and distracting literal matches hurt.
- *Transfer:* our small models have little world knowledge, so latent-association needles are not feasible. Every one of our retrieval tasks is therefore a **literal-match** test. **This is a limitation to state, not something to fix** (inference).

### 2.2 Associative-recall synthetics from the efficient-architecture literature

**Zoology / MQAR.** Arora et al. 2023 ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927), [read-section] via PDF text; code [code-inspected] `third_party/zoology/zoology/data/multiquery_ar.py`) (established):
- Pile pretraining at 70M/160M/360M for 10B tokens.
- Attention beats gated convolutions by ≥ +2.14 / +0.59 / +0.35 PPL at those three scales.
- **AR hits** are "tokens in the final position of a bigram which previously appeared in context, but ≤1250× during training", about 6.4% of tokens. They account for **82%** of the gap.
- A 70M attention model beats a 1.4B Hyena on AR-hit PPL (2.41 vs 3.43).
- At 360M, a BaseConv with **3 attention layers** beats the pure Transformer.
- MQAR generator (code):
  - Keys come from vocab[1:V/2] and values from vocab[V/2:V], with no repeats within an example. KV pairs come first.
  - Queries are placed at power-law-distributed gaps (`power_a=0.01`). Non-query positions are filled with random tokens (`random_non_queries=True`).
  - The example config uses V=8,192. Training sets are (64, 4 KV) … (256, 64 KV). Test sets go up to (1024, 256 KV).
  - Gated convolutions need model dimension to scale with sequence length to solve MQAR. Attention solves it at constant dimension.
- *Transfer:* (a) the **AR-hit slice is a zero-shot, prompt-free long-range-recall metric** that is already validated at 70M. (b) The MQAR generator gives a cheap mechanistic capacity probe.

**Based.** Arora et al. 2024 ([arXiv 2402.18668](https://arxiv.org/abs/2402.18668), [abstract-only]) (established):
- There is a recall vs recurrent-state-size trade-off.
- Sliding window plus linear attention traces a Pareto frontier.
- Based beats Mamba by 6.22 points on recall-intensive tasks at up to 1.3B.
- I did not read the body. The task names SWDE/FDA/SQuAD come from the Gated DeltaNet and JRT papers below, not from my reading of Based.

**Mamba-2.** Dao & Gu 2024 ([arXiv 2405.21060](https://arxiv.org/html/2405.21060), [read-section]) (established):
- MQAR is run at T ∈ {256, 512, 1024} with T/4 KV pairs and V=8,192, with random-token filler. A larger state (N=16→64→256) helps.
- **Hybrid ablation at 350M with 48 layers.** Perplexity by number of attention layers: 0 → 8.60, 1 → 8.38, 2 → 8.32, 3 → 8.29, 4 → 8.29, 5 → 8.28, **6 → 8.26**, 7 → 8.27, 9 → 8.28, 11 → 8.30, 15 → 8.34, 24 → 8.50. Transformer++ scores 8.68.
- The authors conclude that "around 10% of the total number of layers being attention performs best".
- *Note (inference):* that is a **perplexity** result. Perplexity is a weak proxy for long-context recall (Section 2.5), so it does not answer our H1.

**Repeat After Me (phonebook, copying).** Jelassi et al., ICML 2024 ([arXiv 2402.01032](https://arxiv.org/html/2402.01032), [read-section]) (established):
- Phonebook lines look like "John Powell: 609-323-7777". The prompt is the book plus two few-shot examples plus a query. Accuracy is string-level over 10×64 samples.
- Pythia 410M beats Mamba 2.8B once the book has ≥70 entries.
- Copying was trained at ~160M parameters (Transformer: 12 layers, d=1024; Mamba: 24 layers, d=1024, state size 32) with a 30-token vocab. Transformers needed ~100× fewer samples.
- Theory: a GSSM with fewer than L·log D − 1 bits of state has error >1/2 on copying uniform strings.
- *Transfer:* (a) **copy-after-gap with random strings** is a clean, zero-shot-scorable probe of inference-state capacity. (b) The phonebook is a natural-format MK-NIAH.

**Samba.** Ren et al. 2024 ([arXiv 2406.07522](https://arxiv.org/html/2406.07522), [read-section]) (established):
- The architecture interleaves Mamba, MLP, SWA (window 2,048) and MLP, at 421M–3.8B.
- Passkey: fine-tuned 500 steps at 4K, perfect to 256K.
- Phonebook: 20–480 entries, fine-tuned 100 steps at 4K, evaluated to 8K.
- Length extrapolation is measured as Proof-Pile perplexity up to 1M.
- *Transfer:* this is precedent for a **short identical fine-tune** (option ii) before synthetic evaluation.

**Gated DeltaNet.** Yang, Kautz & Hatamizadeh 2024/25 ([arXiv 2412.06464](https://arxiv.org/html/2412.06464), [read-section]; repo README [code-inspected]) (established):
- Zero-shot RULER S-NIAH-1/2/3 at 1K–8K for 1.3B models trained on 100B FineWeb-Edu tokens at 4K.
- Example: S-NIAH-2 (number in essay) at 1K/2K/4K/8K scored Mamba2 99.4/98.8/56.2/17.0, DeltaNet 98.4/45.6/18.6/14.4 and Gated DeltaNet 100/99.8/92.2/29.6.
- Recall-intensive tasks (SWDE, SQuAD, FDA, TriviaQA, NQ, DROP) were truncated to 2K. H1/H2 hybrids averaged 39.0/40.1, Transformer++ 37.0 and Mamba2 29.8.
- The README warns that these retrieval tasks "are highly prompt-sensitive for instruction-untuned models in zero-shot settings". It points to the JRT prompt scripts rather than plain lm-eval task names.
- *Transfer:* (a) even at 1.3B, zero-shot S-NIAH on recurrent models collapses within 2–4× the training length, so the length grid must straddle the training length. (b) Base-model prompt sensitivity is a real risk, which argues for cloze formats plus logprob scoring.

**Jamba.** Lieber et al. 2024 ([arXiv 2403.19887](https://arxiv.org/html/2403.19887), [read-section]) (established):
- At 1.3B with 250B tokens, attention:Mamba ratios of 1:3 and 1:7 were "virtually" identical on HellaSwag, WinoGrande, NQ and C4/Books/Code log-prob.
- Pure Mamba failed format adherence in ICL (IMDB 48.8 vs attention 84.1 vs hybrid 90.9).
- NIAH up to 256K is shown only as a figure.
- *Transfer:* (a) log-prob on long-document domains (Books/Code) is used as an evaluation axis. (b) **Format-following failures confound generation-based scoring for recurrent models.** Logprob scoring sidesteps this (inference).

**Systematic Analysis of Hybrid Linear Attention.** Wang, Zhu et al. 2025 ([arXiv 2507.06457](https://arxiv.org/html/2507.06457v1), [read-section]) (established):
- 72 models: 340M (20B tokens) and 1.3B (100B tokens), on FineWeb-Edu.
- Six linear variants × ratios {pure linear, 24:1, 12:1, 6:1, 3:1}, with zero-shot RULER subsets.
- At 1.3B, the average RULER over 4 families was 0.256 (pure linear), 0.338, 0.340, 0.390 and 0.397 (3:1). Transformer ≈ 0.42.
- LM averages are flat at 0.472–0.480.
- **340M recall was "insignificant" and omitted.** The text mentions a 2,048-token window but does not state RULER context lengths or sample counts.
- *Transfer:* this is the closest prior work to our RQ A. It is the main evidence that **zero-shot RULER-style evaluation will be at the floor at our scale.** It also shows that ratio effects on LM metrics are nearly invisible while recall effects are large.

**RNNs are not Transformers (Yet).** Wen, Dang & Lyu 2024 ([arXiv 2402.18510](https://arxiv.org/abs/2402.18510), [abstract-only]) (established):
- The bottleneck for RNNs is in-context retrieval.
- Adding a **single Transformer layer** (or RAG) closes the representational gap for CoT tasks.
- This is theoretical motivation for H1.

**MAD.** Poli et al. 2024 ([arXiv 2403.17844](https://arxiv.org/abs/2403.17844), [abstract-only]) (established):
- Small synthetic token-manipulation tasks (including recall and compression) correlate with compute-optimal perplexity across more than 500 models from 70M to 7B.
- I did not access the task list.

**k-hop induction heads.** Sanford, Hsu & Telgarsky 2024 ([arXiv 2402.09268](https://arxiv.org/abs/2402.09268), [abstract-only]):
- The abstract claims logarithmic depth suffices for transformers on "basic computational tasks".
- I could **not** confirm the k-hop task definition from the abstract page. I use "k-hop chains" below as my own task design (inference), motivated by RULER VT.

### 2.3 Natural-language and "reasoning-in-a-haystack" benchmarks

**BABILong.** Kuratov et al. 2024 ([arXiv 2406.10149](https://arxiv.org/html/2406.10149), [read-section]; code [code-inspected]; HF card [huggingface.co/datasets/RMT-team/babilong](https://huggingface.co/datasets/RMT-team/babilong) [abstract-only]) (established):
- 20 bAbI tasks (qa1 single supporting fact, qa2 two, qa3 three, qa4/qa5 argument relations, qa6 yes/no, qa7 counting, qa8 lists/sets, …) hidden in PG19 sentences, in natural order.
- Public splits run 0K–10M with 100 samples per task per length. A 1,000-sample set exists up to 32K–128K (the paper's text varies).
- Satisfactory is >85% and failure is <30%.
- LLMs effectively use 10–20% of their context.
- **Fine-tuned GPT-2 (137M) RMT/ARMT and fine-tuned Mamba-130M solve the tasks.**
- Code: `NoiseInjectionDataset` splices **token** lists. It samples background sentences until `sample_size − task_len` tokens, inserts facts between sentences at sorted random positions and optionally restricts them to a `[task_start_pct, task_end_pct]` window. The answer check is "target substring in the first sentence of the output".
- lm-eval-harness has `babilong` (0K) and `babilong_longctx` (qa1–qa5 up to 128K) task groups ([code-inspected]).
- *Transfer:* this is the **only** natural-filler multi-hop benchmark with small-model precedent, and that precedent requires **fine-tuning**. The 0K split is a ready-made "reasoning without retrieval" control.

**LongBench.** Bai et al., ACL 2024 ([arXiv 2308.14508](https://arxiv.org/abs/2308.14508), [abstract-only]): 21 datasets in 6 categories, EN/ZH, average about 6.7K English words. Gated DeltaNet reports LongBench averages of 11–18 for 1.3B models ([arXiv 2412.06464](https://arxiv.org/html/2412.06464)). At 20–60M this would be at the floor and dominated by noise (inference).

**LongBench v2.** Bai et al. 2024 ([arXiv 2412.15204](https://arxiv.org/abs/2412.15204), [abstract-only]): 503 multiple-choice questions, 8K–2M words. Human experts score 53.7% and the best direct model 50.1%. **Not usable** at our scale.

**InfiniteBench.** Zhang et al. 2024 ([arXiv 2402.13718](https://arxiv.org/abs/2402.13718) [abstract-only]; [GitHub README](https://github.com/OpenBMB/InfiniteBench) [read-section]): 12 tasks with average inputs of 44K–193K tokens. Synthetic tasks include Retrieve.PassKey (590), Retrieve.Number (590), Retrieve.KV (500) and Math.Find (350). Lengths exceed our 32K maximum. Only the *task ideas* transfer.

**HELMET.** Yen et al. 2024/25 ([arXiv 2410.02694](https://arxiv.org/html/2410.02694), [read-section]; code [code-inspected]) (established):
- 7 categories (RAG, citations, re-ranking, many-shot ICL, long QA, summarisation, synthetic recall) at 8K–128K.
- **NIAH-style synthetics do not reliably predict downstream performance.** No synthetic task reaches an average Spearman >0.8 at 128K across 35 models. Harder recall with distractor needles correlates better.
- The recall set is `ruler_niah_mk_2, ruler_niah_mk_3, ruler_niah_mv, json_kv`, with `max_test_samples: 100` and `shots: 2` (`configs/recall.yaml`).
- *Transfer:* prefer **distractor-dense** variants (MK-2/MK-3/JSON-KV-like) over plain NIAH. The 100-sample default is too small for us.

**Michelangelo / Latent Structure Queries.** Vodrahalli et al. 2024 ([arXiv 2409.12640](https://arxiv.org/html/2409.12640), [read-section]) (established):
- Latent List: track a Python list through operations amid `print('Do nothing.')` filler.
- MRCR: reproduce the k-th earlier response on a topic in a long synthetic conversation, scored with difflib's ratio.
- IDK: about 70% of instances are unanswerable.
- Frontier models drop sharply before 32K.
- I did not see a code release in the part I read. An open MRCR-style dataset exists at [huggingface.co/datasets/openai/mrcr](https://huggingface.co/datasets/openai/mrcr) (search-result level only, [abstract-only]), but I could not confirm its relationship to Michelangelo.
- *Transfer:* the **"latent state tracking under updates"** idea transfers as our "key-update / conflicting values" task. The natural-language formats would sit at the floor.

### 2.4 Effective context length

- **RULER's definition** (established, above) compares each model to a reference model's threshold.
- **STRING.** An et al. 2024 ([arXiv 2410.18745](https://arxiv.org/html/2410.18745), [read-section]) (established):
  - Effective length falls short because relative-position frequencies during training are left-skewed. With a 2,048 window, positions ≥1,536 make up under 5% of position occurrences.
  - Measurement: a 4-needle NIAH with lengths increased in 128-token steps, a 2-of-4 retrieved criterion and 500 trials per length.
  - *Transfer (inference):* **for full-attention layers, eval lengths beyond the training length mostly measure RoPE extrapolation, not attention's intrinsic capability.** H5 must therefore be tested within the trained length, or after a long-context stage that is identical across arms.
- **BABILong** (established): LLMs use ~10–20% of their context.

### 2.5 Perplexity as a long-context metric

- **Can Perplexity Reflect…?** Hu et al. 2024 ([arXiv 2405.06105](https://arxiv.org/abs/2405.06105), [abstract-only]): "no correlation between PPL and LLMs' long-text understanding ability". PPL mostly reflects local context.
- **LongPPL.** Fang et al. 2024 ([arXiv 2410.23771](https://arxiv.org/html/2410.23771), [read-section]):
  - Key tokens satisfy LSD = logp(long) − logp(short, K=4096) > 2 and LCL = logp(long) > −2, as judged by an evaluator LM (default Qwen2-72B-Instruct, or Llama-3.1-8B).
  - Pearson correlation of LongPPL with LongBench / LongEval / RULER is −0.96 / −0.90 / −0.90. For standard PPL it is −0.18 / 0.24 / 0.27.
  - *Transfer:* the idea of averaging loss only over tokens that *need* long context is sound. **Using one of our own models as the evaluator would bias results toward that architecture** (inference). I propose a model-free proxy (Section 5, LM-3).
- **ProLong.** Gao et al. 2024 ([arXiv 2410.02660](https://arxiv.org/html/2410.02660), [read-section]):
  - "Making design decisions based on perplexity (PPL) is not optimal for long-context downstream tasks". PG19 PPL improved with more long data while downstream performance got worse.
  - NIAH saturates.
  - "Long-context evaluation shows clearer signals when it is conducted after SFT."
- **Zoology's AR-hit slice** (above) is the counter-example: a *targeted* slice of perplexity that does track recall gaps at 70M–360M.
- **Stuffed Mamba** (established): per-position LM loss reveals Mamba-2 "state collapse" beyond the training length. Loss rises past T_train, which makes per-position loss a necessary diagnostic for extrapolation.
- **Synthesis (inference):** aggregate perplexity will barely separate our ratio arms, as both Mamba-2's 8.26–8.60 spread and 2507.06457's flat LM averages suggest. **Position-resolved and recall-token-resolved loss should separate them.** Report both and pre-register the resolved ones as primary LM metrics.

### 2.6 Training-time exposure to synthetic tasks

**Birdie.** Blouir et al. 2024/25 ([arXiv 2411.01030](https://arxiv.org/html/2411.01030), [read-section]) (established):
- Mixing copying, deshuffling, selective copying and infilling into pretraining (with RL-chosen mixture weights) at 1.4B and ~32B tokens let SSMs reach 100% on single-number phonebook retrieval. Next-token-prediction SSMs failed.
- **Fixed mixtures (UL2, Fixed Ratio) did not yield phonebook retrieval** in their Gated SSM ablation.
- A second-stage pretraining attempt failed.
- *Caution:* the inclusion strategy matters and can fail silently. A pilot is mandatory.

---

## 3. What is measurable at 20–60M parameters?

| Candidate | Zero-shot at 20–60M? | With identical synthetic exposure? | Verdict |
|---|---|---|---|
| LongBench / v2, InfiniteBench, HELMET (non-synthetic), Michelangelo, NarrativeQA | Floor (inference; 2507.06457 shows a floor even for RULER at 340M) | Would need instruction tuning plus knowledge we lack | **Exclude** |
| RULER as-is (instruction prompts, string-match generation) | Floor (inference + 2507.06457) | Possible if trained on RULER-format data, but then train/eval share templates | Cross-check only |
| Per-position LM loss on long held-out docs | Yes | n/a | **Tier 0, primary LM metric** |
| AR-hit / long-range-repeat-token loss | Yes (Zoology at 70M) | n/a | **Tier 0, primary** |
| Copy-after-gap loss (natural and random strings) | Yes (copying is induction; inference) | n/a | **Tier 0** |
| Cloze passkey | Yes (Stuffed Mamba 36M, after LR/checkpoint selection on passkey) | Saturates | Tier 0 sanity |
| Cloze single-key KV with distractors | Plausible via logprob ranking (hypothesis, pilot) | Yes | **Tier 0 and Tier 1** |
| Dense multi-key KV (MQAR-like), conflicting values | Unlikely zero-shot | Yes (MQAR solved by 2-layer models in Zoology) | **Tier 1** |
| k-hop chains, variable tracking | Floor | Yes (hypothesis; tune k) | **Tier 1** |
| Counting, frequency comparison, multi-value | Floor | Yes (hypothesis) | **Tier 1** |
| BABILong qa1–qa3 | Floor | Yes (GPT-2 137M fine-tuned; Mamba-130M fine-tuned) | Tier 2 (optional) |

---

## 4. How to make the synthetic tasks measurable: three options compared

### Option (i): synthetic task data inside the pretraining mixture

Mechanism:
- Procedurally generate task sequences and interleave them at a fixed token fraction f.
- f = 2–5% is suggested (hypothesis, to be piloted). It could apply throughout pretraining or only in the final long-context phase.
- The token stream (seeded order) is identical for every arm, so "data fixed" holds by construction.

**Pros**
- No extra stage.
- Every arm sees the same exposure, so the result answers "which architecture *learns and uses* the capability given identical data".
- Most modern recipes include such data in annealing/mid-training anyway (inference).
- Can be combined with a long-context phase at 8K–32K, which also fixes the extrapolation confound.

**Cons and risks**
1. Evaluation becomes **in-distribution algorithm execution**, not emergent long-context ability. That is acceptable for RQ A–C if stated.
2. It changes the base LM. Measure the LM-loss cost: with and without synthetic data, in the pilot only.
3. **Leakage:** identical KV pairs, keys, filler passages or templates in train and eval.
4. Architectures may learn at different speeds, so a single f favours fast learners. Report learning curves (task accuracy vs tokens) from checkpoints.
5. Birdie's negative result for fixed mixtures.

### Option (ii): a short, identical fine-tuning stage on synthetic tasks

Mechanism:
- Same steps, LR schedule, batch and data for every arm.
- Suggested: 500–2,000 steps of a multi-task synthetic mix at the target length, with LR at ~10–30% of the peak pretraining LR. This is a hypothesis, with Samba's 100–500 steps as precedent.

**Pros**
- The base LM stays clean. Tier 0 runs on the base checkpoint.
- Precedents: Samba, BABILong, ProLong ("clearer signals after SFT").
- Cheap enough to sweep the budget.

**Cons**
1. The measurement becomes "**fine-tunability**". A fixed budget can confound capability with optimisation speed. Mitigate by reporting accuracy at ≥3 budgets (e.g., 250 / 1,000 / 4,000 steps) and the area under that curve.
2. Catastrophic forgetting of LM ability. Report LM loss after fine-tuning.
3. One more hyperparameter set that could, in principle, be tuned per arm. **It must be frozen and shared.**

### Option (iii): zero-shot log-probability scoring only

**Pros**
- No contamination.
- Measures exactly what pretraining produced.
- Continuous metrics avoid floor clipping.

**Cons**
1. Only induction/copy-style tasks get off the floor (Section 3).
2. Multi-hop and aggregation will almost certainly read zero for every arm (inference), which leaves H3 untestable.
3. Cloze templates still carry prompt sensitivity (Gated DeltaNet README warning).

### Recommendation

- **Use (iii) for Tier 0 in every condition.**
- **Use one of (i)/(ii) for Tier 1, chosen by a cheap pilot.**
  - Train only the two extreme arms (r=0 all-linear and r=1 all-full) at the smallest size.
  - Measure Tier 1 at the in-distribution length.
  - Pick the option where (a) the all-full arm reaches 70–95% on each headline task at the in-distribution length, neither floor nor ceiling, and (b) the gap between the extremes is ≥3× the planned CI half-width.
  - If both options qualify, prefer **(i) applied in the long-context mid-training phase**, because it serves the length-extrapolation fix at the same time (inference).
- Whichever is chosen, **difficulty knobs (number of KV pairs, hops, distractor density) are calibrated on the pilot and then frozen before the main sweep**. Recalibrating after seeing ratio results would be a forking-paths error.

### Leakage controls (mandatory for (i) and (ii))

| Risk | Control |
|---|---|
| Identical items in train and eval | Hash every (key, value, template-id) tuple. Train keys come from word list A and eval keys from list B. Value spaces are disjoint (e.g., train values from one numeric range or symbol set, eval from another). Assert zero overlap in a unit test. |
| Filler overlap | Train filler comes from the training corpus. Eval filler comes from **held-out** documents (a held-out shard of the pretraining corpus, plus PG19 *test* books, 100 books, Apache-2.0, [HF card](https://huggingface.co/datasets/deepmind/pg19) [abstract-only]). |
| Template memorisation | Train on templates T1–T3. Evaluate on T1 (in-distribution) and an unseen paraphrase T4 (OOD), and report both. |
| Shortcut solutions | Every retrieval task has same-type distractors (other KV pairs with the same value format). VT/chains have ≥1 distractor chain. See the "does not measure / shortcut" columns in Section 5. |
| Difficulty and length leakage | Train on lengths ≤ T_train, hops ≤ k_train, KV count ≤ K_train. Evaluate ID cells *and* OOD cells (more hops, more keys, longer). Report them separately and never average them together. |
| Position leakage | Train needle depths uniform over [0,1]. Evaluate the full depth grid. |

---

## 5. Proposed benchmark suite

### 5.0 Shared mechanics

**Formats are cloze completions, not instructions.** Every item ends with a prefix whose natural continuation is the answer, for example `The code for amber-falcon is` → ` 4 1 7 9`. The same items therefore serve zero-shot (Tier 0) and post-exposure (Tier 1).

**Answer tokenisation.**
- Answers are built **from single tokens of our tokenizer**.
- Values are sequences of k single-token symbols drawn from a fixed "value vocabulary". For example, if the tokenizer splits digits individually, use 4–5 space-separated digits. Otherwise use a curated list of single-token words.
- This makes teacher-forced exact match identical to greedy exact match, with no tokenisation-ambiguity loophole (inference).
- Verify with a unit test that `decode(encode(" "+v)) == " "+v` and that the token count is k.

**Exact tokens-at-length.** Build each sample as a list of token IDs:

```
[BOS] + prefix_tokens + haystack_tokens_with_needles + query_tokens + answer_tokens
```

1. Pre-tokenise each needle, query and answer *separately*, with a leading space or newline that matches training conventions.
2. Compute the filler budget F = T − (len(BOS) + len(prefix) + Σlen(needles) + len(query) + len(answer)).
3. Take a contiguous span of exactly F tokens from a held-out document. Use synthetic filler for noise variants.
4. Insert needles at target token offsets, snapped to the nearest sentence boundary (newline/period token) within ±32 tokens. Record the **actual** needle start index.
5. Assert `len(sample) == T` for T ∈ {1024, 2048, 4096, 8192, 16384, 32768}.
6. Never detokenise and retokenise, because BPE merges across splice boundaries would change counts.

T counts every token through the last answer token. Scoring reads logits at positions T−len(answer)−1 … T−2.

**Position variables recorded per item.**
- depth = needle_start / F.
- distance d = query_start − needle_end.
- For multi-needle items, the min, max and spread of needle positions.

**Depth grid.**
- 11 nominal depths {0, 0.1, …, 1.0}. Items are drawn stratified: uniform within each of 10 deciles, plus exact-edge items at depth 0 and depth 1.
- Primary analysis: logistic regression of correctness on depth (natural spline, 4 df) × log T. This avoids underpowered per-cell estimates.

**Distance-controlled design (for H5).**
- Separately, hold d fixed at {256, 1K, 4K, 16K} while varying total T. The needle sits d tokens before the query, and the rest is irrelevant prefix.
- This separates "retrieval distance" from "amount of irrelevant context". The second matters for fixed-state models through state interference.
- **This decomposition is the cleanest test of H5 (hypothesis).**

**Query branching (cheap and statistically cleaner).**
- Prefill each context once.
- Snapshot the inference state: the KV cache for full-attention layers and the recurrent state for linear layers.
- Score Q independent queries from that same snapshot, so query j never sees query i's answer.
- Recommended Q = 4 per context, at different depths.
- Correlation within a context is handled by a **cluster bootstrap over contexts** (Section 6). This cuts prefill compute by about Q×.
- *Implementation check:* branched scoring must equal separate full forward passes to within fp tolerance. Unit-test it.

**Metrics per item (all from one forward pass).**
1. **EM_tf.** Every answer token is the argmax given the gold prefix. Equals greedy exact match under canonical tokenisation.
2. **logp_gold.** Sum of log-probs of the answer tokens. Report the mean and the per-token mean.
3. **Rank accuracy.** Score every candidate in C = {gold} ∪ {all other values present in the context} ∪ {M absent random values, M=4} by summed log-prob (all candidates have equal token length). Correct if gold ranks first.
   - Chance = 1/|C|.
   - Also report **binding error rate** (top candidate is an in-context value of the wrong key) and **absent error rate** (top candidate is not in the context). The first means "stored but mis-bound". The second means "not stored / not retrieved" (inference about interpretation).
   - For the conflicting-values task, also report the **stale-value rate**.

Headline metric: **EM_tf** (binary, binomial CI). Rank accuracy and logp_gold are secondary and continuous, and they are the more sensitive metrics near the floor.

**Lengths.** T ∈ {1K, 2K, 4K, 8K, 16K, 32K}. Each cell is tagged **ID** (T ≤ the longest length any arm was trained on) or **EXTRAP**. Headline H5 claims use ID cells only.

**Controls run on every task.**
- **C0, short control:** the same item with no filler, so T is minimal (about 64–256 tokens). This measures the task logic alone (as BABILong's 0K split does).
- **C1, chance:** the analytic chance level for rank accuracy.
- **C2, oracle position:** the same item at T with all relevant facts placed immediately before the query (d ≈ 0). Comparing C2 with a random-depth item isolates retrieval over distance from handling a long input.

### 5.1 Retrieval family

| ID | Task | Generator spec | Difficulty knobs (frozen after pilot) | Metric | What it does **not** measure / shortcut risks |
|---|---|---|---|---|---|
| **R0** | Cloze passkey (sanity) | Filler is the repeated "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." (Landmark/RULER noise). Needle: `The pass key is {v}. Remember it. {v} is the pass key.` Query: `What is the pass key? The pass key is` → ` {v}` (5 single-token digits). | depth, T | EM_tf, logp | No key binding: one anomaly in uniform filler, so "copy the only number" solves it. Expect a ceiling for attention arms. Not a headline metric. |
| **R1** | Single-key retrieval in natural filler with same-type distractors (S-NIAH+) | Filler is held-out natural text. Insert 1 target plus D distractor needles of the form `The code for {key} is {v}.` Keys are adjective-noun word pairs (word list B for eval). Query: `The code for {key} is` → ` {v}`. | D ∈ {0, 3, 15} (sweep), depth, T | EM_tf, rank acc (chance 1/(D+5)), binding error | D=0 has the copy-the-only-number shortcut, so D=0 is a diagnostic only and D≥3 is the headline. Literal lexical match only (NoLiMa caveat). One fact. |
| **R2** | Dense multi-key KV (MQAR-in-text; RULER MK-2/3 analogue) | The haystack is **entirely** `key: v` lines (no natural filler). N_kv = floor(F / tokens_per_line), all keys unique. Q=4 branched queries per context at stratified depths. | Density is fixed by T (N_kv grows with T, e.g., ≈100 pairs at 1K and ≈3,000 at 32K with ~10 tokens/line). A second variant, **R2-sparse**, uses fixed N_kv=64 pairs spread in natural filler, to separate the number of facts from length. | EM_tf, rank acc (chance ≈1/(N_kv+4) if all in-context values are candidates; cap the candidate set at 32 sampled in-context values + gold), binding error | Capacity-limited by design. It conflates "number of facts" with "length", which is why R2-sparse exists. Not natural language. |
| **R3** | Conflicting values / key updates | One key is assigned U times at increasing positions (`The code for {key} is now {v_u}.`) among distractor needles and filler. Query asks for the **latest** value (primary). A secondary variant asks for the **first** value. | U ∈ {2, 4}; gap between updates; T | EM_tf, **stale-value rate**, rank acc | Tests overwrite/recency, which may favour delta-rule linear layers (hypothesis). It does not test "which value is true", only the order-based rule. |
| **R4** | Position sweep (cross-cutting) | Applied to R1 (D=3) and R2-sparse at T ∈ {4K, 32K} (or the longest ID T) on the full 11-depth grid, plus the distance-controlled design. | depth, d | EM_tf by depth/distance | Isolates where information is used. It does not measure whether several facts are combined. |

Phonebook (Jelassi) is equivalent to R2 with names and numbers. Use R2's format so that keys and values are tokenizer-controlled.

### 5.2 Multi-hop family

| ID | Task | Generator spec | Knobs | Metric | Does **not** measure / shortcuts |
|---|---|---|---|---|---|
| **M1** | k-hop key chain (dereference form; primary) | Chain: `{k_0} is {v}.` then `{k_1} is {k_0}.` … `{k_m} is {k_{m-1}}.` (keys are single-token-pair names from list B). The query asks for the value of the **last** link: `{k_m} is` → ` {v}`. The model must follow m links back to the number. Include **≥2 distractor chains** of equal length (different keys and values) and natural filler. | hops m ∈ {1, 2, 4} (ID); m = 6 OOD; **link order** ∈ {forward (root first), shuffled, reversed}; T; spread of links (all in one 10% window vs spanning the whole context) | EM_tf, rank acc over {v of each chain} ∪ absent (chance ≈ 1/(#chains+4)), "wrong-chain" rate | m=1 reduces to R1. Comparing m=1 with m>1 at the same T and positions isolates composition from retrieval. Link order interacts with recurrent state (JRT), so report it as a factor rather than averaging over it. It does not test natural-language reasoning. |
| **M2** | Variable tracking (RULER VT, set-output) | RULER-style `VAR ABCDE = 12345` / `VAR FGHIJ = VAR ABCDE` with 1 target chain + **3 distractor chains** (RULER's default config has 1 chain in a noise haystack, which is shortcut-able). Query: `Variables equal to 12345:` → the chain's variables in **chain order**. | hops ∈ {2, 4}; chains = 4; filler = noise or natural | Per-variable teacher-forced accuracy (ordered), all-correct EM | Without distractor chains, "list every VAR" solves it (inference from RULER code). The ordered output adds a sequencing demand. Secondary to M1. |
| **M3** | BABILong qa2/qa3 (optional, Tier 2) | Official generator (`third_party/babilong/babilong/babilong_utils.py`, `NoiseInjectionDataset`) with **our tokenizer**, PG19 test filler for eval and train filler for fine-tuning. | T ∈ {0K, 1K, 4K, 16K}; qa1/qa2/qa3 | Rank accuracy over the bAbI location set (answers are from a closed vocabulary of ~6 locations, so logprob ranking is natural) | Natural language but templated. Facts keep natural order. It requires fine-tuning on the bAbI train set, so it measures learned reasoning plus retrieval. |

### 5.3 Integration / aggregation family

| ID | Task | Generator spec | Knobs | Metric | Does **not** measure / shortcuts |
|---|---|---|---|---|---|
| **A1** | Counting | A marker sentence `I saw a {animal}.` appears c times for the target animal, with 4 distractor animals at their own counts, spread over natural filler. Query: `Number of times I saw a {target}:` → ` {c}` (c ∈ 1…9, single token). | c ∈ 1..9 uniform; spread (uniform over context vs clustered); T | EM_tf, rank acc over 1..9 (chance 1/9), mean absolute count error | Needs *every* occurrence. Partial retrieval yields an under-count, so the signed error is informative. Exact counting may be hard for both architectures at 50M (risk of floor; calibrate c_max in the pilot). |
| **A2** | Frequency comparison (CWE-lite) | A RULER-CWE-style numbered word list, `1. apple 2. stone …`, with words from list B. Two probe words X, Y with count ratio ρ. Query: `More frequent: {X} or {Y}? Answer:` → ` {winner}`. | ρ ∈ {3:1, 2:1, 1.5:1}; total words set by T | Binary rank accuracy (chance 0.5), so the CI is tightest | Relative frequency, not exact counts. With 2 candidates the floor is 50%, so CIs must be judged against 0.5. |
| **A3** | Multi-value gathering (RULER MV analogue) | One key has V values in separate needles `One code for {key} is {v_j}.` spread across context, plus distractor keys. Query: `All codes for {key}, in order:` → ` {v_1} ; {v_2} ; … ; {v_V}` (order of appearance). | V ∈ {2, 4}; spread; T | Per-value teacher-forced accuracy (teacher-forcing earlier values isolates each value's retrieval), all-correct EM | Teacher-forcing removes compounding errors, which differs from free generation. The order requirement adds sequencing. |
| **A4** | Common-word extraction (RULER CWE, generation) | RULER parameters (freq_cw=30, freq_ucw=3, num_cw=10 at ≥4K; 6/1 below 4K). | as RULER | `string_match_all` with greedy generation | Diagnostic and cross-check only. A 10-item set answer requires generation and format following. |

### 5.4 Language-modelling metrics (Tier 0, all checkpoints)

| ID | Metric | Data | Spec |
|---|---|---|---|
| **LM-1** | Per-position loss curve | (a) ≥300 held-out docs of ≥32K tokens from the pretraining corpus's held-out shard. (b) PG19 test (100 books), each cut to its first 32,768 tokens; drop books shorter than that. | Evaluate each doc at the full T_max context (no sliding window). Mean token loss in log-spaced buckets [0,64), [64,128) … [16K,32K). Report the curve, plus the **long-context gain** L[256,512) − L[16K,32K), and the **tail-slope**, a regression of loss on log(position) over [1K,32K). Docs are the bootstrap unit. |
| **LM-2** | Aggregate validation loss at the training length | Same held-out shard | The standard LM-quality covariate. Expect small differences between ratios (Mamba-2 table; 2507.06457). |
| **LM-3** | **Long-range repeat-token loss**, distance-resolved (generalises Zoology AR hits) | Same docs | Token t is a *repeat token* if the n-gram (n=4) ending at t occurred earlier in the same doc. Its **distance** Δ = t − (end of the most recent prior occurrence). Following Zoology, exclude n-grams whose corpus frequency is > a threshold (e.g., top 1% most frequent 4-grams in a 1B-token sample), so memorised phrases do not count. Report mean loss on repeat tokens bucketed by Δ ∈ [64,256), [256,1K), [1K,4K), [4K,16K), [16K,32K), alongside non-repeat tokens at the same positions. **Model-free** (avoids LongPPL's evaluator bias). |
| **LM-4** | Copy-after-gap | Synthetic | Sequence = P + G + P. P is (a) a 256-token held-out passage or (b) 256 uniform-random tokens from a 4K-token subset of the vocab. G is held-out filler of length G ∈ {0, 512, 2K, 8K, ~30K}. Metric: mean loss on the second P (tokens 9–256) relative to the first P. Variant (b) removes LM priors, so it is pure inference-state capacity (Jelassi). |

Interpretation guardrail (inference): LM-1 beyond the training length conflates extrapolation (RoPE, or state collapse as in Stuffed Mamba) with long-range use. Always mark the T_train boundary on plots.

---

## 6. Sample sizes and CI mathematics

Wilson 95% CI **half-widths** in percentage points, z=1.96. The center is (p + z²/2n)/(1 + z²/n), and the half-width is z·√(p(1−p)/n + z²/4n²)/(1 + z²/n). These were computed by a throwaway script that is not in the repo, and the values are reproducible from the formula:

| n | p=0.05 | p=0.10 | p=0.25 | p=0.50 | p=0.90 |
|---|---|---|---|---|---|
| 50 | 6.6 | 8.5 | 11.7 | 13.4 | 8.5 |
| 100 | 4.5 | 6.0 | 8.4 | 9.6 | 6.0 |
| 200 | 3.1 | 4.2 | 6.0 | 6.9 | 4.2 |
| 400 | 2.2 | 3.0 | 4.2 | 4.9 | 3.0 |
| 500 | 1.9 | 2.6 | 3.8 | 4.4 | 2.6 |
| 1000 | 1.4 | 1.9 | 2.7 | 3.1 | 1.9 |

Required n for half-width h at p=0.5 (worst case), using n = z²·p(1−p)/h²:
- h = 10 pp → 97
- h = 5 pp → 385
- h = 4 pp → 601
- h = 3 pp → 1,068
- h = 2 pp → 2,401

At p=0.9: h = 5 pp → 139, h = 3 pp → 385.

**Comparing two arms** (α=0.05, power 0.8):
- **Unpaired**, n per arm: 50→60% needs 388. 80→90% needs 199. 90→95% needs 435. 50→55% needs 1,565.
- **Paired on identical items** (McNemar), n = (z√p_d + z_β√(p_d − Δ²))²/Δ²:
  - Δ=10 pp with discordance p_d=0.2 needs **155**, and with p_d=0.3 needs **234**.
  - Δ=5 pp with p_d=0.1 needs 312, and with p_d=0.2 needs 626.
- **All arms must therefore be evaluated on exactly the same generated items** (same seeds), and analysed paired.

**Clustered queries** (branched Q per context). Design effect DE = 1 + (Q−1)·ICC.
- Q=4, ICC=0.2 → DE=1.6, so 100 contexts ≈ 250 effective items.
- Q=8, ICC=0.5 → DE=4.5, so 100 contexts ≈ 178 effective items.
- Estimate ICC in the pilot. Use a cluster bootstrap (resample contexts) for all CIs.

**Recommended allocation per model** (inference):

| Cell type | Contexts | Queries/context | Effective n (ICC 0.2) | 95% half-width at p=.5 |
|---|---|---|---|---|
| Headline (task × T), R1/R2/M1/A1/A3 | 400 | 4 (R1, R2, A1 via distinct targets); 1 (M1, A3) | ~1,000 / 400 | ~3.1 / 4.9 pp |
| A2 (binary) | 400 | 4 | ~1,000 | ~3.1 pp |
| Position sweep (R4), T ∈ {4K, 32K} | 1,100 (100/depth) | 1 | 100 per depth / 1,100 total | 9.6 pp per depth; pooled by regression |
| Controls C0/C2 | 200 | 1 | 200 | 6.9 pp |
| Dev suite (intermediate checkpoints) | 100 at T ∈ {1K, 4K, 16K} | 4 | ~250 | ~6 pp |

**Training-seed variance (inference, important).** Synthetic-task competence in small models can switch on abruptly, so the variance between seeds can exceed item-sampling variance.
- Item-level CIs condition on one trained model and **understate** uncertainty about the architecture.
- Plan ≥2 seeds (preferably 3) for the arms that carry H1/H2 (e.g., r=0, the smallest hybrid, r=1/4, r=1).
- Report across-seed spread separately.
- Suggested model: mixed-effects logistic regression, correct ~ arm × log2(T) × task + depth spline + (1|item) + (1|seed).

**Effective context length (ours).** The largest ID T where accuracy ≥ 0.85 × the model's own C0 (short-control) accuracy *and* the lower CI bound is ≥ chance + 10 pp. Use a bootstrap CI over contexts. This adapts RULER's thresholding idea to a no-reference-model setting.

---

## 7. Compute estimate for evaluation (inference, to be measured)

- Per-token forward FLOPs ≈ 2N + 2·n_full·T·d_model (causal attention average). For N≈50M, 12 full layers, d=512, T=32K this is ≈ 1e8 + 4e8 = 5e8 FLOPs/token. Linear layers add O(d²) per token.
- Full suite per model ≈ 6 lengths × (8 task-variants × 400 contexts) plus position sweeps plus LM docs ≈ 3e8 tokens. That is ≈ 1.5e17 FLOPs for the all-full arm.
- At an assumed 30 TFLOP/s effective bf16 on the RTX 5080 (**unmeasured**), that is ≈ 1.4 h per all-full model. Hybrids and linear arms would be much less.
- With query branching (Q=4) and a dev suite for checkpoints, evaluation should stay well under 15% of the training budget (inference).
- **Timing benchmarks (prefill/decode) need an idle GPU.** The CARLA sharing noted for this machine does not affect accuracy evaluation, but it does affect latency measurements.

---

## 8. Reuse vs build

| Component | Recommendation | Why |
|---|---|---|
| Task logic and word lists | **Borrow from RULER** (`scripts/data/synthetic/*.py`, Apache-2.0) and the Zoology MQAR generator (`zoology/data/multiquery_ar.py`) | Proven task definitions. Citable lineage. |
| Generator | **Custom** `src/eval/generators/` that works in token space, with exact-T, branched queries, disjoint train/eval namespaces and logprob scoring | RULER builds strings with ≤-length fitting, instruction prompts and generation-based string matching. None of that fits a 50M base model. |
| Runner | **Custom** forward-pass scorer with state snapshot/branching | lm-eval's RULER tasks are `generate_until` with HF tokenizer requirements (`lm_eval/tasks/ruler/*.yaml`, [code-inspected]). Wrapping custom hybrid models is possible via an LM subclass, but slower and generation-based. |
| Cross-validation | **lm-eval-harness `ruler` and `babilong_longctx`** on 2–3 public small checkpoints (e.g., a Pythia and a Mamba of ~130–160M; exact choices are to be decided and are not claims), compared with our re-implementation of `niah_single_2` / `niah_multikey_2` on the same tokenizer | Detects generator bugs. Gives an external anchor for floors. |
| BABILong | **Reuse the official generator** (`NoiseInjectionDataset`) with our tokenizer | It already splices tokens. Train split available for fine-tuning. |
| LM-3 repeat-token loss | **Custom** (~80 LOC). The n-gram index is per-document. Corpus frequency comes from a 1B-token count table. | No existing implementation found (I did not search exhaustively). |

**Freezing protocol.** Before the main sweep, commit the generator code hash, all seeds, difficulty knobs and the item files (JSONL of token IDs plus metadata) to `experiments/`. Every arm is evaluated on byte-identical item files.

---

## 9. Mapping to hypotheses

| Hypothesis | Primary measurements | Decisive contrast |
|---|---|---|
| H1 partial sufficiency | R1(D≥3), R2, M1, A1 at ID lengths | Smallest-r hybrid vs r=1: fraction of the (r=1 − r=0) gap recovered, with paired bootstrap CI |
| H2 diminishing returns | Same, across all r | Monotone/concave fit of accuracy vs r (isotonic + logistic) |
| H3 task dependence | The r needed to reach 90% of r=1 per family | Overlap of CIs for "r_90" across R / M / A |
| H4 placement | Same tasks, equal r, different placement | Paired comparison on identical items |
| H5 length dependence | Distance-controlled design + accuracy × log T interaction, ID cells only | arm × log T coefficient per task family |
| H6 frontier | Capability (above) vs measured inference-state memory / prefill / decode (other agents) | Pareto-dominance test with CIs |

---

## 10. Open questions and risks

1. **Tokenizer choice** determines single-token value vocabularies. Is it a project decision owned by another agent? Confirm digit-splitting behaviour.
2. **Training context length.** If T_train < 32K for full-attention arms, most of 8K–32K is extrapolation, and H5 then confounds RoPE behaviour with capability. A uniform long-context phase is the fix, but it is outside my scope.
3. **Option (i) vs (ii).** This is unresolved until the pilot runs. Birdie shows fixed mixtures can fail for SSMs.
4. **Floor risk for aggregation (A1/A3) and multi-hop m≥4** even after exposure. Calibrate in the pilot.
5. **ICC and seed variance** are unknown until the pilot. Sample sizes above assume ICC ≈ 0.2.
6. **Literal-match only.** All retrieval is lexical (NoLiMa). Conclusions are about literal recall, not semantic retrieval.
7. **Synthetic-to-natural validity.** HELMET shows NIAH-style tasks correlate poorly with downstream tasks at frontier scale. Our claims should be framed as capability-specific, not "long-context ability" in general.
