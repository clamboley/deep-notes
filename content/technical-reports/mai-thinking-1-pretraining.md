---
title: "MAI-Thinking-1: Pretraining the Base Model"
tags: [pretraining, scaling-laws, mixture-of-experts, language-models, paper]
date: 2026-06-25
draft: false
source: https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
aliases: [MAI-Thinking-1, MAI-Base-1, MAI Pretraining, Efficiency Gain, EGFLOPs, EGTime, Scaling Ladder, rank non-invariance]
---

> [!abstract] TL;DR
> MAI-Thinking-1 (Microsoft AI) is a 35B-active / 1T-total MoE reasoning model trained **from scratch, with no distillation** from third-party models, on 30T tokens of in-house-processed human data. The pretraining contribution is a *methodology*, not a trick: every architecture and data decision is validated on a **scaling ladder** (models at constant tokens-per-parameter, config fixed by a single knob $L$) and scored by **efficiency gain** — how much more compute the baseline would need to match a candidate, fit from a scaling law $L = AC^{-\alpha}+E$. Two flavors, `EGFLOPs` (fair fight, ignores implementation maturity) and `EGTime` (what pays off on today's stack), keep the comparisons honest. Development is driven by cheap **NLL/bits-per-byte** evals rather than accuracy. The most transferable lesson: the *rank-invariance hypothesis* for data mixtures (small-scale ordering predicts large-scale) **can fail**, so mixtures are judged by their scaling trajectory across the ladder.

## Context

This is the first model from Microsoft AI's "hill-climbing machine" — the explicit framing that you don't optimize a model, you optimize a *repeatable process* (data pipelines + training infra + RL environments + evals + safety tests). Three design principles drive nearly every choice:

1. **Capabilities are learned, not inherited** — no distillation from other models; RL later starts from a base with zero reasoning traces. Imitated intelligence is held to lack steerability and robustness for long climbs.
2. **Simplicity is sustainable** — simple recipes, clean data, standard components that ride well-optimized kernels.
3. **Scientific rigor avoids shortcuts** — every decision must be falsifiable via ladders, ablations, and evals.

The base model is **MAI-Base-1** (34.7B active / 962B total): a decoder-only Transformer with interleaved local/global attention and alternating dense/MoE blocks, trained on 8K GB200 GPUs. After a 30T-token main phase and 3.55T tokens of mid-training, it reaches a 256K context. This note covers everything up to (but not including) the RL climb — see [[MAI-Thinking-1 RL Climb]] for post-training and [[MAI-Thinking-1 Systems]] for the distributed-training machinery.

Lineage: the ladder and efficiency-gain methodology are the operational descendants of [[Kaplan 2020]] and [[Chinchilla]] — they take the power-law worldview and turn it into a daily decision procedure.

## The Scaling Ladder

All ablations are run with scaling behavior as the *primary* object of study, because the benefit of an innovation often shrinks as compute grows. The ladder makes "scale this up" unambiguous by collapsing the entire model config to a **single parameter**: the number of layers $L$.

Everything else is derived:

| Quantity | Rule |
|---|---|
| Hidden size | $D = L \times \tfrac{256}{3}$ (fixed aspect ratio) |
| Query heads | $L$, rounded up to a multiple of 16 (tensor-parallel friendly) |
| Layer constraint | $L$ a multiple of 6, so the last attention layer is global (5:1 local:global) |
| Dense FFN | $2\times$ hidden expansion |
| LatentMoE | $2\times$ compression, $3\times$ per-expert expansion |

The ladder rungs run from L12 (365M active / 3.9B total) up to L78 (35.6B / 1015B), and MAI-Base-1 is a lightly-revised L78.

Two knobs control how the ladder is used:

- **Tokens-per-parameter (TPP)** is held *constant* across rungs for any given ablation, and chosen to fit the question. Architecture ablations sit near Chinchilla-optimal (**100–200 TPP**); the production run is deliberately **over-trained at 500–1,000 TPP** to yield a compact model for heavy inference. This is the [[Chinchilla]] result used in reverse: compute-optimal is the wrong target when you'll serve the model a lot.
- **The loop closes**: ablate on the current ladder → accepted changes become the *next baseline ladder* → repeat. That iteration is the "machine."

## Efficiency Gain (EG)

The core metric for "is this change worth it." Fit a scaling law to the baseline ladder:

$$L = f(C) = A\,C^{-\alpha} + E$$

where $C$ is training cost (FLOPs or time), $A$ scales the reducible loss, $\alpha$ is the decay exponent, and $E$ is the irreducible loss. For a candidate run reaching loss $L'$ at cost $C'$, ask: *how much would the baseline have to spend to reach $L'$?* That is $f^{-1}(L')$. Then

$$\boxed{\;\text{EG} = \frac{f^{-1}(L')}{C'}\;}$$

So **EG = 1.3 means the baseline needs 30% more compute** to match the candidate's loss. They fit EG-vs-cost *curves* (not single points) specifically to keep only improvements that **persist as scale grows**.

The genuinely useful idea is the two cost definitions:

> [!note] EGFLOPs vs EGTime
> - **EGFLOPs** uses FLOPs as cost and *intentionally ignores wall-clock efficiency (MFU)*. Established architectures have accumulated years of kernel optimization; a wall-clock comparison unfairly punishes a newer variant. EGFLOPs asks: *"if both got equal implementation effort, which models better?"*
> - **EGTime** uses wall-clock and answers the orthogonal question: *"given my current stack, what actually pays off today?"*

This split does real work in the sparsity ablation: MoE-in-every-layer with a shared expert reaches `EGFLOPs ≈ 1.03` (slightly better modeling per FLOP) but `EGTime < 1` (slower in practice). So they keep the interleaved high-sparsity-MoE + dense-FFN layout because it wins on the metric that matters for their hardware — a decision that is *legible* precisely because the two costs were separated. EG is also the metric they use to validate subtle wins like attention zero-init (below).

## Architecture (overview)

Mostly conservative, to stay on well-optimized paths (FlashAttention-4, Ulysses context parallelism). Highlights:

![[assets/mai-thinking-1/base-architecture.png]]
*MAI-Base-1 alternates local/global attention, dense FFNs, and sparse MoE blocks; the latent MoE path routes each token to 8 of 512 experts after a shared down-projection.*

- **Interleaved sparsity**: alternate a *high*-sparsity MoE layer (top-8 of 512 experts) with a *zero*-sparsity dense FFN. Scales comparably to medium-sparsity-everywhere on EGFLOPs but is faster (EGTime). Secondary finding: every-layer-MoE leans heavily on shared experts, but adding shared experts to the *interleaved* layout barely helps — the dense layers already do that job.
- **Periodic attention (Gemma-3 style)**: 5 local (sliding-window 512, RoPE base 10k) to 1 global. Global layers use **no position encoding** (comparable to RoPE, cheaper). GQA with 8 KV heads, head dim 128, QK-norm.
- **LatentMoE**: a shared down-projection *before* the all-to-all dispatch; routing decisions still use the *original* (uncompressed) representation. Shrinks the communicated tensors.
- Tied embeddings, no biases, RMSNorm at input and output of each block, SwiGLU, off-the-shelf `o200k_base` tokenizer (200k vocab) chosen for tooling integration over a marginally better in-house one.

Load balancing and the dropless-MoE decision are systems-flavored and covered in [[MAI-Thinking-1 Systems#MoE load balancing and dropless routing]].

## Evaluation: NLL / Bits-Per-Byte over Accuracy

For *development*, they evaluate almost entirely with next-token **NLL** (reported as cross-tokenizer-comparable **bits-per-byte** for model-vs-model), not multiple-choice or generative accuracy. Reasons, all practical:

- **Cost.** NLL is the same teacher-forced objective as pretraining — no autoregressive generation, no judge model — so ~40 benchmarks run cheaply and *consistently* on every experiment.
- **Robustness to confounds.** MCQ ability (parsing an A/B/C/D format) emerges only at surprisingly large scale, so MCQ accuracy is noisy on small rungs. Accuracy is also hostage to formatting: MATH's `\boxed{}` requirement and MBPP's `\n` vs `\r\n` mismatch between prompt and problem swing scores when pretraining-data formatting drifts. Teacher-forcing with a ground-truth prefix limits error compounding.
- **Construction cost.** A good Q&A benchmark needs difficulty calibration, dedup, and expert iteration; an NLL benchmark can start from any topical corpus and improve incrementally.

Category scores aggregate with **explicit weights** — the weights *are* the priority statement:

$$\text{Target} = 0.5\,\text{Code} + 0.175\,\text{STEM} + 0.175\,\text{Math} + 0.1\,\text{General} + 0.05\,\text{Multilingual}$$

Math is elevated to its own category (deliberately up-weighted) separate from STEM. Raw NLLs are normalized against a fixed in-house reference model before averaging.

> [!warning] Decontamination is load-bearing
> Eval leakage (especially via GitHub) produces *counterintuitive* results — e.g., nominally-coding data improving long-tail general-knowledge evals. They strip everything from `huggingface.co` and mirrors, apply 20-gram fuzzy dedup at 80% similarity, and — because that's imperfect — maintain private benchmarks they're confident aren't on the web.

Caveat the authors are honest about: which pretraining NLL metrics best predict *downstream* performance remains an open problem. NLL is the best available high-signal proxy, not a solved one.

## Pretraining Data and the Rank Non-Invariance Lesson

30T tokens, entirely in-house-processed from HTML, web PDFs, books/journals, and public GitHub — **no open-source training sets** and **no LM-generated synthetic data** (with active effort to remove AI-generated content), all in service of "learned not inherited."

Dedup is treated as first-order, with the framing that large sparse models memorize and that *predictive scaling is sensitive to the count of unique tokens* — bigger models exhaust novelty earlier and scale worse on low-diversity corpora. The stack: boilerplate removal → exact (hash) → fuzzy (MinHash LSH @ 0.8) → templated-page skeletonization → **semantic** dedup (embedding clusters, keep N representatives). A subtle cross-dataset step uses a global drop-order so a duplicate survives only in the highest-priority source — meaning *changing one dataset shifts data into/out of another even if you added nothing*.

> [!example] The single most transferable finding: rank non-invariance
> The **rank-invariance hypothesis** — that the relative ordering of two data mixtures is preserved as compute scales — is what makes cheap small-scale mixture search trustworthy. **It can fail.** A `stem-heavy-mix` beat a `code-heavy-mix` on held-out STEM NLL at small scale (as predicted), but at 23B-active / ~20T tokens the curves *crossed mid-training* and code-heavy won. Root cause on inspection: two high-quality but **low-diversity** STEM sources held 11.8% weight in stem-heavy vs 0.3% in code-heavy — great for small models, a liability at scale (the "exhaust novelty earlier" effect). **Takeaway:** judge a candidate mixture by its scaling *trajectory* across the ladder, not a single-scale point estimate.

![[assets/mai-thinking-1/rank-non-invariance.png]]
*The small-scale ordering of data mixtures can flip at large scale: `stem-heavy-mix` wins early, but `code-heavy-mix` overtakes during the 23B-active run.*

### Mixture optimization

A hierarchical alternating search over ~10 high-level categories:

- **Local search**: vary weights *within* one category, freeze the rest (e.g., code files vs PRs vs commits).
- **Global search**: freeze each category's internal makeup, vary the category weights.

Any dataset is capped at **8 epochs** to avoid overfitting / diminishing returns. The top candidates are re-validated at ~2.8× the mixing-search compute, and the winner is trusted only if it *stops changing with scale*. The final composition reveals the bet — code and math reasoning over breadth:

| Source family | Mix % | Training tokens | Avg. epochs |
|---|---|---|---|
| Code | 54.6 | 16.4T | 2.22× |
| STEM | 15.8 | 4.7T | 2.17× |
| Math | 5.4 | 1.6T | **5.28×** (most repeated) |
| Web text | 14.9 | 4.5T | 0.55× |
| PDFs | 4.7 | 1.4T | 0.53× |
| Books & journals | 3.1 | 0.9T | 1.65× |
| Multilingual (other) | 1.6 | 0.5T | **0.06×** (most downsampled) |

~300B unique math tokens are pushed the hardest (5.28 epochs), while web/PDF are each seen *less than once* even over a 30T run.

## Mid-training

3.55T tokens (3.4T at 64K context, then 150B at 256K), drawn *entirely from the pretraining corpus* (no new sources), re-weighted and re-packed at longer sequence lengths to minimize distribution shift. The mixture biases hard toward reasoning: **STEM/math → 35%, code → 55%, background → 10%**. Two ideas worth keeping:

- **Bloom's-taxonomy filtering** for STEM PDFs: keep documents at cognitive level "Analyze" or above — structured reasoning, not bare facts.
- **Memorization-aware epoch capping**: estimate per-source memorization by the fraction of a source's validation-loss improvement that comes from *near-certain* tokens (NLL $< 0.01$); a high fraction signals memorization, so cap that source's exposure harder. A more principled handle than raw epoch counts.

## Training-Recipe Highlights

A few non-standard choices, each validated on the ladder:

- **Attention zero-init.** At init, the attention softmax is near-uniform → causal mean-pooling → collapsed, correlated token representations → severe MoE routing imbalance that *worsens with depth*. Fix: initialize attention output to zero (zero the output RMSNorm gains), so the model *starts* as a stack of per-token FFNs and lets attention "kick in" gradually. Reduces initial imbalance and improves EG.
- **Shallower LR decay.** Cosine from $2\times10^{-4}$ to $2\times10^{-5}$ — a $0.1\times$ final-to-peak ratio rather than the usual $0.01\times$ — because decaying *less* improved *post-RL* results. (The optimization target is the eventual RL outcome, not pretraining loss.)
- **High dropout (0.15)** at each layer output, used as complementary regularization alongside weight decay.
- AdamW with $\beta_1=0.95,\ \beta_2=0.925$; weight decay 0.1, reduced on attention (0.01) and embeddings (0.005); grad-norm clip 1.0; global batch 134M tokens.
- **Numerical precision** is carefully zoned (BF16 default; FP8 for GEMMs; FP32 for sensitive spots including the *entire residual stream*) — see [[MAI-Thinking-1 Systems#Numerical precision recipe]] and [[FP8]].

The 30T loss curve had several early spikes that recovered with **no interventions and no skipped batches**; they traced the spikes to coding data correlating with expert imbalance under dropless routing.

## Results

Reported as bits-per-byte on held-out tasks (lower is better). MAI-Base-1 beats contemporaneous base models of similar *active* parameter count across Code/QA/STEM/Math; the only stronger model in the comparison (DeepSeek-V4-Pro) carries 1.4× active and 1.6× total parameters. Their previous-generation 23B model (also 30T tokens) is included as a progress yardstick.

## Related

- [[MAI-Thinking-1 RL Climb]] — post-training: GRPO modifications, self-distillation, the three specialists → consolidation
- [[MAI-Thinking-1 Systems]] — YOLO, MoE load balancing & dropless routing, determinism, goodput
- [[Kaplan 2020]] — the power laws the ladder/EG methodology operationalizes
- [[Chinchilla]] — compute-optimal allocation; here used *in reverse* (over-train for inference efficiency)
- [[FLOPs and MFU]] — the $6ND$ budget and MFU, the cost axes EG is built on
- [[FP8]] — the low-precision formats used in the numerics recipe

## References

1. The Microsoft AI Team, *MAI-Thinking-1: Building a Hill-Climbing Machine* (2026) — https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
2. Kaplan, J. et al., *Scaling Laws for Neural Language Models*, OpenAI (2020) — https://arxiv.org/abs/2001.08361
3. Hoffmann, J. et al., *Training Compute-Optimal Large Language Models*, DeepMind (2022) — https://arxiv.org/abs/2203.15556
