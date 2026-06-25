---
title: "MAI-Thinking-1: The Reinforcement Learning Climb"
tags: [reinforcement-learning, post-training, reasoning, language-models, paper]
date: 2026-06-25
draft: false
source: https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
aliases: [MAI-Thinking-1 RL Climb, RL Climb, MAI RL, self-distillation, adaptive entropy control, outer ratio clip, GRPO modifications]
---

> [!abstract] TL;DR
> MAI-Thinking-1's reasoning is learned by RL **from scratch** — the base model has *zero* reasoning traces — which makes **multi-thousand-step stability** the central problem. The recipe is GRPO (token-level policy gradient) plus two stabilizers: **adaptive entropy control** (an integral controller that widens/narrows the upper clip bound to hold a target entropy) and an **outer ratio clip** (a hard cap on all branches to kill gradient-norm spikes). The backbone enabler is **self-distillation**: periodically SFT a fresh checkpoint on the RL run's own rollouts to reset numerics, recover from collapses, and carry progress to new base models. Three independent specialists (STEM, agentic/SWE, helpfulness+safety) are trained, distilled into one model via SFT, then finished with a light RL pass. Reward engineering uses **lexicographic** and **gated** aggregation so safety is never traded for quality.

## Context

Pre- and mid-training give broad predictive competence but don't specify *how* to behave, solve long-horizon tasks, or allocate inference-time compute. RL fixes that — but because this is Microsoft AI's first in-house reasoning model and it starts from a checkpoint with **no exposure to reasoning traces**, the model must invent chain-of-thought against task feedback. Everything in this note is downstream of one constraint: *keep a log-linear climb going for thousands of steps without collapse*.

See [[MAI-Thinking-1 Pretraining]] for the base model (MAI-Base-1) this starts from, and [[MAI-Thinking-1 Systems#RL infrastructure (Rocket)]] for the async RL infra and the train/inference numerics gap that stability depends on.

## The Objective: GRPO + Two Modifications

Start from a policy $\pi_\theta$. For a prompt $q$, sample a group of $G$ responses; each gets a scalar reward $R_i$. The advantage is group-normalized, $A_i = (R_i - \text{mean}(R_{1:G}))/\text{std}(R_{1:G})$, shared across all tokens of response $i$. The objective is GRPO with token-level policy gradient (normalization over *all tokens in the global batch*, so every token contributes equally regardless of response length):

$$J(\theta) = \mathbb{E}\left[\frac{1}{\sum_i |y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|} \min\big(r_{i,t}A_i,\ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)A_i\big)\right]$$

with importance ratio $r_{i,t}(\theta) = \pi_\theta(y_{i,t}\mid q, y_{i,<t}) / \pi_{\text{old}}(y_{i,t}\mid q, y_{i,<t})$.

### Modification 1 — Adaptive entropy control

Use asymmetric clip bounds, with the *upper* bound (the dangerous one) controlled by a relaxation term $k$ on top of a base $\epsilon$:

$$r^{\text{tr}}_{i,t}(\theta) = \text{clip}\big(r_{i,t},\ 1-\epsilon,\ (1-\epsilon)^{-1} + k\big)$$

Too-wide an upper bound explodes entropy; too-tight collapses it. So $k$ is regulated **online by a simple integral controller**. Each step, estimate per-token entropy with an importance-weighted estimator $\hat{H}(\pi_\theta)$, then:

$$k \leftarrow \text{clip}\big(k + \delta\cdot\text{sign}(H^\star - \hat{H}(\pi_\theta)),\ 0,\ k_{\max}\big)$$

When entropy is too low, widen the upper clip so the policy can push alternative tokens harder; when too high, tighten. This is an automatic entropy regularizer **without** an explicit entropy bonus (which underperformed). Initialize $k=0$ so the initial clip interval is symmetric in log-ratio space. Production values: $\epsilon=0.6$, $k_{\max}=2.5$, $\delta=0.25$, target $H^\star=0.3$.

### Modification 2 — Outer ratio clip

Standard GRPO deliberately leaves two branches unclipped (the self-correcting cases). Those occasionally cause **catastrophic gradient-norm spikes**, so they add a hard outer clip on *all* branches:

$$r^{\text{out}}_{i,t}(\theta) = \text{clip}(r_{i,t},\ r_{\min},\ r_{\max})$$

with $r_{\max}$ large (50) and $r_{\min}$ unconstrained (0) — in the spirit of dual-clip PPO. It discards extreme old/new probability discrepancies while preserving normal trust-region behavior. Result: fewer spikes, more stable climbing.

## Reward Design

A shared decomposition across all three climbs:

$$R(q, y_i) = R_{\text{task}}(q, y_i) + w_{\text{lang}}\cdot R_{\text{lang}}(y_i) - w_{\text{len}}\cdot R_{\text{len}}(y_i)$$

- **Language consistency** penalizes non-English words in the CoT — mixed-language CoTs *correlate with train/inference logprob divergence* and destabilize training (a stability concern, not aesthetics). $w_{\text{lang}}=0.5$, per-word penalty $\alpha=0.005$. (top-p sampling alone also largely prevents stray low-probability foreign tokens.)
- **Length penalty** is difficulty-aware: $R_{\text{len}}(y_i) = \rho_q \cdot |y_i| / \ell_{\max}$, where $\rho_q$ is the problem's pass rate. **Hard problems (low pass rate) get a weaker penalty** so the model may reason longer; easy problems get penalized into concision. $w_{\text{len}}=0.25$, removed entirely at the 128k stage.

## Sampling Strategy

Three efficiency/stability levers:

- **Early exit.** Sample $G_{\text{early}}=16$ first and compute the pass rate; only if it's in $[0.05, 0.8]$ spend the full $G=128$. Then a *second* filter $[0.1, 0.8]$ removes low-variance groups (all-right or all-wrong give no relative signal).
- **Top-p masking** ($p=0.97$). Reuse the rollout's top-p truncation mask during training and set out-of-nucleus logits to $-\infty$ before softmax. **Backpropagating through tokens outside the sampled nucleus causes catastrophic off-policy mismatch** and divergence within a few steps; masking prevents it (at the cost of mask storage/replay).
- **Length curriculum.** Cap rollouts at 8k early, doubling 16k → 32k → 64k → 128k as capability grows — long traces are rarely needed in the low-performance regime and are expensive.

## Self-Distillation — the Backbone

The most important and most under-explained idea. **Mechanic:** collect rollouts generated *during* RL, SFT a mid-trained checkpoint on them, and resume RL from that SFT'd model. Four distinct uses:

1. Move from a raw text prompt to the native chat format (just reformat the SFT data).
2. **Recover from collapses** — and the key insight: resuming from a *pre-collapse* checkpoint often fails, because instabilities are embedded in the parameters many steps before the visible collapse. Self-distillation re-seeds clean numerics while keeping the discovered behavior.
3. Carry progress forward when a new base/mid-trained checkpoint arrives.
4. Filter out reward-hacking traces during the SFT.

![[assets/mai-thinking-1/stem-climb-self-distillation.png]]
*During the STEM climb, self-distillation resets are visible as star-marked handoffs that preserve progress while recovering from numerical collapses and moving to longer contexts.*

> [!tip] Self-distillation best practices (reusable)
> - **~1M traces is enough.** More risks over-constraining the policy and killing exploration once RL resumes.
> - Training on *incorrect*-final-answer traces works about as well as success-only; they used success-only because RL produces them in abundance.
> - Use traces from *later* checkpoints, but spanning a *range* of strong checkpoints — diversity beats a single final policy for downstream exploration.
> - For a fixed token budget, **prompt diversity beats traces-per-prompt**, and plain random sampling beat biased selection.
> - Mix in mid-training data to avoid forgetting long-context behavior.
>
> Two hyperparameters matter during self-distillation: high dropout (0.15, raises entropy / prevents collapse) and a *large* MoE load-balance coefficient ($10^{-2}$ vs $10^{-5}$ during RL), because narrow RL distributions cause expert imbalance — and balancing during self-distillation carries over since the contexts come from the RL run itself.

## Three Specialists → One Model

Train three independent teachers on the same recipe but different prompt distributions and rewards: **STEM/competitive-code**, **agentic/tool-use**, and **helpfulness+safety**. Then distill all three into one model via SFT, and finish with a lightweight RL pass → MAI-Thinking-1.

![[assets/mai-thinking-1/rl-climbs-overview.png]]
*The final model comes from three specialist RL teachers, trace-distillation SFT into one consolidated model, and a final lightweight RL climb.*

The **consolidation SFT mixture** is balanced *by sample weight*, even though the token distribution skews to long reasoning traces:

| Capability | Sample weight | Token weight |
|---|---|---|
| STEM & coding | 56% | 89% |
| Agentic | 11% | 9% |
| Helpfulness & safety | 33% | 2% |

Balancing by *sample* weight is what matters; the token skew doesn't hurt helpfulness. The final consolidation RL keeps a small amount of STEM/coding data because reasoning otherwise degrades slowly over a helpfulness-focused climb. A useful transfer asymmetry: **mixing STEM tasks into the agentic climb stabilizes it and transfers positively to SWE, while agentic tasks don't transfer back to single-pass STEM.**

## Domain Pipelines (where the work actually is)

### STEM data

A four-phase pipeline turning textbooks/PDFs/competition archives into verifiable $(q,a)$ or $(q, \{\text{tests}\})$ pairs: hierarchical parsing → QA pairing → curation → scoring. Two standout steps: **MCQ/proof → open-ended conversion** (MCQ is guessable, giving an unreliable reward; proofs are hard to verify), done with 3× consensus; and a **blind-grading guard against bad ground truth** — for problems the strongest model tier rarely solves, present that model's consensus answer and the stored ground truth to a judge in random order, and if the judge prefers the model's answer, drop the item as having a suspect label. Result: >5M samples, 550k+ hard. Competitive coding is separate (160k problems, 17 languages, with runtime/memory constraints) because comprehensive test cases don't exist in unstructured PDFs.

### Agentic / SWE

A ReAct loop over a **Sandbox Execution Environment (SEE)** — a fresh, network-isolated container per task. The SWE environment-building funnel is brutal, and the survival rates calibrate expectations:

| Stage | Surviving |
|---|---|
| Public GitHub PRs (start) | 102M |
| Merged, <15 files, code+test changes, linked issue | 4.87M |
| Automatic agentic env building (LLM writes Dockerfiles, validated) | 2.08M (42.8%) |
| Reference grading (F2P / P2P test extraction) | 745k (15.3%) |
| Env + grader re-validation in the training sandbox | **265k (5.5%)** across 94k repos |

Roughly 1 in 18 candidate PRs survives. Discarded-but-executable environments are reused via synthetic problem generation.

> [!warning] Reward-hacking prevention in SWE (a copyable checklist)
> - **Internet search**: cut network access (or allowlist the minimum) so the agent can't retrieve the PR's golden solution.
> - **Local git history**: *time-travel* the repo by scrubbing all commits/refs/branches after the base commit, so the fix can't be dug out of `.git` — while keeping git itself usable.
> - **Test tampering**: reset all agent-modified test files before grading, hide test changes until grading, and run an LLM monitor for subtler tricks (e.g., monkey-patching the test framework).

General tool-use environments are different in character: >50 tools each, mostly synthetic (>150 environments, 130k tasks), with environment-specific personas for diversity and deliberately-included *no-tool-needed* tasks to curb over-eager tool calling.

### Helpfulness & safety reward combination

Quality isn't machine-verifiable, so they combine a trained **reward model** (a post-trained MAI-Base-1 predicting preference *as text tokens* over k-way side-by-sides; inference uses cyclic permutation of candidates to debias position), **AI judges** (the fast, adaptable lever), and **verifiable rewards** wherever a constraint is checkable. The combination logic is the clever part, because naive summation lets the largest-magnitude reward dominate and lets a fluent-but-unsafe response score positively:

- **Lexicographic shaping**: a secondary reward influences the gradient only when the primary is tied *within the rollout group* (scale-invariant by construction).
- **Gated application**: safety gates everything — a policy-non-compliant response gets the minimum reward and is never graded on quality.

The gating is justified by an audit stat: **87.8% of policy-non-compliant responses received reward-model scores ≥ 3**, so without the gate the RM would happily reward unsafe-but-fluent output. Honesty uses a five-bucket scheme (`CONFIDENT_CORRECT` … `CONFIDENT_INCORRECT`) that penalizes confident hallucination most, treats abstention as neutral, and gives unconfident-but-correct a *reduced* reward to discourage over-hedging.

## Key Hyperparameters (RL)

- AdamW, $\beta_1=\beta_2=0.95$, $\epsilon=10^{-15}$, no weight decay; constant LR $10^{-6}$ (lowered to $9\times10^{-7}$ at long lengths to reduce off-policiness), no warmup.
- Global batch 7040 (packed), ≤12000 unpacked sequences. 5 gradient steps between inference-model updates; discard rollouts more than 8 inference updates stale (40 gradient steps).
- MoE load-balance coefficient $10^{-5}$ during RL.
- Self-distillation SFT: packed seqs, global batch 2048, seq len 128k, AdamW wd 0.001, cosine LR (max $1.7\times10^{-5}$, min $5.2\times10^{-6}$, 2% warmup).

## Related

- [[MAI-Thinking-1 Pretraining]] — the base model and the data/eval methodology this builds on
- [[MAI-Thinking-1 Systems]] — async RL (Rocket), the train/inference numerics gap, weight transfer
- [[FLOPs and MFU]] — compute accounting referenced by the RL infra

## References

1. The Microsoft AI Team, *MAI-Thinking-1: Building a Hill-Climbing Machine* (2026) — https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
2. Shao, Z. et al., *DeepSeekMath* (GRPO) (2024) — https://arxiv.org/abs/2402.03300
3. Yu, Q. et al., *DAPO: An Open-Source LLM RL System at Scale* (2025) — https://arxiv.org/abs/2503.14476
