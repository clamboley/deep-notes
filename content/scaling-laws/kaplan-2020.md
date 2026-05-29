---
title: Scaling Laws for Neural Language Models
tags: [scaling-laws, language-models, pretraining, paper]
date: 2026-05-29
draft: false
source: https://arxiv.org/abs/2001.08361
aliases: [Kaplan 2020, OpenAI Scaling Laws, Kaplan et al. 2020]
---

> [!abstract] TL;DR
> Language model performance follows smooth power laws in model size $N$, dataset size $D$, and compute $C$, each over many orders of magnitude. These laws let you predict loss before training and prescribe how to allocate a compute budget: per Kaplan, scale model size much faster than data (later revised by [[Chinchilla (Hoffmann et al. 2022)]]).

## Context

January 2020, OpenAI. The transformer recipe was established, but scaling was largely heuristic. Kaplan et al. ran a systematic sweep of transformer LMs on WebText2 — varying $N$ from ~768 to 1.5B parameters, $D$ up to ~22B tokens, $C$ over many orders of magnitude — and found performance follows remarkably clean power laws across all three axes. These laws directly informed GPT-3's design six months later and kicked off the era of deliberate, law-guided scaling.

Critical lineage: builds on the transformer architecture, directly motivates [[GPT-3]], and was revised by [[Chinchilla (Hoffmann et al. 2022)]] two years later.

## What are Scaling Laws

A **scaling law** is an empirical power-law relationship between model test loss $L$ (cross-entropy in nats) and a resource. On a log-log plot, $L$ vs. scale falls on a nearly straight line:

$$L \propto X^{-\alpha}$$

where $X \in \{N, D, C\}$ and $\alpha > 0$ is the scaling exponent. The key property: $\alpha$ stays roughly constant over many orders of magnitude, so you can measure cheaply at small scale and extrapolate confidently to large scale.

## Key Findings

![[kaplan-fig1.png]]
*Figure 1: Test loss follows power laws in N, D, and compute C across many orders of magnitude.*

### Power laws in N, D, C independently

When the other resources are not the bottleneck:

| Resource | Law | Exponent |
|----------|-----|----------|
| Model size | $L(N) = (N_c/N)^{\alpha_N}$ | $\alpha_N \approx 0.076$ |
| Dataset size | $L(D) = (D_c/D)^{\alpha_D}$ | $\alpha_D \approx 0.095$ |
| Compute (optimal) | $L(C_\text{min}) = (C_c/C_\text{min})^{\alpha_C}$ | $\alpha_C \approx 0.050$ |

$N_c$, $D_c$, $C_c$ are dataset-dependent fitting constants (specific to WebText2 with their tokenizer). The exponents are the universal, transferable part.

### Joint N and D

When both are finite, they interact — increasing $N$ beyond what $D$ can support yields diminishing returns:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

Verification: as $N \to \infty$ the first term vanishes, giving $L \to (D_c/D)^{\alpha_D}$; as $D \to \infty$ the second term vanishes, giving $L \to (N_c/N)^{\alpha_N}$. The formula interpolates smoothly between both bottleneck regimes.

### FLOPs formula

$$C \approx 6ND$$

Total training FLOPs for a transformer with $N$ non-embedding parameters trained on $D$ tokens. Factor of 6 = ~2N FLOPs/token forward + ~4N backward. *(Full derivation in [[FLOPs and MFU]].)*

### Architecture insensitivity

At fixed total $N$, changing depth/width ratio, number of heads, or FFN multiplier has much less effect on loss than changing $N$ itself. What matters is total parameter count, not how it is distributed. The optimal depth scales roughly as $d_\text{model} \propto \sqrt{N}$, but this is a weak effect in practice.

### Sample efficiency

![[kaplan-fig5.png]]
*Figure 5: Larger models reach any given loss level with fewer training tokens — they are more sample-efficient.*

Each gradient step carries more signal when the model has higher capacity. This is why a 10× bigger model trained for 1/10 the steps often beats the smaller model at the same total compute.

## Optimal Compute Allocation

![[kaplan-fig7.png]]
*Figure 7: For a given compute budget, both optimal model size and optimal dataset size follow power laws in C.*

Given a compute budget $C$ (FLOPs), the compute-optimal allocation is:

$$N_\text{opt} \propto C^{0.73}, \quad D_\text{opt} \propto C^{0.27}$$

**Practical rule**: doubling compute → scale model by $2^{0.73} \approx 1.66\times$, data by $2^{0.27} \approx 1.20\times$. Model size grows roughly 8× faster than dataset size per decade of compute.

**Key corollary**: fully converging a model is wasteful. For a large compute budget, run a bigger model for fewer steps rather than a smaller model to convergence. The early training of a bigger model beats the late training of a smaller one at equal FLOPs.

> [!warning] Chinchilla revision
> These exponents were later shown to be biased. [[Chinchilla (Hoffmann et al. 2022)]] (2022) found $N_\text{opt} \propto C^{0.50}$ and $D_\text{opt} \propto C^{0.50}$ — scale model and data equally. The bias in Kaplan: small models were trained to convergence while large models were stopped early, inflating the apparent benefit of model size.

## Things I Had to Untangle

> [!question] What is $C_\text{min}$ vs. $C$?
> $C$ is the actual FLOPs used in a training run. $C_\text{min}$ is the minimum FLOPs to reach a given loss assuming optimal training: the right batch size ($B = B_\text{crit}$) and a well-tuned LR schedule. Any real run has $C \geq C_\text{min}$; the gap grows when batch size or LR are suboptimal. The paper derives the power law for $C_\text{min}$ specifically, not for an arbitrary run.

> [!tip] Why is $\alpha_C \approx 0.050$ smaller than $\alpha_N \approx 0.076$?
> Because a compute-optimal run scales $N$ and $D$ jointly. The exponent $\alpha_C$ reflects this combined improvement, which is slower per log unit of $C$ than improving $N$ alone (you also need more data). Do not compare these exponents across variables to rank "which resource matters more."

> [!question] $\alpha_D > \alpha_N$ — does that mean data scales better than parameters?
> Marginally per unit of log-scale, yes. But this comparison ignores the different costs of acquiring data vs. parameters, and the fitting constants $N_c$, $D_c$ set the crossover point. In practice, the compute allocation determines the effective ratio.

## Open Questions

- [ ] Do these power laws hold beyond ~1T parameters? The paper reached ~1.5B — extrapolating 3+ orders of magnitude is a significant leap.
- [ ] The laws are for cross-entropy loss. How do they map to downstream task performance? ([[Beyond Neural Scaling Laws]] argues the relationship is complex, task-dependent, and can show discontinuous jumps.)
- [ ] Do these laws generalize to non-autoregressive or non-transformer architectures? (See [[LeJEPA]] and [[Looped Language Models]] for contrast.)

## Related

- [[Chinchilla (Hoffmann et al. 2022)]] — the must-read follow-up; corrects the compute-optimal allocation
- [[GPT-3]] — the first large model designed using these laws
- [[FLOPs and MFU]] — full derivation of the 6ND formula and how to compute hardware utilization
- [[Beyond Neural Scaling Laws]] — whether power laws hold or break at extreme scale

## References

1. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D., *Scaling Laws for Neural Language Models*, OpenAI (2020) — https://arxiv.org/abs/2001.08361
