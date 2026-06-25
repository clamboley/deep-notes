---
title: Training Compute-Optimal Large Language Models
tags: [scaling-laws, language-models, pretraining, paper]
date: 2026-06-01
draft: false
source: https://arxiv.org/abs/2203.15556
aliases: [Chinchilla, Hoffmann 2022, Chinchilla scaling laws, DeepMind scaling laws]
---

> [!abstract] TL;DR
> Kaplan's compute-optimal allocation was wrong: model size and training tokens should scale equally with compute ($N_{opt} \propto C^{0.5}$, $D_{opt} \propto C^{0.5}$), not 0.73/0.27. The practical rule is ~20 tokens per parameter. Chinchilla (70B, 1.4T tokens) uses the same compute as Gopher (280B, 300B tokens) and outperforms it — and every larger model of the era — at 4× lower inference cost.

## Context

March 2022, DeepMind. The post-Kaplan era had settled into a pattern: when compute scales, scale the model. GPT-3, Jurassic, Gopher — all trained on ~300B tokens regardless of model size. Hoffmann et al. ran 400+ training runs across 70M–16B parameters and showed this was systematically suboptimal.

The paper sits at the center of the scaling law lineage: it directly corrects [[Kaplan 2020]] and its prescription shaped every serious model trained after 2022 — Llama, Mistral, and essentially all open models.

## The Kaplan Revision

Kaplan: given a 10× compute increase, grow model 5.5×, data 1.8×. Chinchilla: **grow both equally.**

Two methodological flaws caused Kaplan's bias:

1. **Fixed cosine schedule.** Kaplan used a schedule decayed over 130B tokens for all runs. For runs trained on $D' \ll 130\text{B}$ tokens, the intermediate loss overestimates what a properly-scheduled run would achieve at $D'$ — making data-limited training appear more costly than it is, artificially inflating the benefit of model size.

2. **Small model regime.** Most of Kaplan's runs were under 100M parameters, where there's curvature in the FLOP-loss frontier. Extrapolating from this regime inflates the model-size exponent. Chinchilla's analysis is dominated by models >500M.

## Three Approaches to the Compute-Optimal Frontier

All three independently estimate $N_{opt}(C)$ and $D_{opt}(C)$, and all converge on equal scaling:

| Approach | $a$: $N_{opt} \propto C^a$ | $b$: $D_{opt} \propto C^b$ |
|---|---|---|
| 1. Training curve envelope | 0.50 | 0.50 |
| 2. IsoFLOP profiles | 0.49 | 0.51 |
| 3. Parametric fit | 0.46 | 0.54 |
| Kaplan et al. | 0.73 | 0.27 |

### Approach 1: Training curve envelope

Train model families (70M–10B+) each for 4 different token horizons (cosine cycle lengths spanning 16×). For each FLOP budget, find which model + token count achieves the lowest loss. Fit power laws to the resulting $(C, N_{opt})$ and $(C, D_{opt})$ series.

![[assets/chinchilla-2022/chinchilla-fig2.png]]
*Figure 2: Left: training loss curves for all runs. Center and right: the lower envelope of minimum-loss points, with optimal N and D fit as power laws in FLOPs. Green lines project the Gopher budget ($5.76 \times 10^{23}$ FLOPs) to ~67B parameters and ~1.5T tokens.*

### Approach 2: IsoFLOP profiles

Fix 9 FLOP budgets ($6 \times 10^{18}$ to $3 \times 10^{21}$). For each budget, train models of varying size with cosine schedules matched to that budget's token count. Plot final loss vs. $N$ — each curve has a clear valley. Fit a parabola to each isoFLOP slice to extract $N_{opt}$.

![[assets/chinchilla-2022/chinchilla-fig3.png]]
*Figure 3: Left: each isoFLOP slice has a clear loss minimum at a particular model size — the valley shifts right as compute grows. Center and right: optimal N and D extracted from those minima, fit as power laws in FLOPs.*

### Approach 3: Parametric loss model

Fit all final losses to:

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- $E$: irreducible loss — entropy of the data distribution
- $A/N^\alpha$: underfitting from finite model capacity
- $B/D^\beta$: underfitting from finite training (not trained to convergence)

Parameters $(A, B, E, \alpha, \beta)$ estimated by minimizing Huber loss ($\delta = 10^{-3}$) via L-BFGS over all runs. The efficient frontier follows from minimizing $\hat{L}$ subject to $C = 6ND$:

$$N_{opt}(C) = G\!\left(\frac{C}{6}\right)^a, \quad D_{opt}(C) = G^{-1}\!\left(\frac{C}{6}\right)^b$$

where $G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}$, $a = \frac{\beta}{\alpha+\beta}$, $b = \frac{\alpha}{\alpha+\beta}$.

![[assets/chinchilla-2022/chinchilla-fig4.png]]
*Figure 4: Left: contour plot of $\hat{L}(N,D)$ with the efficient frontier (blue) threading through the lowest-FLOP point on each iso-loss contour. Right: isoFLOP slices showing the parabolic structure. This approach predicts a slightly smaller optimal (~40B) than Approaches 1 and 2, due to the parametric fit weighting high-compute points more heavily.*

## Optimal Compute Allocation

The practical rule: **~20 tokens per parameter** for compute-optimal training ($D_{opt} \approx 20N$). Combined with $C = 6ND$, this gives closed-form answers for all three planning scenarios:

| Given | Optimal N | Optimal D | Compute C |
|---|---|---|---|
| Budget $C$ | $\sqrt{C/120}$ | $20 N_{opt}$ | — |
| Data $D$ | $D / 20$ | — | $0.3\, D^2$ |
| Model size $N$ | — | $20N$ | $120\, N^2$ |

Verification: Gopher's budget $C = 5.76 \times 10^{23}$ → $N_{opt} = \sqrt{5.76 \times 10^{23}/120} \approx 69\text{B}$. Paper says 40–70B, Chinchilla is 70B. ✓

**Concrete reference points:**

| Parameters | Compute (FLOPs) | Tokens |
|---|---|---|
| 1B | $1.2 \times 10^{20}$ | 20B |
| 7B | $5.9 \times 10^{21}$ | 140B |
| 10B | $1.2 \times 10^{22}$ | 200B |
| 70B | $5.9 \times 10^{23}$ | 1.4T |
| 175B | $3.7 \times 10^{24}$ | 3.5T |

> [!warning] Train-optimal ≠ inference-optimal
> These prescriptions minimize *training* FLOPs. If you're serving many inference requests, the right move is to train a *smaller* model on *more* tokens than Chinchilla suggests — amortize the extra training cost over inference calls. This is the explicit reasoning behind LLaMA. The Chinchilla frontier is a lower bound on training compute, not a guide to total deployment cost.

## Chinchilla: Validation at Scale

70B parameters, 1.4T tokens — same compute as Gopher (280B, 300B tokens). Architecture matches Gopher except $d_{model} = 8192$ (vs. 16384), 64 heads (vs. 128), and a smaller batch size. Uses AdamW instead of Adam (better fine-tuning) and a slightly modified SentencePiece tokenizer (94.15% token overlap with Gopher's).

**Results vs. larger contemporaries:**

| Benchmark | Chinchilla 70B | Gopher 280B | GPT-3 175B | MT-NLG 530B |
|---|---|---|---|---|
| MMLU 5-shot | **67.6%** | 60.0% | 43.9% | — |
| BIG-bench avg | **65.1%** | 54.4% | — | — |
| LAMBADA 0-shot | **77.4%** | 74.5% | 76.2% | 76.6% |
| RACE-h few-shot | **82.3%** | 71.6% | 46.8% | 47.9% |
| Wikitext-103 ppl | **7.16** | 7.75 | — | — |

Chinchilla outperforms models 2–8× larger on nearly every task, and at 4× lower inference cost. The MMLU result (67.6%) exceeded the expert forecast for June 2023 (63.4%) made three months prior.

## Related

- [[Kaplan 2020]] — the paper this corrects; source of the original 0.73/0.27 exponents
- [[GPT-3]] — canonical undertrained model by Chinchilla standards: 175B on 300B tokens vs. the ~3.7T tokens Chinchilla would prescribe
- [[FLOPs and MFU]] — derivation of the $C \approx 6ND$ identity used throughout

## References

1. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de las Casas, D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O., & Sifre, L., *Training Compute-Optimal Large Language Models*, DeepMind (2022) — https://arxiv.org/abs/2203.15556
