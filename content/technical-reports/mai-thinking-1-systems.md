---
title: "MAI-Thinking-1: Training & Inference Systems"
tags: [training, infrastructure, mixture-of-experts, hardware, distributed-training, paper]
date: 2026-06-25
draft: false
source: https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
aliases: [MAI-Thinking-1 Systems, YOLO framework, Rocket RL, dropless MoE, global-batch load balancing, hill-climbing machine, goodput, NVLink SHARP]
---

> [!abstract] TL;DR
> The cluster is treated as an *active part of model development*, not a backdrop. MAI-Base-1 trained on **8K GB200 GPUs** on a single logical cluster using **YOLO**, an in-house from-scratch training framework. Key systems ideas: **global-batch load balancing** (the *aggregation* of expert frequencies matters far more than the loss type), **fully dropless MoE** (no token dropped, variable-size all-to-all), a carefully *zoned* FP8/FP32 numerics recipe, and **bitwise determinism** as a first-class property (which is why they *disable NVLink SHARP*). Each architecture generation (v2→v5) improved efficiency-gain but transiently tanked MFU, recovered by 20+ optimizations. RL runs on **Rocket** (async, SGLang inference + YOLO learner), where the train/inference **numerics gap** is the controlling stability risk. Final run: **90% goodput** at 8K GPUs; **MAIA-200** serves the model at >40% higher tokens/watt than GB200.

## Context

In training, the objective is to maximize useful FLOPs per wall-clock day while preserving numerical correctness, deterministic recovery, high MFU, and high goodput. In inference, maximize useful tokens per second *and per watt*. The same principles apply to both: topology matters, memory movement matters, silent correctness failures are unacceptable, and aggregate throughput only counts when it becomes reliable progress.

This note is the infra companion to [[MAI-Thinking-1 Pretraining]] (the model/data) and [[MAI-Thinking-1 RL Climb]] (the RL recipe that the async infra and numerics control below make possible).

## YOLO — the Training Framework

Built from scratch on PyTorch for full-stack control (kernels → parallelism → scheduling), drawing on Megatron-Core / DeepSpeed / TorchTitan but co-designed with the model. It implements the whole training loop and supports pretraining, mid-training, SFT, and the learner side of RL. Notable design choices:

- **Descriptive sharding annotations** (like JAX/DTensor) that *deliberately do not auto-insert communication* — to avoid accidental synchronization points. This gives per-tensor control: tensor-parallel for embedding/loss/attention weights, but not for MLP/MoE weights; different data-parallel degrees for MoE vs non-MoE tensors (*parallel folding*).
- **ZeRO stages 1–3 from scratch**, always storing parameters *sharded* regardless of stage (which makes distributed AdamW trivial — optimizer state is automatically sharded too). The stages differ only in when gathered-parameter/gradient buffers are cleared.
- Custom kernels (Triton/CUDA/CuteDSL/CUTLASS): FP8 GEMMs with delayed scaling, grouped GEMMs for MoE, fused quantization with scale-factor swizzling, casts fused into RMSNorm.
- Activation checkpointing + asynchronous activation offload to host memory (pinned, on dedicated CUDA streams).

## MoE Load Balancing and Dropless Routing

This is where two frequently-confused MoE concepts live.

### Aggregating expert frequencies (global-batch load balancing)

The router sends each token to its top-k experts; an auxiliary loss pushes that assignment toward uniform. The GShard/Switch-style loss is, schematically,

$$\mathcal{L}_{\text{aux}} \propto N \cdot \sum_{i} f_i \, P_i$$

where $f_i$ is the **fraction of tokens actually dispatched to expert $i$** (the hard count — the *frequency*) and $P_i$ is the mean router probability mass on expert $i$ (the soft, differentiable part). The question is: **over which set of tokens do you compute $f_i$?**

- The naive default computes $f_i$ *locally* — per micro-batch, per data-parallel worker — because that needs no communication.
- **Global-batch aggregation** instead all-reduces the per-expert token counts across *all* DP workers and *all* gradient-accumulation micro-batches, so $f_i$ reflects the true distribution over the entire optimizer step.

Why it matters more than the loss formulation: balancing per-micro-batch forces *every small chunk* to be uniform, which fights legitimate specialization (a micro-batch that happens to be all code shouldn't be punished for using code experts) and is a high-variance estimate. Global aggregation only requires balance *in aggregate*, frees individual sequences to specialize, and gives a low-variance frequency. With an accurate global $f_i$, the GShard loss and the loss-free bias-adjustment variant converge to similar behavior — hence: **aggregation dominates the loss type.**

### Dropless MoE

The classic efficiency trick fixes an **expert capacity** $C = (\text{capacity factor})\times \text{tokens}/\text{num experts}$, so every expert processes a static, identically-shaped tensor (hardware-friendly: fixed shapes, fixed-size all-to-all). The catch: when an expert is over-subscribed, the **overflow tokens are dropped** — they receive no expert transformation at that layer and carry only their residual forward.

**Dropless** means *no token is ever dropped*: every token reaches every expert it selected, regardless of imbalance. The price is ragged computation — grouped GEMMs over variable per-expert token counts and **variable-size all-to-all** — plus a synchronization step to communicate the actual token counts before sizing buffers. Two payoffs the authors cared about:

> [!note] Why they went fully dropless
> 1. Dropping wastes forward compute *and* information.
> 2. Capacity-based dropping can introduce **causal leakage**: if which tokens get dropped depends on position/order (the buffer filling up in sequence), a token's fate can be influenced by later tokens, corrupting the autoregressive objective. Dropless sidesteps the whole class of bug.
>
> Critically, they observed that **load-balancing conclusions can flip depending on expert capacity** — even careful low-rate dropping gives different answers than a dropless setting. To make ablations trustworthy, they standardized on dropless.

To bound memory under imbalance, they built a **static-memory dropless** mode: cap tokens per *round* but run multiple `dispatch → compute → collect` rounds until everything is processed (capped per pass, nothing dropped overall), with per-expert-per-round recompute in the backward pass to avoid storing lopsided activations.

## Numerical Precision Recipe

Default weights/activations are **BF16**. **FP8** is used for GEMMs (E4M3 in the forward, E5M2 for the data-gradient), with BF16 weight-gradient compute and FP32 accumulation; all FP8 ops use delayed scaling with a 1024-step abs-max history. **FP32** is reserved for the genuinely sensitive spots:

- *All* pre-softmax activations: attention scores, MoE router logits, output logits.
- MoE combine, and the **entire residual stream from embedding to output**.
- Embedding / RMSNorm / router weights; the full optimizer (main params + momentum + all AdamW math); DP all-reduce and micro-batch gradient-accumulation buffers.

Stochastic rounding is applied on downcasts (gradients flowing from the FP32 residual stream into lower-precision layer compute). For the format landscape these choices sit in, see [[FP8]].

## Determinism (and Why NVLink SHARP Is Disabled)

Determinism is a first-class infrastructure property, not just model code. For fixed hardware topology, config, and software version, **two runs produce bitwise-identical models** — accepted even at an MFU cost, because it makes "did this change do anything?" answerable and powers a "golden config" regression-test suite.

Floating-point accumulation is non-associative ($(a+b)+c \neq a+(b+c)$ under rounding), so reproducibility requires controlling *the order* of every sum:

- **GPU kernels**: deterministic accumulations in prescribed orders rather than unordered atomics — e.g., a two-stage tiled RMSNorm backward (partial sums then a fixed-order finalization), and a *stable sort* for MoE top-k tie-breaking.
- **Network collectives**: this is where **NVLink SHARP** comes in. SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) is NVIDIA's **in-network reduction** — the NVSwitch fabric performs the summation *in-switch* as data flows through (exposed via NCCL's NVLS algorithm). It's faster (less data movement, arithmetic offloaded from the SMs), but the accumulation order is set by the switch's aggregation tree and dynamic packet-arrival timing, which varies run-to-run with congestion. That makes the collective non-bitwise-reproducible. **Disabling NVLink SHARP** forces the all-reduce back onto a GPU-side ring/tree path with a pinned accumulation order, and they hold the NCCL topology constant so intra-rack reductions always combine in the same sequence — at the cost of the speed SHARP would give.

It's the network-collective instance of the same principle behind the deterministic kernels: anywhere a sum's order can drift, nail it down and eat the MFU cost.

## Architecture / Infra Co-Design (v2 → v5)

The headline pattern: **every architecture change improved efficiency-gain (EG) but tanked the *initial* MFU** when run on the previous generation's stack, and it took 20+ optimizations per generation to recover MFU above 20%. EG and MFU pull against each other transiently; you accept the temporary MFU hit because EG is the durable modeling win.

![[assets/mai-thinking-1/mfu-eg-codesign.png]]
*Across v2-v5, model changes improve EG while MFU initially drops and then recovers after infrastructure work.*

| Version | Active/Total | Layers | Top-k/Experts | Capacity | MFU (init → final) | EG vs v2 |
|---|---|---|---|---|---|---|
| v2 (first GB200) | 23B/600B | 54 | 4/192 | 2 | 18% → 22% | 1.00× |
| v3 (dropless MoE) | 23B/600B | 54 | 4/192 | ∞ | ~22% | 1.40× |
| v4 (more experts, LatentMoE, 8K GPUs) | 23B/611B | 66 | 8/512 | ∞ | 16% → 20% | 1.69× |
| v5 (MAI-Base-1, scale up) | 35B/1T | 78 | 8/512 | ∞ | → 20% | 1.69× |

Representative fixes: v2 used GPU Direct RDMA, a custom block-sparse attention backend (FA2's deterministic mode was inefficient on GB200 and FA4 wasn't ready), ZeRO-2, and a Triton expert-encode kernel (10% → ~80% HBM bandwidth). v4's smaller LatentMoE GEMMs made CPU launch overhead dominant (fixed with FA4 deterministic kernels + batching). v5 hit a ZeRO-3 all-gather bottleneck in the backward pass, solved by activation offloading so it could revert to ZeRO-2.

## Fault Tolerance and Checkpointing

- **Distributed checkpoint (DCP)**, rewritten for ~10× faster saves and lower CPU/GC overhead, with pre-computed/cached save plans off the critical path. Asynchronous: copy tensors device→host in the training process, then hand off to a separate checkpointing process while training proceeds; at most one checkpoint in flight.
- Local Azure Blob storage; replicated state is loaded once and broadcast via NCCL to avoid single-blob fan-in hotspots.
- **Rapid recovery** via in-job restarts using Ray actors with hot standbys (no pod recreation). A nice operational detail: they explicitly **clear OS caches on GB200 actor restarts** to avoid OOM crash-loops from residual GPU memory, and validate determinism after restart against historical loss.

## RL Infrastructure (Rocket)

Async distributed RL, built because open-source frameworks didn't scale to async RL across thousands of GPUs. **YOLO is the learner, SGLang the inference engine**, with a controller + problem-workers + rollout-workers + router/inference topology; off-policy for large runs, on-policy reserved for debugging.

![[assets/mai-thinking-1/rocket-framework.png]]
*Rocket separates task sampling, async control, rollout generation, inference serving, learner pools, checkpointing, and weight transfer.*

- **Inference dominates.** The inference:learner GPU ratio reaches **5:1**; the largest job ran 4096 inference / 768 learner chips on GB300. Single-turn workloads are KV-cache-bound, so they *disable* prefix caching (to let sliding-window tokens evict during 128k generations) and lean on expert + data parallelism, DeepEP, and EPLB. Multi-turn workloads are prefill-heavy, so they lean *hard* on prefix caching (**97–98% hit rate**). Same loop, opposite caching strategy.
- **The numerics gap is the named stability risk.** Different kernels/scheduling/parallelism between YOLO and SGLang cause small per-token logprob discrepancies that compound over long rollouts and destabilize the importance-sampling correction. The mitigation is almost anticlimactic: **BF16 on both sides** (smaller gap than lower-precision alternatives), plus MoE routing replay and top-p mask replay. Matching the two engines numerically matters more than squeezing precision.
- **Weight transfer.** In async RL, fresh learner weights must reach the inference fleet every $k$ steps, and the two sides shard the same tensors differently. They compile a **transfer plan once** (intersect the two sharding layouts; emit per-sub-shard entries with required dtype casts / layout permutations), targeting an idealized 1-learner/1-inference topology that expands at runtime to all live replicas (so it survives replica churn). Because DP replicates rather than shards, they enlist only a subset of DP groups and run transfers in parallel — a 36-server fleet does four 9-server transfers at once, improving throughput and containing failure blast radius.

## Goodput and MFU Metrics

Two production KPIs, tied to [[FLOPs and MFU]]:

$$\text{MFU} = \frac{\text{FLOP}}{t_{\text{step}}\cdot \text{FLOP}_{\text{spec}}}, \qquad \text{goodput} = \frac{\text{ideal training duration}}{\text{actual wall-clock duration}}$$

MFU is normalized against the GB200 BF16 dense throughput ($\text{FLOP}_{\text{spec}} = 2.5\times10^{15}$ FLOP/s per GPU). **Goodput** decomposes the gap into named overhead categories, each with an owner, detection signal, and quantified FLOP cost — so a failure's true cost (restart + recomputation + startup + placement perturbation + post-recovery MFU drop) is explicit, and an MFU drop is treated as a production incident even while the job keeps running.

The final run hit **90.0% goodput at 8K GPUs** with 51h total overhead. The instructive part is *where the remaining overhead lives*: recomputation dropped to 15% and non-stepping time to 27%, so **MFU-drop became the single largest overhead (35%)** — driven by checkpointing, network degradation, memory pressure, and hardware-health transitions. The reliability problems got solved and exposed an efficiency problem underneath: the classic systems progression.

## Inference and MAIA-200

Inference efficiency is treated as a first-class objective across model, serving engine, and hardware. Deploying MAI-Thinking-1 on Microsoft's **MAIA-200** accelerator delivers **>40% higher token-generation throughput per rack-power budget** than a GB200 deployment — the perf-per-watt number that governs serving economics at scale.

## Related

- [[MAI-Thinking-1 Pretraining]] — the model and data this infra trains
- [[MAI-Thinking-1 RL Climb]] — the async RL recipe that depends on the numerics gap and weight transfer here
- [[FLOPs and MFU]] — the $6ND$ budget, MFU definition, and why MFU < 100% in practice
- [[FP8]] — the low-precision formats used in the numerics recipe

## References

1. The Microsoft AI Team, *MAI-Thinking-1: Building a Hill-Climbing Machine* (2026) — https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf
2. Rajbhandari, S. et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — https://arxiv.org/abs/1910.02054
3. Zheng, L. et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)
