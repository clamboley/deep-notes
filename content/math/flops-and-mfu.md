---
title: FLOPs and MFU
tags: [compute, training, hardware, fundamentals]
date: 2026-06-01
draft: false
aliases: [FLOPs, MFU, Model FLOPs Utilization, floating point operations]
---

> [!abstract] TL;DR
> A FLOP is one floating-point operation. For transformer training: count 2 FLOPs per multiply-accumulate in each matmul, multiply by 3 for forward+backward, sum over all linear layers and the attention mechanism. Under the approximation that sequence length $S \ll d_{model}$, this collapses to $C \approx 6ND$ — the formula used in [[Kaplan 2020]] and [[Chinchilla]]. MFU measures what fraction of peak hardware throughput you actually use.

## What Is a FLOP

A **FLOP** (floating-point operation) is one arithmetic operation — a multiply or an add — on a floating-point scalar. In practice, hardware executes fused multiply-adds (FMA): one multiply followed by one accumulate, counted as **2 FLOPs**. All FLOP counts below use this convention.

This means: for a dot product of two vectors of size $n$, you perform $n$ FMAs = $2n$ FLOPs.

## FLOPs for nn.Linear(I, O)

A linear layer computes $\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$ where $\mathbf{x} \in \mathbb{R}^{T \times I}$, $\mathbf{W} \in \mathbb{R}^{O \times I}$, $\mathbf{b} \in \mathbb{R}^O$, $\mathbf{y} \in \mathbb{R}^{T \times O}$ and $T$ is the number of tokens.

### Forward pass

Each output element $y_{t,o} = \sum_{i=1}^{I} x_{t,i}\, W_{o,i} + b_o$ requires:
- $I$ multiplications
- $I - 1$ additions (the sum) + 1 addition (the bias) $\approx I$ additions

So **2I FLOPs per output element**. With $T \times O$ output elements:

$$C_{\text{fwd}} = 2 \cdot T \cdot I \cdot O$$

### Backward pass

Given $\partial L / \partial \mathbf{y}$ of shape $(T, O)$, three gradients are needed:

**Gradient w.r.t. input** $\partial L/\partial \mathbf{x} = (\partial L/\partial \mathbf{y})\, \mathbf{W}$:

$$\frac{\partial L}{\partial x_{t,i}} = \sum_{o=1}^{O} \frac{\partial L}{\partial y_{t,o}} W_{o,i}$$

This is another $(T, O) \times (O, I)$ matmul — identical cost to the forward:

$$C_{\partial x} = 2 \cdot T \cdot O \cdot I$$

**Gradient w.r.t. weights** $\partial L/\partial \mathbf{W} = (\partial L/\partial \mathbf{y})^\top \mathbf{x}$:

$$\frac{\partial L}{\partial W_{o,i}} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_{t,o}}\, x_{t,i}$$

Again a matmul, $(O, T) \times (T, I)$, same cost:

$$C_{\partial W} = 2 \cdot T \cdot O \cdot I$$

**Gradient w.r.t. bias** $\partial L/\partial \mathbf{b} = \sum_t \partial L/\partial \mathbf{y}_{t,\cdot}$: just $T$ additions per output dim, $\approx TO$ FLOPs — negligible.

### Summary

$$C_{\text{bwd}} \approx 4\,T\,I\,O \qquad C_{\text{total}} = C_{\text{fwd}} + C_{\text{bwd}} = 6\,T\,I\,O$$

**The backward is exactly 2× the forward.** Forward + backward = 3× forward = $6TIO$.

Per token ($T = 1$): a Linear(I, O) costs **$2IO$ FLOPs forward, $6IO$ FLOPs for training**.

## FLOPs for the Attention Mechanism

For one multi-head attention layer: $d = d_\text{model}$, $h$ heads, $d_k = d_v = d/h$, sequence length $S$.

### Linear projections

Four projections, each Linear$(d, d)$: Q, K, V (inputs), and the output projection O.

$$C_\text{proj} = 4 \times 2d^2 = 8d^2 \quad \text{FLOPs per token (forward)}$$

### QK attention scores

$$\text{scores} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{S \times S}$$

For a single head: $\mathbf{Q} \in \mathbb{R}^{S \times d_k}$, $\mathbf{K} \in \mathbb{R}^{S \times d_k}$. The matmul costs $2 \cdot S \cdot S \cdot d_k$ FLOPs. Across $h$ heads, with $h \cdot d_k = d$:

$$C_{QK} = h \cdot 2S^2 d_k = 2S^2 d \quad \Longrightarrow \quad 2Sd \;\text{ FLOPs per token}$$

### Attention-weighted values

$$\text{out} = \text{softmax}(\text{scores})\,\mathbf{V}$$

Same shape as the QK matmul — identical cost:

$$C_{AV} = 2S^2 d \quad \Longrightarrow \quad 2Sd \;\text{ FLOPs per token}$$

Softmax is elementwise and negligible in practice.

### Total attention FLOPs per token (forward)

$$C_\text{attn} = \underbrace{8d^2}_{\text{projections}} + \underbrace{4Sd}_{\text{QK}^\top + AV}$$

## FLOPs for a Full Transformer Layer

A standard decoder layer: multi-head attention + MLP with 4× expansion (Linear$(d, 4d)$ → activation → Linear$(4d, d)$). LayerNorm and residuals are elementwise and negligible.

**MLP FLOPs per token (forward):**

$$C_\text{MLP} = 2 \cdot d \cdot 4d + 2 \cdot 4d \cdot d = 16d^2$$

**Total per layer per token (forward):**

$$C_\text{layer} = 8d^2 + 4Sd + 16d^2 = 24d^2 + 4Sd$$

## The 6ND Formula

### Parameter count

Non-embedding parameters per layer (ignoring biases and layer norms):
- Q, K, V, O projections: $4d^2$
- MLP up + down: $4d^2 + 4d^2 = 8d^2$
- **Total: $12d^2$ per layer**

For $L$ layers: $N \approx 12Ld^2$.

### Derivation

Total forward FLOPs for $D$ tokens over $L$ layers:

$$C_\text{fwd} = D \cdot L \cdot (24d^2 + 4Sd) = D\,(2 \cdot \underbrace{12Ld^2}_{N} + 4SLd) = 2ND + 4SLdD$$

Forward + backward (factor 3):

$$\boxed{C \approx 6ND + 12DSLd}$$

**Under the approximation $S \ll d$**, the attention term vanishes:

$$C \approx 6ND$$

The factor **6** = 2 (forward) + 2 (grad_input) + 2 (grad_weight), one matmul each.

### Hypotheses made

1. **Short context: $S \ll d$.** The attention quadratic term $4Sd$ per layer is dropped. More precisely, the approximation holds when $4Sd \ll 24d^2$, i.e. $S \ll 6d$.
2. **Non-embedding parameters only.** The embedding table ($V \times d$) and unembedding layer (also $V \times d$ at the output) are excluded from $N$. They can be significant for large vocabularies.
3. **Standard MHA.** Full Q, K, V projections each of size $d \times d$. GQA/MQA reduce the K and V projections.
4. **No activation recomputation.** Gradient checkpointing reruns the forward pass during the backward, adding $\approx 1\times$ forward cost, making the real cost closer to $8ND$.
5. **Biases and layer norms negligible.** True in practice — they're $O(d)$ vs $O(d^2)$ for the matmuls.

### Where it fails

**Long contexts.** The full formula is $C \approx 6D(N + 2LSd)$. The attention correction is $\frac{2LSd}{N} = \frac{2LSd}{12Ld^2} = \frac{S}{6d}$. Concretely:

| $d_\text{model}$ | Context length where correction ≥ 10% |
|---|---|
| 4096 (≈7B) | $S \gtrsim 2{,}500$ |
| 8192 (≈70B) | $S \gtrsim 5{,}000$ |
| 12288 (≈175B) | $S \gtrsim 7{,}000$ |

Modern long-context models (32K–128K tokens) significantly exceed these thresholds. 6ND underestimates their training compute.

**GQA/MQA.** Grouped-query attention uses $n_{kv}$ KV heads instead of $h$. The K and V projections shrink from $d^2$ to $d \cdot (d/h) \cdot n_{kv}$ each, reducing the per-layer parameter count and linear-projection FLOPs. The 12 in $N \approx 12Ld^2$ becomes smaller.

**Unembedding layer.** The final LM head is Linear($d$, $V$) with $V \approx 32{,}000$–$128{,}000$. At $2dV$ FLOPs/token forward, for $V = 128{,}000$ and $d = 4096$ this is $10^9$ FLOPs/token — comparable to a full transformer layer in a small model. Often excluded from $N$ but never from the actual compute.

**Inference.** No backward pass — just $2N$ FLOPs per token during prefill. Autoregressive decoding generates one token per step using a KV cache; the $2N$ formula still applies per generated token, but the effective compute per token is now dominated by memory bandwidth, not FLOPs (the batch size is typically 1).

**Activation recomputation.** If you recompute activations to save memory (gradient checkpointing), the forward is run twice, giving $C \approx 8ND$ instead of $6ND$. Some frameworks checkpoint selectively (e.g., only attention activations), landing somewhere in between.

## MFU — Model FLOPs Utilization

MFU measures what fraction of the hardware's peak throughput is actually being used for model computation:

$$\text{MFU} = \frac{\text{actual FLOPs/s}}{\text{hardware peak FLOPs/s}} = \frac{\text{tokens/s} \times 6N}{\text{peak FLOPS}}$$

where "peak FLOPS" is the accelerator's rated BF16/FP16 tensor-core throughput (the relevant dtype for training).

**Why MFU < 100% in practice:**
- Memory bandwidth bottlenecks (weight loads, KV cache I/O)
- All-reduce and other collective communication in distributed training (TP, DP)
- Pipeline bubbles in pipeline parallelism
- Non-matmul operations (layer norm, softmax, elementwise activations) — fast but not tensor-core-bound
- Kernel launch overhead, scheduler inefficiency

**Reference values (H200, BF16):**

| MFU | Assessment |
|---|---|
| > 30% | Good |
| ≈ 40% | Excellent |
| > 50% | Exceptional — only at very large batch sizes on dense matmuls |

MFU is the right metric to report alongside training throughput — tokens/s alone is meaningless without the model size and hardware context.

## Related

- [[Kaplan 2020]] — uses $C \approx 6ND$ as the compute budget formula throughout
- [[Chinchilla]] — same formula; optimal allocation derived by minimizing $L(N, D)$ subject to $C = 6ND$

## References

1. Kaplan, J. et al., *Scaling Laws for Neural Language Models*, OpenAI (2020) — https://arxiv.org/abs/2001.08361  
2. Hoffmann, J. et al., *Training Compute-Optimal Large Language Models*, DeepMind (2022) — https://arxiv.org/abs/2203.15556  
3. Korthikanti, V. et al., *Reducing Activation Recomputation in Large Transformer Models*, NVIDIA (2022) — https://arxiv.org/abs/2205.05198
