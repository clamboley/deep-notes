---
title: Low-Precision Number Formats (FP8, MXFP8, NVFP4)
tags: [quantization, hardware, inference, training, formats]
date: 2026-06-10
draft: false
aliases: [FP8, MXFP8, NVFP4, E4M3, E5M2, E2M1, microscaling, low-precision formats]
---

> [!abstract] TL;DR
> BF16 → FP8 → MXFP8 → NVFP4 is a coherent progression: shrink the format, shrink the scaling granularity to compensate. FP8 uses one FP32 scale per tensor; MXFP8 drops to a power-of-two scale per 32 elements; NVFP4 uses 4 bits (E2M1) with a fractional FP8 scale per 16 elements plus a per-tensor FP32 global factor. All three are natively accelerated on Blackwell tensor cores. 1.58-bit (BitNet b1.58) sits orthogonally — training-time ternary quantization rather than a post-training format conversion.

## Floating-Point Format Basics

A floating-point scalar is stored as **sign + exponent + mantissa**, written ExMy (x exponent bits, y mantissa bits). The exponent controls **dynamic range** — the span from smallest to largest representable magnitude. The mantissa controls **precision** — how finely values are resolved within each power-of-two interval. Cutting bits forces a tradeoff: fewer exponent bits means overflow and underflow risk; fewer mantissa bits means coarser quantization steps.

![[fp8-format-comparison.png]]
*FP16 (E5M10), BF16 (E8M7), FP8 E4M3, and FP8 E5M2 side by side. BF16 matches FP32's 8-bit exponent at the cost of mantissa precision; E4M3 preserves more mantissa than E5M2 but has narrower dynamic range.*

## BF16 — the Training Baseline

BF16 is **E8M7**: 1 sign + 8 exponent + 7 mantissa bits. The key design choice is keeping the full FP32 exponent (range ~1e-38 to 1e38) while discarding 16 of FP32's 23 mantissa bits. FP16 (E5M10) has more precision but a 5-bit exponent whose range (±65,504) is too narrow for gradients — overflow and underflow during training are why FP16 fell out of favor. BF16 remains the standard training dtype.

## FP8 — Two Formats for Two Jobs

The FP8 spec (OCP, 2022) defines two variants used simultaneously within the same model:

- **E4M3** (range ~±448): higher mantissa precision, used for forward pass weights and activations.
- **E5M2** (range ~±57,344): wider dynamic range, used for gradients in the backward pass where magnitudes vary more.

Neither format can represent its tensor's full distribution in raw form, so each tensor gets a **per-tensor FP32 scaling factor**: a single scalar that remaps the global maximum absolute value into FP8 range before quantization. This is the format's central weakness — the scale must accommodate the tensor's peak value, wasting precision wherever values fall well below it.

In practice, **delayed scaling** is the standard recipe: track the per-tensor maximum absolute value over prior iterations and use those historical statistics to set the current scale, avoiding the need to scan the current tensor before quantizing it.

## MXFP8 — Microscaling

MXFP8 is part of the OCP Microscaling (MX) standard (2023, backed by Microsoft, Intel, NVIDIA, AMD). The central change: replace the single per-tensor scale with a distinct scaling factor for each **block of 32 consecutive values**.

Two consequences follow directly. Because each block adapts to its own local magnitude, the higher-precision **E4M3** format can be used throughout — no need to fall back to E5M2 even for gradients. Block scales are stored as **E8M0**, a pure 8-bit exponent with no mantissa, representing only powers of two. This is deliberate: power-of-two scales reduce the scale multiply to a bit shift, and hardware can ingest it natively without dequantizing first.

MXFP8 is accelerated on Blackwell (SM 10.0+), where tensor cores handle the 32-element block and its E8M0 scale directly. Realizing the theoretical ~2× throughput gain over BF16 requires careful kernel design to avoid memory-bandwidth bottlenecks.

![[fp8-vs-mxfp8-scaling-factors.png]]
*FP8 assigns a single FP32 scale to the entire tensor (left); MXFP8 assigns a distinct E8M0 scale to each block of 32 consecutive values (right). The finer granularity lets each block adapt to its local magnitude rather than being dominated by the tensor's global maximum.*

## NVFP4 — 4-Bit Floating Point

NVFP4 uses **E2M1** (1 sign + 2 exponent + 1 mantissa bit), representing roughly 8 distinct magnitudes spanning ±6: {0, 0.5, 1, 1.5, 2, 3, 4, 6} plus signs. That is brutally few representable values, so NVFP4 compensates with a **two-level scaling scheme**:

1. **Per-block FP8 scale (E4M3):** each block of 16 contiguous values gets its own E4M3 scaling factor. The reconstruction is $x = x_q \times s_\text{block} \times s_\text{tensor}$.
2. **Per-tensor FP32 scale:** a single FP32 scalar maps the overall tensor distribution before the per-block scales handle the residual local variation.

**Why E4M3 block scales rather than E8M0?** E4M3 supports fractional (non-power-of-two) scale values, so each block scale can be chosen to minimize quantization error jointly across its 16 values. NVIDIA's analysis shows E4M3 block scales achieve MSE = 0.08 vs. MSE = 0.72 for E8M0 scales on the same data — nearly an order-of-magnitude improvement.

![[e4m3-vs-e8m0-quantization.gif]]
*The same input values quantized with E8M0 (power-of-two scale, coarse) vs. E4M3 (fractional scale, finer match). E8M0's constrained scale choices force larger rounding errors; E4M3 finds the scale that minimizes block-level MSE.*

![[mxfp4-vs-nvfp4.gif]]
*MXFP4 (top): 32-value blocks with E8M0 power-of-two scales. NVFP4 (bottom): 16-value blocks with E4M3 fractional scales plus a global FP32 factor. Finer blocks and fractional scales together give substantially lower quantization error.*

### NVFP4 vs. MXFP4

|  | MXFP4 | NVFP4 |
|--|-------|-------|
| Block size | 32 values | 16 values |
| Block scale | E8M0 (power-of-two) | E4M3 (fractional) |
| Global scale | — | FP32 per tensor |
| Effective bits/value | ~4 | ~4.5 |

The 0.5-bit overhead comes from the E4M3 block scale: one 8-bit scalar per 16 values adds 0.5 bits/value. Net result: NVFP4 is 3.5× smaller than FP16 and 1.8× smaller than FP8. The 16-element blocks give twice as many opportunities to match local dynamic range as MXFP4's 32-element blocks.

### Accuracy

On DeepSeek-R1-0528 across MMLU-PRO, GPQA Diamond, LIVECODEBENCH, SCICODE, Math-500, and AIME 2024, NVFP4 shows ≤1% degradation from FP8. The advantage over INT4 at the same bit width: transformer weight and activation distributions span many orders of magnitude, and the floating-point per-block exponent preserves that structure better than INT4's fixed linear range.

![[nvfp4-scaling.gif]]
*NVFP4 two-level scaling: 4-bit E2M1 values at the innermost level, E4M3 FP8 block scales applied per 16 values, and a global FP32 tensor scale.*

### Hardware and Tooling

Blackwell fifth-generation tensor cores handle NVFP4 natively — grouping, scaling, and 4-bit matmul are all done in hardware. Post-training quantization to NVFP4 is supported via TensorRT Model Optimizer and LLM Compressor; pre-quantized checkpoints (DeepSeek-R1-0528, Llama 3.1-405B-Instruct) are available under the `nvidia/` namespace on Hugging Face.

## 1.58-bit — A Different Category

BitNet b1.58 constrains every weight to ternary {-1, 0, 1} (log₂3 ≈ 1.58 bits/param), matching BF16 perplexity at the same model size and token count. This is not post-training quantization — it requires quantization-aware training from scratch. With weights in {-1, 0, 1}, matrix multiplications collapse into additions and subtractions, enabling much deeper hardware efficiency gains than FP8/FP4 without changing the storage format. See [[BitNet b1.58]] for a full treatment.

## The Throughline

BF16 → FP8 (per-tensor FP32 scale) → MXFP8 (per-32-block E8M0 scale) → NVFP4 (per-16-block E4M3 scale + per-tensor FP32) is one continuous lineage: each step shrinks the number format, then compensates by making the scaling granularity finer and the scale format more expressive. BitNet sits orthogonally — it eliminates the mantissa and exponent entirely, but accepts training-time overhead to do so.

## Related

- [[BitNet b1.58]] — ternary weight quantization requiring training from scratch
- [[FLOPs and MFU]] — hardware throughput context for why lower-precision formats matter

## References

1. NVIDIA Developer Blog, *Floating Point 8: An Introduction to Efficient Lower-Precision AI Training* — https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/
2. NVIDIA Developer Blog, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference* — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
3. Rouhani, B. et al., *Microscaling Data Formats for Deep Learning*, Microsoft (2023) — https://arxiv.org/abs/2310.10537
4. Ma, S. et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*, Microsoft Research (2024) — https://arxiv.org/abs/2402.17764
