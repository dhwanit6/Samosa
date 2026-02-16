# EcoHybrid Diffusion: Theory and Rationale

## Objective

Given clean sequence `x0`, timestep `t`, and corruption operator `q(xt | x0, t)`:

1. Corrupt tokens by masking with rate `r(t)`.
2. Predict original tokens from `xt`.
3. Train with masked-token cross-entropy:

`L = E_{x0, t, m} [ sum_i m_i * CE(p_theta(x0_i | xt, t), x0_i) / sum_i m_i ]`

where `m_i` indicates masked positions.

## Why this architecture

Standard diffusion denoisers often use full self-attention (`O(T^2)`), which is expensive on local hardware for long sequences.

EcoHybrid replaces this with:

1. **Local mixer** (depthwise temporal conv): `O(T * k * d)`
2. **Token -> memory attention** with `M` memory slots: `O(T * M * d)`
3. **Memory -> token attention** with `M` slots: `O(T * M * d)`

Total per-layer complexity:

`O(T * k * d + 2 * T * M * d + T * d_ff * d)`

For `M << T`, global context cost is near-linear in `T`.

## Nature-inspired mapping

- Local conv branch: local sensory processing (nearby context).
- Memory slots: compact working memory / hippocampal summary.
- Iterative denoising: progressive refinement, not one-shot prediction.

The architecture tries to mimic multi-scale cognition:
- fast local pattern processing,
- small global memory bottleneck,
- iterative correction.

## Why this is not novelty for novelty

This design is a direct response to deployment constraints:

1. Lower memory movement and compute than full bidirectional attention.
2. Preserves global coherence through memory slots.
3. Compatible with diffusion decoding and confidence-based early reveal.

## What to measure first

1. Tokens/sec vs baseline diffusion transformer at same parameter count.
2. Quality on Hindi/Hinglish prompt set (human preference + error buckets).
3. Latency vs sequence length scaling.
4. Effect of memory slots (`M=8, 16, 32`) on quality/speed.
