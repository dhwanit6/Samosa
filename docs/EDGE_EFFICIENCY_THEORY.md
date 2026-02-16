# Edge Efficiency Theory (Primary EcoHybrid)

## Objective

Maximize edge-time utility:

\[
\text{Utility} = \frac{\text{Quality}}{\text{Latency} \times \text{Memory} \times \text{Energy}}
\]

In practice we optimize latency first while constraining quality drop.

## Why diffusion feels expensive

For masked diffusion decoding with \(S\) denoising steps and target length \(T\):

\[
\text{Cost}_{\text{full}} \propto S \cdot T
\]

per decode pass through the denoiser backbone.  
If generation is blockwise with block size \(B\), prompt length \(P\), and \(K=\lceil N/B \rceil\) blocks:

\[
\text{Cost}_{\text{block-full-context}} \propto S \sum_{k=1}^{K} (P + kB)
\]

which grows superlinearly in generated tokens \(N\).

## New integrated idea: Frozen Context + Incremental Memory Update

EcoHybrid already uses memory slots (\(M \ll T\)).  
We treat memory slots as a compact recurrent state:

1. Encode prompt once into memory state \(m_0\).
2. Decode each new block against \(m_k\) (no full prompt replay).
3. Ingest finalized block tokens to update memory:
   \[
   m_{k+1} = U(m_k, y_k)
   \]

Approximate cost:

\[
\text{Cost}_{\text{frozen}} \propto K \cdot (S \cdot B \cdot M + B \cdot M)
\]

With fixed \(M\), this is near-linear in \(N\), avoiding repeated full-context passes.

## Second integrated idea: Progressive Depth Denoising

Not all reverse steps need full model depth.  
Use a depth schedule from \(L_{\min}\) to \(L\):

\[
L_s = L_{\min} + \left\lceil (L - L_{\min}) \cdot \left(1 - \frac{s}{S-1}\right) \right\rceil
\]

where \(s\) counts reverse steps from \(S-1 \rightarrow 0\).  
Early high-noise steps use fewer layers; later low-noise steps recover full capacity.

Implemented as `min_decode_layers` in generation paths.

## Quality proxy used in code

`edge_benchmark.py` reports pseudo log-likelihood for generated continuation:

\[
\text{PLL}(y) = \frac{1}{|y|} \sum_i \log p_\theta(y_i \mid y, t=0)
\]

This is a **ranking proxy** (not calibrated perplexity), useful for comparing decode modes under identical checkpoints.

## Experiments to run

1. Throughput/quality mode ablation:
   - `full`
   - `block`
   - `block_frozen`
2. Step sweep:
   - `steps in {4, 6, 8, 12}`
3. Block size sweep:
   - `block_size in {32, 64, 96}`
4. Quantization sweep (CPU):
   - fp32 vs dynamic int8
5. Progressive depth sweep:
   - `min_decode_layers in {0, 2, 4}`

Report:
- mean tok/s
- p50 latency
- mean pseudo log-likelihood

Use `edge_benchmark.py` to keep comparisons identical and reproducible.
