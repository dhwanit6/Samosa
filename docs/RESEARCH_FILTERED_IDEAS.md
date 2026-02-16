# Research Filter (Papers + Community)

## Selection rule

Keep only ideas that are:
1. compatible with current architecture,
2. measurable with local experiments,
3. likely to improve edge latency/throughput without large quality collapse.

## High-signal ideas kept

1. Blockwise diffusion decoding with cached context representation.
   - Why kept: directly targets repeated full-sequence compute.
   - Status: implemented (`frozen_context` + incremental context-memory update).

2. Progressive depth denoising (fewer layers early, full layers late).
   - Why kept: aligns compute with denoising difficulty by step.
   - Status: implemented (`min_decode_layers`).

3. Edge benchmarking with fixed decode modes and a quality proxy.
   - Why kept: prevents subjective tuning; reproducible and comparable.
   - Status: implemented (`edge_benchmark.py`).

4. Quantization for CPU deployment.
   - Why kept: direct practical speed/memory benefit on edge CPUs.
   - Status: integrated in sampler/benchmark (`--quantize_dynamic`).

## Ideas deferred (not yet implemented)

1. AR-style speculative decoding for diffusion.
   - Reason: not directly drop-in for masked iterative denoisers; needs a draft-verifier design specific to diffusion.

2. KV-cache-centric optimizations.
   - Reason: diffusion denoising does not naturally use causal KV cache; requires architecture redesign.

3. Theorem-proving-specialized tiny models.
   - Reason: requires domain dataset + verifier loop; outside immediate infra refactor.

## Community signal handling

Reddit/forum discussions were used as hypothesis generation only.
Final implementation decisions were gated by paper-level evidence and codebase fit.
