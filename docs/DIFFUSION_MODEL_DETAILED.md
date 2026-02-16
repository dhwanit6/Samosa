# Diffusion Model Detailed Design (Aria)

## 1. Goal

Build a diffusion language model that is:

1. Efficient on local hardware.
2. Robust on long-form generation.
3. Compatible with Hindi, Hinglish, and mixed-script text.
4. Grounded in clear math, not novelty-only architecture choices.

This document describes what is implemented in this repo, why it is implemented, and which higher-impact theories are intentionally left for later experiments.

## 2. What Is Implemented

There are two diffusion denoiser paths:

1. `train/diffusion/model.py`: baseline masked diffusion Transformer denoiser.
2. `train/diffusion/eco_hybrid.py`: non-standard EcoHybrid denoiser (local conv + memory slots).

Training entry points:

1. `python -m diffusion.train_diffusion`
2. `python -m diffusion.train_eco_hybrid`

Sampling entry point:

1. `python -m diffusion.sample_diffusion`

## 3. Core Diffusion Objective

Let `x0` be clean token ids, `t` diffusion timestep, `xt` corrupted sequence.

Corruption process:

1. Sample timestep `t ~ Uniform(0, T-1)`.
2. Compute mask rate `r(t)` using a cosine schedule.
3. Mask random positions with probability `r(t)` using a dedicated mask token id.

Training objective:

`L = E_{x0, t, m}[ sum_i m_i * CE(p_theta(x0_i | xt, t), x0_i) / sum_i m_i ]`

where:

1. `m_i = 1` if token `i` is masked at timestep `t`.
2. CE is token-level cross entropy.
3. Loss is only applied on masked positions.

Reason:

1. This focuses the model on denoising and avoids wasting loss on already visible tokens.
2. Works well with iterative unmasking at inference.

## 4. Inference Procedure

Both diffusion models use confidence-based iterative reveal:

1. Initialize generation region as all `[MASK]`.
2. For each reverse step:
1. Predict token distributions.
2. Choose candidate tokens (sampled or greedy).
3. Reveal only the most confident subset.
3. Final step reveals all remaining masked tokens.
4. Fallback pass fills any unresolved masks.

This is similar in spirit to MaskGIT-style decoding, with conservative reveal schedules for stability.

## 5. Baseline Denoiser (`model.py`)

### 5.1 Architecture

1. Token embedding + positional embedding.
2. Timestep embedding added to sequence states.
3. Stacked bidirectional Transformer blocks.
4. Tied output head (same matrix as token embedding).

### 5.2 Why keep this baseline

1. It is a stable reference.
2. It provides a control arm for ablation.
3. Any "new" architecture should beat this on speed-quality tradeoff.

### 5.3 Cost profile

Per layer attention is `O(T^2 * d)` for sequence length `T`, hidden size `d`.

This is the main bottleneck for local long-sequence diffusion inference.

## 6. EcoHybrid Denoiser (`eco_hybrid.py`)

### 6.1 Architecture summary

Each block does:

1. Local temporal mixing via depthwise conv.
2. Token -> memory cross-attention (tokens query compact memory slots).
3. Memory -> token cross-attention (memory updates from token stream).
4. Token FFN update.

### 6.2 Why this is non-standard but justified

It explicitly splits the problem:

1. Local detail: handled by depthwise temporal conv.
2. Global coherence: handled by small learned memory slots.

This replaces global token-token attention with token-memory interactions.

### 6.3 Complexity intuition

Let:

1. `T` = sequence length
2. `d` = hidden dim
3. `k` = conv kernel size
4. `M` = number of memory slots, where `M << T`

Per block dominant terms:

1. Local conv: `O(T * k * d)`
2. Token-memory attention: `O(T * M * d)`
3. Memory-token attention: `O(T * M * d)`
4. FFN: `O(T * d * d_ff)`

Compared to full attention `O(T^2 * d)`, global mixing becomes near-linear in `T` when `M` is small.

### 6.4 Nature-inspired mapping (engineering version)

1. Local conv branch: local sensory integration.
2. Memory slots: compact working memory.
3. Iterative denoising: repeated correction loop.

This is not biology mimicry for style; it is a compute budget strategy:

1. Keep global channel narrow (`M` slots).
2. Keep local processing cheap and parallel.

## 7. Training Config and Practical Start

Recommended first real run:

```bash
cd train
python -m diffusion.train_eco_hybrid \
  --data_dir data/processed \
  --output_dir checkpoints/eco_hybrid_hi_v1 \
  --seq_len 512 \
  --batch_size 4 \
  --grad_accum 8 \
  --d_model 512 \
  --n_layers 10 \
  --n_heads 8 \
  --d_ff 1536 \
  --memory_slots 16 \
  --conv_kernel 7 \
  --timesteps 32 \
  --sample_steps 12 \
  --max_steps 5000 \
  --compile
```

Low-memory fallback:

1. `d_model=384`
2. `n_layers=8`
3. `memory_slots=8`
4. `seq_len=384`

## 8. What This Model Does Well Right Now

Expected strengths:

1. Better scaling than full-attention diffusion at longer `T`.
2. Lower memory pressure with small `M`.
3. Structured path to high-throughput iterative generation.

Expected weaknesses:

1. May miss fine-grained long-range interactions compared to full bidirectional attention.
2. Confidence-based reveal can create local inconsistencies if confidence is miscalibrated.
3. Still a research path versus mature autoregressive serving.

## 9. Experimental Theories Not Implemented Yet

These are high-value ideas that are mathematically plausible but intentionally not in current code.

### 9.1 Adaptive per-token step budget

Idea:

1. Use uncertainty to allocate more denoising steps only to hard positions.
2. Easy positions stop early.

Potential gain:

1. Lower average decoding steps.
2. Faster wall-of-text generation.

Risk:

1. Error propagation if early-stopped tokens are wrong.

### 9.2 Learnable memory routing (token -> subset of slots)

Idea:

1. Instead of attending all memory slots, route tokens to top-k memory slots.
2. Slot competition regularized by load-balancing loss.

Potential gain:

1. Lower memory-attention compute.
2. Better specialization of slots.

Risk:

1. Slot collapse, unstable routing.

### 9.3 Multi-timescale memory banks

Idea:

1. Short-term slots updated every layer.
2. Slow slots updated every N layers with lower learning rate.

Potential gain:

1. Better long-context coherence.
2. Reduced drift in global semantic state.

Risk:

1. More tuning complexity.

### 9.4 AR-assisted diffusion verifier (hybrid decode)

Idea:

1. Diffusion drafts a large chunk quickly.
2. Small AR verifier validates or repairs low-confidence spans.

Potential gain:

1. Better reliability with high throughput.

Risk:

1. Two-model orchestration overhead.

### 9.5 Flow-step distillation for fewer denoise steps

Idea:

1. Distill many-step denoiser into fewer-step model.
2. Keep quality near teacher while reducing decoding depth.

Potential gain:

1. Major generation speedup.

Risk:

1. Distillation collapse on multilingual/code-mixed edge cases.

### 9.6 Retrieval-conditioned memory slots

Idea:

1. Inject retrieved factual embeddings directly into memory slots.
2. Denoiser conditions on both prompt and retrieved evidence.

Potential gain:

1. Better factuality without huge param increases.

Risk:

1. Retrieval noise contaminates generation.

### 9.7 Contrastive confidence calibration

Idea:

1. Train confidence head against correctness labels from held-out trajectories.
2. Use calibrated confidence for reveal order and early stopping.

Potential gain:

1. More reliable iterative reveal policy.

Risk:

1. Extra training cost and pipeline complexity.

### 9.8 Script-aware corruption policy

Idea:

1. Corrupt Devanagari, Roman Hindi, and English spans with different schedules.
2. Preserve fragile grapheme clusters and key markers.

Potential gain:

1. Better code-mix robustness.
2. Better transliteration consistency.

Risk:

1. Requires robust script/span detection in preprocessing.

## 10. Near-Term Experiment Plan (Recommended)

Order by expected ROI:

1. Baseline comparison:
1. `model.py` vs `eco_hybrid.py` at matched params and tokens.
2. Memory slot sweep:
1. `M in {8, 16, 32}`.
3. Decode-step sweep:
1. `sample_steps in {8, 12, 16}`.
4. Confidence threshold sweep:
1. `confidence_stop in {0.95, 0.98, 0.99}`.
5. Script-aware eval:
1. Hindi native script, Hinglish Roman, mixed-script prompts.

Track all runs with:

1. tokens/sec
2. wall-clock latency
3. loss/perplexity proxy
4. human preference on a fixed Indic eval set
5. script-consistency error rate

## 11. Decision Rules

Keep EcoHybrid as primary diffusion path only if:

1. It beats baseline diffusion on throughput by meaningful margin.
2. It does not regress badly on Hindi/Hinglish quality.
3. It remains stable under low-hardware constraints.

Otherwise:

1. Revert diffusion to baseline denoiser for stability.
2. Keep EcoHybrid as R&D branch for targeted improvements.

## 12. Relationship to Main Product Path

Current recommendation remains:

1. AR/recurrent model is the main product path.
2. Diffusion is parallel R&D for high-throughput long text generation.
3. Hybrid serving (diffusion draft + AR verifier) is the most practical medium-term target.
