

# Honest Review of the Diffusion Design

I'm going to split this into three parts: what's technically sound, what has problems, and the strategic question of whether you should be doing this at all right now.

---

## Part 1: What's Genuinely Good Here

Your document shows real engineering maturity. I want to acknowledge that specifically.

```
Things most people get wrong that you got RIGHT:

✓ Decision rules with explicit kill criteria
  "Keep EcoHybrid only if it beats baseline by meaningful margin"
  → Most researchers fall in love with their architecture.
     You built in the exit ramp. That's rare.

✓ Baseline control arm
  "Any new architecture should beat this on speed-quality tradeoff"
  → model.py exists specifically to tell you if eco_hybrid.py
     is actually doing anything useful. Correct methodology.

✓ Honest weakness documentation
  "May miss fine-grained long-range interactions"
  "Confidence-based reveal can create local inconsistencies"
  → You're not selling yourself a fantasy.

✓ Theories explicitly separated from implementation
  Section 9 says "these are plausible but NOT in code"
  → This prevents architecture astronaut syndrome.

✓ The math is correct
  The complexity analysis of O(T*M*d) vs O(T²*d) is right.
  The training objective (masked CE, loss only on masked positions)
  is standard MDLM/masked diffusion. Nothing broken here.
```

---

## Part 2: Technical Critique

### 2.1 The Masking Schedule Matters More Than The Architecture

```python
# Your corruption process:
# 1. Sample t ~ Uniform(0, T-1)
# 2. Compute mask rate r(t) using cosine schedule
# 3. Mask random positions with probability r(t)

# This is fine for English. For Indic languages, it's subtly broken.
```

The problem with uniform random masking for Hindi/Gujarati/Odia:

```
Hindi word: "स्वतंत्रता" (independence)
Devanagari: This is 5 Unicode codepoints but ONE semantic unit.
Your tokenizer might encode it as 1-3 tokens.

If you mask the MIDDLE token of a multi-token Devanagari word,
the model sees:

  [स्व] [MASK] [ता]

This is like seeing "ind_____nce" in English.
The model CAN figure it out, but you're wasting capacity
on a trivially solvable sub-problem.

WORSE for Roman Hindi:
"independence" is one English word.
"swatantrata" is the Roman Hindi equivalent.
Your tokenizer might split it as: ["sw", "at", "ant", "rata"]

Masking token 2: ["sw", MASK, "ant", "rata"]
The model now has to figure out that "at" goes there,
which is a character-level puzzle, not a language modeling task.
```

**What to do instead:**

```python
# data/diffusion_corruption.py

import torch

def span_masking(token_ids, mask_rate, mask_token_id, 
                 mean_span_length=3, script_boundaries=None):
    """
    Mask contiguous SPANS, not random individual tokens.
    This forces the model to do real language modeling
    (predict meaningful chunks) instead of character puzzles.
    
    Optionally align span boundaries to script/word boundaries
    for Indic text.
    """
    B, T = token_ids.shape
    masked = token_ids.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=token_ids.device)
    
    for b in range(B):
        num_to_mask = int(T * mask_rate)
        masked_count = 0
        
        while masked_count < num_to_mask:
            # Geometric distribution for span length
            span_len = min(
                torch.distributions.Geometric(1.0 / mean_span_length).sample().int().item() + 1,
                T - masked_count
            )
            
            # Random start position
            start = torch.randint(0, T - span_len + 1, (1,)).item()
            
            # Optional: snap to word/script boundaries
            if script_boundaries is not None:
                start = snap_to_boundary(start, script_boundaries[b])
            
            mask[b, start:start + span_len] = True
            masked_count += span_len
    
    masked[mask] = mask_token_id
    return masked, mask


def script_aware_masking(token_ids, mask_rate, mask_token_id,
                         script_labels):
    """
    Your Section 9.8 idea — implement it NOW, not later.
    It's simple and high-impact for Indic languages.
    
    Different corruption rates for different scripts in the same sequence:
    - Devanagari spans: mask at rate r(t) * 0.8 (preserve more structure)
    - Roman Indic spans: mask at rate r(t) * 1.0 (standard)
    - English spans: mask at rate r(t) * 1.2 (these are easier, mask more)
    
    This forces the model to spend more capacity on harder
    (Indic) tokens and less on easier (English) tokens.
    """
    B, T = token_ids.shape
    masked = token_ids.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=token_ids.device)
    
    script_multipliers = {
        "devanagari": 0.8,
        "gujarati": 0.8,
        "odia": 0.8,
        "roman_indic": 1.0,
        "english": 1.2,
    }
    
    for b in range(B):
        for pos in range(T):
            script = script_labels[b][pos]
            effective_rate = mask_rate * script_multipliers.get(script, 1.0)
            if torch.rand(1).item() < effective_rate:
                mask[b, pos] = True
                masked[b, pos] = mask_token_id
    
    return masked, mask
```

**Move Section 9.8 to Section 6. It's not a "future theory." It's a fundamental requirement for your target languages.**

### 2.2 The Confidence-Based Reveal Has a Known Failure Mode

```
Your decode procedure:
  1. Predict all masked positions
  2. Reveal the most confident predictions
  3. Repeat until all revealed

The problem: CONFIDENCE IS NOT CORRECTNESS.

Example in Hindi diffusion:

Step 1: All masked → "_____ _____ _____ _____ _____"
Step 2: Model is 99% confident position 3 is "है" (hai)
        Model is 60% confident position 1 is "मैं" (main)
        → Reveals position 3 first: "_____ _____ है _____ _____"
Step 3: Now position 1 SHOULD be "मैं" but the model sees "है"
        at position 3 and gets confused about verb agreement.
        Predicts "वह" (vah/he) with 95% confidence instead.
        → "वह _____ है _____ _____"
Step 4: Now the sentence is grammatically committed to third person
        even though the original intent was first person.

Result: "वह बाज़ार जा रहा है" instead of "मैं बाज़ार जा रहा हूँ"
        Grammatically perfect. Semantically wrong.
```

**Fix: Reveal in linguistically meaningful chunks, not by raw confidence.**

```python
# diffusion/sampling.py

def structured_reveal(logits, current_mask, step, total_steps,
                      token_ids, tokenizer):
    """
    Instead of revealing by confidence alone, use a hybrid strategy:
    
    Phase 1 (steps 0-40%): Reveal ANCHOR tokens first
      - Nouns, verbs, named entities (high semantic content)
      - These set the "skeleton" of the sentence
      
    Phase 2 (steps 40-80%): Reveal DEPENDENT tokens
      - Adjectives, adverbs, postpositions
      - These flesh out the skeleton
      
    Phase 3 (steps 80-100%): Reveal FUNCTION tokens
      - Particles, conjunctions, auxiliary verbs
      - These handle agreement and grammar
      
    This mimics how humans construct sentences:
    idea → structure → grammar
    
    For Hindi specifically:
    Reveal order: Subject/Object → Main verb → Auxiliary → Postpositions
    This respects SOV word order and ensures agreement is correct.
    """
    B, T, V = logits.shape
    progress = step / total_steps
    
    confidences = torch.softmax(logits, dim=-1).max(dim=-1).values  # (B, T)
    candidates = torch.argmax(logits, dim=-1)  # (B, T)
    
    # How many to reveal this step
    # Cosine schedule: slow start, fast middle, slow end
    reveal_ratio = cosine_reveal_schedule(step, total_steps)
    num_reveal = int(current_mask.sum() * reveal_ratio)
    
    if progress < 0.4:
        # Phase 1: Boost confidence of content words
        content_bonus = estimate_content_word_score(candidates, tokenizer)
        adjusted_confidence = confidences + 0.3 * content_bonus
    elif progress < 0.8:
        # Phase 2: Standard confidence
        adjusted_confidence = confidences
    else:
        # Phase 3: Boost function words, penalize remaining content words
        # (if a content word hasn't been revealed by now, it's uncertain — be careful)
        function_bonus = estimate_function_word_score(candidates, tokenizer)
        adjusted_confidence = confidences + 0.2 * function_bonus
    
    # Only consider currently masked positions
    adjusted_confidence = adjusted_confidence * current_mask.float()
    
    # Reveal top-k
    _, reveal_indices = adjusted_confidence.topk(num_reveal, dim=-1)
    
    new_mask = current_mask.clone()
    new_tokens = token_ids.clone()
    for b in range(B):
        for idx in reveal_indices[b]:
            new_mask[b, idx] = False
            new_tokens[b, idx] = candidates[b, idx]
    
    return new_tokens, new_mask
```

### 2.3 The EcoHybrid Memory Architecture — What Worries Me

```
Your architecture per block:
  1. Depthwise conv (local)
  2. Token → Memory cross-attention
  3. Memory → Token cross-attention  
  4. Token FFN

The concern: Information bottleneck.

With M=16 memory slots and d=512:
  Memory bank = 16 × 512 = 8,192 floats
  
This bank must carry ALL global information for the ENTIRE sequence.

For a 512-token Hindi paragraph, global coherence requires tracking:
  - Subject-verb agreement across clauses
  - Pronoun reference chains
  - Topic continuity
  - Tense consistency
  - Code-switching boundaries (Hindi→English→Hindi)

16 slots MIGHT work for simple text.
16 slots will probably FAIL for:
  - Complex compound sentences (common in Hindi formal writing)
  - Multi-party dialogue (who said what)
  - Narrative with flashbacks or embedded stories

Your ablation plan says sweep M ∈ {8, 16, 32}.
I'd add M ∈ {48, 64} and watch quality carefully.
```

**But the deeper issue is:**

```
Is the memory bottleneck actually what you WANT for diffusion?

In autoregressive models, you process left-to-right.
Global context means "everything before this token."
Memory slots are a reasonable compression.

In diffusion, you process ALL positions SIMULTANEOUSLY.
The model needs to coordinate ACROSS the whole sequence
at EVERY denoising step.

Position 5 needs to know what position 495 is becoming
because they might be coreferent ("Ram... he... his...").

With full attention: position 5 directly sees position 495.
With memory slots: position 5 writes to memory, position 495 
reads from memory. Two hops. Information loss at each hop.

For diffusion specifically, this double-hop might hurt MORE
than it would for AR, because the "what position 495 is becoming"
changes every denoising step. The memory slots are always
one step behind.
```

**Recommendation:**

```python
# Consider a HYBRID approach within EcoHybrid itself:
# - Most layers: local conv + memory slots (fast)
# - Every 4th layer: full sliding window attention (accurate)
# 
# This is exactly what Aria's AR model does (RWKV + shared attention)!
# Apply the same principle to the diffusion denoiser.

class HybridEcoBlock(nn.Module):
    def __init__(self, d_model, n_heads, memory_slots, conv_kernel,
                 use_full_attention=False, window_size=128):
        super().__init__()
        
        self.local_conv = DepthwiseConv1d(d_model, conv_kernel)
        
        if use_full_attention:
            # Every 4th layer: real bidirectional attention
            # (bidirectional because diffusion is NOT causal!)
            self.global_mix = BidirectionalSlidingWindowAttention(
                d_model, n_heads, window_size
            )
        else:
            # Other layers: memory slot compression
            self.global_mix = MemorySlotAttention(
                d_model, n_heads, memory_slots
            )
        
        self.ffn = FFN(d_model)


class EcoHybridDenoiser(nn.Module):
    def __init__(self, n_layers=10, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridEcoBlock(
                use_full_attention=(i % 4 == 3),  # Layer 3, 7, 11...
                **kwargs
            )
            for i in range(n_layers)
        ])
```

### 2.4 Timestep Conditioning — You're Underselling Its Importance

```python
# Your document says:
# "Timestep embedding added to sequence states"

# This is almost certainly implemented as:
t_emb = self.time_mlp(timestep_embedding(t))  # (B, d)
x = x + t_emb.unsqueeze(1)                     # Add to all positions

# This is the MINIMUM viable approach and it's weak.
# The model gets ONE global signal about noise level.
```

For Indic diffusion, timestep conditioning should be richer:

```python
class AdaptiveTimestepModulation(nn.Module):
    """
    AdaLN-Zero from DiT (Peebles & Xie, 2023).
    Instead of adding timestep to hidden states,
    use timestep to MODULATE the layer norm parameters.
    
    This gives each layer a different "lens" for each noise level:
    - High noise: layers focus on broad semantic recovery
    - Low noise: layers focus on grammatical fine-tuning
    
    Critical for Hindi because:
    - At high noise, the model needs to recover word ORDER (SOV vs SVO)
    - At low noise, the model needs to fix verb AGREEMENT (gender, number)
    - These are fundamentally different tasks requiring different processing
    """
    def __init__(self, d_model, d_time):
        super().__init__()
        # Predict 6 modulation parameters per layer:
        # gamma1, beta1 (pre-attention norm)
        # gamma2, beta2 (pre-FFN norm)  
        # alpha1, alpha2 (residual gates)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time, 6 * d_model)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, t_emb):
        return self.adaLN_modulation(t_emb).chunk(6, dim=-1)


class ModulatedEcoBlock(nn.Module):
    def __init__(self, d_model, d_time, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.local_conv = DepthwiseConv1d(d_model, kwargs['conv_kernel'])
        self.global_mix = MemorySlotAttention(d_model, **kwargs)
        self.ffn = FFN(d_model)
        self.adaLN = AdaptiveTimestepModulation(d_model, d_time)
    
    def forward(self, x, t_emb):
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = self.adaLN(t_emb)
        
        # Modulated pre-norm for attention path
        h = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h = self.local_conv(h)
        h = self.global_mix(h)
        x = x + alpha1.unsqueeze(1) * h
        
        # Modulated pre-norm for FFN path
        h = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h
        
        return x
```

### 2.5 Tied Output Head — Probably Wrong for Diffusion

```
Your doc: "Tied output head (same matrix as token embedding)"

For AR models, weight tying works because:
  - Input and output are in the same "space" (next token prediction)
  - Saves parameters, acts as regularization

For diffusion denoisers, tying is LESS justified because:
  - Input is CORRUPTED tokens (with mask tokens mixed in)
  - Output is CLEAN token predictions
  - The input embedding must handle [MASK] token well
  - The output projection must handle the FULL vocabulary distribution
  - These are somewhat different tasks

At 95M params you want to save every parameter, so tying is
a reasonable trade-off. But if you see the model struggling
to distinguish between similar tokens (common in Devanagari
where many characters look similar to the model), try untying.
```
.


---

## Part 4: If You Proceed With Diffusion, Here's What To Prioritize

### The 3 Changes That Matter Most (Do These First, Ignore Everything Else)

```
1. SPAN MASKING instead of random token masking
   Impact: Major quality improvement for Indic languages
   Effort: 2 hours to implement
   
2. AdaLN-Zero timestep conditioning instead of additive
   Impact: Significant quality improvement, proven in DiT
   Effort: 3 hours to implement
   
3. Script-aware corruption (your Section 9.8, moved to launch)
   Impact: Major for Hindi/Gujarati/Odia robustness
   Effort: 4 hours to implement (needs script detection in tokenizer)
```

### The 3 Things To NOT Do Yet

```
1. Adaptive per-token step budget (Section 9.1)
   → Premature optimization. Get fixed-step working first.

2. Learnable memory routing (Section 9.2)
   → Adds instability. Prove uniform routing works first.

3. AR-assisted diffusion verifier (Section 9.4)
   → Two-model orchestration before either model works? No.
```

### The Eval That Actually Matters

```python
# Don't just track perplexity. Track THESE for Indic diffusion:

INDIC_DIFFUSION_EVALS = {
    
    "script_consistency": {
        "test": "Generate 100 Hindi sentences. Count how many have "
                "mixed Devanagari + random Unicode garbage.",
        "why": "Diffusion models can blend scripts within a word. "
               "AR models can't (they commit left-to-right). "
               "This is diffusion's Achilles heel for Indic.",
        "target": "<2% script-inconsistent outputs",
    },
    
    "verb_agreement": {
        "test": "Generate sentences with subject-verb pairs. "
                "Check gender/number agreement in Hindi. "
                "'लड़का जा रहा है' (correct) vs 'लड़का जा रही है' (wrong)",
        "why": "Diffusion reveals tokens non-sequentially. "
               "Subject might be revealed before verb or vice versa. "
               "Agreement errors are the #1 quality signal.",
        "target": "<5% agreement errors",
    },
    
    "transliteration_coherence": {
        "test": "Prompt with Roman Hindi. Does the OUTPUT stay in "
                "Roman Hindi or randomly switch to Devanagari mid-word?",
        "why": "The model must learn that 'mai' and 'मैं' are the same "
               "concept and not mix scripts within a generation.",
        "target": "95%+ script-consistent outputs",
    },
    
    "code_switch_naturalness": {
        "test": "Generate Hinglish. Are English words inserted at "
                "natural syntactic boundaries?",
        "example_good": "yaar kal ka match was amazing",
        "example_bad": "yaar kal ka mat was ch amazing",
        "why": "Diffusion's parallel generation can put English words "
               "in syntactically wrong positions.",
        "target": "Human eval: 4/5 naturalness",
    },
}
```

---

## The Actual Bottom Line

```
The diffusion design document: B+ (solid engineering, good instincts)
The EcoHybrid idea: B  (interesting but unproven, memory bottleneck risk)
The experimental plan: A- (proper methodology, good ablation design)

## 1. The Foundational Math Problem: Your Timesteps Are Wrong

```
Your design: T = 32 discrete timesteps
This is the single most consequential mistake in the document.
```

Here's why. With T=32 and a cosine schedule:

```
t=0  → mask rate ≈ 0.0%   (clean)
t=1  → mask rate ≈ 0.2%   (1 token masked in 512)
t=2  → mask rate ≈ 1.0%
t=5  → mask rate ≈ 6%
t=10 → mask rate ≈ 23%
t=16 → mask rate ≈ 50%
t=22 → mask rate ≈ 77%
t=27 → mask rate ≈ 94%
t=31 → mask rate ≈ 100%  (fully masked)
```

The model must learn **32 different behaviors** — one per noise level. Each noise level requires fundamentally different processing:

```
t=2  (1% masked): "Spot the ONE missing word and fill it in"
     → Basically a cloze task. Trivial.

t=16 (50% masked): "Read a sentence with half the words gone, reconstruct"
     → Hard language modeling. Requires grammar + semantics.

t=28 (96% masked): "Given 20 random surviving tokens from a 512-token
     paragraph, figure out what the entire paragraph should be"
     → Nearly unconditional generation. Extremely hard.
```

With only 32 levels, the **jumps between adjacent timesteps are enormous**. Between t=15 and t=17, the mask rate jumps from ~46% to ~54%. The model's optimal strategy changes discontinuously. It can't interpolate smoothly between behaviors.

**What actually works:**

```python
# diffusion/schedule.py

import torch
import math

class ContinuousTimeSchedule:
    """
    Train with CONTINUOUS time t ∈ [0, 1].
    Discretize ONLY during sampling.
    
    During training:
      t = random uniform in [0, 1]
      mask_rate = cosine_schedule(t)
    
    During inference:
      Use 12-16 discrete steps spaced along the schedule.
    
    This lets the model learn a SMOOTH mapping from noise level
    to denoising behavior, instead of memorizing 32 discrete modes.
    """
    
    def __init__(self, schedule_type="cosine"):
        self.schedule_type = schedule_type
    
    def mask_rate(self, t):
        """
        t: float tensor in [0, 1]
        Returns: mask probability at time t
        
        t=0 → clean (mask_rate=0)
        t=1 → fully masked (mask_rate=1)
        """
        if self.schedule_type == "cosine":
            # Standard cosine schedule from MDLM
            return torch.cos(t * math.pi / 2)  # Wait — this gives 1→0
            # We want 0→1, so:
            return 1 - torch.cos(t * math.pi / 2)
        
        elif self.schedule_type == "sqrt":
            # Spends more time at high noise levels
            # Better for harder generation tasks
            return torch.sqrt(t)
        
        elif self.schedule_type == "linear":
            return t
    
    def sample_timestep(self, batch_size, device):
        """Sample continuous timestep for training."""
        # LOW-DISCREPANCY SAMPLING — much better than uniform random
        # Stratified sampling ensures all noise levels are covered in every batch
        t = (torch.arange(batch_size, device=device) + torch.rand(batch_size, device=device)) / batch_size
        return t  # shape: (B,)
    
    def get_inference_steps(self, num_steps):
        """Return discrete timesteps for inference."""
        # Go from t=1 (fully masked) to t=0 (clean)
        return torch.linspace(1.0, 0.0, num_steps + 1)


class TimestepEmbedding(torch.nn.Module):
    """
    Sinusoidal embedding for CONTINUOUS timesteps.
    Not a lookup table of 32 embeddings.
    """
    def __init__(self, d_model, max_period=10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        # Project sinusoidal features to model dimension
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, t):
        """
        t: (B,) continuous timesteps in [0, 1]
        Returns: (B, d_model) timestep embeddings
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * self.max_period
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
```

**Impact:** This single change will improve your model quality more than any architecture modification.

---

## 2. The ELBO Weighting You're Not Doing

Your loss:

```
L = (1/|masked|) * Σ_masked CE(predicted, true)
```

This treats every timestep equally. **It shouldn't.** Here's the math for why.

The variational lower bound (ELBO) for masked diffusion decomposes as:

```
log p(x0) ≥ E_t[ (1/T) * Σ_i  mask_rate'(t)/mask_rate(t) * m_i * log p(x0_i | xt, t) ]
```

The key term is `mask_rate'(t) / mask_rate(t)` — the **rate of change** of the mask schedule at time t, divided by the mask rate. This is the importance weight.

Intuitively:
- At timesteps where the schedule changes rapidly, each masked position carries more information → higher weight
- At timesteps where the schedule is flat, less information is revealed per step → lower weight

Without this weighting, your model over-optimizes for high-noise timesteps (where there are many masked tokens and thus many loss terms) and under-optimizes for low-noise timesteps (few masked tokens, few loss terms).

**The problem for Hindi:** Low-noise timesteps are where grammatical agreement is finalized. If the model is weak there, you get sentences like:

```
"लड़की स्कूल जा रहा है" instead of "लड़की स्कूल जा रही है"
                  ^^^                              ^^^^
          masculine (wrong)                  feminine (correct)
```

The gender agreement error happens because the model doesn't invest enough capacity in the "almost clean, fix the details" regime.

```python
# diffusion/loss.py

import torch
import torch.nn.functional as F

def weighted_diffusion_loss(logits, targets, mask, t, schedule):
    """
    ELBO-weighted masked diffusion loss.
    
    Args:
        logits: (B, T, V) model predictions
        targets: (B, T) ground truth token ids
        mask: (B, T) bool — True where token is masked
        t: (B,) continuous timesteps
        schedule: ContinuousTimeSchedule instance
    """
    B, T_seq, V = logits.shape
    
    # Per-token cross entropy (only on masked positions)
    ce = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        reduction='none'
    ).view(B, T_seq)
    
    # Zero out unmasked positions
    ce = ce * mask.float()
    
    # ELBO importance weight per sample
    # d/dt [mask_rate(t)] / mask_rate(t)
    eps = 1e-6
    mask_rate = schedule.mask_rate(t)  # (B,)
    
    # Numerical derivative of mask_rate
    dt = 1e-4
    mask_rate_grad = (schedule.mask_rate(t + dt) - schedule.mask_rate(t - dt)) / (2 * dt)
    
    importance = (mask_rate_grad.abs() / (mask_rate + eps))  # (B,)
    
    # Normalize importance weights so they average to 1 across the batch
    importance = importance / (importance.mean() + eps)
    
    # Weight each sample's loss
    per_sample_loss = ce.sum(dim=1) / (mask.sum(dim=1) + eps)  # (B,)
    weighted_loss = (importance * per_sample_loss).mean()
    
    return weighted_loss
```

---

## 3. The EcoHybrid Compute Reality Check

Let me actually compute the FLOPS for your config. This matters because your stated motivation is efficiency.

```
Config: d=512, n_layers=10, n_heads=8, d_ff=1536, M=16, k=7, T=512

═══════════════════════════════════════════════════════════════
BASELINE (model.py) — Full Bidirectional Attention
═══════════════════════════════════════════════════════════════

Per layer:
  Q,K,V projections:    3 × 512 × 512² = 402M  flops
  Attention scores:      512 × 512 × 512 = 134M  flops
  Attention × V:         512 × 512 × 512 = 134M  flops  
  Output projection:     512 × 512²      = 134M  flops
  FFN (up + down):       2 × 512 × 512 × 1536 = 805M flops
  ─────────────────────────────────────────────────────
  Total per layer:       ~1,609M flops
  
10 layers: ~16.1B flops per forward pass

═══════════════════════════════════════════════════════════════
ECOHYBRID (eco_hybrid.py) — Conv + Memory Slots
═══════════════════════════════════════════════════════════════

Per layer:
  Depthwise conv:        512 × 7 × 512   = 1.8M   flops (negligible)
  
  Token→Memory cross-attn:
    Token Q proj:         512 × 512²      = 134M
    Memory K,V proj:      16 × 512² × 2   = 8.4M
    Attention (T×M):      512 × 16 × 512  = 4.2M
    Output proj:          512 × 512²      = 134M
    Subtotal:             ~281M
  
  Memory→Token cross-attn:
    Memory Q proj:        16 × 512²       = 4.2M
    Token K,V proj:       512 × 512² × 2  = 268M
    Attention (M×T):      16 × 512 × 512  = 4.2M
    Output proj:          16 × 512²       = 4.2M
    Subtotal:             ~281M
  
  FFN:                    2 × 512 × 512 × 1536 = 805M
  ─────────────────────────────────────────────────────
  Total per layer:        ~1,368M flops

10 layers: ~13.7B flops per forward pass

═══════════════════════════════════════════════════════════════
SAVINGS: 16.1B → 13.7B = 15% reduction
═══════════════════════════════════════════════════════════════
```

**15%. That's it.**

The FFN dominates at 805M per layer (59% of EcoHybrid per-layer cost). The attention savings from M=16 memory slots vs T=512 full attention is real but small because the **projection matrices are the same size** — you still need to project all 512 tokens to Q/K/V space.

**The efficiency claim only becomes significant at longer sequences:**

```
At T=2048 (4× longer):
  Baseline attention term:  2048² × 512 = 2.1B per layer
  EcoHybrid memory term:    2 × 2048 × 16 × 512 = 33.5M per layer
  
  Baseline total per layer: ~4.5B
  EcoHybrid total per layer: ~1.4B (same, attention was minor before)
  
  Savings: ~69%   ← NOW we're talking

At T=512 (your current config):
  Savings: ~15%   ← Not worth the quality risk
```

**Engineering conclusion:**

```
At seq_len=512, EcoHybrid's efficiency advantage is marginal.
The quality risk from the memory bottleneck is NOT marginal.
You're taking a real quality hit for a 15% speedup.

EcoHybrid becomes justified at seq_len ≥ 1024.
At your current seq_len=512, just use the baseline.

If you want efficiency at T=512, shrink the FFN instead:
  d_ff=1536 → d_ff=1024 saves 268M/layer = 20% savings
  with ZERO quality risk from information bottleneck.
```

---

## 4. Variable Length Generation: The Missing Design

Your document doesn't address this at all. For a chat model, this is critical.

```
AR model: Generates until it produces <EOS>. Natural variable length.

Diffusion model: Starts with N mask tokens. Must decide N in advance.
  - Too few masks → response truncated
  - Too many masks → model fills with garbage/repetition to fill space

User: "bhai kal ka match kaisa raha?"
Expected response: "yaar maza aa gaya, Kohli ne century maari" (8-10 tokens)
But you initialized 512 mask tokens...
What fills the other 500 positions?
```

Three approaches, each with tradeoffs:

```python
# diffusion/variable_length.py

import torch

class VariableLengthDiffusion:
    """
    Strategy 1: LENGTH PREDICTION + EOS MASKING
    
    1. Small MLP predicts expected response length from prompt
    2. Initialize that many masks
    3. Also allow [EOS] as a denoised output
    4. Everything after first [EOS] is discarded
    """
    
    def __init__(self, denoiser, length_predictor, max_len=512):
        self.denoiser = denoiser
        self.length_pred = length_predictor  # Simple 2-layer MLP
        self.max_len = max_len
    
    def generate(self, prompt_ids, **kwargs):
        # Predict length
        prompt_emb = self.denoiser.embed(prompt_ids).mean(dim=1)
        pred_len = self.length_pred(prompt_emb).clamp(8, self.max_len).int()
        
        # Add 30% buffer (better too long than too short)
        gen_len = int(pred_len.item() * 1.3)
        
        # Initialize: [prompt_tokens] [MASK × gen_len]
        mask_ids = torch.full((1, gen_len), MASK_TOKEN_ID, device=prompt_ids.device)
        input_ids = torch.cat([prompt_ids, mask_ids], dim=1)
        generation_mask = torch.cat([
            torch.zeros_like(prompt_ids, dtype=torch.bool),
            torch.ones(1, gen_len, dtype=torch.bool, device=prompt_ids.device)
        ], dim=1)
        
        # Run denoising
        output = self.denoise_loop(input_ids, generation_mask, **kwargs)
        
        # Truncate at first EOS
        eos_positions = (output == EOS_TOKEN_ID).nonzero(as_tuple=True)[1]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            output = output[:, :first_eos]
        
        return output[:, prompt_ids.shape[1]:]  # Return only generated part
    
    
class SemiAutoregressiveDiffusion:
    """
    Strategy 2: BLOCK-BY-BLOCK GENERATION
    
    Generate in chunks of 64 tokens.
    Each chunk is diffusion-denoised conditioned on all previous chunks.
    Stop when a chunk contains [EOS].
    
    This is the most practical approach:
    - Natural variable length (stop at EOS)
    - Each chunk benefits from parallel diffusion
    - Previous chunks provide strong conditioning
    - Latency = num_chunks × denoise_steps × forward_pass
    
    For 256-token response with 64-token blocks and 12 denoise steps:
    = 4 chunks × 12 steps = 48 forward passes
    vs AR: 256 sequential forward passes
    vs full-sequence diffusion: 12 passes but must guess length
    """
    
    def __init__(self, denoiser, block_size=64, denoise_steps=12):
        self.denoiser = denoiser
        self.block_size = block_size
        self.denoise_steps = denoise_steps
    
    def generate(self, prompt_ids, max_blocks=8):
        generated = prompt_ids.clone()
        
        for block_idx in range(max_blocks):
            # Create new block of masks
            new_block = torch.full(
                (1, self.block_size), 
                MASK_TOKEN_ID, 
                device=prompt_ids.device
            )
            
            # Concat with everything generated so far
            full_seq = torch.cat([generated, new_block], dim=1)
            
            # Mask only covers the new block
            gen_mask = torch.zeros(full_seq.shape, dtype=torch.bool, device=full_seq.device)
            gen_mask[:, -self.block_size:] = True
            
            # Denoise this block
            denoised = self.denoise_loop(full_seq, gen_mask)
            generated = denoised
            
            # Check for EOS in the new block
            new_tokens = generated[:, -self.block_size:]
            if (new_tokens == EOS_TOKEN_ID).any():
                eos_pos = (generated[0] == EOS_TOKEN_ID).nonzero()[0].item()
                generated = generated[:, :eos_pos]
                break
        
        return generated[:, prompt_ids.shape[1]:]
```

**Strategy 2 (block-by-block) is what I'd actually implement.** It naturally handles variable length, gives you the latency benefit of diffusion within each block, and doesn't require a length predictor.

---

## 5. The Denoising Loop: What's Actually Broken

Your current reveal logic has a critical bug pattern for Indic text. Let me show you:

```
Hindi sentence to generate: "मैं बाज़ार जा रहा हूँ"
Tokens: [मैं] [बाज़ार] [जा] [रहा] [हूँ]

Diffusion step 1: All masked → [M] [M] [M] [M] [M]
  Model predicts confidences:
    pos 0: "मैं" (0.72), "वह" (0.15), "तुम" (0.08)...
    pos 1: "बाज़ार" (0.45), "स्कूल" (0.30), "घर" (0.20)...
    pos 2: "जा" (0.81), "आ" (0.10)...
    pos 3: "रहा" (0.40), "रही" (0.38), "रहे" (0.15)...
    pos 4: "हूँ" (0.25), "है" (0.35), "हैं" (0.20)...

  Confidence ranking: pos2 (0.81) > pos0 (0.72) > pos1 (0.45) > pos3 (0.40) > pos4 (0.35)
  
  Reveal top-2: pos2="जा", pos0="मैं"
  → [मैं] [M] [जा] [M] [M]

Step 2: Model now sees "मैं ___ जा ___ ___"
  pos 3: NOW "रहा" confidence jumps to 0.85 (masculine, matches "मैं")
  pos 4: "हूँ" confidence jumps to 0.78 (first person, matches "मैं")
  pos 1: "बाज़ार" confidence rises to 0.65
  
  Reveal top-2: pos3="रहा", pos4="हूँ"
  → [मैं] [M] [जा] [रहा] [हूँ]

Step 3: Reveal remaining: pos1="बाज़ार"
  → [मैं] [बाज़ार] [जा] [रहा] [हूँ] ✓ CORRECT!
```

This worked because **"मैं" was revealed before "रहा"/"हूँ"**, so agreement was resolved correctly. But what if confidence was different?

```
BAD SCENARIO — common with poorly calibrated confidence:

Step 1: Model predicts:
    pos 4: "है" (0.88)  ← VERY confident but WRONG for first person
    pos 2: "जा" (0.81)
    
  Reveal: pos4="है", pos2="जा"
  → [M] [M] [जा] [M] [है]

Step 2: Model sees "___ ___ जा ___ है"
  "है" = third person singular → model now predicts:
    pos 0: "वह" (0.91)  ← Changed from "मैं" to match "है"!
    pos 3: "रहा" (0.87)
  
  Reveal: pos0="वह", pos3="रहा"
  → [वह] [M] [जा] [रहा] [है]

Step 3: pos1="बाज़ार"
  → [वह] [बाज़ार] [जा] [रहा] [है]
  
  "He is going to the market" — grammatically correct but
  the INTENDED meaning was "I am going to the market"
```

**The reveal order determined the meaning.** This is a fundamental property of masked diffusion that doesn't exist in AR models.

**Fix: Dependency-aware reveal ordering**

```python
# diffusion/reveal_policy.py

import torch

class DependencyAwareReveal:
    """
    Instead of pure confidence-based reveal, use a policy that
    respects linguistic dependency structure.
    
    For Hindi (SOV language):
    1. First reveal: Subject + Object (semantic anchors)
    2. Second reveal: Main verb stem (semantic core)
    3. Third reveal: Auxiliaries, postpositions (grammatical agreement)
    
    The key insight: AGREEMENT TOKENS MUST BE REVEALED LAST
    because they depend on everything else.
    
    Agreement tokens in Hindi: हूँ/है/हैं/हो (be-auxiliaries),
    रहा/रही/रहे (progressive markers), ता/ती/ते (habitual markers)
    """
    
    def __init__(self, tokenizer, lang="hi"):
        self.tokenizer = tokenizer
        
        # Tokens that should be revealed LATE (they carry agreement)
        # These are language-specific
        self.agreement_tokens = self._build_agreement_set(lang)
        
        # Tokens that should be revealed EARLY (semantic anchors)
        self.content_indicators = self._build_content_indicators(lang)
    
    def _build_agreement_set(self, lang):
        """Tokens whose identity depends on OTHER tokens."""
        if lang == "hi":
            agreement_words = [
                "हूँ", "है", "हैं", "हो",           # be-forms
                "रहा", "रही", "रहे",              # progressive
                "ता", "ती", "ते",                 # habitual
                "गा", "गी", "गे",                 # future
                "था", "थी", "थे",                 # past be-forms
                "ने", "को", "से", "में", "पर", "के", "की", "का",  # postpositions
                "और", "या", "कि", "जो", "तो",    # conjunctions
            ]
        elif lang == "gu":
            agreement_words = [
                "છે", "છું", "છો",                # be-forms
                "નો", "ની", "નું", "ના",          # postpositions
                "અને", "કે", "પણ",               # conjunctions
            ]
        elif lang == "or":
            agreement_words = [
                "ଅଛି", "ଅଛେ", "ଅଛ",              # be-forms
                "ର", "କୁ", "ରେ",                 # postpositions
            ]
        else:
            return set()
        
        # Convert to token ids
        token_ids = set()
        for word in agreement_words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.update(ids)
        return token_ids
    
    def compute_reveal_priority(self, confidences, candidate_ids, mask, step_progress):
        """
        Adjust reveal priority based on linguistic role.
        
        Args:
            confidences: (B, T) raw model confidence
            candidate_ids: (B, T) predicted token ids
            mask: (B, T) bool — currently masked positions
            step_progress: float in [0, 1] — how far through denoising
        
        Returns:
            priority: (B, T) adjusted scores (higher = reveal sooner)
        """
        B, T = confidences.shape
        priority = confidences.clone()
        
        # Only adjust masked positions
        for b in range(B):
            for pos in range(T):
                if not mask[b, pos]:
                    continue
                
                token_id = candidate_ids[b, pos].item()
                
                if token_id in self.agreement_tokens:
                    # SUPPRESS agreement tokens in early steps
                    # They should be revealed in the final 30% of steps
                    if step_progress < 0.7:
                        priority[b, pos] *= 0.3  # Heavily penalize
                    else:
                        priority[b, pos] *= 1.2  # Slight boost in final steps
                
                elif token_id in self.content_indicators:
                    # BOOST content words in early steps
                    if step_progress < 0.4:
                        priority[b, pos] *= 1.5  # Reveal these first
        
        # Zero out already-revealed positions
        priority = priority * mask.float()
        
        return priority
```

---

## 6. What Your Denoiser Architecture Actually Needs (Not What You Built)

Let me work from first principles. What does a text diffusion denoiser need to do?

```
INPUT:  A partially masked sequence + noise level indicator
OUTPUT: Probability distribution over vocabulary for EVERY masked position

The denoiser must:
1. Read all VISIBLE tokens (bidirectional — not causal!)
2. Understand the POSITIONS of visible vs masked tokens
3. Know the NOISE LEVEL (how many tokens are masked)
4. Predict what goes in EACH masked position
5. Make predictions that are JOINTLY COHERENT
   (not just independently good at each position)

Property #5 is the hardest and most important.
AR models get it for free (each token is conditioned on all previous).
Diffusion models must EXPLICITLY coordinate across positions.
```

Now evaluate EcoHybrid against these requirements:

```
Requirement              EcoHybrid                    Grade
─────────────────────────────────────────────────────────────
1. Bidirectional read    Conv is bidirectional ✓       A
                         Memory attention is bidir ✓
                         
2. Position awareness    Positional embedding...       C
                         WAIT. Your doc doesn't 
                         mention positional encoding
                         at ALL. Is it in the code?
                         
3. Noise level           "Timestep embedding added     C-
                         to sequence states" — weak,
                         should be AdaLN (discussed)
                         
4. Per-position pred     Output head over vocab ✓      B
                         Tied weights (questionable)
                         
5. Joint coherence       16 memory slots must carry    D
                         ALL cross-position info.
                         Bottleneck.
```

**Requirement 2 is terrifying.** Let me address it:

```
If you don't have positional encoding or if it's just standard
sinusoidal/learned position embeddings, you have a problem
specific to diffusion:

The model sees: [the] [MASK] [MASK] [MASK] [sat] [MASK] [the] [MASK]

Without strong positional signal, the model has to figure out:
- Position 1 should be "cat" (subject)
- Position 6 should be "on" (preposition)
- Position 7 should be "mat" (object)

But "the" appears at positions 0 AND 6. If positional encoding
is weak, the model might confuse which "the" is which.

For HINDI, it's even worse because postpositions come AFTER nouns:
[राम] [MASK] [बाज़ार] [MASK] [जा] [MASK] [MASK]

Position 1 = "ने" (ergative marker)
Position 3 = "में" (locative)  
Position 5 = "रहा" (progressive)
Position 6 = "है" (auxiliary)

ALL of these are function words that look similar to the model.
Positional signal is the ONLY way to tell them apart in a
partially masked context.
```

**Use RoPE or ALiBi, not learned position embeddings:**

```python
# diffusion/positional.py

import torch
import math

class RotaryEmbedding(torch.nn.Module):
    """
    RoPE for the diffusion denoiser.
    
    Why RoPE over learned positions for diffusion:
    1. Relative position matters more than absolute
       (the model needs to know "this mask is 3 positions 
        after this visible token")
    2. RoPE encodes this naturally in the attention computation
    3. Generalizes to unseen positions if you later increase seq_len
    4. Works with both full attention AND memory-slot attention
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute for max length
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, q, k, seq_offset=0):
        """Apply rotary embeddings to q and k."""
        T = q.shape[1]
        cos = self.cos_cached[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(2)
        
        q_rot = apply_rotary(q, cos, sin)
        k_rot = apply_rotary(k, cos, sin)
        return q_rot, k_rot


def apply_rotary(x, cos, sin):
    """x: (B, T, H, D)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

---

## 7. The Memory Slot Initialization Problem Nobody Talks About

In your EcoHybrid, memory slots are **learned parameters**:

```python
self.memory_slots = nn.Parameter(torch.randn(M, d_model) * 0.02)
```

At the start of every forward pass, these slots are the **same** regardless of the input. The cross-attention updates them, but the starting point is static.

**Why this is a problem for diffusion specifically:**

```
The memory slots need to represent DIFFERENT things at different noise levels:

At t≈1.0 (fully masked):
  The input is [MASK MASK MASK MASK MASK MASK...]
  Memory slots should initialize to: "generic language priors"
  (common Hindi sentence structures, frequent word patterns)

At t≈0.1 (almost clean):
  The input is [मैं बाज़ार MASK रहा MASK]
  Memory slots should initialize to: "this specific sentence's semantics"
  (subject is first person, location is market, progressive tense)

But your memory starts as the SAME learned vector regardless!
The cross-attention has to do ALL the work of adapting slots
to both the noise level and the content.
```

**Fix: Condition memory slot initialization on timestep AND a summary of visible tokens:**

```python
# model/conditioned_memory.py

import torch
import torch.nn as nn

class ConditionedMemorySlots(nn.Module):
    """
    Memory slots whose initialization depends on:
    1. The current noise level (timestep)
    2. A summary of visible (non-masked) tokens
    
    This gives the memory a "head start" — it knows roughly
    what kind of sentence it's working with before the
    cross-attention iterations begin.
    """
    
    def __init__(self, n_slots, d_model, d_time):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        
        # Base memory (learned, like before)
        self.base_memory = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        
        # Timestep-dependent modulation
        self.time_to_memory = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_slots * d_model),
        )
        
        # Visible-token summary → memory adjustment
        self.summary_proj = nn.Linear(d_model, n_slots * d_model)
        
    def forward(self, x, mask, t_emb):
        """
        Args:
            x: (B, T, d_model) token embeddings (masked tokens are mask_emb)
            mask: (B, T) bool — True where masked
            t_emb: (B, d_time) timestep embedding
        
        Returns:
            memory: (B, M, d_model) initialized memory slots
        """
        B = x.shape[0]
        
        # Start with base memory
        memory = self.base_memory.unsqueeze(0).expand(B, -1, -1)  # (B, M, d)
        
        # Add timestep conditioning
        time_mod = self.time_to_memory(t_emb)  # (B, M*d)
        time_mod = time_mod.view(B, self.n_slots, self.d_model)
        memory = memory + time_mod
        
        # Add visible-token summary
        # Mean-pool over VISIBLE (non-masked) tokens only
        visible_mask = (~mask).float().unsqueeze(-1)  # (B, T, 1)
        visible_sum = (x * visible_mask).sum(dim=1)   # (B, d)
        visible_count = visible_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        visible_mean = visible_sum / visible_count     # (B, d)
        
        summary_mod = self.summary_proj(visible_mean)  # (B, M*d)
        summary_mod = summary_mod.view(B, self.n_slots, self.d_model)
        memory = memory + summary_mod
        
        return memory
```

---

## 8. The Mask Token Embedding Problem

This is subtle and almost everyone gets it wrong:

```python
# Most implementations:
self.token_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
# Mask token is just another embedding vector.
```

**The problem:**

```
At t≈1.0 (fully masked), your input is:
[mask_emb, mask_emb, mask_emb, mask_emb, ...]

Every position has the SAME embedding (plus positional encoding).

The model's first layer receives near-identical vectors at every position.
With attention, every position attends to every other position equally
(because Q·K is the same everywhere).

This is an information-poor starting point.
The model has to rely ENTIRELY on positional encoding to differentiate
positions in the fully masked regime.

For EcoHybrid this is WORSE because:
- Conv over identical vectors = identical output
- Memory cross-attention with identical queries = identical updates
```

**Fix: Noise the mask embeddings**

```python
# model/mask_embedding.py

import torch
import torch.nn as nn

class StochasticMaskEmbedding(nn.Module):
    """
    Instead of one fixed mask embedding, add learned noise
    to break symmetry between masked positions.
    
    Each masked position gets:
    mask_emb + positional_encoding + small_random_perturbation
    
    The perturbation is drawn from a learned distribution,
    not fixed. This gives the model diverse "starting points"
    for denoising different positions.
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.base_mask_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Learned noise scale (starts small, model can increase if helpful)
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        
        # Learned noise direction basis (not random noise — structured noise)
        self.noise_basis = nn.Parameter(torch.randn(8, d_model) * 0.01)
    
    def forward(self, num_masks, device):
        """
        Returns: (num_masks, d_model) — diverse mask embeddings
        """
        # Random coefficients for noise basis
        coeffs = torch.randn(num_masks, 8, device=device) * self.noise_scale
        noise = coeffs @ self.noise_basis  # (num_masks, d_model)
        
        return self.base_mask_emb.unsqueeze(0) + noise
```

---

## 9. What Your Training Loop Should Actually Look Like

Putting it all together. This is the training loop I would implement:

```python
# diffusion/train_loop.py

import torch
import torch.nn.functional as F

def train_step(model, batch, schedule, optimizer, step, total_steps):
    """
    One training step for masked diffusion.
    
    Key differences from a naive implementation:
    1. Continuous time sampling (not discrete T=32)
    2. Low-discrepancy timestep sampling across batch
    3. ELBO importance weighting
    4. Span masking (not random token masking)
    5. Loss tracking per noise-level bucket for diagnostics
    """
    model.train()
    x0 = batch["input_ids"]  # (B, T) clean token ids
    B, T = x0.shape
    device = x0.device
    
    # ─── Continuous timestep sampling ───
    # Stratified: each sample in the batch gets a different noise level
    # Ensures every noise regime is covered in every batch
    t = (torch.arange(B, device=device).float() + torch.rand(B, device=device)) / B
    # t shape: (B,) values spread across [0, 1]
    
    # ─── Compute mask rate per sample ───
    mask_rate = schedule.mask_rate(t)  # (B,)
    
    # ─── Create corrupted sequence (span masking) ───
    xt = x0.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    
    for b in range(B):
        num_to_mask = max(1, int(T * mask_rate[b].item()))
        masked_so_far = 0
        
        while masked_so_far < num_to_mask:
            # Geometric span length (mean=3 for natural word-level spans)
            span_len = min(
                int(torch.distributions.Geometric(0.33).sample().item()) + 1,
                num_to_mask - masked_so_far,
                T
            )
            start = torch.randint(0, max(1, T - span_len + 1), (1,)).item()
            mask[b, start:start + span_len] = True
            masked_so_far = mask[b].sum().item()
    
    xt[mask] = MASK_TOKEN_ID
    
    # ─── Forward pass ───
    logits = model(xt, t)  # (B, T, V)
    
    # ─── Masked cross-entropy with ELBO weighting ───
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        x0.view(-1),
        reduction='none'
    ).view(B, T)
    
    # Loss only on masked positions
    masked_ce = ce * mask.float()
    per_sample_loss = masked_ce.sum(dim=1) / mask.sum(dim=1).float().clamp(min=1)
    
    # ELBO importance weights
    eps = 1e-6
    dt = 1e-4
    dmask_dt = (schedule.mask_rate(t + dt) - schedule.mask_rate(t - dt)) / (2 * dt)
    importance = dmask_dt.abs() / (mask_rate + eps)
    importance = importance / importance.mean()  # normalize
    
    loss = (importance * per_sample_loss).mean()
    
    # ─── Backward + optimize ───
    loss.backward()
    
    # Gradient clipping (important for diffusion stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # ─── Diagnostics: loss per noise bucket ───
    with torch.no_grad():
        buckets = {}
        for b in range(B):
            bucket_name = f"t={t[b].item():.1f}"
            buckets[bucket_name] = per_sample_loss[b].item()
    
    return {
        "loss": loss.item(),
        "mean_mask_rate": mask_rate.mean().item(),
        "loss_per_bucket": buckets,
        "grad_norm": get_grad_norm(model),
    }
```

---

## 10. The Inference Loop Done Right

```python
# diffusion/sample.py

import torch

@torch.no_grad()
def generate(model, prompt_ids, gen_length, schedule,
             num_steps=16, temperature=0.8, 
             reveal_policy=None, block_size=64):
    """
    Block-wise semi-autoregressive diffusion generation.
    
    Generates in blocks of `block_size` tokens.
    Each block is denoised in `num_steps` reverse diffusion steps.
    Stops when [EOS] is generated.
    """
    device = prompt_ids.device
    generated = prompt_ids.clone()
    max_blocks = gen_length // block_size + 1
    
    for block_idx in range(max_blocks):
        # ─── Initialize new block of masks ───
        new_masks = torch.full(
            (1, block_size), MASK_TOKEN_ID, 
            dtype=torch.long, device=device
        )
        full_seq = torch.cat([generated, new_masks], dim=1)
        
        # Track which positions are still masked
        is_masked = torch.zeros(full_seq.shape[1], dtype=torch.bool, device=device)
        is_masked[-block_size:] = True
        
        # ─── Reverse diffusion steps ───
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for step_idx in range(num_steps):
            t_now = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            
            # How many masks should remain AFTER this step
            target_mask_rate = schedule.mask_rate(t_next)
            current_masked = is_masked.sum().item()
            target_masked = max(0, int(block_size * target_mask_rate.item()))
            num_to_reveal = current_masked - target_masked
            
            if num_to_reveal <= 0:
                continue
            
            # Forward pass
            t_batch = t_now.unsqueeze(0)  # (1,)
            logits = model(full_seq, t_batch)  # (1, T_full, V)
            
            # Only look at currently masked positions
            masked_logits = logits[0, is_masked]  # (num_masked, V)
            
            # Temperature-scaled sampling
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            candidates = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidences = probs.gather(1, candidates.unsqueeze(-1)).squeeze(-1)
            
            # ─── Reveal policy ───
            if reveal_policy is not None:
                # Dependency-aware: suppress agreement tokens early
                step_progress = step_idx / num_steps
                adjusted_conf = reveal_policy.adjust(
                    confidences, candidates, step_progress
                )
            else:
                adjusted_conf = confidences
            
            # Reveal the top-k most confident (or policy-adjusted)
            _, reveal_order = adjusted_conf.topk(min(num_to_reveal, len(adjusted_conf)))
            
            # Map back to full sequence positions
            masked_positions = is_masked.nonzero().squeeze(-1)
            for idx in reveal_order:
                pos = masked_positions[idx]
                full_seq[0, pos] = candidates[idx]
                is_masked[pos] = False
        
        # ─── Reveal any remaining masks (final cleanup) ───
        if is_masked.any():
            t_batch = torch.tensor([0.0], device=device)
            logits = model(full_seq, t_batch)
            remaining = is_masked.nonzero().squeeze(-1)
            for pos in remaining:
                full_seq[0, pos] = logits[0, pos].argmax()
                is_masked[pos] = False
        
        generated = full_seq
        
        # ─── Check for EOS ───
        new_block = generated[0, -block_size:]
        if (new_block == EOS_TOKEN_ID).any():
            eos_pos = (generated[0] == EOS_TOKEN_ID).nonzero()
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                generated = generated[:, :first_eos]
            break
    
    # Return only the generated portion
    return generated[:, prompt_ids.shape[1]:]
```

---

## 11. Temperature Annealing — Small Detail, Big Impact

```python
# WRONG: Fixed temperature across all denoising steps
temperature = 0.8  # Same for step 1 and step 12

# RIGHT: Anneal temperature across steps
def get_temperature(step, total_steps, t_start=1.2, t_end=0.5):
    """
    Early steps (high noise): high temperature
      → Explore diverse possibilities, don't commit yet
    
    Late steps (low noise): low temperature  
      → Commit to specific tokens, ensure consistency
    
    This is analogous to simulated annealing:
    explore broadly first, then narrow down.
    
    For Hindi, this is especially important because:
    - Early: "could be मैं or वह or हम — keep options open"
    - Late: "it's definitely मैं, now make everything agree"
    """
    progress = step / total_steps
    return t_start + (t_end - t_start) * progress
```

---

## 12. The Experiments That Actually Matter

Forget sweeping memory slots 8/16/32. Here's what you should measure first:

```
═══════════════════════════════════════════════════════════════
EXPERIMENT 1: Does diffusion work on Hindi AT ALL?
═══════════════════════════════════════════════════════════════
Model: baseline (model.py), NOT EcoHybrid
Data: 10M tokens of Hindi Wikipedia (small, fast)
Config: d=256, layers=6, heads=4 (tiny model, ~10M params)
Steps: 5000

Eval: Generate 100 Hindi sentences from prompts.
  Metric 1: % grammatically valid Hindi (human eval)
  Metric 2: % consistent script (no Devanagari-Latin mixing)
  Metric 3: % correct verb agreement

Expected outcome: 40-60% grammatically valid
This establishes your FLOOR. Everything else builds from here.
Time: ~2 hours on T4


═══════════════════════════════════════════════════════════════
EXPERIMENT 2: Continuous vs discrete timesteps
═══════════════════════════════════════════════════════════════
Model A: baseline + T=32 discrete (your current design)
Model B: baseline + continuous t ∈ [0,1] (my recommendation)
Everything else identical.

Eval: Same Hindi generation eval.
Expected: Model B wins by 5-15% on all metrics.
Time: ~4 hours on T4 (two runs)

THIS EXPERIMENT DETERMINES WHETHER YOU CONTINUE.
If even the baseline can't generate valid Hindi, the architecture
doesn't matter.


═══════════════════════════════════════════════════════════════
EXPERIMENT 3: Span masking vs token masking
═══════════════════════════════════════════════════════════════
Only run this AFTER Experiment 2 confirms basic viability.
Model: baseline + continuous time
Variant A: random token masking
Variant B: span masking (mean span length 3)

Expected: Span masking produces more coherent words.
Devanagari conjunct errors should drop significantly.
Time: ~4 hours


═══════════════════════════════════════════════════════════════
EXPERIMENT 4: Baseline vs EcoHybrid
═══════════════════════════════════════════════════════════════
Only run this AFTER Experiments 2-3.
At matched parameter count (~10M), does EcoHybrid match baseline?

If EcoHybrid LOSES on quality at T=512:
  → Kill EcoHybrid at this sequence length.
  → Only revisit at T≥1024 where efficiency matters.

If EcoHybrid MATCHES baseline:
  → Proceed with EcoHybrid + conditioned memory + RoPE.


═══════════════════════════════════════════════════════════════
EXPERIMENT 5: Hindi-specific eval battery  
═══════════════════════════════════════════════════════════════
Only after you have a model that generates passable Hindi.

Test cases (20 prompts each):
  a) Complete: "मैं आज बाज़ार ___"  (simple completion)
  b) Complete: "राम ने श्याम को ___" (case marker test)
  c) Hinglish: "bhai aaj ka match ___" (code-switch test)
  d) Gujarati: "આજે હું ___" (script test)
  e) Long: "भारत एक विविधताओं वाला देश है जहाँ ___" (coherence test)
```

---

## 13. The Architecture I Would Actually Build

If I were starting your diffusion denoiser from scratch, knowing everything above:

```python
# diffusion/denoiser_v2.py — What I'd actually build

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionDenoiserV2(nn.Module):
    """
    Key design choices:
    1. Continuous time, sinusoidal embedding + AdaLN-Zero
    2. Standard bidirectional attention (NOT EcoHybrid — at T=512 it's not worth it)
    3. RoPE positional encoding
    4. Stochastic mask embeddings
    5. Span-aware architecture
    6. UNTIED output head
    
    Save EcoHybrid for when you scale to T≥1024.
    At T=512, full attention is fine and MORE RELIABLE.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=384,       # Smaller than your 512 — see why below
        n_layers=8,
        n_heads=6,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Token embeddings (separate from output head)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.mask_emb = StochasticMaskEmbedding(d_model)
        
        # Positional encoding (RoPE — applied in attention)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Timestep embedding (continuous)
        d_time = d_model * 4
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_time),
            nn.GELU(),
            nn.Linear(d_time, d_time),
        )
        
        # Transformer blocks with AdaLN-Zero
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(d_model, n_heads, d_ff, d_time, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head (UNTIED — separate from input embedding)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize output projection small
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(n_layers))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x_ids, t, mask=None):
        """
        Args:
            x_ids: (B, T) token ids (with MASK_TOKEN_ID at masked positions)
            t: (B,) continuous timesteps in [0, 1]
            mask: (B, T) bool — optional, True where token is [MASK]
                  If None, inferred from x_ids == MASK_TOKEN_ID
        
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x_ids.shape
        
        if mask is None:
            mask = (x_ids == MASK_TOKEN_ID)
        
        # ─── Token embedding ───
        # Non-masked tokens: normal embedding
        # Masked tokens: stochastic mask embedding
        x = self.token_emb(x_ids)  # (B, T, d)
        
        # Replace masked position embeddings with stochastic ones
        num_masked = mask.sum().item()
        if num_masked > 0:
            mask_embeddings = self.mask_emb(num_masked, x.device)
            x[mask] = mask_embeddings
        
        # ─── Timestep conditioning ───
        t_emb = self.time_emb(t)  # (B, d_time)
        
        # ─── Transformer blocks with AdaLN ───
        for block in self.blocks:
            x = block(x, t_emb, self.rope)
        
        # ─── Output projection ───
        x = self.out_norm(x)
        logits = self.out_proj(x)  # (B, T, V)
        
        return logits


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Norm Zero (DiT-style).
    Bidirectional attention (NOT causal — diffusion is not causal).
    """
    
    def __init__(self, d_model, n_heads, d_ff, d_time, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # AdaLN-Zero modulation (6 parameters from timestep)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time, 6 * d_model),
        )
        # Initialize to identity (zero gate → residual passthrough)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x, t_emb, rope):
        """
        x: (B, T, d_model)
        t_emb: (B, d_time)
        rope: RotaryEmbedding module
        """
        # AdaLN modulation parameters
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = \
            self.adaLN(t_emb).chunk(6, dim=-1)  # Each: (B, d_model)
        
        # ─── Attention path ───
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        
        B, T, d = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, H, D)
        
        # Apply RoPE
        q, k = rope(q, k)
        
        # Bidirectional attention (NO causal mask!)
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bthd,bshd->bhts', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('bhts,bshd->bthd', attn, v)
        out = out.reshape(B, T, d)
        out = self.out_proj(out)
        
        x = x + alpha1.unsqueeze(1) * out
        
        # ─── FFN path ───
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h
        
        return x


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, max_period=10000):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(half) / half)
        )
    
    def forward(self, t):
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * 10000
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class StochasticMaskEmbedding(nn.Module):
    def __init__(self, d_model, n_basis=8):
        super().__init__()
        self.base = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.basis = nn.Parameter(torch.randn(n_basis, d_model) * 0.01)
    
    def forward(self, num_masks, device):
        coeffs = torch.randn(num_masks, self.basis.shape[0], device=device)
        noise = coeffs @ self.basis
        return self.base.unsqueeze(0) + self.scale * noise


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())
    
    def forward(self, q, k, offset=0):
        T = q.shape[1]
        cos = self.cos_cache[offset:offset+T].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cache[offset:offset+T].unsqueeze(0).unsqueeze(2)
        
        def rotate(x):
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], -1)
        
        return rotate(q), rotate(k)
```

**Why d=384 instead of your d=512:**

```
At d=384, n_layers=8, n_heads=6:
  Params: ~25M (fits on T4 with longer sequences and larger batches)
  FFN: d_ff=1024 → 384×1024×2 = 786K per layer
  
At d=512, n_layers=10, n_heads=8:
  Params: ~60M (tighter on T4, smaller batches)
  FFN: d_ff=1536 → 512×1536×2 = 1.57M per layer

For your first experiments, the 25M model trains 2-3× faster
and you'll iterate 2-3× faster on ideas.

Scale up AFTER you've proven the approach works.
A good 25M model tells you more than a broken 60M model.
```

---

## 14. Summary: The 7 Critical Fixes

```
Priority  Fix                                      Impact    Effort
─────────────────────────────────────────────────────────────────────
  P0      Continuous time (not T=32)                Huge      2 hours
  P0      ELBO importance weighting                 Large     1 hour
  P1      Span masking (not random token)           Large     2 hours
  P1      AdaLN-Zero timestep conditioning          Large     3 hours
  P1      RoPE positional encoding                  Medium    1 hour
  P2      Stochastic mask embeddings                Medium    1 hour
  P2      Block-wise generation for variable length Medium    3 hours
─────────────────────────────────────────────────────────────────────
  Total implementation time: ~13 hours

  Things to DROP for now:
  ✗ EcoHybrid (use baseline at T=512, revisit at T≥1024)
  ✗ All Section 9 ideas (premature)
  ✗ d=512 model size (start at d=384)
  ✗ Memory slot sweeps (no memory slots yet)
```

Build the v2 denoiser with these fixes, train for 5000 steps on Hindi Wikipedia on your T4, and generate 100 sentences. That result tells you whether this entire diffusion direction is viable for Indic languages. Everything else follows from that answer.