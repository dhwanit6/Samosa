# Colab Runbook: EcoHybrid Diffusion (Auto Data + 500-Step Check)

This runbook is for fast validation in Colab with automatic Hindi data setup.

## 1. Colab Setup

1. In Colab: `Runtime -> Change runtime type -> GPU`.
2. Run:

```bash
!git clone <your-repo-url> Aria
%cd Aria
!pip install -r requirements.txt
```

## 2. Start One-Shot Training

From repo root:

```bash
%cd train
!python -m diffusion.colab_train_eco --max_steps 500 --save_interval 100 --log_interval 10 --compile
```

What this command does automatically:

1. Detects GPU and picks default `batch_size`, `grad_accum`, `seq_len`.
2. Downloads Hindi data if missing.
3. Trains tokenizer if missing.
4. Preprocesses to `data/processed/train.bin`.
5. Starts EcoHybrid training and saves checkpoints.

## 3. Minimal "Does It Work?" Criteria

### By step ~20

1. Loss is finite (no NaN/inf).
2. `mask` metric is in a sensible range (typically between `0.15` and `0.9` depending on sampled timesteps).
3. Throughput is stable (no repeated stalls).

### By step ~100

1. Loss should show downward trend from early steps.
2. Checkpoint files should exist under `train/checkpoints/eco_hybrid_hi/`.

### By step ~500 (minimum meaningful check)

1. Loss decreased materially from step 1 (rough target: >=10-20% relative drop; exact value depends on data mix/hardware).
2. Sampling produces coherent short continuations for Hindi/Hinglish prompts.
3. No mode collapse (not repeating the same few tokens across prompts).

## 4. Sampling at Step 500

```bash
!python -m diffusion.sample_eco_hybrid \
  --ckpt checkpoints/eco_hybrid_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --prompt "mai kal bazaar gaya tha" \
  --max_new_tokens 80 \
  --steps 12 \
  --temperature 1.0 \
  --top_k 50
```

Try multiple prompts:

1. `mai kal bazaar gaya tha aur`
2. `भारत में आज मौसम`
3. `yaar mujhe samajh nahi aa raha`
4. `मुझे एक अच्छी कहानी सुनाओ`

## 5. Recommended First Hyperparameter Set

Use defaults first:

1. `d_model=512`
2. `n_layers=10`
3. `n_heads=8`
4. `memory_slots=16`
5. `seq_len=512` (or auto from GPU)

Only reduce if OOM:

1. `--d_model 384`
2. `--n_layers 8`
3. `--memory_slots 8`
4. `--seq_len 384`

## 6. Common Failure Modes

### OOM

1. Reduce `seq_len` first.
2. Then reduce `batch_size`.
3. Then reduce `d_model` or `n_layers`.

### Loss not improving

1. Verify dataset built correctly (`data/processed/meta.json` exists and `train_tokens > 0`).
2. Run 500-1000 steps before judging (diffusion can be noisy step-to-step).
3. Try lower LR (`--lr 2e-4`) if unstable.

### Bad samples despite lower loss

1. Increase `--steps` at sampling (e.g., `16`).
2. Lower `temperature` (e.g., `0.8`).
3. Adjust `top_k` (e.g., `30-80`).

## 7. Next After 500 Steps

1. Continue to 2k-5k steps.
2. Compare against baseline diffusion model (`train_diffusion.py`) at matched parameter budget.
3. Track:
1. loss trend
2. tokens/sec
3. sample quality on fixed prompt set
4. Hindi/Hinglish script-mix robustness
