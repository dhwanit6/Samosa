# Diffusion LM (Low-Hardware Track)

This folder contains a practical masked diffusion language model for local hardware experiments.

## Why this exists

Autoregressive decoding is still the most mature path, but diffusion decoding can expose parallel token refinement and very high throughput potential. This implementation is an R&D track to evaluate that tradeoff for Indic-first models.

## Implementation choices

- **Discrete masked diffusion** over token IDs.
- **Bidirectional denoiser** (Transformer blocks, non-causal attention).
- **Cosine mask schedule** from `min_mask_rate` to `max_mask_rate`.
- **Confidence-based iterative unmasking** (MaskGIT-style).
- **Tied token embedding/head** for memory efficiency.
- **Reserved mask token**: `vocab_size = base_vocab + 1`, with mask id at `base_vocab`.

## Files

- `config.py`: model and diffusion hyperparameters.
- `model.py`: denoiser, corruption process, training loss, sampling loop.
- `eco_hybrid.py`: non-standard hybrid denoiser (local conv + memory-slot attention).
- `train_diffusion.py`: standalone trainer on `train.bin`.
- `train_eco_hybrid.py`: trainer for the eco-hybrid architecture.
- `sample_diffusion.py`: checkpoint sampling utility.
- `sample_eco_hybrid.py`: sampling utility for EcoHybrid checkpoints.
- `ECO_HYBRID_THEORY.md`: objective, complexity, and architectural rationale.
- `DIFFUSION_MODEL_DETAILED.md`: full design, training flow, limitations, and future theory backlog.
- `COLAB_ECO_RUNBOOK.md`: Colab run steps + 500-step success criteria.
- `colab_train_eco.py`: one-shot Colab runner with auto data setup.

## Quick start

Train:

```bash
cd train
python -m diffusion.train_diffusion \
  --data_dir data/processed \
  --output_dir checkpoints/diffusion_hi \
  --seq_len 512 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_steps 5000 \
  --time_mode continuous \
  --timestep_sampling stratified \
  --masking_strategy span
```

Train (recommended non-standard path):

```bash
cd train
python -m diffusion.train_eco_hybrid \
  --data_dir data/processed \
  --output_dir checkpoints/eco_hybrid_hi \
  --seq_len 512 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_steps 5000 \
  --time_mode continuous \
  --timestep_sampling stratified \
  --masking_strategy span
```

Colab one-shot (auto data setup):

```bash
cd train
python -m diffusion.colab_train_eco --max_steps 500 --save_interval 100 --log_interval 10 --compile
```

Run matched ablations (baseline vs Eco, discrete/continuous, token/span):

```bash
cd train
python -m diffusion.run_experiments \
  --data_dir data/processed \
  --output_root checkpoints/ablations \
  --max_steps 2000 \
  --d_model 384 \
  --n_layers 8 \
  --n_heads 8 \
  --d_ff 1024 \
  --execute
```

Sample:

```bash
cd train
python -m diffusion.sample_diffusion \
  --ckpt checkpoints/diffusion_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --prompt "mai kal bazaar gaya tha" \
  --max_new_tokens 128 \
  --steps 16
```

Sample (EcoHybrid):

```bash
cd train
python -m diffusion.sample_eco_hybrid \
  --ckpt checkpoints/eco_hybrid_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --prompt "mai kal bazaar gaya tha" \
  --max_new_tokens 128 \
  --steps 12
```

## Notes

- This is a research baseline, not a production-ready replacement for AR decoding.
- For now, keep AR model as product path and evaluate diffusion as a parallel speed/quality experiment.
- If compute is tight, start with `train_eco_hybrid.py` and reduce `memory_slots` or `d_model`.
