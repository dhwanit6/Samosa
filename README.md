# Edge Diffusion LM (Gujarati + Gujlish First)

## Repository Layout

Core runtime:
- `eco_hybrid.py` (primary model, edge-first)
- `model.py` (backup model)
- `config.py`
- `train.py`
- `sample.py`
- `reasoning_layer.py` (multi-candidate reasoning + verifier + gating)
- `safety_layer.py` (prompt/output safety checks)
- `preference_layer.py` (warmth/harmony/ecology preference scoring)

Utility runners:
- `tools/colab_train_eco.py`
- `tools/run_experiments.py`
- `tools/edge_benchmark.py`
- `tools/build_alignment_dataset.py`

Documentation:
- `docs/claude_opinion.md`
- `docs/ECO_HYBRID_THEORY.md`
- `docs/EDGE_EFFICIENCY_THEORY.md`
- `docs/DIFFUSION_MODEL_DETAILED.md`
- `docs/RESEARCH_FILTERED_IDEAS.md`
- `docs/COLAB_ECO_RUNBOOK.md`

Compatibility wrappers remain at repo root so old commands still work:
- `colab_train_eco.py`, `run_experiments.py`, `edge_benchmark.py`, `build_alignment_dataset.py`
- `train_eco_hybrid.py`, `train_diffusion.py`, `sample_eco_hybrid.py`, `sample_diffusion.py`

## Train

Primary:

```bash
python -m train \
  --model primary \
  --data_dir data/processed \
  --output_dir checkpoints/eco_hybrid_gu \
  --edge_profile laptop \
  --max_steps 5000
```

## Colab One-Shot (Gujarati + Roman Gujarati + Gujlish)

```bash
python -m tools.colab_train_eco \
  --language gu \
  --include_romanized \
  --include_gujlish \
  --quality_profile strict \
  --max_steps 500 \
  --save_interval 100 \
  --log_interval 10 \
  --compile
```

Recommended stabilization knobs for first 5k steps:
- `--lr_schedule cosine --warmup_steps 300 --min_lr_ratio 0.10`
- `--mask_curriculum_steps 1500 --start_min_mask_rate 0.02 --start_max_mask_rate 0.40`

If data is missing, it bootstraps corpus + tokenizer and writes:
- `data/processed/train.bin`
- `data/processed/meta.json`

## Fast Sampling

```bash
python -m sample \
  --ckpt checkpoints/eco_hybrid_gu/latest.pt \
  --tokenizer data/tokenizer/aria_gu.model \
  --model primary \
  --prompt "kem cho? aaje su plan che?" \
  --max_new_tokens 128 \
  --edge_profile max \
  --blockwise \
  --frozen_context \
  --min_decode_layers 3
```

## Reasoning Layer

Always-on reasoning:

```bash
python -m sample \
  --ckpt checkpoints/eco_hybrid_gu/latest.pt \
  --tokenizer data/tokenizer/aria_gu.model \
  --model primary \
  --prompt "aa problem nu best solution su che?" \
  --reasoning \
  --reasoning_candidates 6 \
  --reasoning_verbose
```

Confidence-gated reasoning (recommended default):

```bash
python -m sample \
  --ckpt checkpoints/eco_hybrid_gu/latest.pt \
  --tokenizer data/tokenizer/aria_gu.model \
  --model primary \
  --prompt "mane career decision ma madad kar" \
  --reasoning \
  --reasoning_gate \
  --reasoning_gate_min_confidence 0.62 \
  --reasoning_gate_max_repetition 0.55 \
  --reasoning_gate_min_preference 0.00 \
  --safety_layer
```

This keeps latency close to fast-path when base confidence is good, and escalates to multi-candidate reasoning only when needed.

## Alignment Dataset Build (SFT + Preference + Safety)

```bash
python -m tools.build_alignment_dataset \
  --corpus_path data/raw/gu_bootstrap.txt \
  --out_dir data/alignment \
  --val_ratio 0.05
```

Outputs:
- `data/alignment/sft_train.jsonl`
- `data/alignment/sft_val.jsonl`
- `data/alignment/pref_train.jsonl`
- `data/alignment/pref_val.jsonl`

## Benchmark

```bash
python -m tools.edge_benchmark \
  --ckpt checkpoints/eco_hybrid_gu/latest.pt \
  --tokenizer data/tokenizer/aria_gu.model \
  --model primary \
  --modes full block block_frozen \
  --runs_per_prompt 3
```
