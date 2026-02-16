# Edge Diffusion LM (Primary + Backup)

This repo is now intentionally lean:

- `eco_hybrid.py`: **primary model** (edge-first diffusion denoiser)
- `model.py`: **backup model** (baseline bidirectional diffusion transformer)
- `train.py`: canonical training entrypoint for both models
- `sample.py`: canonical sampling/benchmark entrypoint for both models
- `colab_train_eco.py`: one-shot Colab bootstrap + training for primary model
- `EDGE_EFFICIENCY_THEORY.md`: math-grounded rationale + experiment protocol
- `RESEARCH_FILTERED_IDEAS.md`: kept/rejected ideas from papers + community scan

Legacy script names (`train_eco_hybrid.py`, `train_diffusion.py`, `sample_eco_hybrid.py`, `sample_diffusion.py`) are compatibility wrappers to the unified entrypoints.

## Train

Primary (recommended):

```bash
python -m train \
  --model primary \
  --data_dir data/processed \
  --output_dir checkpoints/eco_hybrid_hi \
  --edge_profile laptop \
  --max_steps 5000
```

Backup:

```bash
python -m train \
  --model backup \
  --data_dir data/processed \
  --output_dir checkpoints/diffusion_hi \
  --edge_profile laptop \
  --max_steps 5000
```

## Colab one-shot

```bash
python -m colab_train_eco --max_steps 500 --save_interval 100 --log_interval 10 --compile
```

If data is missing, this script auto-downloads Hindi text, trains SentencePiece, and writes:
- `data/processed/train.bin`
- `data/processed/meta.json`

## Sample (edge-focused)

```bash
python -m sample \
  --ckpt checkpoints/eco_hybrid_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --model primary \
  --prompt "mai kal bazaar gaya tha" \
  --max_new_tokens 128 \
  --edge_profile max \
  --blockwise \
  --frozen_context \
  --min_decode_layers 3
```

`--frozen_context` enables cached context-memory decoding for the primary model to reduce repeated full-context computation in blockwise mode.

## Benchmark decoding speed

```bash
python -m sample \
  --ckpt checkpoints/eco_hybrid_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --model primary \
  --prompt "namaste" \
  --max_new_tokens 128 \
  --blockwise \
  --frozen_context \
  --benchmark_runs 5 \
  --warmup_runs 1
```

Or run a structured mode comparison:

```bash
python -m edge_benchmark \
  --ckpt checkpoints/eco_hybrid_hi/latest.pt \
  --tokenizer data/tokenizer/aria_hindi.model \
  --model primary \
  --modes full block block_frozen \
  --runs_per_prompt 3
```
