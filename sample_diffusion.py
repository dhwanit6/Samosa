"""
Sample text from a trained diffusion language model checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from diffusion.config import DiffusionLMConfig
from diffusion.model import DiffusionLanguageModel


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_model(ckpt_path: Path, device: torch.device) -> DiffusionLanguageModel:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = DiffusionLMConfig(**ckpt["config"])
    model = DiffusionLanguageModel(config).to(device)
    model.load_state_dict(_normalize_state_dict_keys(ckpt["model"]))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Diffusion LM sampling")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--tokenizer", type=str, required=True, help="SentencePiece model path")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperature_end",
        type=float,
        default=None,
        help="Optional final-step temperature for annealing",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--eos_token_id", type=int, default=None)
    parser.add_argument("--blockwise", action="store_true")
    parser.add_argument("--block_size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    prompt_ids = sp.encode(args.prompt, out_type=int) if args.prompt else []
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = model.generate(
        prompt_ids=prompt,
        max_new_tokens=args.max_new_tokens,
        num_steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        temperature_end=args.temperature_end,
        eos_token_id=args.eos_token_id,
        blockwise=args.blockwise,
        block_size=args.block_size,
    )
    text = sp.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
