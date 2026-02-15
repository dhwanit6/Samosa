"""
Sample text from an EcoHybrid diffusion checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

# Support multiple repo layouts and invocation styles:
# - python -m diffusion.sample_eco_hybrid
# - python -m sample_eco_hybrid
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from diffusion.eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM
except ModuleNotFoundError:
    from eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM  # type: ignore


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_model(
    ckpt_path: Path, device: torch.device, quantize_dynamic: bool = False
) -> EcoHybridDiffusionLM:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = EcoHybridConfig(**ckpt["config"])
    model = EcoHybridDiffusionLM(cfg).to(device)
    model.load_state_dict(_normalize_state_dict_keys(ckpt["model"]))
    if quantize_dynamic:
        if device.type != "cpu":
            print("[WARN] Dynamic quantization is CPU-only. Ignoring --quantize_dynamic on non-CPU device.")
        else:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    return model


def _apply_edge_profile(args: argparse.Namespace) -> None:
    if args.edge_profile == "none":
        return
    if args.edge_profile == "balanced":
        args.steps = min(args.steps, 8)
        args.temperature = min(args.temperature, 0.8)
        if args.temperature_end is None:
            args.temperature_end = 0.6
        args.top_k = 0
        args.blockwise = True
        args.block_size = args.block_size or 64
        args.calibrated_confidence = False
        args.final_fill_threshold = max(args.final_fill_threshold, 16)
    elif args.edge_profile == "max":
        args.steps = min(args.steps, 6)
        args.temperature = 0.0
        args.temperature_end = None
        args.top_k = 0
        args.blockwise = True
        args.block_size = args.block_size or 64
        args.calibrated_confidence = False
        args.final_fill_threshold = max(args.final_fill_threshold, 32)
        args.quantize_dynamic = True


def main():
    parser = argparse.ArgumentParser(description="Sample EcoHybrid diffusion model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--tokenizer", type=str, required=True, help="SentencePiece model path")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=12)
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
    parser.add_argument("--quantize_dynamic", action="store_true", help="Apply CPU int8 dynamic quantization")
    parser.add_argument(
        "--edge_profile",
        type=str,
        default="none",
        choices=["none", "balanced", "max"],
        help="Edge-focused decoding presets",
    )
    parser.add_argument(
        "--calibrated_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use probability-calibrated confidence in greedy decode path",
    )
    parser.add_argument(
        "--final_fill_threshold",
        type=int,
        default=16,
        help="Auto-fill remaining masks when count is below this threshold",
    )
    args = parser.parse_args()
    _apply_edge_profile(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device, quantize_dynamic=args.quantize_dynamic)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    prompt_ids = sp.encode(args.prompt, out_type=int) if args.prompt else []
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = model.generate(
        prompt_ids=prompt,
        max_new_tokens=args.max_new_tokens,
        num_steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        temperature_end=args.temperature_end,
        eos_token_id=args.eos_token_id,
        blockwise=args.blockwise,
        block_size=args.block_size,
        calibrated_confidence=args.calibrated_confidence,
        final_fill_threshold=args.final_fill_threshold,
    )
    print(sp.decode(gen[0].tolist()))


if __name__ == "__main__":
    main()
