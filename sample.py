"""
Unified sampling entrypoint for diffusion checkpoints.

Supports:
1) primary model (EcoHybrid) checkpoints
2) backup model (baseline diffusion transformer) checkpoints
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

# Make unicode decoding output robust on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Support multiple repo layouts and invocation styles.
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from diffusion.eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM
    from diffusion.config import DiffusionLMConfig
    from diffusion.model import DiffusionLanguageModel
except ModuleNotFoundError:
    from eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM  # type: ignore
    from config import DiffusionLMConfig  # type: ignore
    from model import DiffusionLanguageModel  # type: ignore


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
    return state_dict


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
        args.frozen_context = True
        args.min_decode_layers = max(args.min_decode_layers, 4)
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
        args.frozen_context = True
        args.min_decode_layers = max(args.min_decode_layers, 3)


def _load_model(
    ckpt_path: Path,
    device: torch.device,
    model_type: str,
    quantize_dynamic: bool,
):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = _normalize_state_dict_keys(ckpt["model"])
    cfg_data = ckpt["config"]

    if model_type == "auto":
        model_type = str(ckpt.get("model_type", "primary"))
        if model_type not in {"primary", "backup"}:
            # Backward compatibility with older checkpoints.
            model_type = "primary" if "memory_slots" in cfg_data else "backup"

    if model_type == "primary":
        cfg = EcoHybridConfig(**cfg_data)
        model = EcoHybridDiffusionLM(cfg).to(device)
    else:
        cfg = DiffusionLMConfig(**cfg_data)
        model = DiffusionLanguageModel(cfg).to(device)

    model.load_state_dict(state)
    if quantize_dynamic:
        if device.type != "cpu":
            print("[WARN] Dynamic quantization is CPU-only. Ignoring --quantize_dynamic on non-CPU device.")
        else:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    return model, model_type


def main() -> None:
    p = argparse.ArgumentParser(description="Unified diffusion sampler")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--tokenizer", type=str, required=True, help="SentencePiece model path")
    p.add_argument("--model", type=str, default="auto", choices=["auto", "primary", "backup"])
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_end", type=float, default=None)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--eos_token_id", type=int, default=None)
    p.add_argument("--blockwise", action="store_true")
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument(
        "--frozen_context",
        action="store_true",
        help="Primary model only: cache prompt context as memory state for faster blockwise decode",
    )
    p.add_argument("--quantize_dynamic", action="store_true", help="Apply CPU int8 dynamic quantization")
    p.add_argument(
        "--min_decode_layers",
        type=int,
        default=0,
        help="Primary model only: start denoising with this many layers and ramp to full depth",
    )
    p.add_argument("--edge_profile", type=str, default="none", choices=["none", "balanced", "max"])
    p.add_argument(
        "--calibrated_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use probability-calibrated confidence in greedy decode path",
    )
    p.add_argument("--final_fill_threshold", type=int, default=16)
    p.add_argument("--benchmark_runs", type=int, default=1, help="Run N decode passes and report tok/s")
    p.add_argument("--warmup_runs", type=int, default=0, help="Warmup decode passes before benchmarking")
    args = p.parse_args()
    _apply_edge_profile(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_model(
        ckpt_path=Path(args.ckpt),
        device=device,
        model_type=args.model,
        quantize_dynamic=args.quantize_dynamic,
    )
    print(f"[sampler] model={model_type} | device={device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    prompt_ids = sp.encode(args.prompt, out_type=int) if args.prompt else []
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen_kwargs = dict(
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
    if model_type == "primary":
        gen_kwargs["frozen_context"] = args.frozen_context
        gen_kwargs["min_decode_layers"] = args.min_decode_layers

    for _ in range(max(args.warmup_runs, 0)):
        _ = model.generate(**gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()

    runs = max(1, int(args.benchmark_runs))
    timings = []
    out = None
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    assert out is not None
    if runs > 1:
        avg_s = sum(timings) / len(timings)
        tps = out.shape[1] / max(avg_s, 1e-8)
        print(
            f"[bench] runs={runs} | mean={avg_s:.3f}s | "
            f"p50={sorted(timings)[len(timings)//2]:.3f}s | tok/s={tps:,.1f}"
        )
    ids = out[0].tolist()
    vocab_n = int(sp.vocab_size())
    unk = int(sp.unk_id())
    bad = sum(1 for tok in ids if tok < 0 or tok >= vocab_n)
    if bad > 0:
        print(
            f"[WARN] {bad} generated token ids were outside tokenizer range [0, {vocab_n - 1}] "
            f"and were mapped to <unk>."
        )
        ids = [tok if 0 <= tok < vocab_n else unk for tok in ids]
    print(sp.decode(ids))


if __name__ == "__main__":
    main()
