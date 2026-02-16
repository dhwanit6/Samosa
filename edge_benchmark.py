"""
Edge decoding benchmark for unified diffusion models.

Measures:
1) latency / tokens-per-second
2) pseudo log-likelihood of generated continuation under t=0 denoiser

The pseudo log-likelihood is a ranking proxy, not calibrated perplexity.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

# Support both:
# - python -m diffusion.edge_benchmark
# - python -m edge_benchmark
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


def _t0(cfg, batch_size: int, device: torch.device) -> torch.Tensor:
    if getattr(cfg, "time_mode", "discrete") == "continuous":
        return torch.zeros((batch_size,), dtype=torch.float32, device=device)
    return torch.zeros((batch_size,), dtype=torch.long, device=device)


@torch.inference_mode()
def _pseudo_logp(model, full_ids: torch.Tensor, prompt_len: int) -> float:
    if full_ids.shape[1] <= prompt_len:
        return float("nan")
    logits = model.forward(full_ids, _t0(model.cfg, full_ids.shape[0], full_ids.device))
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, full_ids.unsqueeze(-1)).squeeze(-1)
    cont_lp = tok_lp[:, prompt_len:]
    return float(cont_lp.mean().item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        p = Path(args.prompts_file)
        lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
        prompts.extend([line for line in lines if line])
    if not prompts:
        prompts = ["namaste", "mai kal bazaar gaya tha", "aaj mausam kaisa hai"]
    return prompts


def _mode_kwargs(mode: str) -> dict[str, bool]:
    if mode == "full":
        return {"blockwise": False, "frozen_context": False}
    if mode == "block":
        return {"blockwise": True, "frozen_context": False}
    if mode == "block_frozen":
        return {"blockwise": True, "frozen_context": True}
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser(description="Edge decoding benchmark")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--model", type=str, default="auto", choices=["auto", "primary", "backup"])
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--prompts_file", type=str, default="")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["full", "block", "block_frozen"],
        choices=["full", "block", "block_frozen"],
    )
    p.add_argument("--runs_per_prompt", type=int, default=3)
    p.add_argument("--warmup_runs", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--temperature_end", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument(
        "--min_decode_layers",
        type=int,
        default=0,
        help="Primary model only: start denoising with this many layers and ramp to full depth",
    )
    p.add_argument(
        "--calibrated_confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use probability-calibrated confidence in greedy decode path",
    )
    p.add_argument("--final_fill_threshold", type=int, default=16)
    p.add_argument("--quantize_dynamic", action="store_true")
    p.add_argument("--out_json", type=str, default="benchmarks/edge_benchmark.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = _load_model(
        ckpt_path=Path(args.ckpt),
        device=device,
        model_type=args.model,
        quantize_dynamic=args.quantize_dynamic,
    )
    print(f"[bench] model={model_type} | device={device}")

    modes = list(args.modes)
    if model_type != "primary":
        modes = [m for m in modes if m != "block_frozen"]
        if "block_frozen" in args.modes:
            print("[WARN] block_frozen is primary-only. Skipping for backup model.")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    prompts = _read_prompts(args)

    results: dict[str, dict] = {}
    with torch.inference_mode():
        for mode in modes:
            cfg = _mode_kwargs(mode)
            latencies: list[float] = []
            tok_s: list[float] = []
            pll: list[float] = []
            generated_lens: list[int] = []

            for prompt_text in prompts:
                prompt_ids = sp.encode(prompt_text, out_type=int)
                prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                gen_kwargs = dict(
                    prompt_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_steps=args.steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    temperature_end=args.temperature_end,
                    eos_token_id=None,
                    blockwise=cfg["blockwise"],
                    block_size=args.block_size,
                    calibrated_confidence=args.calibrated_confidence,
                    final_fill_threshold=args.final_fill_threshold,
                )
                if model_type == "primary":
                    gen_kwargs["frozen_context"] = cfg["frozen_context"]
                    gen_kwargs["min_decode_layers"] = args.min_decode_layers

                for _ in range(max(0, args.warmup_runs)):
                    _ = model.generate(**gen_kwargs)
                    _sync(device)

                for _ in range(max(1, args.runs_per_prompt)):
                    _sync(device)
                    t0 = time.perf_counter()
                    out = model.generate(**gen_kwargs)
                    _sync(device)
                    dt = max(time.perf_counter() - t0, 1e-8)

                    n_new = int(out.shape[1])
                    latencies.append(dt)
                    tok_s.append(n_new / dt)
                    generated_lens.append(n_new)

                    full = torch.cat([prompt, out], dim=1)
                    pll.append(_pseudo_logp(model, full, prompt_len=prompt.shape[1]))

            mean_latency = float(sum(latencies) / len(latencies))
            mean_tok_s = float(sum(tok_s) / len(tok_s))
            p50_latency = float(statistics.median(latencies))
            mean_pll = float(sum(pll) / len(pll))
            mean_len = float(sum(generated_lens) / len(generated_lens))

            results[mode] = {
                "runs": len(latencies),
                "mean_latency_s": mean_latency,
                "p50_latency_s": p50_latency,
                "mean_tok_s": mean_tok_s,
                "mean_generated_tokens": mean_len,
                "mean_pseudo_logp": mean_pll,
            }
            print(
                f"[{mode}] runs={len(latencies)} | mean={mean_latency:.3f}s | "
                f"tok/s={mean_tok_s:,.1f} | pseudo_logp={mean_pll:.4f}"
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_type,
        "device": str(device),
        "prompts": prompts,
        "args": vars(args),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
