"""
Unified sampling entrypoint for diffusion checkpoints.

Supports:
1) primary model (EcoHybrid) checkpoints
2) backup model (baseline diffusion transformer) checkpoints
"""
from __future__ import annotations

import argparse
import inspect
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
    from diffusion.reasoning_layer import (
        ReasoningLayerConfig,
        candidate_diagnostics,
        generate_with_reasoning,
    )
    from diffusion.safety_layer import SafetyConfig, evaluate_text, safe_refusal
    from diffusion.preference_layer import PreferenceConfig, score_preference_text
except ModuleNotFoundError:
    from eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM  # type: ignore
    from config import DiffusionLMConfig  # type: ignore
    from model import DiffusionLanguageModel  # type: ignore
    from reasoning_layer import (  # type: ignore
        ReasoningLayerConfig,
        candidate_diagnostics,
        generate_with_reasoning,
    )
    from safety_layer import SafetyConfig, evaluate_text, safe_refusal  # type: ignore
    from preference_layer import PreferenceConfig, score_preference_text  # type: ignore


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
    return state_dict


def _build_config(config_cls, cfg_data: dict, model_type: str):
    try:
        return config_cls(**cfg_data)
    except TypeError:
        # Backward/forward compatibility: ignore unknown checkpoint keys.
        params = inspect.signature(config_cls).parameters
        filtered = {k: v for k, v in cfg_data.items() if k in params}
        dropped = sorted(k for k in cfg_data.keys() if k not in params)
        if dropped:
            print(
                f"[WARN] Dropping {len(dropped)} unknown config keys for {model_type} model: "
                + ", ".join(dropped)
            )
        return config_cls(**filtered)


def _model_cfg(model):
    return getattr(model, "cfg", getattr(model, "config", None))


def _sanitize_prompt_ids(
    prompt_ids: list[int],
    model_vocab_size: int,
    replacement_id: int,
) -> tuple[list[int], int]:
    if model_vocab_size <= 0:
        return prompt_ids, 0
    fixed: list[int] = []
    replaced = 0
    for tid in prompt_ids:
        if 0 <= tid < model_vocab_size:
            fixed.append(tid)
        else:
            fixed.append(replacement_id)
            replaced += 1
    return fixed, replaced


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
        cfg = _build_config(EcoHybridConfig, cfg_data, model_type="primary")
        model = EcoHybridDiffusionLM(cfg).to(device)
    else:
        cfg = _build_config(DiffusionLMConfig, cfg_data, model_type="backup")
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
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="Run multi-candidate reasoning layer (self-consistency + verifier scoring)",
    )
    p.add_argument("--reasoning_candidates", type=int, default=4)
    p.add_argument("--reasoning_temperature_spread", type=float, default=0.35)
    p.add_argument("--reasoning_confidence_weight", type=float, default=0.25)
    p.add_argument("--reasoning_consensus_weight", type=float, default=0.20)
    p.add_argument("--reasoning_repetition_weight", type=float, default=0.20)
    p.add_argument("--reasoning_mask_penalty_weight", type=float, default=0.50)
    p.add_argument("--reasoning_preference_weight", type=float, default=0.35)
    p.add_argument(
        "--reasoning_gate",
        action="store_true",
        help="Run fast base decode first; trigger reasoning only if diagnostics indicate risk",
    )
    p.add_argument(
        "--reasoning_gate_min_confidence",
        type=float,
        default=0.62,
        help="Trigger reasoning if base confidence falls below this value",
    )
    p.add_argument(
        "--reasoning_gate_max_repetition",
        type=float,
        default=0.55,
        help="Trigger reasoning if base repetition rate exceeds this value",
    )
    p.add_argument(
        "--reasoning_gate_max_mask_rate",
        type=float,
        default=0.00,
        help="Trigger reasoning if base output still contains mask tokens above this rate",
    )
    p.add_argument(
        "--reasoning_gate_min_pll",
        type=float,
        default=None,
        help="Optional: trigger reasoning if base pseudo-logp is below this threshold",
    )
    p.add_argument(
        "--reasoning_gate_min_preference",
        type=float,
        default=None,
        help="Optional: trigger reasoning if base preference score is below this threshold",
    )
    p.add_argument(
        "--reasoning_verbose",
        action="store_true",
        help="Print full per-candidate reasoning diagnostics",
    )
    p.add_argument(
        "--preference_alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable preference scoring for warmth/harmony/ecology in reranking",
    )
    p.add_argument("--pref_warmth_weight", type=float, default=0.35)
    p.add_argument("--pref_humility_weight", type=float, default=0.20)
    p.add_argument("--pref_harmony_weight", type=float, default=0.20)
    p.add_argument("--pref_ecology_weight", type=float, default=0.15)
    p.add_argument("--pref_toxicity_penalty_weight", type=float, default=0.40)
    p.add_argument(
        "--safety_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt/output safety checks",
    )
    p.add_argument("--safety_block_threshold", type=float, default=0.70)
    p.add_argument("--safety_monitor_threshold", type=float, default=0.35)
    p.add_argument(
        "--safety_language_mode",
        type=str,
        default="gujlish",
        choices=["gujarati", "roman", "gujlish"],
    )
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
    model_cfg = _model_cfg(model)
    if model_cfg is None:
        raise RuntimeError("Could not resolve model config after loading checkpoint")
    model_vocab_n = int(getattr(model_cfg, "vocab_size", 0))
    tok_vocab_n = int(sp.vocab_size())
    if model_vocab_n > 0 and tok_vocab_n != model_vocab_n:
        print(
            f"[WARN] tokenizer vocab ({tok_vocab_n}) != model vocab ({model_vocab_n}). "
            "Out-of-range prompt ids will be mapped to a safe token."
        )

    prompt_ids = sp.encode(args.prompt, out_type=int) if args.prompt else []
    tok_unk = int(sp.unk_id())
    replacement_id = tok_unk if 0 <= tok_unk < max(model_vocab_n, 1) else (1 if model_vocab_n > 1 else 0)
    prompt_ids, replaced = _sanitize_prompt_ids(
        prompt_ids=prompt_ids,
        model_vocab_size=model_vocab_n,
        replacement_id=replacement_id,
    )
    if replaced > 0:
        print(f"[WARN] mapped {replaced} prompt token ids outside model vocab to id {replacement_id}.")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    safety_cfg = SafetyConfig(
        block_threshold=float(args.safety_block_threshold),
        monitor_threshold=float(args.safety_monitor_threshold),
    )
    if args.safety_layer and args.prompt:
        prompt_decision = evaluate_text(args.prompt, safety_cfg)
        if prompt_decision.blocked:
            cats = ",".join(prompt_decision.categories) if prompt_decision.categories else "unknown"
            print(
                f"[safety] blocked prompt | score={prompt_decision.score:.3f} "
                f"| categories={cats}"
            )
            print(safe_refusal(args.safety_language_mode))
            return

    pref_cfg = PreferenceConfig(
        warmth_weight=float(args.pref_warmth_weight),
        humility_weight=float(args.pref_humility_weight),
        harmony_weight=float(args.pref_harmony_weight),
        ecology_weight=float(args.pref_ecology_weight),
        toxicity_penalty_weight=float(args.pref_toxicity_penalty_weight),
    )

    vocab_n = int(sp.vocab_size())
    unk = int(sp.unk_id())

    def _decode_ids(ids_tensor: torch.Tensor) -> tuple[str, int]:
        ids = ids_tensor[0].tolist()
        bad = sum(1 for tok in ids if tok < 0 or tok >= vocab_n)
        if bad > 0:
            ids = [tok if 0 <= tok < vocab_n else unk for tok in ids]
        return sp.decode(ids), bad

    def _preference_extra_score(ids_tensor: torch.Tensor) -> float:
        text, _ = _decode_ids(ids_tensor)
        return float(score_preference_text(text, pref_cfg))

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

    reasoning_cfg = None
    if args.reasoning:
        reasoning_cfg = ReasoningLayerConfig(
            num_candidates=max(1, int(args.reasoning_candidates)),
            temperature_spread=max(0.0, float(args.reasoning_temperature_spread)),
            confidence_weight=float(args.reasoning_confidence_weight),
            consensus_weight=float(args.reasoning_consensus_weight),
            repetition_weight=float(args.reasoning_repetition_weight),
            mask_penalty_weight=float(args.reasoning_mask_penalty_weight),
            preference_weight=float(args.reasoning_preference_weight),
        )
        if args.benchmark_runs > 1:
            print(
                f"[reasoning] enabled with {reasoning_cfg.num_candidates} candidates "
                f"(benchmark run cost scales by ~x{reasoning_cfg.num_candidates})"
            )

    def _run_once() -> tuple[torch.Tensor, dict | None]:
        if reasoning_cfg is None:
            out_ids = model.generate(**gen_kwargs)
            rep = None
            if args.preference_alignment:
                rep = {"preference_only": True, "preference": _preference_extra_score(out_ids)}
            return out_ids, rep

        extra_score_fn = _preference_extra_score if args.preference_alignment else None
        if not args.reasoning_gate:
            out_ids, report = generate_with_reasoning(
                model=model,
                prompt_ids=prompt,
                generate_kwargs=gen_kwargs,
                cfg=reasoning_cfg,
                extra_score_fn=extra_score_fn,
            )
            return out_ids, report

        base_out = model.generate(**gen_kwargs)
        base_diag = candidate_diagnostics(
            model=model,
            prompt_ids=prompt,
            continuation_ids=base_out,
        )
        base_pref = _preference_extra_score(base_out) if args.preference_alignment else 0.0
        trigger = (
            base_diag["confidence"] < float(args.reasoning_gate_min_confidence)
            or base_diag["repetition_rate"] > float(args.reasoning_gate_max_repetition)
            or base_diag["mask_rate"] > float(args.reasoning_gate_max_mask_rate)
        )
        if args.reasoning_gate_min_pll is not None:
            trigger = trigger or (base_diag["pll"] < float(args.reasoning_gate_min_pll))
        if args.reasoning_gate_min_preference is not None:
            trigger = trigger or (base_pref < float(args.reasoning_gate_min_preference))

        gate_report = {
            "gated": True,
            "triggered": bool(trigger),
            "base": {
                **{k: float(v) for k, v in base_diag.items()},
                "preference": float(base_pref),
            },
            "thresholds": {
                "min_confidence": float(args.reasoning_gate_min_confidence),
                "max_repetition": float(args.reasoning_gate_max_repetition),
                "max_mask_rate": float(args.reasoning_gate_max_mask_rate),
                "min_pll": (
                    None if args.reasoning_gate_min_pll is None else float(args.reasoning_gate_min_pll)
                ),
                "min_preference": (
                    None
                    if args.reasoning_gate_min_preference is None
                    else float(args.reasoning_gate_min_preference)
                ),
            },
        }
        if not trigger:
            gate_report["reason"] = "base_candidate_passed_gate"
            return base_out, gate_report

        out_ids, report = generate_with_reasoning(
            model=model,
            prompt_ids=prompt,
            generate_kwargs=gen_kwargs,
            cfg=reasoning_cfg,
            seed_candidates=[(base_out, float(gen_kwargs["temperature"]))],
            extra_score_fn=extra_score_fn,
        )
        report["gate"] = gate_report
        return out_ids, report

    for _ in range(max(args.warmup_runs, 0)):
        _run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()

    runs = max(1, int(args.benchmark_runs))
    timings = []
    out = None
    reason_report = None
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, reason_report = _run_once()
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
    if reason_report is not None:
        if bool(reason_report.get("gated", False)):
            base = reason_report["base"]
            print(
                "[reasoning-gate] "
                f"triggered={reason_report['triggered']} | "
                f"base_conf={base['confidence']:.4f} | "
                f"base_rep={base['repetition_rate']:.4f} | "
                f"base_mask={base['mask_rate']:.4f} | "
                f"base_pll={base['pll']:.4f} | "
                f"base_pref={base['preference']:.4f}"
            )
            if not reason_report["triggered"]:
                print("[reasoning-gate] base candidate accepted; skipped multi-candidate reasoning.")
        elif bool(reason_report.get("preference_only", False)):
            print(f"[preference] score={reason_report['preference']:.4f}")
        else:
            if "gate" in reason_report:
                gate = reason_report["gate"]
                base = gate["base"]
                print(
                    "[reasoning-gate] "
                    f"triggered={gate['triggered']} | "
                    f"base_conf={base['confidence']:.4f} | "
                    f"base_rep={base['repetition_rate']:.4f} | "
                    f"base_mask={base['mask_rate']:.4f} | "
                    f"base_pll={base['pll']:.4f} | "
                    f"base_pref={base['preference']:.4f}"
                )
            print(
                f"[reasoning] selected candidate {reason_report['best_index']} "
                f"with score {reason_report['best_score']:.3f}"
            )
        if args.reasoning_verbose and ("candidates" in reason_report):
            for c in reason_report["candidates"]:
                print(
                    "  "
                    f"idx={c['idx']} temp={c['temperature']:.3f} score={c['score']:.3f} "
                    f"pll={c['pll']:.4f} conf={c['confidence']:.4f} "
                    f"cons={c['consensus']:.4f} rep={c['repetition_rate']:.4f} "
                    f"mask={c['mask_rate']:.4f} pref={c['preference']:.4f}"
                )

    text_out, bad = _decode_ids(out)
    if bad > 0:
        print(
            f"[WARN] {bad} generated token ids were outside tokenizer range [0, {vocab_n - 1}] "
            f"and were mapped to <unk>."
        )

    if args.safety_layer:
        out_decision = evaluate_text(text_out, safety_cfg)
        if out_decision.blocked:
            cats = ",".join(out_decision.categories) if out_decision.categories else "unknown"
            print(
                f"[safety] blocked generated output | score={out_decision.score:.3f} "
                f"| categories={cats}"
            )
            print(safe_refusal(args.safety_language_mode))
            return
        if out_decision.score >= safety_cfg.monitor_threshold:
            cats = ",".join(out_decision.categories) if out_decision.categories else "none"
            print(
                f"[safety] monitor warning | score={out_decision.score:.3f} "
                f"| categories={cats}"
            )

    print(text_out)


if __name__ == "__main__":
    main()
