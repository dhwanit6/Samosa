"""
Reasoning layer for diffusion decoding.

This module adds an inference-time deliberation pass:
1) generate multiple candidate continuations
2) score each candidate with model-grounded signals
3) pick the highest-scoring candidate

The goal is higher answer reliability without retraining the base model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ReasoningLayerConfig:
    num_candidates: int = 4
    temperature_spread: float = 0.35
    confidence_weight: float = 0.25
    consensus_weight: float = 0.20
    repetition_weight: float = 0.20
    mask_penalty_weight: float = 0.50
    preference_weight: float = 0.35


def _cfg_from_model(model):
    return getattr(model, "cfg", getattr(model, "config", None))


def _t0(cfg, batch_size: int, device: torch.device) -> torch.Tensor:
    time_mode = getattr(cfg, "time_mode", "discrete")
    if time_mode == "continuous":
        return torch.zeros((batch_size,), dtype=torch.float32, device=device)
    return torch.zeros((batch_size,), dtype=torch.long, device=device)


def _zscore(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _repetition_rate(ids: torch.Tensor, pad_id: int) -> float:
    valid = ids[ids.ne(pad_id)]
    if valid.numel() == 0:
        return 1.0
    unique = valid.unique().numel()
    return 1.0 - float(unique) / float(valid.numel())


def _jaccard(a: torch.Tensor, b: torch.Tensor, pad_id: int) -> float:
    aset = set(a[a.ne(pad_id)].tolist())
    bset = set(b[b.ne(pad_id)].tolist())
    if not aset and not bset:
        return 1.0
    union = len(aset | bset)
    if union == 0:
        return 0.0
    return float(len(aset & bset)) / float(union)


@torch.inference_mode()
def candidate_diagnostics(
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> dict[str, float]:
    """
    Compute model-grounded diagnostics for a generated continuation.
    """
    if prompt_ids.shape[0] != 1 or continuation_ids.shape[0] != 1:
        raise ValueError("candidate_diagnostics currently supports batch size 1")

    model_cfg = _cfg_from_model(model)
    if model_cfg is None:
        raise RuntimeError("Could not resolve model config for reasoning diagnostics")
    pad_id = int(getattr(model_cfg, "pad_token_id", 0))
    mask_id = int(getattr(model_cfg, "mask_token_id", -1))

    full = torch.cat([prompt_ids, continuation_ids], dim=1)
    logits = model.forward(full, _t0(model_cfg, full.shape[0], full.device))
    logp = F.log_softmax(logits.float(), dim=-1)

    prompt_len = prompt_ids.shape[1]
    cont_logp = logp[:, prompt_len:, :]
    cont_ids = continuation_ids.unsqueeze(-1)
    pll = float(cont_logp.gather(-1, cont_ids).squeeze(-1).mean().item())
    conf = float(cont_logp.exp().amax(dim=-1).mean().item())
    rep = _repetition_rate(continuation_ids[0], pad_id=pad_id)
    mask_rate = float(continuation_ids.eq(mask_id).float().mean().item()) if mask_id >= 0 else 0.0

    return {
        "pll": pll,
        "confidence": conf,
        "repetition_rate": rep,
        "mask_rate": mask_rate,
    }


@torch.inference_mode()
def generate_with_reasoning(
    model,
    prompt_ids: torch.Tensor,
    generate_kwargs: dict,
    cfg: ReasoningLayerConfig,
    seed_candidates: list[tuple[torch.Tensor, float]] | None = None,
    extra_score_fn=None,
) -> tuple[torch.Tensor, dict]:
    """
    Returns:
        best_ids: [1, T]
        report: reasoning diagnostics
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("Reasoning layer currently supports batch size 1")

    n = max(1, int(cfg.num_candidates))
    base_temp = float(generate_kwargs.get("temperature", 1.0))
    spread = max(0.0, float(cfg.temperature_spread))
    temperatures: list[float] = []
    for i in range(n):
        alpha = i / max(n - 1, 1)
        if base_temp > 0.0:
            scale = (1.0 - spread * 0.5) + spread * alpha
            t = max(0.0, base_temp * scale)
        else:
            # Preserve deterministic mode for first candidate; inject small diversity after.
            t = 0.0 if i == 0 else 0.15 + 0.20 * alpha
        temperatures.append(t)

    model_cfg = _cfg_from_model(model)
    if model_cfg is None:
        raise RuntimeError("Could not resolve model config for reasoning layer")
    pad_id = int(getattr(model_cfg, "pad_token_id", 0))

    candidates: list[torch.Tensor] = []
    candidate_temps: list[float] = []
    pll_vals: list[float] = []
    conf_vals: list[float] = []
    rep_vals: list[float] = []
    mask_vals: list[float] = []
    extra_vals: list[float] = []

    if seed_candidates:
        for c, temp in seed_candidates:
            candidates.append(c)
            candidate_temps.append(float(temp))

    temp_cursor = 0
    while len(candidates) < n:
        t = temperatures[temp_cursor]
        temp_cursor += 1
        local_kwargs = dict(generate_kwargs)
        local_kwargs["temperature"] = t
        out = model.generate(**local_kwargs)
        candidates.append(out)
        candidate_temps.append(float(t))

    for out in candidates:
        d = candidate_diagnostics(model, prompt_ids=prompt_ids, continuation_ids=out)
        pll_vals.append(float(d["pll"]))
        conf_vals.append(float(d["confidence"]))
        rep_vals.append(float(d["repetition_rate"]))
        mask_vals.append(float(d["mask_rate"]))
        if extra_score_fn is not None:
            try:
                extra_vals.append(float(extra_score_fn(out)))
            except Exception:
                extra_vals.append(0.0)
        else:
            extra_vals.append(0.0)

    # Self-consistency via pairwise token-set overlap.
    cons_vals = [0.0 for _ in range(n)]
    if n > 1:
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                s += _jaccard(candidates[i][0], candidates[j][0], pad_id=pad_id)
            cons_vals[i] = s / max(n - 1, 1)

    z_pll = _zscore(pll_vals)
    z_conf = _zscore(conf_vals)
    z_cons = _zscore(cons_vals)
    z_rep = _zscore(rep_vals)
    z_mask = _zscore(mask_vals)
    z_extra = _zscore(extra_vals)

    scores: list[float] = []
    for i in range(n):
        score = (
            z_pll[i]
            + float(cfg.confidence_weight) * z_conf[i]
            + float(cfg.consensus_weight) * z_cons[i]
            - float(cfg.repetition_weight) * z_rep[i]
            - float(cfg.mask_penalty_weight) * z_mask[i]
            + float(cfg.preference_weight) * z_extra[i]
        )
        scores.append(score)

    best_idx = max(range(n), key=lambda i: scores[i])
    report = {
        "best_index": int(best_idx),
        "best_score": float(scores[best_idx]),
        "candidates": [
            {
                "idx": i,
                "temperature": float(candidate_temps[i]),
                "score": float(scores[i]),
                "pll": float(pll_vals[i]),
                "confidence": float(conf_vals[i]),
                "consensus": float(cons_vals[i]),
                "repetition_rate": float(rep_vals[i]),
                "mask_rate": float(mask_vals[i]),
                "preference": float(extra_vals[i]),
            }
            for i in range(n)
        ],
    }
    return candidates[best_idx], report
