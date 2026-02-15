"""
Efficient masked diffusion language model for local hardware.

Core idea:
- Train by masking random tokens at a timestep-dependent rate.
- Predict original tokens from partially masked sequences.
- Generate by iterative confidence-based unmasking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import DiffusionLMConfig
except ImportError:
    from config import DiffusionLMConfig  # type: ignore


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    timesteps: [B] int or float tensor
    returns: [B, dim]
    """
    half = dim // 2
    device = timesteps.device
    exponent = -math.log(10_000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    freqs = torch.exp(exponent)  # [half]
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * norm).to(in_dtype) * self.weight


class DenoiserBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class DiffusionOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    masked_ratio: float | None = None


class DiffusionLanguageModel(nn.Module):
    """
    Discrete masked diffusion language model.

    Training:
        x_t = mask(x_0, rate(t))
        predict x_0 from x_t

    Inference:
        start from [MASK] and iteratively reveal high-confidence tokens.
    """

    def __init__(self, config: DiffusionLMConfig):
        super().__init__()
        self.config = config
        c = config

        self.token_embed = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_embed = nn.Embedding(c.max_seq_len, c.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(c.d_model, c.d_model),
            nn.SiLU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.blocks = nn.ModuleList(
            [DenoiserBlock(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)]
        )
        self.final_norm = RMSNorm(c.d_model)

        # Tied embedding head for parameter efficiency.
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=self.config.init_std)
        nn.init.normal_(self.pos_embed.weight, std=self.config.init_std)
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalized_t(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.config.time_mode == "continuous":
            return timesteps.float().clamp(0.0, 1.0)
        return timesteps.float() / max(self.config.timesteps - 1, 1)

    def _sample_train_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.config.time_mode == "discrete":
            return torch.randint(
                low=0,
                high=self.config.timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        if self.config.timestep_sampling == "stratified":
            base = (
                torch.arange(batch_size, device=device, dtype=torch.float32)
                + torch.rand(batch_size, device=device)
            ) / max(batch_size, 1)
            perm = torch.randperm(batch_size, device=device)
            return base[perm]
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def _inference_timesteps(
        self,
        reverse_step: int,
        total_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        frac = reverse_step / max(total_steps - 1, 1)
        if self.config.time_mode == "continuous":
            return torch.full((batch_size,), float(frac), device=device, dtype=torch.float32)
        t_idx = int(round(frac * (self.config.timesteps - 1)))
        return torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

    def _step_temperature(
        self, reverse_step: int, total_steps: int, temperature: float, temperature_end: float | None
    ) -> float:
        if temperature_end is None:
            return temperature
        progress = 1.0 - (reverse_step / max(total_steps - 1, 1))
        return temperature + (temperature_end - temperature) * progress

    def _sample_spans(
        self,
        valid_mask: torch.Tensor,
        target: int,
    ) -> torch.Tensor:
        seq_len = valid_mask.shape[0]
        masked = torch.zeros_like(valid_mask, dtype=torch.bool)
        if target <= 0:
            return masked

        mean_span = max(float(self.config.mean_span_length), 1.0)
        p = min(1.0, 1.0 / mean_span)
        remaining = target
        # A small over-sampling factor avoids many fallback rounds.
        n_spans = max(1, math.ceil(target / mean_span) * 2)
        starts = torch.randint(0, seq_len, (n_spans,), device=valid_mask.device)
        if p >= 1.0:
            lengths = torch.ones((n_spans,), device=valid_mask.device, dtype=torch.long)
        else:
            u = torch.rand((n_spans,), device=valid_mask.device).clamp_(1e-6, 1 - 1e-6)
            lengths = torch.floor(torch.log1p(-u) / math.log1p(-p)).to(torch.long) + 1
            lengths = lengths.clamp(min=1)

        for start, span_len in zip(starts.tolist(), lengths.tolist()):
            if remaining <= 0:
                break
            end = min(seq_len, start + span_len)
            span_valid = valid_mask[start:end]
            if not span_valid.any():
                continue
            new = span_valid & (~masked[start:end])
            masked[start:end] |= span_valid
            remaining -= int(new.sum().item())

        if remaining > 0:
            # Fill any shortfall by random token masking over unmasked valid positions.
            eligible = valid_mask & (~masked)
            eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
            if eligible_idx.numel() > 0:
                take = min(remaining, int(eligible_idx.numel()))
                pick = eligible_idx[
                    torch.randperm(eligible_idx.numel(), device=eligible_idx.device)[:take]
                ]
                masked[pick] = True

        return masked

    def _build_mask(
        self,
        valid_mask: torch.Tensor,
        rates: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.masking_strategy == "token":
            rand = torch.rand(valid_mask.shape, device=valid_mask.device)
            return (rand < rates.unsqueeze(1)) & valid_mask

        bsz = valid_mask.shape[0]
        masked = torch.zeros_like(valid_mask, dtype=torch.bool)
        for b in range(bsz):
            valid_idx = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
            valid_count = int(valid_idx.numel())
            if valid_count == 0:
                continue
            target = max(1, int(round(valid_count * float(rates[b].item()))))
            target = min(target, valid_count)
            masked[b] = self._sample_spans(valid_mask[b], target)
        return masked

    def _mask_rate(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Cosine schedule over normalized timestep in [0, 1].
        """
        t = self._normalized_t(timesteps)
        curve = 0.5 - 0.5 * torch.cos(math.pi * t)  # smooth 0 -> 1
        return self.config.min_mask_rate + curve * (
            self.config.max_mask_rate - self.config.min_mask_rate
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        timesteps: [B] integer timestep ids
        attention_mask: [B, T] where 1 means valid token
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions).unsqueeze(0)

        t_scaled = self._normalized_t(timesteps) * max(self.config.timesteps - 1, 1)
        t_emb = timestep_embedding(t_scaled, self.config.d_model)
        h = h + self.time_mlp(t_emb).unsqueeze(1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.final_norm(h)
        return self.lm_head(h)

    def corrupt(
        self,
        clean_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Corrupt clean tokens by replacing random positions with mask token.
        Returns (noisy_ids, masked_positions, mask_rates).
        """
        rates = self._mask_rate(timesteps)  # [B]
        non_pad = clean_ids.ne(self.config.pad_token_id)
        valid = non_pad
        if attention_mask is not None:
            valid = valid & attention_mask.bool()

        masked = self._build_mask(valid, rates)

        # Ensure at least one masked token per sample when possible.
        valid_counts = valid.sum(dim=1)
        no_mask = masked.sum(dim=1).eq(0) & valid_counts.gt(0)
        if no_mask.any():
            for b in no_mask.nonzero(as_tuple=False).squeeze(-1):
                valid_idx = valid[b].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    ridx = torch.randint(valid_idx.numel(), (1,), device=valid_idx.device)
                    masked[b, valid_idx[ridx]] = True

        noisy = clean_ids.clone()
        noisy[masked] = self.config.mask_token_id
        return noisy, masked, rates

    def compute_loss(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        """
        Diffusion objective:
        - sample timestep t
        - corrupt input according to t
        - predict original token only on corrupted positions
        """
        bsz = clean_ids.shape[0]
        timesteps = self._sample_train_timesteps(bsz, clean_ids.device)
        noisy, masked_pos, _ = self.corrupt(clean_ids, timesteps, attention_mask)
        logits = self.forward(noisy, timesteps, attention_mask)

        ce = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            clean_ids.reshape(-1),
            reduction="none",
        ).reshape_as(clean_ids)

        target_mask = masked_pos
        if attention_mask is not None:
            target_mask = target_mask & attention_mask.bool()

        denom = target_mask.sum().clamp(min=1)
        loss = (ce * target_mask.float()).sum() / denom
        masked_ratio = float(target_mask.float().mean().item())
        return DiffusionOutput(logits=logits, loss=loss, masked_ratio=masked_ratio)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        num_steps: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        temperature_end: float | None = None,
        eos_token_id: int | None = None,
        blockwise: bool = False,
        block_size: int | None = None,
        calibrated_confidence: bool = True,
        final_fill_threshold: int = 16,
    ) -> torch.Tensor:
        """
        Confidence-based iterative unmasking (MaskGIT-style).
        Returns generated continuation tokens.
        """
        self.eval()
        steps = num_steps or self.config.sample_steps
        steps = max(2, steps)

        if prompt_ids is None:
            prompt_ids = torch.empty((1, 0), dtype=torch.long, device=self.token_embed.weight.device)

        bsz, prompt_len = prompt_ids.shape
        if blockwise and bsz != 1:
            raise ValueError("blockwise generation currently supports batch size 1")

        if blockwise:
            chunk_size = block_size or self.config.block_size
            if chunk_size < 1:
                raise ValueError("block_size must be >= 1")
            generated = prompt_ids.clone()
            remaining = max_new_tokens
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                total_len = generated.shape[1] + chunk
                if total_len > self.config.max_seq_len:
                    raise ValueError(
                        f"requested length ({total_len}) exceeds max_seq_len ({self.config.max_seq_len})"
                    )
                tokens = torch.full(
                    (bsz, total_len),
                    self.config.mask_token_id,
                    dtype=torch.long,
                    device=generated.device,
                )
                fixed = torch.zeros_like(tokens, dtype=torch.bool)
                tokens[:, : generated.shape[1]] = generated
                fixed[:, : generated.shape[1]] = True
                tokens = self._decode_tokens(
                    tokens=tokens,
                    fixed=fixed,
                    steps=steps,
                    temperature=temperature,
                    top_k=top_k,
                    temperature_end=temperature_end,
                    calibrated_confidence=calibrated_confidence,
                    final_fill_threshold=final_fill_threshold,
                )
                new_tokens = tokens[:, -chunk:]
                generated = torch.cat([generated, new_tokens], dim=1)
                remaining -= chunk
                if eos_token_id is not None:
                    eos = (new_tokens[0] == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
                    if eos.numel() > 0:
                        cut = generated.shape[1] - chunk + int(eos[0].item()) + 1
                        generated = generated[:, :cut]
                        break
            return generated[:, prompt_len:]

        total_len = prompt_len + max_new_tokens
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"requested length ({total_len}) exceeds max_seq_len ({self.config.max_seq_len})"
            )

        tokens = torch.full(
            (bsz, total_len), self.config.mask_token_id, dtype=torch.long, device=prompt_ids.device
        )
        fixed = torch.zeros_like(tokens, dtype=torch.bool)
        if prompt_len > 0:
            tokens[:, :prompt_len] = prompt_ids
            fixed[:, :prompt_len] = True

        tokens = self._decode_tokens(
            tokens=tokens,
            fixed=fixed,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            temperature_end=temperature_end,
            calibrated_confidence=calibrated_confidence,
            final_fill_threshold=final_fill_threshold,
        )
        continuation = tokens[:, prompt_len:]
        if eos_token_id is not None:
            for b in range(bsz):
                eos = (continuation[b] == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
                if eos.numel() > 0:
                    continuation[b, int(eos[0].item()) + 1 :] = self.config.pad_token_id
        return continuation

    def _decode_tokens(
        self,
        tokens: torch.Tensor,
        fixed: torch.Tensor,
        steps: int,
        temperature: float,
        top_k: int,
        temperature_end: float | None,
        calibrated_confidence: bool,
        final_fill_threshold: int,
    ) -> torch.Tensor:
        bsz, total_len = tokens.shape
        for s in range(steps - 1, -1, -1):
            t = self._inference_timesteps(
                reverse_step=s,
                total_steps=steps,
                batch_size=bsz,
                device=tokens.device,
            )
            logits = self.forward(tokens, t)
            # Never generate the special mask token as content.
            logits[..., self.config.mask_token_id] = -torch.inf

            step_temp = self._step_temperature(
                reverse_step=s,
                total_steps=steps,
                temperature=temperature,
                temperature_end=temperature_end,
            )
            if step_temp <= 0:
                pred = logits.argmax(dim=-1)
                if calibrated_confidence:
                    probs = F.softmax(logits.float(), dim=-1)
                    conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
                else:
                    conf = logits.float().amax(dim=-1)
            else:
                scaled = logits / step_temp
                if top_k > 0:
                    k = min(top_k, scaled.shape[-1])
                    topk_vals, topk_idx = scaled.topk(k, dim=-1)
                    probs = F.softmax(topk_vals, dim=-1)
                    sampled = torch.multinomial(probs.reshape(-1, k), 1).reshape(bsz, total_len)
                    pred = topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                    conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                else:
                    probs = F.softmax(scaled, dim=-1)
                    pred = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]), 1
                    ).reshape(bsz, total_len)
                    conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

            masked_now = tokens.eq(self.config.mask_token_id) & (~fixed)
            if not masked_now.any():
                break

            # Reveal high-confidence subset; reveal all at final step.
            for b in range(bsz):
                idx = masked_now[b].nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                if s == 0:
                    choose = idx
                else:
                    n_rem = idx.numel()
                    n_reveal = max(1, math.ceil(n_rem / (s + 1)))
                    vals = conf[b, idx]
                    best = vals.topk(min(n_reveal, n_rem), dim=0).indices
                    choose = idx[best]
                tokens[b, choose] = pred[b, choose]
                remain = tokens[b].eq(self.config.mask_token_id) & (~fixed[b])
                if not remain.any():
                    continue

                # Fast path for edge devices: finish tiny remainder immediately.
                if final_fill_threshold > 0 and int(remain.sum().item()) <= final_fill_threshold:
                    tokens[b, remain] = pred[b, remain]
                    continue

                # Confidence-gated early finalize (requires calibrated probabilities).
                if (
                    calibrated_confidence
                    and self.config.confidence_stop > 0.0
                    and conf[b, remain].mean().item() >= self.config.confidence_stop
                ):
                    tokens[b, remain] = pred[b, remain]

        # Fallback in case some mask tokens remain.
        remaining = tokens.eq(self.config.mask_token_id) & (~fixed)
        if remaining.any():
            t0 = self._inference_timesteps(
                reverse_step=0,
                total_steps=steps,
                batch_size=bsz,
                device=tokens.device,
            )
            logits = self.forward(tokens, t0)
            logits[..., self.config.mask_token_id] = -torch.inf
            tokens[remaining] = logits.argmax(dim=-1)[remaining]
        return tokens
