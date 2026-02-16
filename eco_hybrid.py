"""
EcoHybrid Diffusion LM
----------------------

An efficiency-first denoiser that avoids full O(T^2) self-attention.

Key idea:
- Local context via depthwise temporal convolution.
- Global context via a small set of learned memory slots.
- Cross-attention is only token<->memory (O(T*M), M << T).

This gives a diffusion model that is less generic than standard
Transformer denoisers and is designed for low-hardware experimentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EcoHybridConfig:
    # Vocabulary
    vocab_size: int = 32001
    mask_token_id: int = 32000
    pad_token_id: int = 0

    # Backbone
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    max_seq_len: int = 1024
    memory_slots: int = 16
    conv_kernel_size: int = 7
    dropout: float = 0.0

    # Diffusion process
    timesteps: int = 32
    min_mask_rate: float = 0.05
    max_mask_rate: float = 0.95
    time_mode: str = "discrete"  # discrete | continuous
    timestep_sampling: str = "uniform"  # uniform | stratified
    masking_strategy: str = "token"  # token | span
    mean_span_length: float = 3.0
    sample_steps: int = 12

    # Decoding behavior
    confidence_stop: float = 0.98
    block_size: int = 64

    init_std: float = 0.02

    def __post_init__(self):
        if self.mask_token_id >= self.vocab_size:
            raise ValueError("mask_token_id must be < vocab_size")
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("pad_token_id must be < vocab_size")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.memory_slots < 2:
            raise ValueError("memory_slots must be >= 2")
        if self.conv_kernel_size < 3 or self.conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd and >= 3")
        if not (0.0 <= self.min_mask_rate < self.max_mask_rate <= 1.0):
            raise ValueError("mask rates must satisfy 0 <= min < max <= 1")
        if not (0.0 <= self.confidence_stop < 1.0):
            raise ValueError("confidence_stop must be in [0, 1)")
        if self.time_mode not in {"discrete", "continuous"}:
            raise ValueError("time_mode must be one of: discrete, continuous")
        if self.timestep_sampling not in {"uniform", "stratified"}:
            raise ValueError("timestep_sampling must be one of: uniform, stratified")
        if self.masking_strategy not in {"token", "span"}:
            raise ValueError("masking_strategy must be one of: token, span")
        if self.mean_span_length <= 0:
            raise ValueError("mean_span_length must be > 0")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(10_000.0) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
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


class LocalConvMixer(nn.Module):
    """Depthwise temporal conv + pointwise projection."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pw = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T] -> [B, T, C]
        y = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.pw(F.silu(y))
        return self.dropout(y)


class EcoHybridBlock(nn.Module):
    def __init__(self, cfg: EcoHybridConfig):
        super().__init__()
        self.norm_local = RMSNorm(cfg.d_model)
        self.local_mixer = LocalConvMixer(cfg.d_model, cfg.conv_kernel_size, cfg.dropout)

        self.norm_token = RMSNorm(cfg.d_model)
        self.token_to_mem = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        self.norm_mem = RMSNorm(cfg.d_model)
        self.mem_to_token = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        self.norm_ff = RMSNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) local channel-time mixing
        tokens = tokens + self.local_mixer(self.norm_local(tokens))

        # 2) token attends to compact memory
        q_tok = self.norm_token(tokens)
        attn_tok, _ = self.token_to_mem(q_tok, memory, memory, need_weights=False)
        tokens = tokens + attn_tok

        # 3) memory absorbs global summary from token stream
        q_mem = self.norm_mem(memory)
        upd_mem, _ = self.mem_to_token(
            q_mem, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False
        )
        memory = memory + upd_mem

        # 4) final token FFN
        tokens = tokens + self.ff(self.norm_ff(tokens))
        return tokens, memory


class EcoHybridDiffusionLM(nn.Module):
    """
    Diffusion LM with local convolutions + memory-slot attention.
    """

    def __init__(self, cfg: EcoHybridConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.memory_slots = nn.Parameter(torch.empty(cfg.memory_slots, cfg.d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([EcoHybridBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=self.cfg.init_std)
        nn.init.normal_(self.pos_embed.weight, std=self.cfg.init_std)
        nn.init.normal_(self.memory_slots, std=self.cfg.init_std)
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _run_blocks(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        active_layers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_total = len(self.blocks)
        if active_layers is None:
            n_use = n_total
        else:
            n_use = max(1, min(int(active_layers), n_total))
        for idx in range(n_use):
            tokens, memory = self.blocks[idx](tokens, memory, key_padding_mask=key_padding_mask)
        return tokens, memory

    def _normalized_t(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.cfg.time_mode == "continuous":
            return timesteps.float().clamp(0.0, 1.0)
        return timesteps.float() / max(self.cfg.timesteps - 1, 1)

    def _sample_train_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.cfg.time_mode == "discrete":
            return torch.randint(
                low=0,
                high=self.cfg.timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
        if self.cfg.timestep_sampling == "stratified":
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
        if self.cfg.time_mode == "continuous":
            return torch.full((batch_size,), float(frac), dtype=torch.float32, device=device)
        t_idx = int(round(frac * (self.cfg.timesteps - 1)))
        return torch.full((batch_size,), t_idx, dtype=torch.long, device=device)

    def _step_temperature(
        self, reverse_step: int, total_steps: int, temperature: float, temperature_end: float | None
    ) -> float:
        if temperature_end is None:
            return temperature
        progress = 1.0 - (reverse_step / max(total_steps - 1, 1))
        return temperature + (temperature_end - temperature) * progress

    def _sample_spans(self, valid_mask: torch.Tensor, target: int) -> torch.Tensor:
        seq_len = valid_mask.shape[0]
        masked = torch.zeros_like(valid_mask, dtype=torch.bool)
        if target <= 0:
            return masked

        mean_span = max(float(self.cfg.mean_span_length), 1.0)
        p = min(1.0, 1.0 / mean_span)
        remaining = target
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
            eligible = valid_mask & (~masked)
            eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
            if eligible_idx.numel() > 0:
                take = min(remaining, int(eligible_idx.numel()))
                pick = eligible_idx[
                    torch.randperm(eligible_idx.numel(), device=eligible_idx.device)[:take]
                ]
                masked[pick] = True

        return masked

    def _build_mask(self, valid_mask: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        if self.cfg.masking_strategy == "token":
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

    def _mask_rate(self, t: torch.Tensor) -> torch.Tensor:
        x = self._normalized_t(t)
        curve = 0.5 - 0.5 * torch.cos(math.pi * x)
        return self.cfg.min_mask_rate + curve * (self.cfg.max_mask_rate - self.cfg.min_mask_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        active_layers: int | None = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.cfg.max_seq_len})"
            )

        pos = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(pos).unsqueeze(0)
        t_scaled = self._normalized_t(timesteps) * max(self.cfg.timesteps - 1, 1)
        t = self.time_mlp(timestep_embedding(t_scaled, self.cfg.d_model)).unsqueeze(1)
        h = h + t

        mem = self.memory_slots.unsqueeze(0).expand(bsz, -1, -1) + t
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        h, mem = self._run_blocks(
            tokens=h,
            memory=mem,
            key_padding_mask=key_padding_mask,
            active_layers=active_layers,
        )

        h = self.final_norm(h)
        return self.head(h)

    def _encode_context_memory(self, context_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a fixed context once and return memory slots.
        Used by frozen-context decoding to avoid repeated full-sequence recompute.
        """
        bsz, seq_len = context_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"context seq_len ({seq_len}) exceeds max_seq_len ({self.cfg.max_seq_len})")

        if self.cfg.time_mode == "continuous":
            t0 = torch.zeros((bsz,), device=context_ids.device, dtype=torch.float32)
        else:
            t0 = torch.zeros((bsz,), device=context_ids.device, dtype=torch.long)
        t = self.time_mlp(timestep_embedding(t0, self.cfg.d_model)).unsqueeze(1)

        mem = self.memory_slots.unsqueeze(0).expand(bsz, -1, -1) + t
        if seq_len == 0:
            return mem

        pos = torch.arange(seq_len, device=context_ids.device)
        h = self.token_embed(context_ids) + self.pos_embed(pos).unsqueeze(0)
        h = h + t
        h, mem = self._run_blocks(
            tokens=h,
            memory=mem,
            key_padding_mask=None,
            active_layers=None,
        )
        return mem

    def _forward_with_context_memory(
        self,
        decode_ids: torch.Tensor,
        timesteps: torch.Tensor,
        context_memory: torch.Tensor,
        pos_offset: int,
        active_layers: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for decode tokens conditioned on pre-encoded context memory.
        """
        bsz, seq_len = decode_ids.shape
        if pos_offset + seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"decode positions ({pos_offset + seq_len}) exceed max_seq_len ({self.cfg.max_seq_len})"
            )
        pos = torch.arange(pos_offset, pos_offset + seq_len, device=decode_ids.device)
        h = self.token_embed(decode_ids) + self.pos_embed(pos).unsqueeze(0)

        t_scaled = self._normalized_t(timesteps) * max(self.cfg.timesteps - 1, 1)
        t = self.time_mlp(timestep_embedding(t_scaled, self.cfg.d_model)).unsqueeze(1)
        h = h + t

        # Time-adapt cached context memory for this denoising step.
        mem = context_memory + t
        h, mem = self._run_blocks(
            tokens=h,
            memory=mem,
            key_padding_mask=None,
            active_layers=active_layers,
        )

        h = self.final_norm(h)
        return self.head(h)

    def _update_context_memory(
        self,
        context_memory: torch.Tensor,
        new_token_ids: torch.Tensor,
        pos_offset: int,
    ) -> torch.Tensor:
        """
        Incrementally ingest finalized decode tokens into cached context memory.
        This makes blockwise frozen-context decoding linear in generated tokens.
        """
        bsz, seq_len = new_token_ids.shape
        if seq_len == 0:
            return context_memory
        if pos_offset + seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"context update positions ({pos_offset + seq_len}) exceed max_seq_len ({self.cfg.max_seq_len})"
            )

        pos = torch.arange(pos_offset, pos_offset + seq_len, device=new_token_ids.device)
        h = self.token_embed(new_token_ids) + self.pos_embed(pos).unsqueeze(0)

        if self.cfg.time_mode == "continuous":
            t0 = torch.zeros((bsz,), device=new_token_ids.device, dtype=torch.float32)
        else:
            t0 = torch.zeros((bsz,), device=new_token_ids.device, dtype=torch.long)
        t_scaled = self._normalized_t(t0) * max(self.cfg.timesteps - 1, 1)
        t = self.time_mlp(timestep_embedding(t_scaled, self.cfg.d_model)).unsqueeze(1)
        h = h + t

        mem = context_memory
        h, mem = self._run_blocks(
            tokens=h,
            memory=mem,
            key_padding_mask=None,
            active_layers=None,
        )
        return mem

    def corrupt(
        self,
        clean_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rates = self._mask_rate(timesteps)  # [B]
        non_pad = clean_ids.ne(self.cfg.pad_token_id)
        valid = non_pad
        if attention_mask is not None:
            valid = valid & attention_mask.bool()

        masked = self._build_mask(valid, rates)

        # Ensure at least one masked token per sample.
        no_mask = masked.sum(dim=1).eq(0) & valid.sum(dim=1).gt(0)
        if no_mask.any():
            for b in no_mask.nonzero(as_tuple=False).squeeze(-1):
                valid_idx = valid[b].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    ridx = torch.randint(valid_idx.numel(), (1,), device=valid_idx.device)
                    masked[b, valid_idx[ridx]] = True

        noisy = clean_ids.clone()
        noisy[masked] = self.cfg.mask_token_id
        return noisy, masked

    def compute_loss(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        bsz = clean_ids.shape[0]
        t = self._sample_train_timesteps(bsz, clean_ids.device)
        noisy, masked = self.corrupt(clean_ids, t, attention_mask=attention_mask)
        logits = self.forward(noisy, t, attention_mask=attention_mask)
        ce = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            clean_ids.reshape(-1),
            reduction="none",
        ).reshape_as(clean_ids)
        target_mask = masked
        if attention_mask is not None:
            target_mask = target_mask & attention_mask.bool()
        denom = target_mask.sum().clamp(min=1)
        loss = (ce * target_mask.float()).sum() / denom
        return loss, float(target_mask.float().mean().item())

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
        frozen_context: bool = False,
        min_decode_layers: int = 0,
    ) -> torch.Tensor:
        self.eval()
        steps = max(2, num_steps or self.cfg.sample_steps)

        if prompt_ids is None:
            prompt_ids = torch.empty((1, 0), dtype=torch.long, device=self.token_embed.weight.device)

        bsz, prompt_len = prompt_ids.shape
        if blockwise and bsz != 1:
            raise ValueError("blockwise generation currently supports batch size 1")

        if blockwise:
            chunk_size = block_size or self.cfg.block_size
            if chunk_size < 1:
                raise ValueError("block_size must be >= 1")
            generated = prompt_ids.clone()
            context_memory = None
            context_len = generated.shape[1]
            if frozen_context:
                context_memory = self._encode_context_memory(generated)
            remaining = max_new_tokens
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                total_len = generated.shape[1] + chunk
                if total_len > self.cfg.max_seq_len:
                    raise ValueError(
                        f"requested length ({total_len}) exceeds max_seq_len ({self.cfg.max_seq_len})"
                    )
                if frozen_context and context_memory is not None:
                    block_tokens = torch.full(
                        (bsz, chunk),
                        self.cfg.mask_token_id,
                        dtype=torch.long,
                        device=generated.device,
                    )
                    block_fixed = torch.zeros_like(block_tokens, dtype=torch.bool)
                    block_tokens = self._decode_tokens(
                        tokens=block_tokens,
                        fixed=block_fixed,
                        steps=steps,
                        temperature=temperature,
                        top_k=top_k,
                        temperature_end=temperature_end,
                        calibrated_confidence=calibrated_confidence,
                        final_fill_threshold=final_fill_threshold,
                        context_memory=context_memory,
                        pos_offset=context_len,
                        min_decode_layers=min_decode_layers,
                    )
                    new_tokens = block_tokens
                else:
                    tokens = torch.full(
                        (bsz, total_len), self.cfg.mask_token_id, dtype=torch.long, device=generated.device
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
                        min_decode_layers=min_decode_layers,
                    )
                    new_tokens = tokens[:, -chunk:]
                generated = torch.cat([generated, new_tokens], dim=1)
                if frozen_context and context_memory is not None:
                    context_memory = self._update_context_memory(
                        context_memory=context_memory,
                        new_token_ids=new_tokens,
                        pos_offset=context_len,
                    )
                    context_len += new_tokens.shape[1]
                remaining -= chunk
                if eos_token_id is not None:
                    eos = (new_tokens[0] == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
                    if eos.numel() > 0:
                        cut = generated.shape[1] - chunk + int(eos[0].item()) + 1
                        generated = generated[:, :cut]
                        break
            return generated[:, prompt_len:]

        total_len = prompt_len + max_new_tokens
        if total_len > self.cfg.max_seq_len:
            raise ValueError(
                f"requested length ({total_len}) exceeds max_seq_len ({self.cfg.max_seq_len})"
            )

        tokens = torch.full(
            (bsz, total_len), self.cfg.mask_token_id, dtype=torch.long, device=prompt_ids.device
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
            min_decode_layers=min_decode_layers,
        )
        continuation = tokens[:, prompt_len:]
        if eos_token_id is not None:
            for b in range(bsz):
                eos = (continuation[b] == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
                if eos.numel() > 0:
                    continuation[b, int(eos[0].item()) + 1 :] = self.cfg.pad_token_id
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
        context_memory: torch.Tensor | None = None,
        pos_offset: int = 0,
        min_decode_layers: int = 0,
    ) -> torch.Tensor:
        bsz, total_len = tokens.shape
        n_total_layers = len(self.blocks)
        min_layers = max(0, min(int(min_decode_layers), n_total_layers))
        for s in range(steps - 1, -1, -1):
            t = self._inference_timesteps(
                reverse_step=s,
                total_steps=steps,
                batch_size=bsz,
                device=tokens.device,
            )
            active_layers = None
            if min_layers > 0 and n_total_layers > 1:
                progress = 1.0 - (s / max(steps - 1, 1))
                span = n_total_layers - min_layers
                active_layers = min_layers + int(math.ceil(span * progress))
                active_layers = max(1, min(active_layers, n_total_layers))
            if context_memory is None:
                logits = self.forward(tokens, t, active_layers=active_layers)
            else:
                logits = self._forward_with_context_memory(
                    decode_ids=tokens,
                    timesteps=t,
                    context_memory=context_memory,
                    pos_offset=pos_offset,
                    active_layers=active_layers,
                )
            logits[..., self.cfg.mask_token_id] = -torch.inf

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

            masked_now = tokens.eq(self.cfg.mask_token_id) & (~fixed)
            if not masked_now.any():
                break

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

                remain = tokens[b].eq(self.cfg.mask_token_id) & (~fixed[b])
                if not remain.any():
                    continue

                if final_fill_threshold > 0 and int(remain.sum().item()) <= final_fill_threshold:
                    tokens[b, remain] = pred[b, remain]
                    continue

                if (
                    calibrated_confidence
                    and self.cfg.confidence_stop > 0.0
                    and conf[b, remain].mean().item() >= self.cfg.confidence_stop
                ):
                    tokens[b, remain] = pred[b, remain]

        remaining = tokens.eq(self.cfg.mask_token_id) & (~fixed)
        if remaining.any():
            t0 = self._inference_timesteps(
                reverse_step=0,
                total_steps=steps,
                batch_size=bsz,
                device=tokens.device,
            )
            if context_memory is None:
                logits = self.forward(tokens, t0)
            else:
                logits = self._forward_with_context_memory(
                    decode_ids=tokens,
                    timesteps=t0,
                    context_memory=context_memory,
                    pos_offset=pos_offset,
                )
            logits[..., self.cfg.mask_token_id] = -torch.inf
            tokens[remaining] = logits.argmax(dim=-1)[remaining]

        return tokens
