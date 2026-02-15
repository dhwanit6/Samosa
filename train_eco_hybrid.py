"""
Train EcoHybrid Diffusion LM on binary token data.

Example:
    python -m diffusion.train_eco_hybrid \
        --data_dir data/processed \
        --output_dir checkpoints/eco_hybrid_hi \
        --seq_len 512 \
        --batch_size 8 \
        --grad_accum 8 \
        --max_steps 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import BinaryTokenDataset
from diffusion.eco_hybrid import EcoHybridConfig, EcoHybridDiffusionLM


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_vocab_size(data_dir: Path) -> int:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "vocab_size" not in meta:
        raise KeyError(f"`vocab_size` missing in {meta_path}")
    return int(meta["vocab_size"])


def train(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    if not train_bin.exists():
        raise FileNotFoundError(f"Missing train data: {train_bin}")

    vocab_base = load_vocab_size(data_dir)
    cfg = EcoHybridConfig(
        vocab_size=vocab_base + 1,
        mask_token_id=vocab_base,
        pad_token_id=0,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        memory_slots=args.memory_slots,
        conv_kernel_size=args.conv_kernel,
        dropout=args.dropout,
        timesteps=args.timesteps,
        min_mask_rate=args.min_mask_rate,
        max_mask_rate=args.max_mask_rate,
        sample_steps=args.sample_steps,
        confidence_stop=args.confidence_stop,
        time_mode=args.time_mode,
        timestep_sampling=args.timestep_sampling,
        masking_strategy=args.masking_strategy,
        mean_span_length=args.mean_span_length,
        block_size=args.block_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EcoHybridDiffusionLM(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    raw_model = _unwrap_model(model)

    ds = BinaryTokenDataset(str(train_bin), seq_len=args.seq_len)
    if len(ds) == 0:
        raise ValueError(
            f"Dataset has no samples for seq_len={args.seq_len}. "
            "Use smaller --seq_len or more data."
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EcoHybrid Diffusion LM Training")
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M")
    print(f"Vocab: {cfg.vocab_size} (base={vocab_base}, mask={cfg.mask_token_id})")
    print(
        f"d={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, "
        f"memory_slots={cfg.memory_slots}, conv_k={cfg.conv_kernel_size}"
    )
    print(f"Seq len={cfg.max_seq_len}, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"Timesteps={cfg.timesteps}, sample_steps={cfg.sample_steps}")
    print(
        f"Time mode: {cfg.time_mode} ({cfg.timestep_sampling}) | "
        f"Masking: {cfg.masking_strategy} (mean_span={cfg.mean_span_length})"
    )
    print("=" * 70)

    model.train()
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_mask = 0.0
    t0 = time.time()

    for step in range(args.max_steps):
        step_loss = 0.0
        step_mask = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            x = batch["input_ids"].to(device, non_blocking=True)
            loss, masked_ratio = model.compute_loss(x)
            (loss / args.grad_accum).backward()
            step_loss += float(loss.item())
            step_mask += masked_ratio

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += step_loss / args.grad_accum
        running_mask += step_mask / args.grad_accum

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = max(time.time() - t0, 1e-6)
            denom = args.log_interval if step > 0 else 1
            avg_loss = running_loss / denom
            avg_mask = running_mask / denom
            ppl = math.exp(min(avg_loss, 20))
            tok_s = args.batch_size * args.grad_accum * args.seq_len * denom / elapsed
            print(
                f"step {step+1:>6d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:>8.1f} | "
                f"mask {avg_mask:.3f} | gnorm {float(grad_norm):.2f} | "
                f"{tok_s:,.0f} tok/s"
            )
            running_loss = 0.0
            running_mask = 0.0
            t0 = time.time()

        if (step + 1) % args.save_interval == 0 or step == args.max_steps - 1:
            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, out_dir / "latest.pt")
            print(f"saved: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EcoHybrid diffusion LM")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--output_dir", type=str, default="checkpoints/eco_hybrid_hi")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--memory_slots", type=int, default=16)
    p.add_argument("--conv_kernel", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--timesteps", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=12)
    p.add_argument("--min_mask_rate", type=float, default=0.05)
    p.add_argument("--max_mask_rate", type=float, default=0.95)
    p.add_argument("--confidence_stop", type=float, default=0.98)
    p.add_argument(
        "--time_mode",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
    )
    p.add_argument(
        "--timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "stratified"],
    )
    p.add_argument(
        "--masking_strategy",
        type=str,
        default="token",
        choices=["token", "span"],
    )
    p.add_argument("--mean_span_length", type=float, default=3.0)
    p.add_argument("--block_size", type=int, default=64)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
