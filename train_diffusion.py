"""
Train masked diffusion language model on binary token data.

Example:
    python -m diffusion.train_diffusion \
        --data_dir data/processed \
        --output_dir checkpoints/diffusion_hi \
        --max_steps 5000 \
        --seq_len 512
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Support multiple repo layouts (root/* and train/*) and invocation styles.
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from data.dataset import BinaryTokenDataset
except ModuleNotFoundError:
    try:
        from train.data.dataset import BinaryTokenDataset  # type: ignore
    except ModuleNotFoundError:
        class BinaryTokenDataset(Dataset):
            """
            Fallback memmap dataset when data.dataset is unavailable.
            Expected train.bin contains contiguous token ids.
            """

            def __init__(self, bin_path: str, seq_len: int):
                self.bin_path = Path(bin_path)
                self.seq_len = int(seq_len)
                if self.seq_len < 1:
                    raise ValueError("seq_len must be >= 1")
                if not self.bin_path.exists():
                    raise FileNotFoundError(f"Missing train data: {self.bin_path}")

                self.meta_path = self.bin_path.with_name("meta.json")
                dtype = self._resolve_dtype()
                self.data = np.memmap(self.bin_path, dtype=dtype, mode="r")
                self.n = max(0, int(self.data.shape[0]) - self.seq_len + 1)

            def _resolve_dtype(self):
                if self.meta_path.exists():
                    try:
                        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                        name = str(meta.get("dtype", "")).lower()
                        if name in {"uint16", "u2"}:
                            return np.uint16
                        if name in {"uint32", "u4"}:
                            return np.uint32
                        if name in {"int32", "i4"}:
                            return np.int32
                    except Exception:
                        pass
                size = self.bin_path.stat().st_size
                return np.uint16 if size % 2 == 0 else np.uint32

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int):
                if idx < 0 or idx >= self.n:
                    raise IndexError(idx)
                x = self.data[idx : idx + self.seq_len]
                x = torch.from_numpy(np.asarray(x, dtype=np.int64))
                return {"input_ids": x}

try:
    from diffusion.config import DiffusionLMConfig
    from diffusion.model import DiffusionLanguageModel
except ModuleNotFoundError:
    from config import DiffusionLMConfig  # type: ignore
    from model import DiffusionLanguageModel  # type: ignore


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _apply_edge_profile(args: argparse.Namespace) -> None:
    if args.edge_profile == "none":
        return
    if args.edge_profile == "tiny":
        args.d_model = 256
        args.n_layers = 6
        args.n_heads = 4
        args.d_ff = 768
        args.timesteps = 16
        args.sample_steps = 8
        args.seq_len = min(args.seq_len, 384)
    elif args.edge_profile == "laptop":
        args.d_model = 320
        args.n_layers = 8
        args.n_heads = 8
        args.d_ff = 960
        args.timesteps = 24
        args.sample_steps = 10


def load_vocab_size(data_dir: Path) -> int:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "vocab_size" not in meta:
        raise KeyError(f"`vocab_size` missing in {meta_path}")
    return int(meta["vocab_size"])


def run_training(args: argparse.Namespace):
    _apply_edge_profile(args)
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    if not train_bin.exists():
        raise FileNotFoundError(f"Missing train data: {train_bin}")

    vocab_base = load_vocab_size(data_dir)
    # Reserve one extra id for diffusion mask token.
    config = DiffusionLMConfig(
        vocab_size=vocab_base + 1,
        mask_token_id=vocab_base,
        pad_token_id=0,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        timesteps=args.timesteps,
        min_mask_rate=args.min_mask_rate,
        max_mask_rate=args.max_mask_rate,
        sample_steps=args.sample_steps,
        time_mode=args.time_mode,
        timestep_sampling=args.timestep_sampling,
        masking_strategy=args.masking_strategy,
        mean_span_length=args.mean_span_length,
        block_size=args.block_size,
        confidence_stop=args.confidence_stop,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionLanguageModel(config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    raw_model = _unwrap_model(model)
    use_amp = device.type == "cuda" and not args.no_amp
    use_bf16 = use_amp and args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_bf16)

    train_ds = BinaryTokenDataset(str(train_bin), seq_len=args.seq_len)
    if len(train_ds) == 0:
        raise ValueError(
            f"Dataset has no samples for seq_len={args.seq_len}. "
            "Use a smaller --seq_len or provide more data."
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Diffusion LM Training")
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M")
    print(f"Vocab: {config.vocab_size} (base={vocab_base}, mask={config.mask_token_id})")
    print(f"Seq len: {args.seq_len} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"Timesteps: {config.timesteps} | Mask rate: {config.min_mask_rate:.2f}-{config.max_mask_rate:.2f}")
    print(
        f"Time mode: {config.time_mode} ({config.timestep_sampling}) | "
        f"Masking: {config.masking_strategy} (mean_span={config.mean_span_length})"
    )
    if use_amp:
        print(f"AMP: enabled ({'bf16' if use_bf16 else 'fp16'})")
    else:
        print("AMP: disabled")
    print("=" * 70)

    model.train()
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_mask_ratio = 0.0
    t0 = time.time()

    for step in range(args.max_steps):
        step_loss = 0.0
        step_mask_ratio = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            x = batch["input_ids"].to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                out = model.compute_loss(x)
                loss = out.loss
            if scaler.is_enabled():
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            step_loss += float(loss.item())
            step_mask_ratio += float(out.masked_ratio or 0.0)

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += step_loss / args.grad_accum
        running_mask_ratio += step_mask_ratio / args.grad_accum

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = max(time.time() - t0, 1e-6)
            avg_loss = running_loss / args.log_interval if step > 0 else running_loss
            avg_mask = running_mask_ratio / args.log_interval if step > 0 else running_mask_ratio
            ppl = math.exp(min(avg_loss, 20))
            tok_per_sec = (
                args.batch_size * args.grad_accum * args.seq_len * args.log_interval / elapsed
                if step > 0
                else args.batch_size * args.grad_accum * args.seq_len / elapsed
            )
            print(
                f"step {step+1:>6d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:>8.1f} | "
                f"mask {avg_mask:.3f} | gnorm {float(grad_norm):.2f} | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            running_loss = 0.0
            running_mask_ratio = 0.0
            t0 = time.time()

        if (step + 1) % args.save_interval == 0 or step == args.max_steps - 1:
            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
            }
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, out_dir / "latest.pt")
            print(f"saved: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion language model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints/diffusion_hi")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--edge_profile",
        type=str,
        default="none",
        choices=["none", "tiny", "laptop"],
        help="Apply edge-optimized model presets",
    )

    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--min_mask_rate", type=float, default=0.05)
    parser.add_argument("--max_mask_rate", type=float, default=0.95)
    parser.add_argument(
        "--time_mode",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "stratified"],
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="token",
        choices=["token", "span"],
    )
    parser.add_argument("--mean_span_length", type=float, default=3.0)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--confidence_stop", type=float, default=0.98)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype when CUDA AMP is enabled",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
