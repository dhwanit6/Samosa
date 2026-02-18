"""
Unified training entrypoint for edge-focused diffusion LMs.

Primary model:
    EcoHybrid (local conv + memory slots) for best throughput/compute efficiency.

Backup model:
    Baseline bidirectional diffusion transformer for stability/regression checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Support multiple repo layouts and invocation styles.
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
            """Fallback memmap dataset when data/dataset.py is unavailable."""

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
                if not self.meta_path.exists():
                    raise FileNotFoundError(
                        f"Missing metadata for fallback dataset loader: {self.meta_path}"
                    )
                try:
                    meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValueError(f"Could not parse metadata JSON: {self.meta_path}") from exc
                name = str(meta.get("dtype", "")).lower()
                if name in {"uint16", "u2"}:
                    return np.uint16
                if name in {"uint32", "u4"}:
                    return np.uint32
                if name in {"int32", "i4"}:
                    return np.int32
                raise ValueError(
                    f"Unsupported or missing dtype in {self.meta_path}: {meta.get('dtype')!r}"
                )

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int):
                if idx < 0 or idx >= self.n:
                    raise IndexError(idx)
                x = self.data[idx : idx + self.seq_len]
                x = torch.from_numpy(np.asarray(x, dtype=np.int64))
                return {"input_ids": x}

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
        args.memory_slots = min(args.memory_slots, 8)
        args.timesteps = 16
        args.sample_steps = 8
        args.seq_len = min(args.seq_len, 384)
    elif args.edge_profile == "laptop":
        args.d_model = 320
        args.n_layers = 8
        args.n_heads = 8
        args.d_ff = 960
        args.memory_slots = min(args.memory_slots, 12)
        args.timesteps = 24
        args.sample_steps = 10
    # Edge presets also switch to generally stronger training noise schedules.
    args.time_mode = "continuous"
    args.timestep_sampling = "stratified"
    args.masking_strategy = "span"
    args.mean_span_length = max(float(args.mean_span_length), 3.0)


def _load_vocab_size(data_dir: Path) -> int:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "vocab_size" not in meta:
        raise KeyError(f"`vocab_size` missing in {meta_path}")
    return int(meta["vocab_size"])


def _load_data_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "vocab_size" not in meta:
        raise KeyError(f"`vocab_size` missing in {meta_path}")
    return meta


def _validate_train_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad_accum must be >= 1")
    if args.seq_len < 1:
        raise ValueError("--seq_len must be >= 1")
    if args.save_interval < 1:
        raise ValueError("--save_interval must be >= 1")
    if args.log_interval < 1:
        raise ValueError("--log_interval must be >= 1")
    if args.timesteps < 2:
        raise ValueError("--timesteps must be >= 2")
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be >= 1")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0")
    if args.d_model < 1 or args.n_layers < 1 or args.n_heads < 1 or args.d_ff < 1:
        raise ValueError("model dimensions/layers must all be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in [0, 1]")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be >= 1")
    if not (0.0 <= args.start_min_mask_rate < args.start_max_mask_rate <= 1.0):
        raise ValueError("start mask rates must satisfy 0 <= start_min < start_max <= 1")
    if not (0.0 <= args.min_mask_rate < args.max_mask_rate <= 1.0):
        raise ValueError("target mask rates must satisfy 0 <= min < max <= 1")


def _build_model_and_config(args: argparse.Namespace, vocab_base: int):
    common = dict(
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
    if args.model == "primary":
        cfg = EcoHybridConfig(
            **common,
            memory_slots=args.memory_slots,
            conv_kernel_size=args.conv_kernel,
        )
        model = EcoHybridDiffusionLM(cfg)
    else:
        cfg = DiffusionLMConfig(**common)
        model = DiffusionLanguageModel(cfg)
    return model, cfg


def _effective_warmup_steps(args: argparse.Namespace) -> int:
    total = max(int(args.max_steps), 1)
    warmup = max(0, int(args.warmup_steps))
    # Warmup must leave room for non-warmup phase; otherwise LR can stay too low for the whole run.
    return min(warmup, max(total - 1, 0))


def _lr_at_step(step: int, args: argparse.Namespace) -> float:
    base_lr = float(args.lr)
    if args.lr_schedule == "constant":
        return base_lr

    warmup = _effective_warmup_steps(args)
    total = max(int(args.max_steps), 1)
    min_lr = base_lr * max(0.0, float(args.min_lr_ratio))

    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup

    if total <= warmup:
        return base_lr

    progress = (step - warmup) / max(total - warmup - 1, 1)
    progress = min(1.0, max(0.0, float(progress)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _apply_mask_curriculum(raw_model: torch.nn.Module, args: argparse.Namespace, step: int) -> None:
    cfg = getattr(raw_model, "cfg", getattr(raw_model, "config", None))
    if cfg is None or not hasattr(cfg, "min_mask_rate") or not hasattr(cfg, "max_mask_rate"):
        return
    steps = max(0, int(args.mask_curriculum_steps))
    if steps <= 0:
        cfg.min_mask_rate = float(args.min_mask_rate)
        cfg.max_mask_rate = float(args.max_mask_rate)
        return

    start_min = float(args.start_min_mask_rate)
    start_max = float(args.start_max_mask_rate)
    target_min = float(args.min_mask_rate)
    target_max = float(args.max_mask_rate)
    alpha = min(1.0, max(0.0, step / max(steps - 1, 1)))
    cfg.min_mask_rate = start_min + alpha * (target_min - start_min)
    cfg.max_mask_rate = start_max + alpha * (target_max - start_max)


def train(args: argparse.Namespace) -> None:
    _apply_edge_profile(args)
    _validate_train_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    if not train_bin.exists():
        raise FileNotFoundError(f"Missing train data: {train_bin}")

    data_meta = _load_data_meta(data_dir)
    vocab_base = int(data_meta["vocab_size"])
    train_tokens_meta = int(data_meta.get("train_tokens", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _build_model_and_config(args, vocab_base)
    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    raw_model = _unwrap_model(model)

    use_amp = device.type == "cuda" and not args.no_amp
    use_bf16 = use_amp and args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_bf16)

    ds = BinaryTokenDataset(str(train_bin), seq_len=args.seq_len)
    if len(ds) == 0:
        raise ValueError(
            f"Dataset has no samples for seq_len={args.seq_len}. "
            "Use smaller --seq_len or more data."
        )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    def _worker_init_fn(worker_id: int) -> None:
        s = int(args.seed) + int(worker_id)
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=loader_gen,
        worker_init_fn=_worker_init_fn if device.type == "cuda" else None,
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
    start_step = 0

    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.resume_latest:
        candidate = out_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        raw_model.load_state_dict(_normalize_state_dict_keys(ckpt["model"]))
        if (not args.reset_optimizer) and ("optimizer" in ckpt):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", -1)) + 1
        print(f"[resume] loaded {resume_path} at step {start_step}")
        if start_step >= args.max_steps:
            print(
                f"[resume] checkpoint step ({start_step}) is already >= max_steps ({args.max_steps}). "
                "No training steps to run."
            )
            return

    print("=" * 70)
    print(f"Unified Diffusion Training | model={args.model}")
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M")
    print(f"Vocab: {cfg.vocab_size} (base={vocab_base}, mask={cfg.mask_token_id})")
    print(f"Seq len={args.seq_len}, batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"Timesteps={cfg.timesteps}, sample_steps={cfg.sample_steps}")
    print(
        f"Time mode: {cfg.time_mode} ({cfg.timestep_sampling}) | "
        f"Masking: {cfg.masking_strategy} (mean_span={cfg.mean_span_length})"
    )
    if args.model == "primary":
        print(
            f"Primary config: memory_slots={cfg.memory_slots}, conv_k={cfg.conv_kernel_size}, "
            f"confidence_stop={cfg.confidence_stop}"
        )
    if train_tokens_meta > 0:
        total_train_tokens = args.batch_size * args.grad_accum * args.seq_len * args.max_steps
        approx_data_passes = total_train_tokens / max(train_tokens_meta, 1)
        print(
            f"Data tokens={train_tokens_meta:,} | planned train tokens={total_train_tokens:,} | "
            f"approx passes={approx_data_passes:.2f}x"
        )
        if approx_data_passes > 8.0:
            print(
                "[WARN] Planned token budget is high relative to corpus size. "
                "Quality may plateau/overfit early; increase corpus size or reduce max_steps."
            )
    print(f"AMP: {'bf16' if use_bf16 else ('fp16' if use_amp else 'disabled')}")
    print(
        f"LR schedule: {args.lr_schedule} | base_lr={args.lr:.2e} | "
        f"warmup={_effective_warmup_steps(args)} (requested={args.warmup_steps}) | "
        f"min_lr_ratio={args.min_lr_ratio:.3f}"
    )
    print(
        f"Mask curriculum: steps={args.mask_curriculum_steps} | "
        f"start=({args.start_min_mask_rate:.3f},{args.start_max_mask_rate:.3f}) -> "
        f"target=({args.min_mask_rate:.3f},{args.max_mask_rate:.3f})"
    )
    print("=" * 70)

    model.train()
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_mask = 0.0
    running_lr = 0.0
    t0 = time.time()
    uniform_nll = math.log(max(2, int(cfg.vocab_size)))

    for step in range(start_step, args.max_steps):
        step_loss = 0.0
        step_mask = 0.0
        step_lr = _lr_at_step(step, args)
        _set_optimizer_lr(optimizer, step_lr)
        _apply_mask_curriculum(raw_model, args, step)

        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            x = batch["input_ids"].to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                out = model.compute_loss(x)
                if isinstance(out, tuple):
                    loss, masked_ratio = out
                else:
                    loss = out.loss
                    masked_ratio = float(out.masked_ratio or 0.0)
            if scaler.is_enabled():
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            step_loss += float(loss.item())
            step_mask += float(masked_ratio)

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
        running_mask += step_mask / args.grad_accum
        running_lr += step_lr

        if (step + 1) % args.log_interval == 0 or step == start_step:
            elapsed = max(time.time() - t0, 1e-6)
            denom = args.log_interval if step > start_step else 1
            avg_loss = running_loss / denom
            avg_mask = running_mask / denom
            avg_lr = running_lr / denom
            ppl = math.exp(min(avg_loss, 20))
            rel_to_uniform = 1.0 - (avg_loss / max(uniform_nll, 1e-6))
            cfg_live = getattr(raw_model, "cfg", getattr(raw_model, "config", None))
            cur_min_mask = float(getattr(cfg_live, "min_mask_rate", args.min_mask_rate))
            cur_max_mask = float(getattr(cfg_live, "max_mask_rate", args.max_mask_rate))
            tok_s = args.batch_size * args.grad_accum * args.seq_len * denom / elapsed
            print(
                f"step {step+1:>6d}/{args.max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:>8.1f} | "
                f"vs_uni {rel_to_uniform:+.3f} | lr {avg_lr:.2e} | "
                f"mask {avg_mask:.3f} ({cur_min_mask:.2f}-{cur_max_mask:.2f}) | "
                f"gnorm {float(grad_norm):.2f} | "
                f"{tok_s:,.0f} tok/s"
            )
            running_loss = 0.0
            running_mask = 0.0
            running_lr = 0.0
            t0 = time.time()

        if (step + 1) % args.save_interval == 0 or step == args.max_steps - 1:
            ckpt = {
                "step": step,
                "model_type": args.model,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, out_dir / "latest.pt")
            print(f"saved: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified diffusion trainer")
    p.add_argument("--model", type=str, default="primary", choices=["primary", "backup"])
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--output_dir", type=str, default="checkpoints/eco_hybrid_hi")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    p.add_argument("--resume_latest", action="store_true", help="Resume from output_dir/latest.pt if present")
    p.add_argument("--reset_optimizer", action="store_true", help="Ignore optimizer state when resuming")

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--memory_slots", type=int, default=16)
    p.add_argument("--conv_kernel", type=int, default=7)
    p.add_argument(
        "--edge_profile",
        type=str,
        default="none",
        choices=["none", "tiny", "laptop"],
        help="Apply edge-optimized model presets",
    )

    p.add_argument("--timesteps", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=12)
    p.add_argument("--min_mask_rate", type=float, default=0.05)
    p.add_argument("--max_mask_rate", type=float, default=0.85)
    p.add_argument("--confidence_stop", type=float, default=0.98)
    p.add_argument("--time_mode", type=str, default="discrete", choices=["discrete", "continuous"])
    p.add_argument("--timestep_sampling", type=str, default="uniform", choices=["uniform", "stratified"])
    p.add_argument("--masking_strategy", type=str, default="token", choices=["token", "span"])
    p.add_argument("--mean_span_length", type=float, default=3.0)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument(
        "--mask_curriculum_steps",
        type=int,
        default=1500,
        help="Linearly ramp mask-rate difficulty during first N steps (0 disables)",
    )
    p.add_argument("--start_min_mask_rate", type=float, default=0.02)
    p.add_argument("--start_max_mask_rate", type=float, default=0.40)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["constant", "cosine"],
        help="Learning-rate schedule",
    )
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--min_lr_ratio", type=float, default=0.10)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
