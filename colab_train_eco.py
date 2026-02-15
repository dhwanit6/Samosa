"""
Colab-ready one-shot runner for EcoHybrid diffusion training.

What it does:
1. Detects GPU and chooses safe defaults.
2. Auto-downloads Hindi data if missing.
3. Auto-trains tokenizer and preprocesses to binary if missing.
4. Trains EcoHybrid diffusion model.

Usage (Colab):
    !git clone <your-repo-url> Aria
    %cd Aria/train
    !python -m diffusion.colab_train_eco --max_steps 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make notebook/script output visible immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Allow importing both as:
# 1) package module: python -m diffusion.colab_train_eco
# 2) top-level module/script: python -m colab_train_eco / python colab_train_eco.py
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from diffusion.train_eco_hybrid import train as train_eco  # noqa: E402
except ModuleNotFoundError:
    from train_eco_hybrid import train as train_eco  # type: ignore # noqa: E402

try:
    from runners.colab_train import _resolve_path, prepare_data, setup_environment  # noqa: E402
except ModuleNotFoundError:
    try:
        from colab_train import _resolve_path, prepare_data, setup_environment  # type: ignore # noqa: E402
    except ModuleNotFoundError:
        def _resolve_path(path: str) -> Path:
            return (Path.cwd() / path).resolve()

        def setup_environment(allow_cpu: bool = False):
            if torch.cuda.is_available():
                # Safe defaults for Colab T4/L4 style GPUs.
                return 4, 8, 512
            if allow_cpu:
                return 1, 1, 256
            raise RuntimeError("No GPU found. Re-run with --allow_cpu to force CPU mode.")

        def prepare_data(data_dir: str, raw_dir: str) -> str:
            data_dir_path = _resolve_path(data_dir)
            train_bin = data_dir_path / "train.bin"
            meta = data_dir_path / "meta.json"
            if not train_bin.exists() or not meta.exists():
                raise FileNotFoundError(
                    "Could not auto-prepare data because runners helper scripts are missing. "
                    f"Expected preprocessed files at {data_dir_path}: train.bin and meta.json."
                )
            return str(train_bin)


def main():
    parser = argparse.ArgumentParser(description="Colab one-shot EcoHybrid diffusion runner")
    parser.add_argument("--skip_setup", action="store_true", help="Skip auto data prep if train.bin exists")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="checkpoints/eco_hybrid_hi")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU-only run.")

    # Training controls
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--compile", action="store_true")

    # Model controls
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--memory_slots", type=int, default=16)
    parser.add_argument("--conv_kernel", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Diffusion controls
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--min_mask_rate", type=float, default=0.05)
    parser.add_argument("--max_mask_rate", type=float, default=0.95)
    parser.add_argument("--confidence_stop", type=float, default=0.98)
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

    args = parser.parse_args()

    # Auto hardware defaults
    try:
        auto_bs, auto_ga, auto_sl = setup_environment(allow_cpu=args.allow_cpu)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    if args.batch_size is None:
        args.batch_size = auto_bs
    if args.grad_accum is None:
        args.grad_accum = auto_ga
    if args.seq_len is None:
        args.seq_len = auto_sl

    data_dir = _resolve_path(args.data_dir)
    train_bin = data_dir / "train.bin"

    if args.skip_setup and train_bin.exists():
        data_path = str(train_bin)
        print(f"[OK] Using existing data: {data_path}")
    else:
        if args.skip_setup and not train_bin.exists():
            print(
                f"[WARN] --skip_setup was set but {train_bin} is missing. "
                "Running full data setup."
            )
        data_path = prepare_data(args.data_dir, args.raw_dir)

    # train_eco expects --data_dir containing train.bin/meta.json.
    args.data_dir = str(Path(data_path).parent)
    train_eco(args)


if __name__ == "__main__":
    main()
