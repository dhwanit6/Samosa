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
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
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
            raw_dir_path = _resolve_path(raw_dir)
            train_bin = data_dir_path / "train.bin"
            meta = data_dir_path / "meta.json"

            if train_bin.exists() and meta.exists():
                return str(train_bin)

            print("[INFO] runners helper scripts not found. Using built-in auto data bootstrap.")
            data_dir_path.mkdir(parents=True, exist_ok=True)
            raw_dir_path.mkdir(parents=True, exist_ok=True)
            data_root = data_dir_path.parent
            tok_dir = data_root / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)

            corpus_path = raw_dir_path / "hindi_bootstrap.txt"
            sp_prefix = tok_dir / "aria_hindi"
            sp_model = sp_prefix.with_suffix(".model")

            def _extract_text(row: dict) -> str:
                for key in ("text", "content", "sentence"):
                    val = row.get(key)
                    if isinstance(val, str):
                        return val
                tr = row.get("translation")
                if isinstance(tr, dict):
                    for key in ("hi", "hin", "en"):
                        val = tr.get(key)
                        if isinstance(val, str):
                            return val
                return ""

            def _download_corpus(max_docs: int = 120_000, max_chars: int = 60_000_000) -> None:
                try:
                    from datasets import load_dataset
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "datasets package is required for built-in data bootstrap. "
                        "Install with: pip install datasets"
                    ) from exc

                # Try high-quality Hindi sources first.
                candidates = [
                    ("wikimedia/wikipedia", "20231101.hi", "train"),
                    ("wikimedia/wikipedia", "20220301.hi", "train"),
                    ("mc4", "hi", "train"),
                ]
                last_err: Exception | None = None

                for name, config, split in candidates:
                    try:
                        print(f"[INFO] Downloading corpus from {name} ({config})...")
                        ds = load_dataset(name, config, split=split, streaming=True)
                        docs = 0
                        chars = 0
                        with corpus_path.open("w", encoding="utf-8") as f:
                            for row in ds:
                                text = _extract_text(row).strip()
                                if len(text) < 40:
                                    continue
                                f.write(text.replace("\n", " ").strip() + "\n")
                                docs += 1
                                chars += len(text)
                                if docs >= max_docs or chars >= max_chars:
                                    break
                        if docs == 0:
                            raise RuntimeError(f"No usable text extracted from {name}:{config}")
                        print(f"[OK] Collected {docs} documents ({chars/1e6:.1f}M chars)")
                        return
                    except Exception as exc:
                        last_err = exc
                        print(f"[WARN] Failed source {name}:{config} -> {exc}")
                raise RuntimeError("All built-in dataset sources failed") from last_err

            def _train_tokenizer(vocab_size: int = 16_000) -> int:
                print("[INFO] Training SentencePiece tokenizer...")
                spm.SentencePieceTrainer.train(
                    input=str(corpus_path),
                    model_prefix=str(sp_prefix),
                    vocab_size=vocab_size,
                    model_type="unigram",
                    character_coverage=0.9995,
                    input_sentence_size=2_000_000,
                    shuffle_input_sentence=True,
                    train_extremely_large_corpus=False,
                    hard_vocab_limit=False,
                    bos_id=-1,
                    eos_id=-1,
                    pad_id=0,
                    unk_id=1,
                )
                sp = spm.SentencePieceProcessor(model_file=str(sp_model))
                return int(sp.vocab_size())

            def _encode_bin(base_vocab: int) -> None:
                print("[INFO] Encoding corpus to train.bin...")
                sp = spm.SentencePieceProcessor(model_file=str(sp_model))
                dtype = np.uint16 if base_vocab < 65_535 else np.uint32
                n_tokens = 0

                with train_bin.open("wb") as out, corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        ids = sp.encode(text, out_type=int)
                        if not ids:
                            continue
                        np.asarray(ids, dtype=dtype).tofile(out)
                        n_tokens += len(ids)

                meta_payload = {
                    "vocab_size": int(base_vocab),
                    "dtype": "uint16" if dtype == np.uint16 else "uint32",
                    "train_tokens": int(n_tokens),
                    "tokenizer_model": str(sp_model),
                }
                meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                print(f"[OK] Wrote {train_bin} ({n_tokens:,} tokens)")

            _download_corpus()
            vocab_size = _train_tokenizer()
            _encode_bin(vocab_size)
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
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--resume_latest", action="store_true", help="Resume from output_dir/latest.pt")
    parser.add_argument("--reset_optimizer", action="store_true", help="Ignore optimizer state when resuming")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype when CUDA AMP is enabled",
    )

    # Model controls
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--memory_slots", type=int, default=16)
    parser.add_argument("--conv_kernel", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--edge_profile",
        type=str,
        default="none",
        choices=["none", "tiny", "laptop"],
        help="Apply edge-optimized model presets",
    )

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
