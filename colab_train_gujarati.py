"""
Gujarati-first Colab profile for EcoHybrid diffusion training.

Primary datasets:
- https://huggingface.co/datasets/ai4bharat/sangraha
- https://huggingface.co/datasets/wikimedia/wikipedia
- https://huggingface.co/datasets/mc4
"""
from __future__ import annotations

import sys

from colab_train_eco import main as eco_main


def _has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _set_default(flag: str, value: str | None = None) -> None:
    if _has_flag(flag):
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def main() -> None:
    _set_default("--language", "gu")
    _set_default("--output_dir", "checkpoints/eco_hybrid_gu")
    _set_default("--quality_profile", "balanced")
    _set_default("--bootstrap_max_docs", "220000")
    _set_default("--bootstrap_max_chars", "120000000")
    _set_default("--bootstrap_min_native_docs", "30000")
    _set_default("--tokenizer_vocab_size", "24000")
    _set_default("--include_romanized")
    _set_default("--romanized_ratio", "0.35")
    _set_default("--include_gujlish")
    _set_default("--gujlish_ratio", "0.20")
    _set_default("--include_values_pack")
    eco_main()


if __name__ == "__main__":
    main()
