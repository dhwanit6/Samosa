"""
Colab-ready one-shot runner for EcoHybrid diffusion training.

What it does:
1. Detects GPU and chooses safe defaults.
2. Auto-downloads Gujarati-centric data if missing.
3. Auto-trains tokenizer and preprocesses to binary if missing.
4. Trains EcoHybrid diffusion model.

Usage (Colab):
    !git clone <your-repo-url> Aria
    %cd Aria
    !python -m colab_train_eco --max_steps 500
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import unicodedata
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
    from diffusion.train import train as train_unified  # noqa: E402
except ModuleNotFoundError:
    from train import train as train_unified  # type: ignore # noqa: E402

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

        _GUJ_CHAR_RE = re.compile(r"[\u0A80-\u0AFF]")
        _LATIN_RE = re.compile(r"[A-Za-z]")
        _URL_RE = re.compile(r"(https?://|www\.)", flags=re.IGNORECASE)
        _SPACE_RE = re.compile(r"\s+")

        _GUJ_TOKEN_HINTS = (
            "che", "chhe", "ane", "nathi", "karvu", "tame", "hu", "shu", "kem", "pan",
            "mate", "chho", "hatu", "hati", "hata", "kai", "koi", "pachi", "aaje", "kal",
        )

        def _normalize_text(s: str) -> str:
            s = unicodedata.normalize("NFKC", s)
            s = s.replace("\u200c", "").replace("\u200d", "")
            s = _SPACE_RE.sub(" ", s).strip()
            return s

        def _char_ratio(text: str, pattern: re.Pattern[str]) -> float:
            if not text:
                return 0.0
            return sum(1 for ch in text if pattern.match(ch)) / max(len(text), 1)

        def _looks_high_quality_gujarati(text: str, strict: bool) -> bool:
            if len(text) < (45 if strict else 28):
                return False
            if _URL_RE.search(text):
                return False

            gu_ratio = _char_ratio(text, _GUJ_CHAR_RE)
            latin_ratio = _char_ratio(text, _LATIN_RE)
            punct_ratio = sum(1 for ch in text if unicodedata.category(ch).startswith("P")) / max(len(text), 1)

            if strict:
                if gu_ratio < 0.42:
                    return False
                if latin_ratio > 0.18:
                    return False
                if punct_ratio > 0.24:
                    return False
            else:
                if gu_ratio < 0.30:
                    return False
                if latin_ratio > 0.30:
                    return False
                if punct_ratio > 0.30:
                    return False

            # Reject obvious duplicated spam patterns.
            if any(ch * 6 in text for ch in ("!", "?", ".", ",", "-", "_", "*")):
                return False

            return True

        def _to_roman_gujarati(text: str) -> str:
            """
            Gujarati -> ASCII-leaning Roman transliteration.
            Uses indic_transliteration when available; falls back to a coarse map.
            """
            try:
                from indic_transliteration import sanscript  # type: ignore
                from indic_transliteration.sanscript import transliterate  # type: ignore

                out = transliterate(text, sanscript.GUJARATI, sanscript.ITRANS)
                out = out.replace("~N", "n").replace(".N", "n").replace(".m", "m")
                out = out.replace("RRi", "ri").replace("RRI", "ri")
                out = _SPACE_RE.sub(" ", out).strip()
                return out
            except Exception:
                # Fallback is intentionally simple and approximate.
                base_map = {
                    "અ": "a", "આ": "aa", "ઇ": "i", "ઈ": "ii", "ઉ": "u", "ઊ": "uu",
                    "ઋ": "ri", "એ": "e", "ઐ": "ai", "ઓ": "o", "ઔ": "au",
                    "ક": "k", "ખ": "kh", "ગ": "g", "ઘ": "gh", "ચ": "ch", "છ": "chh",
                    "જ": "j", "ઝ": "jh", "ટ": "t", "ઠ": "th", "ડ": "d", "ઢ": "dh",
                    "ણ": "n", "ત": "t", "થ": "th", "દ": "d", "ધ": "dh", "ન": "n",
                    "પ": "p", "ફ": "ph", "બ": "b", "ભ": "bh", "મ": "m",
                    "ય": "y", "ર": "r", "લ": "l", "વ": "v", "શ": "sh", "ષ": "sh",
                    "સ": "s", "હ": "h", "ળ": "l",
                    "ા": "aa", "િ": "i", "ી": "ii", "ુ": "u", "ૂ": "uu",
                    "ે": "e", "ૈ": "ai", "ો": "o", "ૌ": "au", "ં": "n", "ઃ": "h",
                    "્": "", "઼": "",
                    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
                    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
                }
                out = "".join(base_map.get(ch, ch) for ch in text)
                out = _SPACE_RE.sub(" ", out).strip()
                return out

        def _looks_reasonable_romanized(text: str) -> bool:
            if len(text) < 20:
                return False
            latin_ratio = _char_ratio(text, _LATIN_RE)
            if latin_ratio < 0.55:
                return False
            low = text.lower()
            hint_hits = sum(1 for tok in _GUJ_TOKEN_HINTS if tok in low)
            return hint_hits >= 1

        def _to_gujlish(roman_text: str) -> str:
            """
            Light Gujarati+English code-mix transformation for chat-style robustness.
            """
            swaps = {
                "ane": "and",
                "pan": "but",
                "che": "is",
                "chhe": "is",
                "nathi": "not",
                "saru": "good",
                "kharab": "bad",
                "madad": "help",
                "samaj": "understand",
                "jaldi": "quick",
                "aaje": "today",
                "kal": "tomorrow",
            }
            toks = roman_text.split()
            if not toks:
                return roman_text
            out = toks[:]
            n_changes = max(1, int(round(0.15 * len(out))))
            idxs = np.random.permutation(len(out))[:n_changes]
            for idx in idxs:
                key = out[idx].lower().strip(".,!?;:")
                if key in swaps:
                    out[idx] = swaps[key]
            if np.random.rand() < 0.30:
                out.append("please")
            if np.random.rand() < 0.20:
                out.append("thanks")
            return _SPACE_RE.sub(" ", " ".join(out)).strip()

        def _alignment_values_pack(include_romanized: bool) -> list[str]:
            """
            Small, high-quality values corpus seed for empathy + ecological ethics.
            """
            native = [
                "હું હંમેશાં સહાનુભૂતિ, સત્ય અને અહિંસા સાથે વાત કરું છું.",
                "હર વ્યક્તિનો માન-સન્માન કરવો એ મારી પ્રથમ જવાબદારી છે.",
                "પ્રકૃતિ, પાણી, વૃક્ષો અને દરેક પ્રાણી માટે દયા રાખવી જોઈએ.",
                "મારા જવાબો શાંતિ, સમતોલતા અને સહકારને પ્રોત્સાહિત કરે છે.",
                "હું અહંકાર નહીં, સેવા અને વિનમ્રતા પસંદ કરું છું.",
                "મતભેદમાં પણ સંવાદ અને કરુણાથી ઉકેલ શોધવો શ્રેષ્ઠ છે.",
                "હું કોઈને નુકસાન કરતું માર્ગદર્શન નથી આપતો.",
                "હું ભય નહીં, આશા અને જવાબદારી તરફ દોરી જાઉં છું.",
                "અન્ય સંસ્કૃતિઓનો આદર કરવો એ માનવતા માટે જરૂરી છે.",
                "ધરતી બધાની છે; વિકાસ અને પર્યાવરણ બંનેનું સંતુલન રાખવું પડે.",
            ]
            if not include_romanized:
                return native
            roman = [
                "Hu hamesha sahanubhuti, satya ane ahimsa sathe vaat karu chu.",
                "Badha loko no maan ane samman rakhvu mari pehli zimmedari che.",
                "Prakruti, paani, vruksho ane praniyo prati daya jaruri che.",
                "Mara jawab shanti, sahkar ane santulan taraf lai jay che.",
                "Hu ahankar karta vinamrata ane seva pasand karu chu.",
                "Matbhed hoy to pan karuna sathe samvad thi ukel shodhvo joiye.",
                "Hu koi ne nuksan thae evu margdarshan aapto nathi.",
                "Dharati badha ni che, etle ecology ane progress banne jaruri che.",
            ]
            return native + roman

        def prepare_data(
            data_dir: str,
            raw_dir: str,
            language: str = "gu",
            include_romanized: bool = True,
            romanized_ratio: float = 0.40,
            include_gujlish: bool = True,
            gujlish_ratio: float = 0.20,
            include_values_pack: bool = True,
            values_pack_repeat: int = 8,
            quality_profile: str = "strict",
            max_docs: int = 160_000,
            max_chars: int = 80_000_000,
            vocab_size: int = 20_000,
        ) -> str:
            data_dir_path = _resolve_path(data_dir)
            raw_dir_path = _resolve_path(raw_dir)
            train_bin = data_dir_path / "train.bin"
            meta = data_dir_path / "meta.json"

            print("[INFO] runners helper scripts not found. Using built-in auto data bootstrap.")
            data_dir_path.mkdir(parents=True, exist_ok=True)
            raw_dir_path.mkdir(parents=True, exist_ok=True)
            data_root = data_dir_path.parent
            tok_dir = data_root / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)

            lang = str(language).strip().lower()
            strict = quality_profile == "strict"
            corpus_path = raw_dir_path / f"{lang}_bootstrap.txt"
            sp_prefix = tok_dir / f"aria_{lang}"
            sp_model = sp_prefix.with_suffix(".model")
            romanized_ratio = max(0.0, min(float(romanized_ratio), 1.0))
            gujlish_ratio = max(0.0, min(float(gujlish_ratio), 1.0))

            if train_bin.exists() and meta.exists():
                try:
                    prev = json.loads(meta.read_text(encoding="utf-8"))
                except Exception:
                    prev = {}
                mismatch: list[str] = []
                if str(prev.get("language", "")).lower() != lang:
                    mismatch.append(f"language={prev.get('language')} -> {lang}")
                if bool(prev.get("include_romanized", False)) != bool(include_romanized):
                    mismatch.append(
                        f"include_romanized={prev.get('include_romanized')} -> {include_romanized}"
                    )
                if bool(prev.get("include_gujlish", False)) != bool(include_gujlish):
                    mismatch.append(f"include_gujlish={prev.get('include_gujlish')} -> {include_gujlish}")
                if str(prev.get("quality_profile", "")).lower() != str(quality_profile).lower():
                    mismatch.append(f"quality_profile={prev.get('quality_profile')} -> {quality_profile}")
                if not mismatch:
                    print(f"[OK] Reusing existing processed data: {train_bin}")
                    return str(train_bin)
                print("[INFO] Existing processed data does not match requested Gujarati setup.")
                print("[INFO] Rebuilding due to: " + "; ".join(mismatch))

            def _extract_text(row: dict) -> str:
                for key in ("text", "content", "sentence"):
                    val = row.get(key)
                    if isinstance(val, str):
                        return val
                tr = row.get("translation")
                if isinstance(tr, dict):
                    for key in (lang, "gu", "hin", "hi", "en"):
                        val = tr.get(key)
                        if isinstance(val, str):
                            return val
                return ""

            def _download_corpus(max_docs: int, max_chars: int) -> tuple[int, int, int]:
                try:
                    from datasets import load_dataset
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "datasets package is required for built-in data bootstrap. "
                        "Install with: pip install datasets"
                    ) from exc

                # Gujarati-first high-quality sources.
                candidates = [
                    ("wikimedia/wikipedia", f"20231101.{lang}", "train"),
                    ("wikimedia/wikipedia", f"20220301.{lang}", "train"),
                    ("mc4", lang, "train"),
                ]
                last_err: Exception | None = None

                seen: set[str] = set()
                kept_native = 0
                kept_roman = 0
                kept_gujlish = 0
                chars = 0

                for name, config, split in candidates:
                    try:
                        print(f"[INFO] Downloading corpus from {name} ({config})...")
                        ds = load_dataset(name, config, split=split, streaming=True)
                        with corpus_path.open("w", encoding="utf-8") as f:
                            for row in ds:
                                raw = _extract_text(row)
                                text = _normalize_text(raw)
                                if not _looks_high_quality_gujarati(text, strict=strict):
                                    continue
                                key = text.lower()
                                if key in seen:
                                    continue
                                seen.add(key)

                                f.write(text + "\n")
                                kept_native += 1
                                chars += len(text)

                                if include_romanized and romanized_ratio > 0.0:
                                    if np.random.rand() < romanized_ratio:
                                        rom = _to_roman_gujarati(text)
                                        rom = _normalize_text(rom)
                                        if _looks_reasonable_romanized(rom):
                                            f.write(rom + "\n")
                                            kept_roman += 1
                                            chars += len(rom)
                                            if include_gujlish and gujlish_ratio > 0.0 and np.random.rand() < gujlish_ratio:
                                                mix = _to_gujlish(rom)
                                                if _looks_reasonable_romanized(mix):
                                                    f.write(mix + "\n")
                                                    kept_gujlish += 1
                                                    chars += len(mix)

                                if kept_native >= max_docs or chars >= max_chars:
                                    break
                        if kept_native == 0:
                            raise RuntimeError(f"No usable text extracted from {name}:{config}")
                        print(
                            f"[OK] Collected native={kept_native:,}, romanized={kept_roman:,}, "
                            f"gujlish={kept_gujlish:,} "
                            f"({chars/1e6:.1f}M chars)"
                        )
                        if include_values_pack and values_pack_repeat > 0:
                            values = _alignment_values_pack(include_romanized=include_romanized)
                            with corpus_path.open("a", encoding="utf-8") as f:
                                for _ in range(max(1, int(values_pack_repeat))):
                                    for line in values:
                                        f.write(_normalize_text(line) + "\n")
                        return kept_native, kept_roman, kept_gujlish
                    except Exception as exc:
                        last_err = exc
                        print(f"[WARN] Failed source {name}:{config} -> {exc}")
                raise RuntimeError("All built-in dataset sources failed") from last_err

            def _train_tokenizer(vocab_size: int) -> int:
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
                    "language": lang,
                    "quality_profile": quality_profile,
                    "include_romanized": bool(include_romanized),
                    "romanized_ratio": float(romanized_ratio),
                    "include_gujlish": bool(include_gujlish),
                    "gujlish_ratio": float(gujlish_ratio),
                    "include_values_pack": bool(include_values_pack),
                    "values_pack_repeat": int(values_pack_repeat),
                }
                meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                print(f"[OK] Wrote {train_bin} ({n_tokens:,} tokens)")

            native_count, roman_count, gujlish_count = _download_corpus(max_docs=max_docs, max_chars=max_chars)
            vocab_size = _train_tokenizer(vocab_size=vocab_size)
            _encode_bin(vocab_size)
            print(
                f"[DATA] language={lang} | native_lines={native_count:,} | "
                f"romanized_lines={roman_count:,} | gujlish_lines={gujlish_count:,} | "
                f"tokenizer_vocab={vocab_size}"
            )
            return str(train_bin)


def main():
    parser = argparse.ArgumentParser(description="Colab one-shot EcoHybrid diffusion runner")
    parser.add_argument("--skip_setup", action="store_true", help="Skip auto data prep if train.bin exists")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="checkpoints/eco_hybrid_gu")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU-only run.")
    parser.add_argument(
        "--language",
        type=str,
        default="gu",
        help="Bootstrap language code for auto data setup (default: gu)",
    )
    parser.add_argument(
        "--include_romanized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Gujarati text in Roman script alongside native script",
    )
    parser.add_argument(
        "--romanized_ratio",
        type=float,
        default=0.40,
        help="Approximate fraction of accepted lines additionally written in Roman script",
    )
    parser.add_argument(
        "--include_gujlish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add Gujarati+English code-mixed Roman lines",
    )
    parser.add_argument(
        "--gujlish_ratio",
        type=float,
        default=0.20,
        help="For romanized lines, probability of adding a code-mixed Gujlish variant",
    )
    parser.add_argument(
        "--include_values_pack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append compact empathy/ecology values corpus for alignment priming",
    )
    parser.add_argument(
        "--values_pack_repeat",
        type=int,
        default=8,
        help="How many times to repeat values pack lines in bootstrap corpus",
    )
    parser.add_argument(
        "--quality_profile",
        type=str,
        default="strict",
        choices=["strict", "balanced"],
        help="strict favors cleaner corpus; balanced favors more coverage",
    )
    parser.add_argument("--bootstrap_max_docs", type=int, default=160_000)
    parser.add_argument("--bootstrap_max_chars", type=int, default=80_000_000)
    parser.add_argument("--tokenizer_vocab_size", type=int, default=20_000)

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
        kwargs = {
            "language": args.language,
            "include_romanized": args.include_romanized,
            "romanized_ratio": args.romanized_ratio,
            "include_gujlish": args.include_gujlish,
            "gujlish_ratio": args.gujlish_ratio,
            "include_values_pack": args.include_values_pack,
            "values_pack_repeat": args.values_pack_repeat,
            "quality_profile": args.quality_profile,
            "max_docs": args.bootstrap_max_docs,
            "max_chars": args.bootstrap_max_chars,
            "vocab_size": args.tokenizer_vocab_size,
        }
        try:
            sig = inspect.signature(prepare_data)
            supported = {
                k: v for k, v in kwargs.items()
                if k in sig.parameters
            }
        except Exception:
            supported = {}
        data_path = prepare_data(args.data_dir, args.raw_dir, **supported)

    # train expects --data_dir containing train.bin/meta.json.
    args.data_dir = str(Path(data_path).parent)
    args.model = "primary"
    train_unified(args)


if __name__ == "__main__":
    main()
