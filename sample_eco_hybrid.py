"""
Compatibility wrapper.

Canonical entrypoint:
    python -m sample --model primary ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Support:
# - python -m diffusion.sample_eco_hybrid
# - python -m sample_eco_hybrid
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from diffusion.sample import main as unified_main
except ModuleNotFoundError:
    from sample import main as unified_main  # type: ignore


def _inject_default_model() -> None:
    for arg in sys.argv[1:]:
        if arg == "--model" or arg.startswith("--model="):
            return
    sys.argv.extend(["--model", "primary"])


if __name__ == "__main__":
    _inject_default_model()
    unified_main()
