"""
Compatibility wrapper.

Canonical entrypoint:
    python -m train --model backup ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Support:
# - python -m diffusion.train_diffusion
# - python -m train_diffusion
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR / "train"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from diffusion.train import parse_args, train
except ModuleNotFoundError:
    from train import parse_args, train  # type: ignore


def main() -> None:
    args = parse_args()
    args.model = "backup"
    if "--output_dir" not in sys.argv:
        args.output_dir = "checkpoints/diffusion_hi"
    train(args)


if __name__ == "__main__":
    main()
