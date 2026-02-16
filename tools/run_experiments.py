"""
Ablation runner for matched diffusion experiments.

This script executes a controlled matrix over:
1. architecture: backup transformer vs primary EcoHybrid
2. timestep mode: discrete vs continuous
3. masking strategy: token vs span

All shared model/training hyperparameters are kept identical across runs
to make speed/quality comparisons fair.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentSpec:
    name: str
    module: str
    args: dict[str, str]


def _cmd_from_spec(spec: ExperimentSpec) -> list[str]:
    cmd = [sys.executable, "-m", spec.module]
    for key, value in spec.args.items():
        cmd.extend([f"--{key}", value])
    return cmd


def _build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    shared = {
        "data_dir": args.data_dir,
        "max_steps": str(args.max_steps),
        "save_interval": str(args.save_interval),
        "log_interval": str(args.log_interval),
        "seq_len": str(args.seq_len),
        "batch_size": str(args.batch_size),
        "grad_accum": str(args.grad_accum),
        "d_model": str(args.d_model),
        "n_layers": str(args.n_layers),
        "n_heads": str(args.n_heads),
        "d_ff": str(args.d_ff),
        "dropout": str(args.dropout),
        "timesteps": str(args.timesteps),
        "sample_steps": str(args.sample_steps),
        "min_mask_rate": str(args.min_mask_rate),
        "max_mask_rate": str(args.max_mask_rate),
        "timestep_sampling": args.timestep_sampling,
        "mean_span_length": str(args.mean_span_length),
        "block_size": str(args.block_size),
        "lr": str(args.lr),
        "weight_decay": str(args.weight_decay),
    }
    specs: list[ExperimentSpec] = []
    for arch in args.architectures:
        for time_mode in args.time_modes:
            for masking in args.masking_strategies:
                run_name = f"{arch}_{time_mode}_{masking}"
                out_dir = str(Path(args.output_root) / run_name)
                run_args = dict(shared)
                run_args.update(
                    {
                        "model": arch,
                        "output_dir": out_dir,
                        "time_mode": time_mode,
                        "masking_strategy": masking,
                    }
                )
                module = "train"
                if arch == "primary":
                    run_args.update(
                        {
                            "memory_slots": str(args.memory_slots),
                            "conv_kernel": str(args.conv_kernel),
                            "confidence_stop": str(args.confidence_stop),
                        }
                    )
                specs.append(ExperimentSpec(name=run_name, module=module, args=run_args))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run matched diffusion experiment matrix")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--output_root", type=str, default="checkpoints/ablations")
    p.add_argument("--execute", action="store_true", help="Execute runs. Default is dry-run.")

    p.add_argument(
        "--architectures",
        nargs="+",
        default=["backup", "primary"],
        choices=["backup", "primary"],
    )
    p.add_argument(
        "--time_modes",
        nargs="+",
        default=["discrete", "continuous"],
        choices=["discrete", "continuous"],
    )
    p.add_argument(
        "--masking_strategies",
        nargs="+",
        default=["token", "span"],
        choices=["token", "span"],
    )

    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--timesteps", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=12)
    p.add_argument("--min_mask_rate", type=float, default=0.05)
    p.add_argument("--max_mask_rate", type=float, default=0.95)
    p.add_argument("--timestep_sampling", type=str, default="stratified", choices=["uniform", "stratified"])
    p.add_argument("--mean_span_length", type=float, default=3.0)
    p.add_argument("--block_size", type=int, default=64)

    p.add_argument("--memory_slots", type=int, default=16)
    p.add_argument("--conv_kernel", type=int, default=7)
    p.add_argument("--confidence_stop", type=float, default=0.98)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = _build_specs(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "architectures": args.architectures,
        "time_modes": args.time_modes,
        "masking_strategies": args.masking_strategies,
        "runs": [{"name": s.name, "module": s.module, "args": s.args} for s in specs],
    }
    manifest_path = out_root / "experiment_plan.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[plan] wrote manifest: {manifest_path}")

    for idx, spec in enumerate(specs, start=1):
        cmd = _cmd_from_spec(spec)
        cmd_str = " ".join(shlex.quote(token) for token in cmd)
        print(f"[{idx}/{len(specs)}] {spec.name}")
        print(f"  {cmd_str}")
        if not args.execute:
            continue
        subprocess.run(cmd, check=True)

    if not args.execute:
        print("[dry-run] no training jobs executed. Re-run with --execute to launch.")


if __name__ == "__main__":
    main()
