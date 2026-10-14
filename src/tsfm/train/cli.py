"""``tsfm pack`` and ``tsfm train``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm.cli import register


def _pack(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("pack", help="pack Parquet splits into memory-mappable training files")
    p.add_argument("--splits", nargs="*", default=["pretrain", "val", "synthetic", "synthetic_val"])

    def run(a: argparse.Namespace) -> int:
        from tsfm.paths import processed_dir
        from tsfm.train.data import pack

        root = processed_dir()
        groups = {
            "pretrain": [
                f for f in (root / "pretrain").glob("*.parquet") if f.stem != "kernelsynth"
            ],
            "val": [f for f in (root / "val").glob("*.parquet") if f.stem != "kernelsynth"],
            "synthetic": [root / "pretrain" / "kernelsynth.parquet"],
            "synthetic_val": [root / "val" / "kernelsynth.parquet"],
        }
        for name in a.splits:
            meta = pack(groups[name], root / "packed" / name)
            print(
                name,
                meta["n_series"],
                "series",
                meta["n_points"],
                "points",
                meta["values_sha256"][:12],
            )
        return 0

    p.set_defaults(func=run)


def _train(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train", help="train (or resume) a run from a YAML config")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--max-steps-this-session", type=int)
    p.add_argument(
        "--time-budget-min", type=float, help="stop and checkpoint after this many minutes"
    )

    def run(a: argparse.Namespace) -> int:
        from tsfm.train.loop import train

        s = train(
            a.config,
            a.run_dir,
            max_steps_this_session=a.max_steps_this_session,
            time_budget_min=a.time_budget_min,
        )
        print(json.dumps(s, indent=1, default=str))
        return 0

    p.set_defaults(func=run)


def _bench(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("bench", help="measure training throughput per config on this device")
    p.add_argument("--configs", nargs="+", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--run-root", type=Path, default=Path("/tmp/tsfm-bench"))

    def run(a: argparse.Namespace) -> int:
        from tsfm.train.bench import run_bench

        print("wrote", run_bench(a.configs, a.steps, a.precision, a.run_root))
        return 0

    p.set_defaults(func=run)


register(_pack)
register(_train)
register(_bench)
