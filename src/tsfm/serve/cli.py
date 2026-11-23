"""``tsfm serve export | bench``: build the ONNX graphs and measure them. Imports are lazy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm.cli import register

DEFAULT_CKPT = Path("results/runs/cpu_s0_seed0/model.pt")
ARTIFACTS = Path("artifacts/serve")
RESULTS = Path("results/serve")


def _add(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("serve", help="export to ONNX, benchmark the backends, register in MLflow")
    p.add_argument("action", choices=["export", "bench", "register"])
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=ARTIFACTS)
    p.add_argument("--threads", type=int, default=1)

    def run(a: argparse.Namespace) -> int:
        if a.action == "export":
            from tsfm.serve.export import export_all

            rep = export_all(a.checkpoint, a.out)
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "export.json").write_text(json.dumps(rep, indent=2) + "\n")
            print(json.dumps({k: rep[k]["parity"] for k in ("fp32", "int8")}, indent=2))
            return 0
        if a.action == "register":
            from tsfm.serve.track import register as track

            print(json.dumps(track(a.checkpoint, a.out, RESULTS), indent=2))
            return 0
        from tsfm.serve.bench import run as bench
        from tsfm.serve.bench import write

        rep = bench(a.checkpoint, a.out, threads=a.threads)
        write(rep, RESULTS / f"bench_cpu_{a.threads}thread.json")
        print(json.dumps(rep["latency"], indent=2))
        print(json.dumps(rep["accuracy"], indent=2))
        return 0

    p.set_defaults(func=run)


register(_add)
