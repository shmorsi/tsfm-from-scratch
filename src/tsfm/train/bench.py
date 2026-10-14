"""Throughput benchmark: time a few optimiser steps per config on the current device.

Used to replace back-of-envelope compute estimates with measurements before committing
to the scaling grid. Results go to ``results/bench/<device>.json``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from tsfm.paths import results_dir
from tsfm.provenance import write_result
from tsfm.train.loop import RunConfig, Trainer, tokens_per_window


def bench_config(path: Path, steps: int, precision: str | None, run_root: Path) -> dict[str, Any]:
    rc = RunConfig.load(path)
    if precision:
        rc.train.precision = precision
    rc.train.steps = 10**9  # never "finish"; we only time steps
    tr = Trainer(rc, run_root / rc.name)
    dev = tr.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):  # warm-up (allocator, kernels)
        tr.train_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        tr.train_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    tokens = rc.train.batch_size * tokens_per_window(rc)
    n_body = tr.model.num_params(body_only=True)
    with open(path) as f:
        target_steps = int(__import__("yaml").safe_load(f)["train"]["steps"])
    return {
        "config": path.name,
        "params_body": n_body,
        "params_total": tr.model.num_params(),
        "seconds_per_step": dt,
        "tokens_per_second": tokens / dt,
        "achieved_tflops_6ND": 6 * n_body * tokens / dt / 1e12,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
        "precision": rc.train.precision,
        "configured_steps": target_steps,
        "projected_hours": target_steps * dt / 3600,
    }


def run_bench(paths: list[Path], steps: int, precision: str | None, run_root: Path) -> Path:
    dev = "cpu"
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0).replace(" ", "_")
    rows = []
    for p in paths:
        r = bench_config(p, steps, precision, run_root)
        rows.append(r)
        print(
            f"{r['config']:22s} {r['params_body'] / 1e6:6.1f}M  {r['seconds_per_step']:.3f}s/step  {r['tokens_per_second']:,.0f} tok/s  {r['achieved_tflops_6ND']:.1f} TFLOP/s  ~{r['projected_hours']:.2f} h",
            flush=True,
        )
    out = results_dir() / "bench" / f"{dev}.json"
    write_result(
        out, {"device": dev, "rows": rows, "timed_steps": steps}, split="none", label="M3-bench"
    )
    return out
