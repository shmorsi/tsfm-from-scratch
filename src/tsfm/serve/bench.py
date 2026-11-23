"""Latency of the three backends, and what int8 quantisation costs in accuracy.

Latency: one request = one call of the engine (pre-processing included), contexts of 512
points, horizon 32, for batch sizes 1, 8 and 64; a warm-up, then timed repetitions; p50 and
p95 in milliseconds and series per second. Threads are pinned so the numbers are comparable.

Accuracy: Gaussian-process series from the repository's own synthetic generator (a fixed seed
that is not the one the pretraining corpus used), the last 32 points held out; the mean scaled
quantile loss (the CRPS approximation the evaluation uses) of each backend.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tsfm.serve.engine import Engine
from tsfm.serve.export import session

SEED = 20261123  # held-out synthetic draws: not the pretraining corpus's seed


def synthetic(n: int, length: int) -> np.ndarray:
    from tsfm.data.synthetic import sample_series

    return np.stack([sample_series(SEED, i, length, 3)[0] for i in range(n)])


def quantile_loss(q: np.ndarray, y: np.ndarray, levels: tuple[float, ...]) -> float:
    """Mean over series of sum_l pinball_l / sum |y| (scaled, as GluonTS's mean weighted quantile loss)."""
    lv = np.asarray(levels)[None, None, :]
    e = y[..., None] - q
    pin = np.maximum(lv * e, (lv - 1) * e)
    return float(np.mean(2 * pin.sum(axis=1).mean(axis=-1) / np.abs(y).sum(axis=1).clip(1e-8)))


def _time(fn: Callable[[], object], reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(1000 * (time.perf_counter() - t))
    return out


def run(
    checkpoint: Path, onnx_dir: Path, threads: int = 1, reps: int = 30, n_series: int = 256
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    eng = Engine.load(checkpoint, None, threads=threads)
    sessions = {
        "onnx_fp32": session(onnx_dir / "model.onnx", threads),
        "onnx_int8": session(onnx_dir / "model.int8.onnx", threads),
    }
    H, C = 32, 512
    data = synthetic(n_series, C + H)
    ctx, y = data[:, :C], data[:, C:]
    lat: dict[str, Any] = {}
    acc: dict[str, float] = {}
    for name in ("torch", "onnx_fp32", "onnx_int8"):
        if name != "torch":
            eng.onnx_session = sessions[name]
        backend = "torch" if name == "torch" else "onnx"
        f, _ = eng.forecast(list(ctx), H, backend)
        acc[name] = quantile_loss(f.quantiles, y, f.levels)
        lat[name] = {}
        for b in (1, 8, 64):
            batch = list(ctx[:b])
            for _ in range(3):
                eng.forecast(batch, H, backend)
            ms = _time(partial(eng.forecast, batch, H, backend), reps)
            p50 = float(np.percentile(ms, 50))
            lat[name][str(b)] = {
                "p50_ms": p50,
                "p95_ms": float(np.percentile(ms, 95)),
                "series_per_s": 1000 * b / p50,
            }
    return {
        "checkpoint": str(checkpoint),
        "threads": threads,
        "reps": reps,
        "context": C,
        "horizon": H,
        "machine": {
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "latency": lat,
        "accuracy": {
            "series": n_series,
            "seed": SEED,
            "quantile_loss": acc,
            "int8_vs_fp32_ratio": acc["onnx_int8"] / acc["onnx_fp32"],
            "onnx_vs_torch_ratio": acc["onnx_fp32"] / acc["torch"],
        },
    }


def write(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
