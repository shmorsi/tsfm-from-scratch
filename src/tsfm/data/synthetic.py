"""Synthetic series from Gaussian processes with random kernel mixtures.

Follows the idea of KernelSynth (Ansari et al., 2024, "Chronos"): draw 1..J kernels from
a bank, combine them with random ``+``/``*``, and sample one path from the zero-mean GP
on an evenly spaced grid. Written from the paper's description; no code is copied.

Every series is a deterministic function of ``(seed, index)``, so any single series can
be regenerated in isolation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tsfm.data.convert import SCHEMA
from tsfm.paths import processed_dir, results_dir
from tsfm.provenance import sha256_file, write_result

# A kernel maps (t, d) -> covariance, with t the (n, 1) time grid and d = t - t.T.
Kernel = Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]

# Periods (in time steps) typical of real calendars: hourly/daily/weekly/monthly/yearly.
PERIODS = (4, 6, 7, 10, 12, 14, 24, 26, 30, 48, 52, 60, 96, 168, 336, 365)


def _linear(c: float) -> Kernel:
    return lambda t, d: (t - c) * (t - c).T


def _rbf(ls: float) -> Kernel:
    return lambda t, d: torch.exp(-0.5 * d**2 / ls**2)


def _periodic(period: float, ls: float) -> Kernel:
    return lambda t, d: torch.exp(-2.0 * torch.sin(torch.pi * d.abs() / period) ** 2 / ls**2)


def _rq(ls: float, alpha: float) -> Kernel:
    return lambda t, d: (1.0 + d**2 / (2 * alpha * ls**2)) ** (-alpha)


def _white(sigma: float) -> Kernel:
    return lambda t, d: sigma**2 * torch.eye(d.shape[0], dtype=d.dtype)


def _const(c: float) -> Kernel:
    return lambda t, d: torch.full_like(d, c)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kernel: Kernel


def kernel_bank(length: int) -> list[KernelSpec]:
    """Hyper-parameters are on the unit time grid t in [0, 1]."""
    bank: list[KernelSpec] = []
    for c in (0.0, 0.5, 1.0):
        bank.append(KernelSpec(f"linear(c={c})", _linear(c)))
    for ls in (0.01, 0.05, 0.1, 0.3, 1.0):
        bank.append(KernelSpec(f"rbf(ls={ls})", _rbf(ls)))
        bank.append(KernelSpec(f"rq(ls={ls})", _rq(ls, 1.0)))
    for p in PERIODS:
        if p * 2 <= length:
            bank.append(KernelSpec(f"periodic(p={p})", _periodic(p / length, 1.0)))
    for s in (0.05, 0.2):
        bank.append(KernelSpec(f"white({s})", _white(s)))
    bank.append(KernelSpec("const(1)", _const(1.0)))
    return bank


def sample_series(seed: int, index: int, length: int, max_kernels: int) -> tuple[np.ndarray, str]:
    """One GP path. Randomness comes only from a NumPy generator seeded by (seed, index);
    the linear algebra runs in float64 PyTorch (much faster than this NumPy build's
    OpenBLAS Cholesky). Bits can differ across BLAS builds, so the shipped file's SHA-256
    is what pins the data, and regeneration elsewhere is checked statistically."""
    rng = np.random.default_rng([seed, index])
    bank = kernel_bank(length)
    t = torch.linspace(0.0, 1.0, length, dtype=torch.float64)[:, None]
    d = t - t.T
    n = int(rng.integers(1, max_kernels + 1))
    picks = [bank[int(i)] for i in rng.integers(0, len(bank), size=n)]
    k = picks[0].kernel(t, d)
    desc = picks[0].name
    for spec in picks[1:]:
        kk = spec.kernel(t, d)
        if rng.random() < 0.5:
            k, desc = k + kk, f"({desc} + {spec.name})"
        else:
            k, desc = k * kk, f"({desc} * {spec.name})"
    k = 0.5 * (k + k.T)
    z = torch.from_numpy(rng.standard_normal(length))
    jitter = 1e-6 * max(float(k.diagonal().mean()), 1e-6)
    eye = torch.eye(length, dtype=torch.float64)
    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(k + jitter * eye)
        if int(info) == 0:
            x = chol @ z
            break
        jitter *= 10
    else:  # numerically singular kernel: use an eigendecomposition instead
        w, v = torch.linalg.eigh(k)
        x = v @ (w.clamp(min=0).sqrt() * z)
    return x.numpy().astype(np.float32), desc


def _gen_chunk(
    args: tuple[int, int, int, int, int],
) -> tuple[list[str], list[np.ndarray], dict[str, int]]:
    seed, lo, hi, length, max_kernels = args
    torch.set_num_threads(1)
    ids: list[str] = []
    targets: list[np.ndarray] = []
    kinds: dict[str, int] = {}
    for i in range(lo, hi):
        x, desc = sample_series(seed, i, length, max_kernels)
        ids.append(f"kernelsynth/{i}")
        targets.append(x)
        n = str(desc.count(" + ") + desc.count(" * ") + 1)
        kinds[n] = kinds.get(n, 0) + 1
    return ids, targets, kinds


def build_synthetic(
    n_series: int | None = None,
    length: int | None = None,
    seed: int | None = None,
    max_kernels: int | None = None,
    val_series: int | None = None,
    workers: int | None = None,
) -> dict[str, object]:
    from tsfm.data.sources import load_data_config

    cfg = load_data_config()["synthetic"]
    n_series = n_series or cfg["n_series"]
    length = length or cfg["length"]
    seed = seed if seed is not None else cfg["seed"]
    max_kernels = max_kernels or cfg["max_kernels"]
    val_series = val_series if val_series is not None else cfg["val_series"]
    out: dict[str, object] = {}
    # Validation series use indices after the training ones, so they never coincide.
    for split, lo, hi in (("pretrain", 0, n_series), ("val", n_series, n_series + val_series)):
        path = processed_dir() / split / "kernelsynth.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path.with_suffix(".parquet.part"), SCHEMA, compression="zstd")
        kinds: dict[str, int] = {}
        chunks = [(seed, c, min(c + 1000, hi), length, max_kernels) for c in range(lo, hi, 1000)]
        # One BLAS thread per worker process: inherited by spawned children before they
        # import numpy, which avoids oversubscribing the cores.
        for var in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            os.environ[var] = "1"
        ex = ProcessPoolExecutor(workers or os.cpu_count() or 1)
        try:
            for ids, targets, chunk_kinds in ex.map(_gen_chunk, chunks):
                for k, v in chunk_kinds.items():
                    kinds[k] = kinds.get(k, 0) + v
                m = len(ids)
                writer.write_table(
                    pa.table(
                        {
                            "item_id": ids,
                            "dataset": ["kernelsynth"] * m,
                            "freq": ["H"] * m,
                            "start": pa.array([0] * m, pa.timestamp("ms")),
                            "target": targets,
                        },
                        schema=SCHEMA,
                    )
                )
        finally:
            # On error, do not wait for (and then discard) the remaining chunks.
            ex.shutdown(wait=True, cancel_futures=True)
        writer.close()
        path.with_suffix(".parquet.part").replace(path)
        out[f"{split}/kernelsynth.parquet"] = {
            "dataset": "kernelsynth",
            "split": split,
            "series_kept": hi - lo,
            "points_kept": (hi - lo) * length,
            "n_kernels_histogram": kinds,
            "sha256": sha256_file(path),
            "seed": seed,
        }
    write_result(
        results_dir() / "data" / "synthetic_manifest.json",
        {"files": out},
        split="none",
        label="M1-synthetic",
    )
    return out
