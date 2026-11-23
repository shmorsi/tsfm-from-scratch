"""HTTP forecasting service: ``uvicorn tsfm.serve.app:app``.

Environment:
    TSFM_CHECKPOINT   training checkpoint (default: the committed CPU model, seed 0)
    TSFM_ONNX         exported graph (default: artifacts/serve/model.onnx if present)
    TSFM_MAX_SERIES   largest batch one request may send (default 256)
    TSFM_THREADS      intra-op threads (default: library default)

``POST /forecast`` takes ``{"series": [[...], ...], "horizon": 24}``; missing values are
``null``. It answers quantiles for each series and says which backend computed them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from tsfm.serve.engine import Engine
from tsfm.serve.monitor import Monitor, synthetic_reference

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = ROOT / "results" / "runs" / "cpu_s0_seed0" / "model.pt"
DEFAULT_ONNX = ROOT / "artifacts" / "serve" / "model.onnx"
MAX_POINTS = 8192
MAX_HORIZON = 512


class ForecastRequest(BaseModel):
    series: Annotated[list[list[float | None]], Field(min_length=1)]
    horizon: Annotated[int, Field(ge=1, le=MAX_HORIZON)] = 32
    backend: str | None = Field(default=None, pattern="^(onnx|torch)$")


class ForecastResponse(BaseModel):
    levels: list[float]
    quantiles: list[list[list[float]]]  # series x horizon x level
    median: list[list[float]]
    backend: str
    milliseconds: float


def _engine() -> Engine:
    ckpt = Path(os.environ.get("TSFM_CHECKPOINT", DEFAULT_CKPT))
    onnx = Path(os.environ.get("TSFM_ONNX", DEFAULT_ONNX))
    threads = int(os.environ["TSFM_THREADS"]) if os.environ.get("TSFM_THREADS") else None
    return Engine.load(ckpt, onnx, threads=threads)


app = FastAPI(
    title="tsfm",
    version="0.1",
    description="Zero-shot probabilistic forecasts from a from-scratch time-series transformer.",
)
ENGINE: Engine | None = None
MONITOR: Monitor | None = None
DEFAULT_REFERENCE = ROOT / "artifacts" / "serve" / "reference_features.npy"


def engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = _engine()
    return ENGINE


def monitor() -> Monitor:
    global MONITOR
    if MONITOR is None:
        ref = Path(os.environ.get("TSFM_REFERENCE", DEFAULT_REFERENCE))
        MONITOR = Monitor(np.load(ref) if ref.exists() else synthetic_reference(n=200))
    return MONITOR


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    e = engine()
    return {
        "parameters": e.model.num_params(),
        "patch_len": e.model.cfg.patch_len,
        "horizon": e.horizon,
        "levels": list(e.model.cfg.quantiles),
        "max_context": e.max_context,
        "onnx": e.onnx_file,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text format: request counts, errors, latency histogram, input-drift PSI."""
    return monitor().prometheus()


@app.get("/drift")
def drift() -> dict[str, object]:
    """Population stability index of recent inputs against the reference, per feature."""
    return monitor().drift()


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        return _forecast(req)
    except HTTPException:
        monitor().errors += 1
        raise


def _forecast(req: ForecastRequest) -> ForecastResponse:
    e = engine()
    if len(req.series) > int(os.environ.get("TSFM_MAX_SERIES", "256")):
        raise HTTPException(413, "too many series in one request")
    ctx = []
    for s in req.series:
        if len(s) > MAX_POINTS:
            raise HTTPException(413, f"a series longer than {MAX_POINTS} points")
        a = np.array([np.nan if v is None else v for v in s], dtype=np.float32)
        if not np.isfinite(a).any():
            raise HTTPException(422, "a series with no observed value")
        ctx.append(a)
    t0 = time.perf_counter()
    try:
        f, used = e.forecast(ctx, req.horizon, req.backend)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    ms = 1000 * (time.perf_counter() - t0)
    monitor().observe(ctx, used, ms)
    return ForecastResponse(
        levels=list(f.levels),
        quantiles=f.quantiles.round(6).tolist(),
        median=f.mean.round(6).tolist(),
        backend=used,
        milliseconds=ms,
    )
