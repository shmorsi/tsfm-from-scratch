"""Scaling fits: validation loss vs training compute (milestone 6).

Each run's ``summary.json`` holds its training compute (``train_flops_6ND``) and its loss
on the same fixed validation batches. We fit a power law ``L = a * C**(-b)`` by least
squares in log-log space and get 95% intervals for ``a``, ``b`` and the fitted curve by a
*paired* bootstrap over validation batches (every resample uses the same batches for all
runs, so the runs stay comparable). With 4-6 points this is a description of these runs,
not an extrapolation claim; the write-up says so.

Losses are only comparable within one tokenizer (pinball vs cross-entropy), so the fit
is done per tokenizer; the tokenizer comparison uses forecast metrics on the validation
split instead (``results/analysis/val_summary.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.paths import results_dir
from tsfm.provenance import write_result


def load_runs(root: Path | None = None) -> list[dict[str, Any]]:
    """Training summaries under ``root``, one dict per run."""
    root = root or results_dir() / "runs"
    runs = []
    for p in sorted(root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if not s.get("done") or not s.get("val_loss_batches"):
            continue
        runs.append(
            {
                "name": p.parent.name,
                "tokenizer": s["config"]["model"]["tokenizer"],
                "seed": s["config"]["seed"],
                "params_body": s["params_body"],
                "train_tokens": s["train_tokens"],
                "flops": s["train_flops_6ND"],
                "val_batches": s["val_loss_batches"],
            }
        )
    return runs


def fit_power_law(c: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    """Least squares on log L = log a - b log C. Returns (a, b)."""
    x, y = np.log(c), np.log(loss)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)


def bootstrap_fit(runs: list[dict[str, Any]], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Fit the power law and a paired bootstrap band over validation batches."""
    c = np.array([r["flops"] for r in runs], dtype=np.float64)
    batches = np.array([r["val_batches"] for r in runs], dtype=np.float64)  # (runs, batches)
    if batches.ndim != 2:
        raise ValueError("runs must have the same number of validation batches")
    a, b = fit_power_law(c, batches.mean(axis=1))
    rng = np.random.default_rng(seed)
    nb = batches.shape[1]
    grid = np.geomspace(c.min(), c.max(), 50)
    bs, curves = [], []
    for _ in range(n):
        idx = rng.integers(0, nb, size=nb)
        aa, bb = fit_power_law(c, batches[:, idx].mean(axis=1))
        bs.append((aa, bb))
        curves.append(aa * grid ** (-bb))
    arr = np.array(bs)
    cv = np.array(curves)
    return {
        "n_runs": len(runs),
        "a": a,
        "b": b,
        "a_ci95": [float(np.quantile(arr[:, 0], 0.025)), float(np.quantile(arr[:, 0], 0.975))],
        "b_ci95": [float(np.quantile(arr[:, 1], 0.025)), float(np.quantile(arr[:, 1], 0.975))],
        "points": [
            {
                "name": r["name"],
                "flops": r["flops"],
                "params_body": r["params_body"],
                "val_loss": float(np.mean(r["val_batches"])),
            }
            for r in runs
        ],
        "curve": {
            "flops": grid.tolist(),
            "fit": (a * grid ** (-b)).tolist(),
            "lo": np.quantile(cv, 0.025, axis=0).tolist(),
            "hi": np.quantile(cv, 0.975, axis=0).tolist(),
        },
    }


def scaling_report(names_prefix: str = "patch_s") -> Path:
    """Write ``results/analysis/scaling.json`` for runs whose names start with ``names_prefix``."""
    runs = [r for r in load_runs() if r["name"].startswith(names_prefix) and r["seed"] == 0]
    out: dict[str, Any] = {"prefix": names_prefix}
    if len(runs) >= 3:
        out["fit"] = bootstrap_fit(runs)
    else:
        out["fit"] = None
        out["note"] = f"only {len(runs)} finished runs; need at least 3 to fit"
    return write_result(
        results_dir() / "analysis" / "scaling.json", out, split="val", label="M6-scaling"
    )
