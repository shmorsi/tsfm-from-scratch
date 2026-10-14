"""Independent NumPy implementations of the official metrics and their aggregation.

These exist to pin down exactly what the official numbers mean (tests check they agree
with GluonTS on random data) and to compute summaries such as the geometric mean of
Seasonal-Naive-normalised scores and bootstrap intervals.

With ``axis=None`` (as upstream calls ``evaluate_model``) GluonTS pools all forecast
windows of a configuration:

* MASE = mean over all (window, step) of |y - median| / s_w, where s_w is the window's
  seasonal error: mean |y_t - y_{t-m}| over that window's observed history.
* "CRPS" (mean_weighted_sum_quantile_loss) = mean over the nine levels q in 0.1..0.9 of
  2 * sum(pinball_q) / sum(|y|), sums over all windows and steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def seasonal_error(past: np.ndarray, m: int) -> float:
    past = np.asarray(past, np.float64)
    if past.size <= m:
        m = 1
    d = np.abs(past[m:] - past[:-m])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else np.nan


def mase(
    ys: Sequence[np.ndarray], medians: Sequence[np.ndarray], pasts: Sequence[np.ndarray], m: int
) -> float:
    num, cnt = 0.0, 0
    for y, f, p in zip(ys, medians, pasts, strict=True):
        s = seasonal_error(p, m)
        e = np.abs(np.asarray(y, np.float64) - f) / s
        ok = np.isfinite(e)
        num += float(e[ok].sum())
        cnt += int(ok.sum())
    return num / cnt


def weighted_quantile_loss(
    ys: Sequence[np.ndarray], qf: Sequence[dict[float, np.ndarray]], levels: Sequence[float]
) -> float:
    out = []
    for q in levels:
        loss, scale = 0.0, 0.0
        for y, f in zip(ys, qf, strict=True):
            y = np.asarray(y, np.float64)
            ok = np.isfinite(y)
            d = y[ok] - f[q][ok]
            loss += float(np.maximum(q * d, (q - 1) * d).sum())
            scale += float(np.abs(y[ok]).sum())
        out.append(2 * loss / scale)
    return float(np.mean(out))


def gmean(x: Sequence[float]) -> float:
    a = np.asarray(x, np.float64)
    if (a <= 0).any() or not np.isfinite(a).all():
        raise ValueError("geometric mean needs positive finite values")
    return float(np.exp(np.log(a).mean()))


def normalised_gmean(model: dict[str, float], baseline: dict[str, float]) -> float:
    """Geometric mean over configs of model / baseline (the leaderboard's aggregate)."""
    if model.keys() != baseline.keys():
        raise ValueError(f"config mismatch: {sorted(model.keys() ^ baseline.keys())[:5]}")
    return gmean([model[k] / baseline[k] for k in sorted(model)])


def paired_bootstrap_ratio(
    a: dict[str, float], b: dict[str, float], ref: dict[str, float], n: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Ratio of normalised geometric means a/b with a 95% bootstrap CI over configs.

    Configs are resampled with replacement (paired: the same draw for a and b). Returns
    (point, lo, hi)."""
    keys = sorted(a)
    la = np.log([a[k] / ref[k] for k in keys])
    lb = np.log([b[k] / ref[k] for k in keys])
    d = la - lb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    boots = d[idx].mean(axis=1)
    return (
        float(np.exp(d.mean())),
        float(np.exp(np.quantile(boots, 0.025))),
        float(np.exp(np.quantile(boots, 0.975))),
    )
