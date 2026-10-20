"""Split-conformal adjustment for the patch model's quantile forecasts.

The head is trained with pinball loss, which does not target a finite-sample coverage
guarantee, and the median roll-out ignores uncertainty from earlier steps. On the
validation split the nominal 80% intervals of the T4 models cover about 65%.

This is conformalized quantile regression (Romano, Patterson, Candès, 2019), fit only
on validation forecasts:

* score ``s = max(q_lo - y, y - q_hi)`` on the mean-absolute-scaled series, so a
  kilowatt series and a count series contribute on the same scale the model uses;
* the expansion for miscoverage ``alpha`` is the ``ceil((n+1)(1-alpha))``-th order
  statistic of those scores (the standard split-conformal quantile);
* each central interval (50/80/90%) gets its own expansion, in scaled units;
* quantiles are then pushed back into a non-decreasing sequence from the median out,
  so a wider tail cannot cross a narrower one. The median itself is never moved, so
  MASE of the median forecast is unchanged.

The same fitted expansions are applied to every test forecast of that seed. Nothing
here is chosen from the test split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tsfm.tokenizers import QUANTILES

# Nominal central interval -> (lower level, upper level, miscoverage alpha).
INTERVALS: dict[int, tuple[float, float, float]] = {
    50: (0.25, 0.75, 0.50),
    80: (0.10, 0.90, 0.20),
    90: (0.05, 0.95, 0.10),
}


@dataclass(frozen=True)
class CQR:
    """Expansions in mean-absolute-scaled units, one per central interval."""

    levels: tuple[float, ...]
    expansion: dict[int, float]  # interval name -> additive shift in scaled units

    def to_dict(self) -> dict[str, object]:
        return {
            "levels": list(self.levels),
            "expansion_scaled": {str(k): float(v) for k, v in sorted(self.expansion.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> CQR:
        raw = d["expansion_scaled"]
        if not isinstance(raw, dict):
            raise TypeError("expansion_scaled must be a dict")
        levels = d["levels"]
        if not isinstance(levels, list):
            raise TypeError("levels must be a list")
        return cls(
            levels=tuple(float(x) for x in levels),
            expansion={int(k): float(v) for k, v in raw.items()},
        )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n+1)(1-alpha))``-th order statistic.

    ``alpha`` is the target miscoverage. With ``n`` finite this is slightly
    conservative. ``scores`` may contain non-finite values; those are dropped.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.sort(np.asarray(scores, np.float64))
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        raise ValueError("no finite conformity scores")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), n)
    return float(s[rank - 1])


def conformity_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """``max(q_lo - y, y - q_hi)`` on finite targets. Negative means the interval covered."""
    y = np.asarray(y, np.float64)
    score = np.maximum(np.asarray(q_lo, np.float64) - y, y - np.asarray(q_hi, np.float64))
    kept = score[np.isfinite(y) & np.isfinite(score)]
    return np.asarray(kept, dtype=np.float64)


def fit_cqr(scores: dict[int, np.ndarray], levels: tuple[float, ...] = QUANTILES) -> CQR:
    """Fit one expansion per interval from validation scores in scaled units."""
    expansion: dict[int, float] = {}
    for name, (_lo, _hi, alpha) in INTERVALS.items():
        if name not in scores:
            raise KeyError(f"missing conformity scores for the {name}% interval")
        expansion[name] = conformal_quantile(scores[name], alpha)
    return CQR(levels=tuple(levels), expansion=expansion)


def enforce_monotone(q: np.ndarray, mid: int) -> np.ndarray:
    """Make ``q`` non-decreasing along the last axis, leaving the median in place.

    Lower levels are pulled down to the next higher level when they cross it; upper
    levels are pushed up. This only ever widens an interval that the independent
    expansions made inconsistent.
    """
    out = np.array(q, dtype=np.float64, copy=True)
    for i in range(mid - 1, -1, -1):
        out[..., i] = np.minimum(out[..., i], out[..., i + 1])
    for i in range(mid + 1, out.shape[-1]):
        out[..., i] = np.maximum(out[..., i], out[..., i - 1])
    return out


def apply_cqr(quantiles: np.ndarray, cqr: CQR) -> np.ndarray:
    """Expand central intervals of scaled quantiles ``(..., Q)`` and restore order.

    The median column is not modified. ``quantiles`` are in the same scaled units as
    the fitted expansions.
    """
    q = np.array(quantiles, dtype=np.float64, copy=True)
    index = {float(level): i for i, level in enumerate(cqr.levels)}
    mid = index[0.5]
    for name, (lo, hi, _alpha) in INTERVALS.items():
        shift = cqr.expansion[name]
        q[..., index[lo]] -= shift
        q[..., index[hi]] += shift
    return enforce_monotone(q, mid)


def scaled_interval_scores(
    y: np.ndarray, quantiles: np.ndarray, scale: float, levels: tuple[float, ...]
) -> dict[int, np.ndarray]:
    """Conformity scores of one forecast, divided by the context scale.

    ``y`` is ``(H,)`` in original units, ``quantiles`` is ``(H, Q)`` in original units.
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"scale must be positive and finite, got {scale}")
    ys = np.asarray(y, np.float64) / scale
    qs = np.asarray(quantiles, np.float64) / scale
    index = {float(level): i for i, level in enumerate(levels)}
    out: dict[int, np.ndarray] = {}
    for name, (lo, hi, _alpha) in INTERVALS.items():
        out[name] = conformity_scores(ys, qs[:, index[lo]], qs[:, index[hi]])
    return out
