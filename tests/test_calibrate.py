"""Split-conformal adjustment: the quantile, the median, and a coverage check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsfm.calibrate import (
    INTERVALS,
    apply_cqr,
    conformal_quantile,
    conformity_scores,
    enforce_monotone,
    fit_cqr,
)
from tsfm.forecast import forecast
from tsfm.model import TSModel
from tsfm.tokenizers import QUANTILES


def test_conformal_quantile_is_an_order_statistic():
    scores = np.arange(10, dtype=np.float64)  # 0..9
    # n=10, alpha=0.2 -> ceil(11 * 0.8) = 9 -> index 8 -> value 8
    assert conformal_quantile(scores, 0.2) == 8.0
    assert conformal_quantile(np.array([1.0, np.nan, 3.0]), 0.5) == 3.0
    with pytest.raises(ValueError, match="finite"):
        conformal_quantile(np.array([np.nan]), 0.2)


def test_negative_score_when_the_interval_covers():
    y = np.array([0.0, 0.0])
    assert np.all(conformity_scores(y, np.array([-1.0, -1.0]), np.array([1.0, 1.0])) < 0)


def test_apply_leaves_the_median_and_stays_ordered():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(size=(4, 5, len(QUANTILES))), axis=-1)
    # Blow the tails up by different amounts so a naive edit can cross.
    scores = {
        50: rng.normal(loc=-0.2, scale=0.1, size=100),
        80: rng.normal(loc=0.5, scale=0.1, size=100),
        90: rng.normal(loc=1.5, scale=0.1, size=100),
    }
    cqr = fit_cqr(scores)
    out = apply_cqr(q, cqr)
    mid = QUANTILES.index(0.5)
    np.testing.assert_allclose(out[..., mid], q[..., mid])
    assert np.all(np.diff(out, axis=-1) >= -1e-12)
    # 80% interval got wider by at least its own expansion (monotone repair can add more).
    lo, hi, _ = INTERVALS[80]
    ilo, ihi = QUANTILES.index(lo), QUANTILES.index(hi)
    width = out[..., ihi] - out[..., ilo]
    raw = q[..., ihi] - q[..., ilo]
    assert np.all(width + 1e-9 >= raw + 2 * cqr.expansion[80] - 1e-6)


def test_monotone_repair_only_widens():
    q = np.array([0.0, 1.0, 0.5, 2.0])  # dip at index 2
    out = enforce_monotone(q, mid=3)
    assert out[-1] == 2.0
    assert np.all(np.diff(out) >= -1e-12)


def test_forecast_adjustment_does_not_move_the_median():
    torch.manual_seed(0)
    from tsfm.model import ModelConfig

    model = TSModel(
        ModelConfig(
            tokenizer="patch",
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=32,
            patch_len=4,
            horizon=8,
        )
    ).eval()
    ctx = [np.linspace(0, 1, 20, dtype=np.float32)]
    raw = forecast(model, ctx, prediction_length=6, max_context=16)
    scores = {k: np.array([0.25]) for k in (50, 80, 90)}
    adjusted = forecast(model, ctx, prediction_length=6, max_context=16, adjustment=fit_cqr(scores))
    np.testing.assert_allclose(adjusted.mean, raw.mean, rtol=1e-5, atol=1e-5)
    assert adjusted.quantiles.shape == raw.quantiles.shape
    assert np.all(np.diff(adjusted.quantiles, axis=-1) >= -1e-6)


def test_exchangeable_scores_cover_at_the_nominal_level():
    """On the calibration sample itself, split conformal covers at least 1-alpha."""
    rng = np.random.default_rng(1)
    n = 500
    # A too-narrow interval around 0: half-width 0.2, truth is N(0, 1).
    y = rng.normal(size=n)
    q_lo = np.full(n, -0.2)
    q_hi = np.full(n, 0.2)
    scores = conformity_scores(y, q_lo, q_hi)
    shift = conformal_quantile(scores, 0.2)
    covered = (y >= q_lo - shift) & (y <= q_hi + shift)
    assert covered.mean() >= 0.8
