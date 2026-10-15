"""Milestones 8-9: realized volatility, scoring, baselines, fine-tuning."""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from tsfm.finance import data as D
from tsfm.finance import study as S


def test_daily_rvol_formula_and_missing_days():
    day_ms = 86_400_000
    t0 = 1_600_000_000_000 // day_ms * day_ms
    ts = t0 + np.arange(0, 3 * 288) * 300_000
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(scale=0.001, size=ts.size)))
    keep = np.ones(ts.size, bool)
    keep[288 + 10 : 288 + 40] = False  # day 2 loses 30 bars -> below 274 -> missing
    dates, rv, n = D.daily_rvol(ts[keep], close[keep])
    assert len(dates) == 3
    r = np.diff(np.log(close[:288]))
    np.testing.assert_allclose(rv[0], 100 * np.sqrt((r**2).sum()))
    assert np.isnan(rv[1])
    assert n[0] == 287  # first bar of the series has no previous close


def test_read_closes_handles_microseconds_and_header(tmp_path: Path):
    rows = "open_time,open,high,low,close\n1735689600000000,1,1,1,2.5,0,0\n1735689900000000,1,1,1,2.6,0,0\n"
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.csv", rows)
    ts, cl = D.read_closes(p)
    assert ts.tolist() == [1735689600000, 1735689900000]
    assert cl.tolist() == [2.5, 2.6]
    _ = io  # keep import for clarity of fixture construction


def test_months_range():
    assert D.months((2019, 11), (2020, 2)) == [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]


def test_regimes():
    assert S.regime_of(date(2020, 3, 12)) == "covid_2020"
    assert S.regime_of(date(2022, 11, 9)) == "ftx_2022"
    assert S.regime_of(date(2020, 2, 10)) == "near_break"  # targets reach into the break
    assert S.regime_of(date(2021, 6, 1)) == "calm"


def test_score_perfect_and_coverage():
    y = np.arange(1, 23, dtype=float)
    q = np.repeat(y[:, None], len(S.LEVELS), axis=1)
    sc = S.score(y, q)
    assert sc["wql"] == 0.0
    assert sc["mae"] == 0.0
    assert sc["cov90"] == 1.0
    y2 = y.copy()
    y2[3] = np.nan
    assert S.score(y2, q)["n"] == 21


def _series(n: int = 1500, seed: int = 0) -> S.Series:
    rng = np.random.default_rng(seed)
    lx = np.zeros(n)
    for t in range(1, n):
        lx[t] = 0.9 * lx[t - 1] + rng.normal(scale=0.3)
    d0 = date(2017, 8, 17)
    return S.Series("SYN", [d0 + timedelta(days=i) for i in range(n)], np.exp(lx + 1.0))


def test_origins_never_reach_past_the_data():
    s = _series(300)
    idx = S.origins(s, s.dates[0], s.dates[-1])
    assert idx[-1] + S.H == len(s.values) - 1


@pytest.mark.parametrize("fn", [S.random_walk, S.HAR()])
def test_baselines_use_no_future(fn):
    s = _series()
    i = 1000
    a = fn([s.values[: i + 1]], S.H)
    v = s.values.copy()
    v[i + 1 :] *= 10  # change the future
    b = fn([v[: i + 1]], S.H)
    np.testing.assert_allclose(a, b)
    assert (np.diff(a[0], axis=-1) >= -1e-12).all()  # quantiles sorted


def test_random_walk_median_is_last_value():
    s = _series()
    q = S.random_walk([s.values[:900]], S.H)
    np.testing.assert_allclose(q[0, :, S.LEVELS.index(0.5)], s.values[899])


def test_har_beats_random_walk_on_mean_reverting_series():
    s = _series(2000)
    rows_rw = S.evaluate(S.random_walk, [s], s.dates[1200], s.dates[1900], step=7)
    rows_har = S.evaluate(S.HAR(), [s], s.dates[1200], s.dates[1900], step=7)
    pooled = lambda rows: sum(r["wql_num"] for r in rows) / sum(r["abs_y"] for r in rows)  # noqa: E731
    assert pooled(rows_har) < pooled(rows_rw)


def test_summarise_ratio_and_interval():
    rows = [
        {
            "symbol": "A",
            "origin": f"2021-01-{i:02d}",
            "regime": "calm",
            "wql_num": 1.0,
            "abs_y": 10.0,
            "wql": 0.1,
            "mae": 1.0,
            "cov50": 0.5,
            "cov80": 0.8,
            "cov90": 0.9,
        }
        for i in range(1, 29)
    ]
    ref = [{**r, "wql_num": 2.0, "wql": 0.2} for r in rows]
    out = S.summarise(rows, ref, n_boot=200)
    g = out["A/all_test"]
    assert g["wql_ratio_vs_ref"] == pytest.approx(0.5)
    assert g["wql_ratio_ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert out["both/calm"]["origins"] == 28


def test_finetune_keeps_best_weights(monkeypatch: pytest.MonkeyPatch):
    from tsfm.finance import finetune as F
    from tsfm.model import ModelConfig, TSModel

    torch.manual_seed(0)
    model = TSModel(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=8, horizon=24, max_seq_len=64)
    ).eval()
    s = _series(n=(S.VAL_END - date(2017, 8, 17)).days + 60)
    model, log = F.finetune(model, [s], steps=20, batch=8, eval_every=10)
    assert [r["step"] for r in log] == [0, 10, 20]
    best = min(r["val_wql"] for r in log)
    rows = S.evaluate(
        F.model_forecaster(model), [s], S.TRAIN_END, S.VAL_END - timedelta(days=S.H), step=5
    )
    assert F.pooled_wql(rows) == pytest.approx(best, rel=1e-5)
