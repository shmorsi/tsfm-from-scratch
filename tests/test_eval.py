"""M4 evaluation: metric semantics, coverage, predictor plumbing, test discipline,
vendored-file integrity and the reproduction of upstream's Seasonal Naive."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tsfm.eval import metrics as M
from tsfm.eval import run as R
from tsfm.eval.gifteval import VENDOR, configs
from tsfm.paths import repo_root, results_dir
from tsfm.provenance import sha256_file

UPSTREAM_SHA256 = {
    "gift_eval_data.py": "5a3ddfd9701418a43a8a8a15090b6e654645888cdd36912e65fd4a48608da35c",
    "dataset_configurations.csv": "f76c589a64440e7c9f4ea2a9083aaa409dc61a38a3e2eef37795d26a2f011f70",
    "dataset_properties.json": "fd9b53a32b3f953f7033fe385610cbdd2ddb02a77d6e320c0e73056e1318ad4a",
    "statsforecast_predictor.py": "fa30be971e2afd3854623abbf46c47bb29dea6cbfac16a6f9544e28e261f79af",
    "upstream_results/Seasonal_Naive/all_results.csv": "d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949",
    "upstream_results/Auto_Arima/all_results.csv": "e08845b070b6968c6de36ab442fd097d3bebf5ac94bbcc886227e52a0dd81af1",
    "upstream_results/Auto_ETS/all_results.csv": "9a38ac782094915c279f348372b91e4ddc69242acc62df741120abc90fd39312",
    "upstream_results/FlowState-9.1M/all_results.csv": "311d5f94fc76c61e1f00a193c0058f90f85c59034a3dee528509daf95efc58cd",
}


@pytest.mark.parametrize(("rel", "sha"), sorted(UPSTREAM_SHA256.items()))
def test_vendored_files_unchanged(rel: str, sha: str):
    assert sha256_file(VENDOR / rel) == sha


def test_97_official_configs():
    assert len(configs()) == 97


# ------------------------------------------------------------------ metric semantics


def _gluonts_scores(ys, pasts, qf, m):
    """Run GluonTS's evaluate_forecasts exactly as evaluate_model does (axis=None)."""
    from gluonts.model.evaluation import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast

    freq = "D"
    fcs, inputs, labels = [], [], []
    for i, (y, p, f) in enumerate(zip(ys, pasts, qf, strict=True)):
        start = pd.Period("2020-01-01", freq) + len(p)
        keys = ["mean", *[str(q) for q in R.OFFICIAL_QUANTILES]]
        arr = np.stack([f[0.5], *[f[q] for q in R.OFFICIAL_QUANTILES]])
        fcs.append(QuantileForecast(arr, start_date=start, forecast_keys=keys, item_id=str(i)))
        inputs.append({"target": p, "start": pd.Period("2020-01-01", freq), "item_id": str(i)})
        labels.append({"target": y, "start": start, "item_id": str(i)})

    class TD:
        input = inputs
        label = labels

        def __iter__(self):
            return iter(zip(inputs, labels, strict=True))

        def __len__(self):
            return len(inputs)

    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

    res = evaluate_forecasts(
        fcs,
        test_data=TD(),
        metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=R.OFFICIAL_QUANTILES)],
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=m,
    )
    return float(res["MASE[0.5]"][0]), float(res["mean_weighted_sum_quantile_loss"][0])


def test_numpy_metrics_match_gluonts():
    rng = np.random.default_rng(0)
    ys, pasts, qf = [], [], []
    for _ in range(7):
        p = rng.normal(size=int(rng.integers(20, 60))).cumsum() + 50
        y = rng.normal(size=12).cumsum() + p[-1]
        y[3] = np.nan  # missing label value
        med = np.full(12, p[-1])
        f = {q: med + (q - 0.5) * 4 for q in R.OFFICIAL_QUANTILES}
        ys.append(y)
        pasts.append(p)
        qf.append(f)
    g_mase, g_crps = _gluonts_scores(ys, pasts, qf, m=7)
    ours_mase = M.mase(ys, [f[0.5] for f in qf], pasts, m=7)
    ours_crps = M.weighted_quantile_loss(ys, qf, R.OFFICIAL_QUANTILES)
    assert ours_mase == pytest.approx(g_mase, rel=1e-9)
    assert ours_crps == pytest.approx(g_crps, rel=1e-9)


def test_gmean_and_bootstrap():
    a = {"x": 0.5, "y": 2.0}
    base = {"x": 1.0, "y": 1.0}
    assert M.normalised_gmean(a, base) == pytest.approx(1.0)
    pt, lo, hi = M.paired_bootstrap_ratio(a, a, base, n=200)
    assert pt == lo == hi == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    keys = [f"c{i}" for i in range(97)]
    b = {k: float(rng.uniform(0.5, 1.5)) for k in keys}
    better = {k: 0.8 * b[k] for k in keys}
    pt, lo, hi = M.paired_bootstrap_ratio(better, b, dict.fromkeys(keys, 1.0), n=500)
    assert pt == pytest.approx(0.8)
    assert lo == pytest.approx(0.8)
    assert hi == pytest.approx(0.8)
    with pytest.raises(ValueError, match="positive"):
        M.gmean([1.0, 0.0])


def test_interval_coverage():
    from gluonts.model.forecast import QuantileForecast

    keys = [str(q) for q in R.ALL_QUANTILES]
    arr = np.stack([np.full(4, q) for q in R.ALL_QUANTILES])  # quantile q predicts value q
    f = QuantileForecast(arr, start_date=pd.Period("2020-01-01", "D"), forecast_keys=keys)
    y = np.array([0.5, 0.15, 0.03, np.nan])  # inside 50%; inside 80%; outside all
    cov = R.interval_coverage([f], [y])
    assert cov == {
        "coverage_50": pytest.approx(1 / 3),
        "coverage_80": pytest.approx(2 / 3),
        "coverage_90": pytest.approx(2 / 3),
    }


# ------------------------------------------------------------------ predictor plumbing


def test_tsfm_predictor_end_to_end_with_gluonts():
    from gluonts.dataset.split import split
    from gluonts.model import evaluate_model

    from tsfm.model import ModelConfig, TSModel

    torch.manual_seed(0)
    model = TSModel(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=8, horizon=16, max_seq_len=32)
    ).eval()
    rng = np.random.default_rng(0)
    data = [
        {
            "item_id": str(i),
            "start": pd.Period("2020-01-01", "h"),
            "target": rng.normal(size=200).astype(np.float32) + 10,
        }
        for i in range(5)
    ]
    _, tmpl = split(data, offset=-40 * 2)
    td = tmpl.generate_instances(prediction_length=40, windows=2, distance=40)
    pred = R.Recorder(R.TSFMPredictor(model, 40, max_context=128, batch_size=3))
    res = evaluate_model(
        pred,
        test_data=td,
        metrics=R.official_metrics(),
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=24,
    )
    assert np.isfinite(res["MASE[0.5]"][0])
    assert len(pred.forecasts) == 10
    f = pred.forecasts[0]
    assert f.quantile(0.05).shape == (40,)
    assert (f.quantile(0.95) >= f.quantile(0.05)).all()


def test_bins_forecast_shapes_and_reproducibility():
    from tsfm.forecast import forecast_bins
    from tsfm.model import ModelConfig, TSModel

    torch.manual_seed(0)
    model = TSModel(
        ModelConfig(tokenizer="bins", d_model=32, n_layers=1, n_heads=2, n_bins=126, max_seq_len=64)
    ).eval()
    ctx = [np.arange(30, dtype=np.float32), np.ones(10, np.float32) * 5]
    a = forecast_bins(model, ctx, 7, 50, n_samples=8, generator=torch.Generator().manual_seed(1))
    b = forecast_bins(model, ctx, 7, 50, n_samples=8, generator=torch.Generator().manual_seed(1))
    assert a.quantiles.shape == (2, 7, len(a.levels))
    np.testing.assert_array_equal(a.quantiles, b.quantiles)
    assert (np.diff(a.quantiles, axis=-1) >= 0).all()


def test_bins_forecast_chunking_matches_one_context_at_a_time():
    """A tiny KV budget forces one context per chunk; each chunk must then match running
    that context alone (same generator stream), so chunking changes memory, not results."""
    from tsfm.forecast import forecast_bins
    from tsfm.model import ModelConfig, TSModel

    torch.manual_seed(0)
    model = TSModel(
        ModelConfig(tokenizer="bins", d_model=32, n_layers=2, n_heads=2, n_bins=126, max_seq_len=64)
    ).eval()
    rng = np.random.default_rng(0)
    ctx = [rng.normal(10, 2, 24).astype(np.float32) for _ in range(3)]
    gen = torch.Generator().manual_seed(3)
    chunked = forecast_bins(model, ctx, 5, 50, n_samples=4, generator=gen, kv_budget_bytes=1)
    gen = torch.Generator().manual_seed(3)
    singles = [forecast_bins(model, [c], 5, 50, n_samples=4, generator=gen) for c in ctx]
    np.testing.assert_array_equal(chunked.quantiles, np.concatenate([s.quantiles for s in singles]))
    np.testing.assert_array_equal(chunked.mean, np.concatenate([s.mean for s in singles]))


def test_short_kv_cache_matches_full_length_cache():
    from tsfm.model import KVCache, ModelConfig, TSModel

    torch.manual_seed(0)
    cfg = ModelConfig(
        tokenizer="bins", d_model=32, n_layers=2, n_heads=2, n_bins=126, max_seq_len=64
    )
    model = TSModel(cfg).eval()
    ids = torch.randint(2, 128, (2, 10))
    outs = []
    for max_len in (None, 13):
        cache = KVCache.empty(cfg, 2, torch.device("cpu"), torch.float32, max_len=max_len)
        steps = [model(ids, cache=cache)[:, -1]]
        for t in range(3):
            steps.append(model(ids[:, t : t + 1], cache=cache)[:, -1])
        outs.append(torch.stack(steps))
    assert cache.k[0].shape[2] == 13
    torch.testing.assert_close(outs[0], outs[1])


def test_patch_rollout_horizon_and_monotone():
    from tsfm.forecast import forecast_patch
    from tsfm.model import ModelConfig, TSModel

    torch.manual_seed(0)
    model = TSModel(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=8, horizon=16, max_seq_len=40)
    ).eval()
    fc = forecast_patch(
        model, [np.random.default_rng(0).normal(size=100).astype(np.float32)], 70, 160
    )
    assert fc.quantiles.shape == (1, 70, 13)
    assert (np.diff(fc.quantiles, axis=-1) >= -1e-6).all()


# ------------------------------------------------------------------ discipline


def test_test_split_scored_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(R, "results_dir", lambda: tmp_path)
    (tmp_path / "eval" / "test" / "seasonal_naive").mkdir(parents=True)
    R.check_test_discipline("seasonal_naive", "test", exploratory=False)  # unfinished: resume
    (tmp_path / "eval" / "test" / "seasonal_naive" / "all_results.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="already scored"):
        R.check_test_discipline("seasonal_naive", "test", exploratory=False)
    R.check_test_discipline("seasonal_naive", "test", exploratory=True)  # allowed, labelled
    R.check_test_discipline("seasonal_naive", "val", exploratory=False)


def test_model_test_scoring_requires_prereg_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(R, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(R, "git_tag_contains_head", lambda tag: False)
    with pytest.raises(SystemExit, match="pre-registration"):
        R.check_test_discipline("tsfm-10m", "test", exploratory=False)
    monkeypatch.setattr(R, "git_tag_contains_head", lambda tag: True)
    R.check_test_discipline("tsfm-10m", "test", exploratory=False)


# ------------------------------------------------------------------ reproduction


def _read(p: Path) -> dict[str, dict[str, str]]:
    with open(p, newline="") as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}


OURS = results_dir() / "eval" / "test" / "seasonal_naive" / "all_results.csv"


@pytest.mark.skipif(not OURS.exists(), reason="seasonal naive not yet scored")
def test_seasonal_naive_reproduces_upstream():
    """All 11 official metrics on all 97 configurations equal upstream's published
    Seasonal Naive (relative difference < 1e-9)."""
    ours = _read(OURS)
    up = _read(VENDOR / "upstream_results" / "Seasonal_Naive" / "all_results.csv")
    assert ours.keys() == up.keys()
    bad = []
    for k in ours:
        for col in R.METRIC_COLUMNS:
            a, b = float(ours[k][col]), float(up[k][col])
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                bad.append((k, col, a, b))
    assert bad == []


def test_repo_root_has_results_dir():
    assert (repo_root() / "results").exists()


# ------------------------------------------------------------------ validation split rules


@pytest.mark.parametrize(
    ("freq", "h"),
    [
        ("A-DEC", 6),
        ("Q-DEC", 8),
        ("M", 12),
        ("MS", 12),
        ("W-SUN", 8),
        ("D", 30),
        ("H", 48),
        ("30T", 48),
        ("10T", 48),
        ("4S", 60),
    ],
)
def test_val_prediction_lengths(freq: str, h: int):
    from tsfm.eval.valsplit import base_prediction_length

    assert base_prediction_length(freq) == h


@pytest.mark.parametrize(
    ("freq", "pf"), [("MS", "M"), ("QS-JAN", "Q-JAN"), ("YS", "A"), ("H", "H"), ("A-DEC", "A-DEC")]
)
def test_period_freq(freq: str, pf: str):
    from tsfm.eval.valsplit import period_freq

    assert period_freq(freq) == pf
    pd.Period("2020-01-01", freq=period_freq(freq))  # must be constructible


def test_val_dataset_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tsfm.data.convert import SCHEMA
    from tsfm.eval import valsplit as V

    monkeypatch.setattr(V, "processed_dir", lambda: tmp_path)
    V._freq_of.cache_clear()
    V._max_len.cache_clear()
    V.val_configs.cache_clear()
    (tmp_path / "val").mkdir()
    rng = np.random.default_rng(0)
    lens = [600, 900, 30]  # the 30-point series is too short for one daily window and is skipped
    t = pa.table(
        {
            "item_id": [f"d/{i}" for i in range(3)],
            "dataset": ["d"] * 3,
            "freq": ["D"] * 3,
            "start": pa.array([0] * 3, pa.timestamp("ms")),
            "target": [rng.normal(size=n).astype(np.float32) for n in lens],
        },
        schema=SCHEMA,
    )
    pq.write_table(t, tmp_path / "val" / "d.parquet")
    cfgs = V.val_configs()
    assert [c.name for c in cfgs] == ["d/D/short"]
    ds = V.load_val(cfgs[0])
    assert ds.prediction_length == 30
    assert ds.windows == math.ceil(0.1 * 600 / 30)  # 10% of the shortest kept series
    inst = list(ds.test_data)
    assert len(inst) == 2 * ds.windows
    assert all(lab["target"].size == 30 for _, lab in inst)
    V._freq_of.cache_clear()
    V._max_len.cache_clear()
    V.val_configs.cache_clear()
