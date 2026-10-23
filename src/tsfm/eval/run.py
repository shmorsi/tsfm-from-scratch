"""One command for evaluation: ``tsfm eval --model NAME --split val|test``.

Official metrics are computed by GluonTS ``evaluate_model`` with exactly the arguments
and metric list of upstream's notebooks. Interval coverage (50/80/90%) is computed by us
from the same forecasts (it is not part of the official score).

Test-set discipline: a model is scored on the test split at most once. A second attempt
is refused unless ``--exploratory`` is given, in which case results go to
``results/eval/exploratory/`` and are labelled as such. Models other than the baselines
can only be scored on the test split once the pre-registration tag exists.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.eval.gifteval import VENDOR, Config, configs, load_dataset
from tsfm.paths import repo_root, results_dir
from tsfm.provenance import config_hash, stamp

OFFICIAL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALL_QUANTILES = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
METRIC_COLUMNS = [
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]
CSV_HEADER = ["dataset", "model", *METRIC_COLUMNS, "domain", "num_variates"]
INTERVALS = {50: (0.25, 0.75), 80: (0.1, 0.9), 90: (0.05, 0.95)}
PREREG_TAG = "prereg-v1"
# Models whose test scores cannot inform our own model choices: classical baselines and
# the published reference checkpoint. These may be scored before the pre-registration tag.
BASELINES = {"seasonal_naive", "naive", "auto_ets", "auto_arima", "auto_theta", "flowstate_9.1m"}


def official_metrics() -> list[Any]:
    """The metric list from upstream notebooks/naive.ipynb, unchanged."""
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=OFFICIAL_QUANTILES),
    ]


# ---------------------------------------------------------------------------- predictors


def statsforecast_predictor(
    kind: str, prediction_length: int, season: int, freq: str, **kw: Any
) -> Any:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive
    from statsforecast_predictor import StatsForecastPredictor

    model_type = {
        "seasonal_naive": SeasonalNaive,
        "naive": Naive,
        "auto_ets": AutoETS,
        "auto_arima": AutoARIMA,
        "auto_theta": AutoTheta,
    }[kind]
    cls = type(
        f"{model_type.__name__}Predictor", (StatsForecastPredictor,), {"ModelType": model_type}
    )
    pred = cls(
        prediction_length,
        season_length=season,
        freq=freq,
        quantile_levels=ALL_QUANTILES,
        batch_size=kw.pop("batch_size", 512),
        **kw,
    )
    # Upstream's wrapper asks statsforecast for a level-0 interval to get the median
    # ("lo-0"); statsforecast >= 2 rejects level 0. A zero-width central interval is the
    # point forecast, so we read the median from "mean" instead. This is exact for these
    # models (symmetric intervals) and tests/test_eval.py checks we reproduce upstream.
    c = pred.config
    c.statsforecast_keys = ["mean" if k == "lo-0" else k for k in c.statsforecast_keys]
    c.intervals = [i for i in c.intervals if i != 0]
    return pred


def statsforecast_factory(kind: str, kw: dict[str, Any], ds: Any, h: int, s: int, f: str) -> Any:
    """Picklable factory (use with functools.partial) for parallel evaluation."""
    return statsforecast_predictor(kind, h, s, f, **dict(kw))


@dataclass
class TSFMPredictor:
    """GluonTS-compatible predictor around our model (duck-typed: ``predict``)."""

    model: Any
    prediction_length: int
    max_context: int
    batch_size: int = 256
    forecast_kwargs: dict[str, Any] | None = None

    def predict(self, dataset: Any, **_: Any) -> Iterator[Any]:
        from gluonts.dataset.util import forecast_start
        from gluonts.model.forecast import QuantileForecast

        from tsfm.forecast import forecast

        batch: list[dict[str, Any]] = []

        def flush() -> Iterator[Any]:
            ctx = [np.asarray(e["target"], np.float32) for e in batch]
            fc = forecast(
                self.model,
                ctx,
                self.prediction_length,
                self.max_context,
                **(self.forecast_kwargs or {}),
            )
            keys = ["mean", *[str(q) for q in fc.levels]]
            for i, e in enumerate(batch):
                arr = np.concatenate([fc.mean[i][None], fc.quantiles[i].T], axis=0)
                yield QuantileForecast(
                    arr, start_date=forecast_start(e), forecast_keys=keys, item_id=e.get("item_id")
                )

        for entry in dataset:
            batch.append(entry)
            if len(batch) == self.batch_size:
                yield from flush()
                batch = []
        if batch:
            yield from flush()


def official_view(f: Any) -> Any:
    """The forecast as upstream's evaluation sees it: only the "mean" key and the nine
    official quantile levels. Metrics that need other levels (MSIS uses 0.025/0.975) are
    then extrapolated by GluonTS from exactly the same keys as for leaderboard models,
    instead of from our extra 0.05/0.95 levels."""
    from gluonts.model.forecast import QuantileForecast

    if not isinstance(f, QuantileForecast):
        return f
    keep = ["mean", *[str(q) for q in OFFICIAL_QUANTILES]]
    idx = [f.forecast_keys.index(k) for k in keep if k in f.forecast_keys]
    keys = [f.forecast_keys[i] for i in idx]
    return QuantileForecast(
        f.forecast_array[idx], start_date=f.start_date, forecast_keys=keys, item_id=f.item_id
    )


class Recorder:
    """Pass-through predictor: hands the official view of each forecast to GluonTS and
    keeps the full forecast (all quantile levels) for coverage."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.prediction_length = inner.prediction_length
        self.forecasts: list[Any] = []

    def predict(self, dataset: Any, **kw: Any) -> Iterator[Any]:
        for f in self.inner.predict(dataset, **kw):
            self.forecasts.append(f)
            yield official_view(f)


def interval_coverage(forecasts: list[Any], labels: list[np.ndarray]) -> dict[str, float]:
    """Pooled fraction of observed label points inside each central interval."""
    hits = {k: 0 for k in INTERVALS}
    total = 0
    for f, y in zip(forecasts, labels, strict=True):
        y = np.asarray(y, np.float64)
        m = np.isfinite(y)
        total += int(m.sum())
        for k, (lo, hi) in INTERVALS.items():
            try:
                ql, qh = f.quantile(lo), f.quantile(hi)
            except (KeyError, ValueError):
                hits[k] = -(10**18)  # level not provided by this forecaster -> NaN below
                continue
            hits[k] += int(((y >= ql) & (y <= qh) & m).sum())
    return {
        f"coverage_{k}": hits[k] / total if total and hits[k] >= 0 else math.nan for k in INTERVALS
    }


# ---------------------------------------------------------------------------- core


PredictorFactory = Callable[[Any, int, int, str], Any]  # (dataset, prediction_length, season, freq)


def evaluate_one(name: str, ds: Any, test_data: Any, factory: PredictorFactory) -> dict[str, Any]:
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    season = get_seasonality(ds.freq)
    pred = Recorder(factory(ds, ds.prediction_length, season, ds.freq))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate_model(
            pred,
            test_data=test_data,
            metrics=official_metrics(),
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season,
        )
    row = {c: float(res[c.removeprefix("eval_metrics/")][0]) for c in METRIC_COLUMNS}
    labels = [lab["target"] for lab in test_data.label]
    row.update(interval_coverage(pred.forecasts, labels))
    row["seconds"] = time.time() - t0
    row["n_instances"] = len(labels)
    return row


def git_tag_contains_head(tag: str) -> bool:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", tag, "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def output_dir(model_name: str, split: str, exploratory: bool) -> Path:
    if exploratory:
        return (
            results_dir()
            / "eval"
            / "exploratory"
            / f"{model_name}-{time.strftime('%Y%m%dT%H%M%S')}-{split}"
        )
    return results_dir() / "eval" / split / model_name


def check_test_discipline(model_name: str, split: str, exploratory: bool) -> None:
    if split != "test" or exploratory:
        return
    out = output_dir(model_name, split, False)
    if (out / "all_results.csv").exists():
        raise SystemExit(
            f"{model_name} was already scored on the test split ({out}). Use --exploratory for any further run."
        )
    if model_name not in BASELINES and not git_tag_contains_head(PREREG_TAG):
        raise SystemExit(
            f"test scoring of {model_name} requires the pre-registration tag {PREREG_TAG} in HEAD's history"
        )


def _eval_config(args: tuple[Any, str, PredictorFactory, str]) -> dict[str, Any]:
    c, split, factory, model_name = args
    ds, test_data = _load(c, split)
    row = evaluate_one(c.name, ds, test_data, factory)
    row.update(
        {"dataset": c.name, "model": model_name, "domain": c.domain, "num_variates": c.num_variates}
    )
    return row


def _partial_path(out: Path, cfg_name: str) -> Path:
    return out / "partial" / (cfg_name.replace("/", "__") + ".json")


def run_eval(
    model_name: str,
    factory: PredictorFactory,
    *,
    split: str,
    model_info: dict[str, Any],
    exploratory: bool = False,
    only: list[str] | None = None,
    jobs: int = 1,
) -> Path:
    """Score every config of ``split``. Each finished config is saved under
    ``<out>/partial/`` so an interrupted run resumes where it stopped. With ``jobs > 1``
    configs run in parallel processes (``factory`` must then be picklable)."""
    check_test_discipline(model_name, split, exploratory)
    if split == "test":
        cfgs: list[Any] = list(configs())
    else:
        from tsfm.eval.valsplit import val_configs

        cfgs = list(val_configs())
    if only:
        cfgs = [c for c in cfgs if c.name in only or c.name.split("/")[0] in only]
    out = output_dir(model_name, split, exploratory)
    out.mkdir(parents=True, exist_ok=True)
    done = {
        c.name: json.loads(_partial_path(out, c.name).read_text())
        for c in cfgs
        if _partial_path(out, c.name).exists()
    }
    todo = [c for c in cfgs if c.name not in done]
    if done:
        print(f"resuming: {len(done)} configs already scored", flush=True)

    def record(row: dict[str, Any]) -> None:
        p = _partial_path(out, row["dataset"])
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(row))
        done[row["dataset"]] = row
        print(
            f"{row['dataset']:45s} MASE {row['eval_metrics/MASE[0.5]']:.4f} CRPS {row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f} cov80 {row['coverage_80']:.3f} ({row['seconds']:.1f}s)",
            flush=True,
        )

    args = [(c, split, factory, model_name) for c in todo]
    if jobs > 1 and len(args) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(jobs) as ex:
            futs = [ex.submit(_eval_config, a) for a in args]
            for f in as_completed(futs):
                record(f.result())
    else:
        for a in args:
            record(_eval_config(a))
    rows = [done[c.name] for c in cfgs]
    _write(
        out,
        model_name,
        rows,
        split=split,
        exploratory=exploratory,
        model_info=model_info,
        partial=bool(only),
    )
    import shutil

    shutil.rmtree(out / "partial", ignore_errors=True)
    return out


def _load(c: Any, split: str) -> tuple[Any, Any]:
    if split == "test":
        assert isinstance(c, Config)
        ds = load_dataset(c.ds_name, c.term)
        return ds, ds.test_data
    from tsfm.eval.valsplit import load_val

    ds = load_val(c)
    return ds, ds.test_data


def _write(
    out: Path,
    model: str,
    rows: list[dict[str, Any]],
    *,
    split: str,
    exploratory: bool,
    model_info: dict[str, Any],
    partial: bool,
) -> None:
    with open(out / "all_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in sorted(rows, key=lambda r: r["dataset"]):
            w.writerow([r[k] for k in CSV_HEADER])
    extra = {
        r["dataset"]: {
            k: r[k] for k in ("coverage_50", "coverage_80", "coverage_90", "seconds", "n_instances")
        }
        for r in rows
    }
    meta = stamp("exploratory" if exploratory else split, f"eval-{model}")
    meta["partial"] = partial
    doc = {
        "_meta": meta,
        "model": model,
        "split": split,
        "model_info": model_info,
        "model_info_hash": config_hash(model_info),
        "per_config": extra,
    }
    (out / "run.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    # Upstream-format config.json (leaderboard fields).
    (out / "config.json").write_text(
        json.dumps(model_info.get("leaderboard", {"model": model}), indent=4) + "\n"
    )
