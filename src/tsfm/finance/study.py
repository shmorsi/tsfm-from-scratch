"""Realized-volatility forecasting study (milestones 8-9). Design: PREREGISTRATION_VOL.md.

Every model forecasts the next ``H`` daily realized-volatility values from each origin
day in the test period, for BTCUSDT and ETHUSDT. Scores per origin (pooled over the
22-step path): weighted quantile loss over the nine official levels (as GIFT-Eval's
"CRPS"), MAE of the median, and coverage of the central 50/80/90% intervals.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.finance.data import SYMBOLS, load_rvol
from tsfm.paths import results_dir
from tsfm.provenance import write_result

H = 22
LEVELS = (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
OFFICIAL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TRAIN_END = date(2019, 6, 30)
VAL_END = date(2019, 12, 31)
TEST_START = date(2020, 1, 1)
REGIMES = {
    "covid_2020": (date(2020, 2, 20), date(2020, 4, 30)),
    "terra_3ac_2022": (date(2022, 5, 1), date(2022, 7, 31)),
    "ftx_2022": (date(2022, 11, 1), date(2022, 12, 31)),
}
BLOCK = 22  # moving-block bootstrap block length (origins)

# forecast_fn(histories, h) -> quantiles (B, h, len(LEVELS)) on the original scale
ForecastFn = Callable[[list[np.ndarray], int], np.ndarray]


def regime_of(origin: date) -> str:
    """Break regime if the origin lies in a break window; "calm" if neither the origin
    nor any of its target days is in a break window; otherwise "near_break"."""
    for name, (a, b) in REGIMES.items():
        if a <= origin <= b:
            return name
    end = origin + timedelta(days=H)
    for a, b in REGIMES.values():
        if origin <= b and end >= a:
            return "near_break"
    return "calm"


@dataclass
class Series:
    sym: str
    dates: list[date]
    values: np.ndarray

    def index(self, d: date) -> int:
        return (d - self.dates[0]).days


def load_all() -> list[Series]:
    return [Series(s, *load_rvol(s)) for s in SYMBOLS]


def origins(s: Series, start: date, end: date, step: int = 1) -> list[int]:
    """Origin indices i (forecast days i+1..i+H) with all targets inside the data."""
    lo, hi = s.index(start), min(s.index(end), len(s.values) - 1 - H)
    return list(range(lo, hi + 1, step))


def score(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """y (H,), q (H, Q). Path-pooled scores; NaN targets are ignored."""
    ok = np.isfinite(y)
    if not ok.any():
        return {
            "wql": np.nan,
            "mae": np.nan,
            "cov50": np.nan,
            "cov80": np.nan,
            "cov90": np.nan,
            "abs_y": np.nan,
        }
    yy, qq = y[ok], q[ok]
    loss = 0.0
    for lv in OFFICIAL:
        d = yy - qq[:, LEVELS.index(lv)]
        loss += float(np.maximum(lv * d, (lv - 1) * d).sum())
    med = qq[:, LEVELS.index(0.5)]
    out = {
        "wql_num": 2 * loss / len(OFFICIAL),
        "abs_y": float(np.abs(yy).sum()),
        "mae": float(np.abs(yy - med).mean()),
        "n": int(ok.sum()),
    }
    for k, (lo, hi) in {"cov50": (0.25, 0.75), "cov80": (0.1, 0.9), "cov90": (0.05, 0.95)}.items():
        out[k] = float(((yy >= qq[:, LEVELS.index(lo)]) & (yy <= qq[:, LEVELS.index(hi)])).mean())
    out["wql"] = out["wql_num"] / out["abs_y"]
    return out


def evaluate(
    fn: ForecastFn, series: list[Series], start: date, end: date, step: int = 1, batch: int = 256
) -> list[dict[str, Any]]:
    rows = []
    for s in series:
        idx = origins(s, start, end, step)
        for b0 in range(0, len(idx), batch):
            ib = idx[b0 : b0 + batch]
            hist = [s.values[: i + 1] for i in ib]
            q = fn(hist, H)
            for k, i in enumerate(ib):
                sc = score(s.values[i + 1 : i + 1 + H], q[k])
                origin = s.dates[i]
                rows.append(
                    {
                        "symbol": s.sym,
                        "origin": origin.isoformat(),
                        "regime": regime_of(origin),
                        **sc,
                    }
                )
    return rows


# --------------------------------------------------------------------------- baselines


def _finite_tail(x: np.ndarray, n: int) -> np.ndarray:
    v = x[np.isfinite(x)]
    return v[-n:]


def random_walk(histories: list[np.ndarray], h: int) -> np.ndarray:
    """Median = last observed value; quantiles from the empirical distribution of
    log(y_{t+k} / y_t) over the trailing two years, for each k."""
    out = np.full((len(histories), h, len(LEVELS)), np.nan)
    for b, x in enumerate(histories):
        lx = np.log(np.where(x > 0, x, np.nan))
        last = _finite_tail(lx, 1)[0]
        tail = lx[-730:]
        for k in range(1, h + 1):
            d = tail[k:] - tail[:-k]
            d = d[np.isfinite(d)]
            qs = np.quantile(d, LEVELS) if d.size >= 30 else np.zeros(len(LEVELS))
            qs = qs - np.median(d) if d.size >= 30 else qs  # median exactly at the last value
            out[b, k - 1] = np.exp(last + qs)
    return out


def _har_features(lx: np.ndarray) -> np.ndarray:
    """[1, x_t, mean_5(x), mean_22(x)] for every t (NaN where not computable)."""
    n = lx.size
    f = np.full((n, 4), np.nan)
    f[:, 0] = 1.0
    f[:, 1] = lx
    for t in range(n):
        if t >= 21:
            w5, w22 = lx[t - 4 : t + 1], lx[t - 21 : t + 1]
            if np.isfinite(w5).sum() >= 3 and np.isfinite(w22).sum() >= 15:
                f[t, 2] = np.nanmean(w5)
                f[t, 3] = np.nanmean(w22)
    return f


class HAR:
    """HAR-RV (Corsi 2009) on log realized volatility, one direct regression per horizon,
    refit every ``refit`` days on all history up to the origin (expanding window).
    Quantiles = fitted value + empirical quantiles of the in-sample residuals."""

    def __init__(self, refit: int = 22):
        self.refit = refit
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, lx: np.ndarray, h: int) -> list[tuple[np.ndarray, np.ndarray]]:
        f = _har_features(lx)
        out = []
        for k in range(1, h + 1):
            X, y = f[:-k], lx[k:]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            resid = y[ok] - X[ok] @ beta
            out.append((beta, np.quantile(resid, LEVELS)))
        return out

    def __call__(self, histories: list[np.ndarray], h: int) -> np.ndarray:
        out = np.full((len(histories), h, len(LEVELS)), np.nan)
        for b, x in enumerate(histories):
            lx = np.log(np.where(x > 0, x, np.nan))
            n = lx.size
            key = (
                n - n % self.refit,
                int(np.nansum(lx[:5]) * 1e6),
            )  # same series, same refit block
            fit_n = key[0]
            if key not in self._cache:
                self._cache[key] = self.fit(lx[:fit_n], h)  # type: ignore[assignment]
            fits = self._cache[key]
            f_now = _har_features(lx[-30:])[-1]
            if not np.isfinite(f_now).all():
                f_now = np.nan_to_num(f_now, nan=float(np.nanmean(lx[-30:])))
            for k, (beta, rq) in enumerate(fits):  # type: ignore[misc]
                out[b, k] = np.exp(f_now @ beta + rq)
        return out


# --------------------------------------------------------------------------- summaries


def summarise(
    rows: list[dict[str, Any]],
    ref: list[dict[str, Any]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per symbol x regime: pooled wQL (sum loss / sum |y|), MAE, coverage; and, against a
    reference model, the wQL ratio with a moving-block bootstrap 95% interval over origins."""
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        for reg in (r["regime"], "all_test"):
            if reg == "near_break":
                continue
            groups.setdefault((r["symbol"], reg), []).append(i)
            groups.setdefault(("both", reg), []).append(i)
    rng = np.random.default_rng(seed)
    for (sym, reg), idx in sorted(groups.items()):
        rr = [rows[i] for i in idx if np.isfinite(rows[i]["wql"])]
        if not rr:
            continue
        num = np.array([r["wql_num"] for r in rr])
        den = np.array([r["abs_y"] for r in rr])
        g: dict[str, Any] = {
            "origins": len(rr),
            "wql": float(num.sum() / den.sum()),
            "mae": float(np.mean([r["mae"] for r in rr])),
            **{k: float(np.mean([r[k] for r in rr])) for k in ("cov50", "cov80", "cov90")},
        }
        if ref is not None:
            refmap = {(r["symbol"], r["origin"]): r for r in ref}
            pairs = [
                (r, refmap[(r["symbol"], r["origin"])])
                for r in rr
                if (r["symbol"], r["origin"]) in refmap
                and np.isfinite(refmap[(r["symbol"], r["origin"])]["wql"])
            ]
            a = np.array([p[0]["wql_num"] for p in pairs])
            b = np.array([p[1]["wql_num"] for p in pairs])
            n = len(pairs)
            if n >= BLOCK:
                starts = rng.integers(0, n - BLOCK + 1, size=(n_boot, int(np.ceil(n / BLOCK))))
                take = (starts[..., None] + np.arange(BLOCK)).reshape(n_boot, -1)[:, :n]
                ratios = a[take].sum(axis=1) / b[take].sum(axis=1)
                g["wql_ratio_vs_ref"] = float(a.sum() / b.sum())
                g["wql_ratio_ci95"] = [
                    float(np.quantile(ratios, 0.025)),
                    float(np.quantile(ratios, 0.975)),
                ]
        out[f"{sym}/{reg}"] = g
    return out


def save(name: str, rows: list[dict[str, Any]], info: dict[str, Any]) -> Path:
    d = results_dir() / "finance" / name
    d.mkdir(parents=True, exist_ok=True)
    keys = [
        "symbol",
        "origin",
        "regime",
        "wql_num",
        "abs_y",
        "wql",
        "mae",
        "cov50",
        "cov80",
        "cov90",
        "n",
    ]
    with open(d / "per_origin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k)) for k in keys}
            )
    write_result(
        d / "run.json",
        {"model": name, "info": info, "n_rows": len(rows)},
        split="test",
        label=f"M8-{name}",
    )
    return d


def load_rows(name: str) -> list[dict[str, Any]]:
    p = results_dir() / "finance" / name / "per_origin.csv"
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    **r,
                    **{
                        k: float(r[k]) if r[k] not in ("", "nan") else np.nan
                        for k in ("wql_num", "abs_y", "wql", "mae", "cov50", "cov80", "cov90")
                    },
                }
            )
    return rows


def report(models: list[str], ref: str = "random_walk") -> Path:
    refrows = load_rows(ref)
    out = {m: summarise(load_rows(m), None if m == ref else refrows) for m in models}
    return write_result(
        results_dir() / "finance" / "summary.json",
        {
            "reference": ref,
            "regimes": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in REGIMES.items()},
            "models": out,
        },
        split="test",
        label="M8-M9-summary",
    )


def baseline_runs() -> None:
    series = load_all()
    end = series[0].dates[-1]
    for name, fn in (("random_walk", random_walk), ("har", HAR())):
        rows = evaluate(fn, series, TEST_START, end)
        save(name, rows, {"kind": "baseline"})
        print(name, len(rows), "origins")


def json_dump(x: Any) -> str:
    return json.dumps(x, indent=1, default=str)
