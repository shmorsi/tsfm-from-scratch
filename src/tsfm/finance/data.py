"""Daily realized volatility from Binance Vision 5-minute spot klines.

Source: https://data.binance.vision (Binance Vision dataset terms: CC BY-NC-SA 4.0,
non-commercial research use; derived data must credit Binance Vision and keep the same
licence). Monthly kline archives are verified against Binance's published ``.CHECKSUM``
SHA-256 files, and the hashes are recorded in ``locks/binance.lock.json``.

Daily realized volatility (percent) for UTC day d:
    rvol_d = 100 * sqrt( sum_i r_i^2 ),  r_i = log(close_i / close_{i-1})
over the 5-minute bars of day d (the first return of a day uses the previous day's last
close). Days with fewer than ``MIN_BARS`` of the 288 bars are missing (NaN).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.paths import raw_dir, repo_root
from tsfm.provenance import sha256_file

BASE = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVAL = "5m"
FIRST_MONTH = {"BTCUSDT": (2017, 8), "ETHUSDT": (2017, 8)}
LAST_MONTH = (2026, 9)
BARS_PER_DAY = 288
MIN_BARS = 274  # 95% of 288


def months(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    """Year-month pairs from ``first`` through ``last``, inclusive."""
    y, m = first
    out = []
    while (y, m) <= last:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def file_name(sym: str, y: int, m: int) -> str:
    """Binance Vision monthly kline zip name."""
    return f"{sym}-{INTERVAL}-{y:04d}-{m:02d}.zip"


def lock_path() -> Path:
    """Path of the Binance download lock."""
    return repo_root() / "locks" / "binance.lock.json"


def local_dir() -> Path:
    """Directory of the downloaded kline zips."""
    return raw_dir() / "binance"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "tsfm"}), timeout=120
    ) as r:
        data: bytes = r.read()
        return data


def download(write_lock: bool = False) -> list[str]:
    """Download every monthly archive, verify against Binance's CHECKSUM (and against our
    lockfile if it exists). With ``write_lock`` the lockfile is (re)written."""
    lock = json.loads(lock_path().read_text()) if lock_path().exists() and not write_lock else None
    entries, errors = [], []
    for sym in SYMBOLS:
        for y, m in months(FIRST_MONTH[sym], LAST_MONTH):
            name = file_name(sym, y, m)
            url = f"{BASE}/{sym}/{INTERVAL}/{name}"
            dest = local_dir() / sym / name
            published = _get(url + ".CHECKSUM").decode().split()[0]
            if lock is not None:
                want = lock["files"].get(f"{sym}/{name}")
                if want != published:
                    errors.append(f"{name}: published checksum {published} != lock {want}")
                    continue
            if not dest.exists() or sha256_file(dest) != published:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_get(url))
            got = sha256_file(dest)
            if got != published:
                errors.append(f"{name}: sha256 {got} != published {published}")
            entries.append((f"{sym}/{name}", got))
    if write_lock and not errors:
        lock_path().write_text(
            json.dumps(
                {
                    "source": BASE,
                    "interval": INTERVAL,
                    "licence": "CC BY-NC-SA 4.0 (Binance Vision dataset terms)",
                    "files": dict(sorted(entries)),
                },
                indent=1,
            )
            + "\n"
        )
    return errors


def read_closes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(open_time_ms, close) arrays from one monthly archive. Binance switched spot
    timestamps from milliseconds to microseconds in 2025; both are handled."""
    with zipfile.ZipFile(path) as z:
        (inner,) = z.namelist()
        text = z.read(inner).decode()
    ts, close = [], []
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].isdigit():
            continue  # header line in some files
        t = int(row[0])
        ts.append(t // 1000 if t > 10**14 else t)
        close.append(float(row[4]))
    return np.asarray(ts, np.int64), np.asarray(close, np.float64)


def daily_rvol(ts_ms: np.ndarray, close: np.ndarray) -> tuple[list[date], np.ndarray, np.ndarray]:
    """Daily realized volatility (percent) and bar counts per UTC day."""
    order = np.argsort(ts_ms)
    ts_ms, close = ts_ms[order], close[order]
    keep = np.concatenate([[True], np.diff(ts_ms) > 0])  # drop duplicate bars
    ts_ms, close = ts_ms[keep], close[keep]
    r = np.diff(np.log(close))
    gap_ok = np.diff(ts_ms) == 5 * 60 * 1000  # only returns between consecutive bars
    day = (ts_ms[1:] // 86_400_000).astype(np.int64)
    days = np.arange(day.min(), day.max() + 1)
    sq = np.bincount(day - days[0], weights=np.where(gap_ok, r**2, 0.0), minlength=days.size)
    n = np.bincount(day - days[0], weights=gap_ok.astype(float), minlength=days.size)
    rv = 100 * np.sqrt(sq)
    rv[n < MIN_BARS] = np.nan
    dates = [date.fromordinal(date(1970, 1, 1).toordinal() + int(d)) for d in days]
    return dates, rv, n


def build() -> dict[str, Any]:
    """Compute daily rvol per symbol; write ``results/finance/rvol_<SYM>.csv`` + summary."""
    out_dir = repo_root() / "results" / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for sym in SYMBOLS:
        ts_all, cl_all = [], []
        for y, m in months(FIRST_MONTH[sym], LAST_MONTH):
            t, c = read_closes(local_dir() / sym / file_name(sym, y, m))
            ts_all.append(t)
            cl_all.append(c)
        dates, rv, n = daily_rvol(np.concatenate(ts_all), np.concatenate(cl_all))
        p = out_dir / f"rvol_{sym}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "rvol_pct", "bars"])
            for d, v, k in zip(dates, rv, n, strict=True):
                w.writerow([d.isoformat(), "" if np.isnan(v) else f"{v:.6f}", int(k)])
        summary[sym] = {
            "first_day": dates[0].isoformat(),
            "last_day": dates[-1].isoformat(),
            "days": len(dates),
            "missing_days": int(np.isnan(rv).sum()),
            "median_rvol_pct": float(np.nanmedian(rv)),
            "sha256": sha256_file(p),
        }
    from tsfm.provenance import write_result

    write_result(
        out_dir / "rvol_summary.json",
        {
            "symbols": summary,
            "min_bars": MIN_BARS,
            "attribution": "Data: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0",
        },
        split="none",
        label="M8-data",
    )
    return summary


def load_rvol(sym: str) -> tuple[list[date], np.ndarray]:
    """Daily dates and realized volatility for one symbol."""
    p = repo_root() / "results" / "finance" / f"rvol_{sym}.csv"
    dates, vals = [], []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(date.fromisoformat(row["date"]))
            vals.append(float(row["rvol_pct"]) if row["rvol_pct"] else np.nan)
    return dates, np.asarray(vals, np.float64)
