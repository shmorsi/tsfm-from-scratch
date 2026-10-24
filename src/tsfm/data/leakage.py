"""Leakage check: does any pretraining or validation series contain GIFT-Eval data?

Upstream describes GiftEvalPretrain as non-leaking. We test that claim on the content of
the series, not their names, with two checks:

1. **Native frequency.** Windows of ``W`` consecutive points of every GIFT-Eval series
   (stride ``G``, whole series, not only the test region) are z-normalised, rounded to
   ``1/Q`` and hashed. Pretraining windows are hashed the same way at stride ``S`` and
   looked up. Because ``gcd(G, S) = 1``, the two window grids coincide at least once in
   every ``G * S`` consecutive offsets, so any shared segment of at least
   ``W + G * S - 1`` points is guaranteed to be found, also when one copy is an affine
   rescaling of the other (units, offsets).
2. **Daily aggregates.** Sub-daily series on both sides are averaged over complete
   calendar days (z-normalisation makes sum vs mean irrelevant) and checked as in (1)
   with ``W_DAY``-day windows. This catches a GIFT-Eval series that appears in the
   pretraining corpus at a different sampling rate.

Low-information windows (missing values, constant, fewer than ``MIN_DISTINCT`` distinct
rounded values, or more than ``MAX_LINEAR_FRACTION`` of the window on a straight line)
are skipped on both sides, otherwise flat, sparse or linearly interpolated segments would
"match" everywhere. (The straight-line rule was added after the first round flagged
unrelated energy-meter series as "matching" SZ_TAXI through interpolated gaps; that
round's report is kept in ``results/data/leakage_round1.json`` and every series it
flagged stays excluded.)

What this does not detect: copies with added noise, warped time axes, or aggregations
other than whole days. The report says so.

Any pretraining series with at least one hit is written to
``locks/leakage_exclusions.json``; the corpus is rebuilt without it and the check is run
again. The committed ``results/data/leakage.json`` must show zero hits on the final corpus.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from tsfm.data.convert import _list_values, iter_raw_series
from tsfm.paths import processed_dir, repo_root, results_dir
from tsfm.provenance import sha256_file, write_result

W = 64
G = 3  # GIFT-Eval stride
S = 8  # pretraining stride (coprime with G)
W_DAY = 28
G_DAY = 1
S_DAY = 4
Q = 100.0
MIN_DISTINCT = 8
MAX_LINEAR_FRACTION = 0.5
CHUNK = 100_000
_R = np.random.default_rng(0x5EED).integers(
    1, 2**63, size=max(W, W_DAY), dtype=np.uint64
) | np.uint64(1)


if np.gcd(G, S) != 1 or np.gcd(G_DAY, S_DAY) != 1:
    raise ValueError("strides must be coprime for the detection guarantee")
MIN_SHARED_GUARANTEED = W + G * S - 1
MIN_SHARED_DAYS_GUARANTEED = W_DAY + G_DAY * S_DAY - 1


def window_hashes(x: np.ndarray, w: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Hashes of informative windows of ``x`` and their start offsets."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < w:
        return np.empty(0, np.uint64), np.empty(0, np.int64)
    views = sliding_window_view(x, w)[::stride]
    starts_all = np.arange(views.shape[0], dtype=np.int64) * stride
    hs, ss = [], []
    r = _R[:w]
    for lo in range(0, views.shape[0], CHUNK):
        v = views[lo : lo + CHUNK]
        ok = np.isfinite(v).all(axis=1)
        if not ok.any():
            continue
        v = v[ok]
        st = starts_all[lo : lo + CHUNK][ok]
        mu = v.mean(axis=1, keepdims=True)
        sd = v.std(axis=1, keepdims=True)
        good = sd[:, 0] > 1e-9 * np.maximum(1.0, np.abs(mu[:, 0]))
        if not good.any():
            continue
        z = (v[good] - mu[good]) / sd[good]
        q = np.rint(z * Q).astype(np.int64)
        srt = np.sort(q, axis=1)
        distinct = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1)
        # Mostly straight-line windows (e.g. linearly interpolated gaps) look identical
        # after z-normalisation whatever series they come from, so they carry no
        # identity: skip windows whose second differences are ~0 over most of the window.
        linear = (np.abs(np.diff(q, n=2, axis=1)) <= 1).mean(axis=1)
        keep = (distinct >= MIN_DISTINCT) & (linear <= MAX_LINEAR_FRACTION)
        with np.errstate(over="ignore"):
            h = (q[keep].view(np.uint64) * r).sum(axis=1, dtype=np.uint64)
        hs.append(h)
        ss.append(st[good][keep])
    if not hs:
        return np.empty(0, np.uint64), np.empty(0, np.int64)
    return np.concatenate(hs), np.concatenate(ss)


def _sub_daily_steps(freq: str) -> int | None:
    """Points per day for fixed sub-daily frequencies, else None."""
    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            off = pd.tseries.frequencies.to_offset(freq)
            ns = pd.Timedelta(off).value  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None
    day = 86_400 * 10**9
    if ns <= 0 or ns >= day or day % ns:
        return None
    return int(day // ns)


def daily_means(x: np.ndarray, start: Any, freq: str) -> np.ndarray | None:
    """Mean over complete calendar days (NaN for incomplete days); None if not sub-daily."""
    import pandas as pd

    per_day = _sub_daily_steps(freq)
    if per_day is None:
        return None
    t0 = pd.Timestamp(start)
    step_ns = 86_400 * 10**9 // per_day
    first = int((t0 - t0.normalize()).value // step_ns)  # position of first point in its day
    idx = (first + np.arange(x.size)) // per_day
    n_days = int(idx[-1]) + 1 if x.size else 0
    obs = np.isfinite(x)
    sums = np.bincount(idx[obs], weights=x[obs].astype(np.float64), minlength=n_days)
    cnts = np.bincount(idx[obs], minlength=n_days)
    out = np.full(n_days, np.nan)
    full = cnts == per_day
    out[full] = sums[full] / per_day
    return out


@dataclass
class HashIndex:
    """Sorted hashes of GIFT-Eval windows with the dataset they came from."""

    hashes: np.ndarray
    ds_id: np.ndarray
    in_test: np.ndarray
    names: list[str]

    def lookup(self, h: np.ndarray) -> np.ndarray:
        """Index into the index arrays for each query hash, or -1."""
        if h.size == 0 or self.hashes.size == 0:
            return np.full(h.shape, -1, np.int64)
        pos = np.searchsorted(self.hashes, h)
        pos = np.clip(pos, 0, self.hashes.size - 1)
        return np.where(self.hashes[pos] == h, pos, -1)


def _gifteval_series() -> Any:
    from tsfm.eval.gifteval import SHORT_DATASETS
    from tsfm.paths import gifteval_dir

    for ds_name in SHORT_DATASETS:
        paths = sorted(
            (gifteval_dir() / ds_name).glob("data-*-of-*.arrow")
        )  # skip HF cache-*.arrow
        if not paths:
            raise FileNotFoundError(f"no arrow files for GIFT-Eval {ds_name}")
        for s in iter_raw_series(paths, ds_name):
            yield ds_name, s


def build_gifteval_index(
    test_lengths: dict[str, int],
) -> tuple[HashIndex, HashIndex, dict[str, Any]]:
    """Hash every GIFT-Eval window at native frequency and as daily means."""
    names: list[str] = []
    native: dict[str, list[np.ndarray]] = defaultdict(list)
    daily: dict[str, list[np.ndarray]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for ds_name, s in _gifteval_series():
        if ds_name not in names:
            names.append(ds_name)
        i = names.index(ds_name)
        n = s.values.size
        h, st = window_hashes(s.values, W, G)
        test_from = n - test_lengths[ds_name]
        native["h"].append(h)
        native["d"].append(np.full(h.size, i, np.int16))
        native["t"].append(st + W > test_from)
        stats["gifteval_series"] += 1
        stats["gifteval_windows"] += h.size
        dm = daily_means(s.values, s.start, s.freq)
        if dm is None and _is_daily(s.freq):
            dm = s.values.astype(np.float64)
        if dm is not None:
            hd, _ = window_hashes(dm, W_DAY, G_DAY)
            daily["h"].append(hd)
            daily["d"].append(np.full(hd.size, i, np.int16))
            daily["t"].append(np.zeros(hd.size, bool))
            stats["gifteval_daily_windows"] += hd.size
    return _finish(native, names), _finish(daily, names), dict(stats)


def _is_daily(freq: str) -> bool:
    return freq.upper() in {"D", "1D", "B"}


def _finish(parts: dict[str, list[np.ndarray]], names: list[str]) -> HashIndex:
    h = np.concatenate(parts["h"]) if parts["h"] else np.empty(0, np.uint64)
    d = np.concatenate(parts["d"]) if parts["d"] else np.empty(0, np.int16)
    t = np.concatenate(parts["t"]) if parts["t"] else np.empty(0, bool)
    order = np.argsort(h, kind="stable")
    return HashIndex(h[order], d[order], t[order], names)


def _corpus_files() -> list[Path]:
    root = processed_dir()
    return sorted(
        p
        for split in ("pretrain", "val")
        for p in (root / split).glob("*.parquet")
        if "kernelsynth" not in p.name
    )


def check_corpus(native: HashIndex, daily: HashIndex) -> dict[str, Any]:
    """Compare the built corpus to the GIFT-Eval hash indexes and return the report."""
    per_file: dict[str, Any] = {}
    leaking: list[dict[str, Any]] = []
    for path in _corpus_files():
        t = pq.read_table(path)
        vals, offs = _list_values(t.column("target"))
        ids = t.column("item_id").to_pylist()
        freqs = t.column("freq").to_pylist()
        starts = t.column("start").to_pylist()
        rep: dict[str, Any] = {
            "series": len(ids),
            "windows": 0,
            "daily_windows": 0,
            "hits": 0,
            "daily_hits": 0,
            "test_region_hits": 0,
            "series_with_hits": 0,
            "matched": Counter(),
        }
        for i, item in enumerate(ids):
            x = vals[offs[i] : offs[i + 1]]
            h, _ = window_hashes(x, W, S)
            m = native.lookup(h)
            hit = m >= 0
            rep["windows"] += h.size
            n_hit = int(hit.sum())
            n_test = int(native.in_test[m[hit]].sum()) if n_hit else 0
            matched = (
                Counter(native.names[int(j)] for j in native.ds_id[m[hit]]) if n_hit else Counter()
            )
            dm = daily_means(x, starts[i], freqs[i])
            if dm is None and _is_daily(freqs[i]):
                dm = x.astype(np.float64)
            n_dhit = 0
            if dm is not None:
                hd, _ = window_hashes(dm, W_DAY, S_DAY)
                md = daily.lookup(hd)
                rep["daily_windows"] += hd.size
                n_dhit = int((md >= 0).sum())
                if n_dhit:
                    matched.update(
                        daily.names[int(j)] + " (daily)" for j in daily.ds_id[md[md >= 0]]
                    )
            if n_hit or n_dhit:
                rep["series_with_hits"] += 1
                rep["hits"] += n_hit
                rep["daily_hits"] += n_dhit
                rep["test_region_hits"] += n_test
                rep["matched"].update(matched)
                leaking.append(
                    {
                        "item_id": item,
                        "file": path.name,
                        "hits": n_hit,
                        "daily_hits": n_dhit,
                        "test_region_hits": n_test,
                        "matched": dict(matched),
                    }
                )
        rep["matched"] = dict(rep["matched"])
        rep["sha256"] = sha256_file(path)
        per_file[f"{path.parent.name}/{path.name}"] = rep
        print(
            f"{path.parent.name}/{path.name:40s} windows {rep['windows']:>11,d} hits {rep['hits']:>7d} daily {rep['daily_hits']:>6d} series {rep['series_with_hits']}"
        )
    return {"files": per_file, "leaking_series": leaking}


def exclusions_path() -> Path:
    """Path of the committed leakage-exclusion list."""
    return repo_root() / "locks" / "leakage_exclusions.json"


def load_exclusions() -> set[str]:
    """Item ids removed from the corpus. Empty if the file is absent."""
    p = exclusions_path()
    if not p.exists():
        return set()
    return set(json.loads(p.read_text())["item_ids"])


def run_leakage_check() -> bool:
    """Run both checks on the built corpus. Returns True iff there are zero hits.

    On hits, the offending series are added to the exclusion list so that
    ``tsfm data build`` drops them; then run this again."""
    lengths_path = repo_root() / "locks" / "gifteval_test_regions.json"
    if lengths_path.exists():
        lengths = json.loads(lengths_path.read_text())
    else:
        from tsfm.eval.gifteval import test_region_lengths

        lengths = test_region_lengths()
        lengths_path.write_text(json.dumps(lengths, indent=1, sort_keys=True) + "\n")
    native, daily, gstats = build_gifteval_index(lengths)
    print(f"GIFT-Eval index: {gstats}")
    report = check_corpus(native, daily)
    n_leak = len(report["leaking_series"])
    totals = {
        "pretrain_series_checked": sum(f["series"] for f in report["files"].values()),
        "pretrain_windows_checked": sum(f["windows"] for f in report["files"].values()),
        "pretrain_daily_windows_checked": sum(f["daily_windows"] for f in report["files"].values()),
        "native_hits": sum(f["hits"] for f in report["files"].values()),
        "daily_hits": sum(f["daily_hits"] for f in report["files"].values()),
        "series_with_hits": n_leak,
        "excluded_series_total": len(
            load_exclusions() | {r["item_id"] for r in report["leaking_series"]}
        ),
        **gstats,
    }
    params = {
        "W": W,
        "G": G,
        "S": S,
        "G_DAY": G_DAY,
        "W_DAY": W_DAY,
        "S_DAY": S_DAY,
        "Q": Q,
        "MIN_DISTINCT": MIN_DISTINCT,
        "MAX_LINEAR_FRACTION": MAX_LINEAR_FRACTION,
        "min_shared_points_guaranteed": MIN_SHARED_GUARANTEED,
        "min_shared_days_guaranteed": MIN_SHARED_DAYS_GUARANTEED,
    }
    write_result(
        results_dir() / "data" / "leakage.json",
        {"params": params, "totals": totals, **report},
        split="none",
        label="M1-leakage",
    )
    if n_leak:
        excl = load_exclusions() | {r["item_id"] for r in report["leaking_series"]}
        exclusions_path().write_text(json.dumps({"item_ids": sorted(excl)}, indent=1) + "\n")
        print(f"{n_leak} leaking series added to {exclusions_path()}; rebuild and re-run.")
    return n_leak == 0
