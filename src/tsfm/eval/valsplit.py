"""Validation split: our held-out pretraining datasets, scored like GIFT-Eval.

Each validation dataset (see ``validation_datasets`` in ``configs/data.yaml``, plus the
held-out synthetic series) is wrapped so that it exposes the same interface as upstream's
``Dataset`` and uses the same rules: prediction length from the frequency (M4 lengths for
yearly/quarterly series, which the general GIFT-Eval map does not cover), the "short"
term for every dataset and the "long" term (x15) for sub-daily ones, and test windows
covering the last 10% of the shortest series (at most 20 windows).

This split is for model selection, scaling fits and development. It is never used to
claim benchmark performance.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tsfm.eval.gifteval import VENDOR
from tsfm.paths import processed_dir

SUBDAILY = {"H", "T", "S"}
MIN_CONTEXT = 16


def _vendor() -> Any:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import gift_eval_data

    return gift_eval_data


def norm_freq(freq: str) -> str:
    from gluonts.time_feature import norm_freq_str
    from pandas.tseries.frequencies import to_offset

    return str(_vendor().maybe_reconvert_freq(norm_freq_str(to_offset(freq).name)))


def base_prediction_length(freq: str) -> int:
    v = _vendor()
    f = norm_freq(freq)
    if f in v.PRED_LENGTH_MAP:
        return int(v.PRED_LENGTH_MAP[f])
    return int(v.M4_PRED_LENGTH_MAP[f])


def period_freq(freq: str) -> str:
    """Pandas Period does not accept start-anchored offsets (MS, QS, YS); use the
    period equivalent so ``Period(start, freq)`` works for every dataset."""
    f = freq.upper()
    for start, period in (("MS", "M"), ("QS", "Q"), ("AS", "A"), ("YS", "A"), ("BMS", "M")):
        if f == start or f.startswith(start + "-"):
            return period + freq[len(start) :]
    return freq


@dataclass(frozen=True)
class ValConfig:
    name: str  # "<dataset>/<freq>/<term>"
    dataset: str
    split_dir: str  # "val"
    term: str
    domain: str = "validation"
    num_variates: int = 1


@cache
def _freq_of(dataset: str) -> str:
    path = processed_dir() / "val" / f"{dataset}.parquet"
    t = pq.read_table(path, columns=["freq"])
    freqs = set(t.column("freq").to_pylist())
    if len(freqs) != 1:
        raise ValueError(f"{dataset}: expected one frequency, got {freqs}")
    return str(freqs.pop())


@cache
def _max_len(dataset: str) -> int:
    t = pq.read_table(processed_dir() / "val" / f"{dataset}.parquet", columns=["target"])
    import pyarrow.compute as pc

    return int(pc.max(pc.list_value_length(t.column("target"))).as_py())


@cache
def val_configs() -> tuple[ValConfig, ...]:
    """Every validation dataset with the "short" term, plus "long" for sub-daily ones;
    a term is skipped if no series is long enough for one window plus some context."""
    out = []
    for p in sorted((processed_dir() / "val").glob("*.parquet")):
        ds = p.stem
        freq = _freq_of(ds)
        terms = ["short", "long"] if norm_freq(freq) in SUBDAILY else ["short"]
        for term in terms:
            h = {"short": 1, "long": 15}[term] * base_prediction_length(freq)
            if _max_len(ds) >= h + MIN_CONTEXT:
                out.append(ValConfig(f"{ds}/{freq}/{term}", ds, "val", term))
    return tuple(out)


class ValDataset:
    """Same interface as upstream ``gift_eval.data.Dataset`` (the parts evaluation uses)."""

    def __init__(self, cfg: ValConfig):
        self.cfg = cfg
        self.term = cfg.term
        t = pq.read_table(processed_dir() / cfg.split_dir / f"{cfg.dataset}.parquet")
        self._rows = t.to_pylist()
        self.freq = _freq_of(cfg.dataset)

    @cached_property
    def prediction_length(self) -> int:
        mult = {"short": 1, "medium": 10, "long": 15}[self.term]
        return mult * base_prediction_length(self.freq)

    @cached_property
    def _entries(self) -> list[dict[str, Any]]:
        import pandas as pd

        h = self.prediction_length
        out = []
        for r in self._rows:
            x = np.asarray(r["target"], np.float32)
            # Series too short for even one window plus some context are skipped.
            if x.size < h + MIN_CONTEXT:
                continue
            out.append(
                {
                    "item_id": r["item_id"],
                    "start": pd.Period(r["start"], freq=period_freq(self.freq)),
                    "target": x,
                }
            )
        return out

    @cached_property
    def windows(self) -> int:
        v = _vendor()
        min_len = min(e["target"].size for e in self._entries)
        w = math.ceil(v.TEST_SPLIT * min_len / self.prediction_length)
        return int(min(max(1, w), v.MAX_WINDOW))

    @property
    def test_data(self) -> Any:
        from gluonts.dataset.split import split

        _, tmpl = split(self._entries, offset=-self.prediction_length * self.windows)
        return tmpl.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )


def load_val(cfg: ValConfig) -> ValDataset:
    return ValDataset(cfg)
