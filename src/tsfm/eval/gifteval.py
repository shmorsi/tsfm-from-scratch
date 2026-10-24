"""The 97 official GIFT-Eval configurations and the upstream data loader.

Naming follows upstream's notebooks exactly: a directory name such as
``kdd_cup_2018_with_missing/H`` becomes the config ``kdd_cup_2018/H/short``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from tsfm.paths import gifteval_dir

VENDOR = Path(__file__).parent / "_vendor"

# Upstream notebooks/naive.ipynb (full lists, as commented there).
SHORT_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]
MED_LONG_DATASETS = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


@dataclass(frozen=True)
class Config:
    """One official GIFT-Eval configuration."""

    name: str  # official config name, e.g. "electricity/15T/short"
    ds_name: str  # upstream Dataset() name / directory, e.g. "electricity/15T"
    term: str
    domain: str
    num_variates: int

    @property
    def key(self) -> str:
        """Dataset key, the first component of the official name."""
        return self.name.split("/")[0]

    @property
    def freq_label(self) -> str:
        """Frequency label, the second component of the official name."""
        return self.name.split("/")[1]


@cache
def properties() -> dict[str, dict[str, Any]]:
    """Upstream dataset properties, keyed by dataset name."""
    props: dict[str, dict[str, Any]] = json.loads((VENDOR / "dataset_properties.json").read_text())
    return props


def _config_name(ds_name: str, term: str) -> str:
    if "/" in ds_name:
        key, freq = ds_name.split("/")
        key = PRETTY_NAMES.get(key.lower(), key.lower())
    else:
        key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
        freq = properties()[key]["frequency"]
    return f"{key}/{freq}/{term}"


@cache
def configs() -> tuple[Config, ...]:
    """Every official short, medium and long configuration this repo scores."""
    out = []
    for ds in SHORT_DATASETS:
        for term in ("short", "medium", "long"):
            if term != "short" and ds not in MED_LONG_DATASETS:
                continue
            name = _config_name(ds, term)
            key = name.split("/")[0]
            p = properties()[key]
            out.append(Config(name, ds, term, p["domain"], int(p["num_variates"])))
    official = official_config_names()
    names = {c.name for c in out}
    if names != official:
        raise AssertionError(f"config mismatch: {sorted(names ^ official)}")
    return tuple(sorted(out, key=lambda c: c.name))


@cache
def official_config_names() -> frozenset[str]:
    """Configuration names from upstream's committed CSV."""
    with open(VENDOR / "dataset_configurations.csv", newline="") as f:
        return frozenset(row["dataset"] for row in csv.DictReader(f))


def load_dataset(ds_name: str, term: str, to_univariate: bool | None = None) -> Any:
    """Upstream ``gift_eval.data.Dataset`` (vendored, unmodified)."""
    os.environ.setdefault("GIFT_EVAL", str(gifteval_dir()))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from gift_eval_data import Dataset

    if to_univariate is None:
        to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
    return Dataset(name=ds_name, term=term, to_univariate=to_univariate)


def test_region_lengths() -> dict[str, int]:
    """For every dataset directory, the number of trailing points covered by the test
    windows of its longest term (``windows * prediction_length``)."""
    out: dict[str, int] = {}
    for c in configs():
        ds = load_dataset(c.ds_name, c.term, to_univariate=False)
        out[c.ds_name] = max(out.get(c.ds_name, 0), int(ds.windows * ds.prediction_length))
    return out
