"""M1 data pipeline: selection, verification, conversion, synthetic data, leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tsfm.data import leakage
from tsfm.data.convert import Rules, convert_dataset, iter_raw_series, shift_start, validate_parquet
from tsfm.data.sources import select_pretrain_files, verify_file
from tsfm.data.synthetic import sample_series
from tsfm.paths import processed_dir, repo_root, results_dir
from tsfm.provenance import sha256_file

# ---------------------------------------------------------------- helpers


def write_hf_arrow(path: Path, rows: list[dict], multivariate: bool = False) -> Path:
    """Write rows in the layout of HF `save_to_disk` (Arrow IPC stream)."""
    tgt = pa.list_(pa.list_(pa.float32())) if multivariate else pa.list_(pa.float32())
    schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("start", pa.timestamp("s")),
            ("freq", pa.string()),
            ("target", tgt),
        ]
    )
    table = pa.table({k: [r[k] for r in rows] for k in schema.names}, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as w:
        w.write_table(table, max_chunksize=2)
    return path


def row(i: int, n: int, *, dims: int = 0, freq: str = "h") -> dict:
    rng = np.random.default_rng(i)
    x = rng.normal(size=(dims, n) if dims else n).astype(np.float32)
    return {
        "item_id": f"s{i}",
        "start": pd.Timestamp("2020-01-01"),
        "freq": freq,
        "target": x.tolist(),
    }


# ---------------------------------------------------------------- rules


def test_rules_reasons():
    r = Rules(min_observed=4)
    assert r.check(np.array([], np.float32)) == "empty"
    assert r.check(np.array([1, 2, np.inf, 3, 4], np.float32)) == "inf"
    assert r.check(np.array([1, 2, np.nan], np.float32)) == "too_short"
    assert r.check(np.ones(10, np.float32)) == "constant"
    assert r.check(np.array([1, 2, 3, 4, 1e13], np.float32)) == "too_large"
    x = np.full(100, np.nan, np.float32)
    x[:5] = [1, 2, 3, 4, 5]
    assert r.check(x) == "mostly_missing"
    assert r.check(np.arange(10, dtype=np.float32)) is None


def test_rules_coerce_yaml_strings():
    assert Rules(max_abs_value="1.0e12").max_abs_value == 1e12  # type: ignore[arg-type]


# ---------------------------------------------------------------- sources


def _f(path: str, size: int = 10) -> dict:
    return {"path": path, "size": size, "oid": "x", "lfs": {"oid": "y"}}


def test_select_pretrain_files_rules():
    files = [
        _f("small/data-00000-of-00001.arrow", 10**6),
        _f("small/dataset_info.json", 5),
        _f("huge/data-00000-of-00001.arrow", 10**10),
        _f("cmip6_2000/data-00000-of-00002.arrow"),
        _f("fam/data-00000-of-00003.arrow"),
        _f("fam/data-00001-of-00003.arrow"),
        _f("fam/data-00002-of-00003.arrow"),
        _f("README.md"),
    ]
    cfg = {
        "pretrain": {
            "include_all_below_mb": 100,
            "exclude_prefixes": ["cmip6_"],
            "exclude": [],
            "sampled_shards": {"fam": [0, 2]},
        }
    }
    got = sorted(e["path"] for e in select_pretrain_files(files, cfg))
    assert got == [
        "fam/data-00000-of-00003.arrow",
        "fam/data-00002-of-00003.arrow",
        "small/data-00000-of-00001.arrow",
        "small/dataset_info.json",
    ]


def test_verify_file_detects_corruption(tmp_path: Path):
    p = tmp_path / "a" / "x.arrow"
    p.parent.mkdir()
    p.write_bytes(b"hello")
    e = {
        "path": "a/x.arrow",
        "size": 5,
        "sha256": hashlib.sha256(b"hello").hexdigest(),
        "git_oid": "",
    }
    assert verify_file(e, tmp_path) is None
    p.write_bytes(b"hellp")
    assert "sha256" in str(verify_file(e, tmp_path))
    p.write_bytes(b"hell")
    assert "size" in str(verify_file(e, tmp_path))
    assert verify_file({**e, "path": "nope"}, tmp_path) == "missing"


def test_verify_small_file_by_git_blob_hash(tmp_path: Path):
    data = b'{"a": 1}'
    (tmp_path / "info.json").write_bytes(data)
    oid = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    e = {"path": "info.json", "size": len(data), "sha256": None, "git_oid": oid}
    assert verify_file(e, tmp_path) is None
    assert verify_file({**e, "git_oid": "0" * 40}, tmp_path) == "git blob hash mismatch"


# ---------------------------------------------------------------- conversion


def test_iter_raw_series_splits_multivariate(tmp_path: Path):
    p = write_hf_arrow(
        tmp_path / "d.arrow", [row(0, 50, dims=3), row(1, 40, dims=3)], multivariate=True
    )
    out = list(iter_raw_series([p], "ds"))
    assert [s.item_id for s in out] == [f"ds/s{i}_dim{d}" for i in range(2) for d in range(3)]
    assert out[0].values.shape == (50,)
    np.testing.assert_array_equal(
        out[4].values, np.asarray(row(1, 40, dims=3)["target"][1], np.float32)
    )


def test_convert_cap_determinism_and_truncation(tmp_path: Path):
    p = write_hf_arrow(tmp_path / "d.arrow", [row(i, 100 + i) for i in range(20)] + [row(99, 5)])
    a = convert_dataset(
        "ds", [p], tmp_path / "a.parquet", split="pretrain", rules=Rules(), max_points=700, seed=1
    )
    b = convert_dataset(
        "ds", [p], tmp_path / "b.parquet", split="pretrain", rules=Rules(), max_points=700, seed=1
    )
    assert sha256_file(tmp_path / "a.parquet") == sha256_file(tmp_path / "b.parquet")
    assert a.points_kept == b.points_kept == 700
    assert a.dropped["too_short"] == 1
    assert validate_parquet(tmp_path / "a.parquet", Rules()) == []
    t = pq.read_table(tmp_path / "a.parquet").to_pylist()
    # A truncated series keeps its most recent points and its start moves accordingly.
    src = {f"ds/s{i}": row(i, 100 + i) for i in range(20)}
    for r in t:
        full = np.asarray(src[r["item_id"]]["target"], np.float32)
        np.testing.assert_array_equal(
            np.asarray(r["target"], np.float32), full[-len(r["target"]) :]
        )
        shift = len(full) - len(r["target"])
        assert pd.Timestamp(r["start"]) == pd.Timestamp("2020-01-01") + pd.Timedelta(hours=shift)


def test_convert_exclusions_do_not_change_selection(tmp_path: Path):
    p = write_hf_arrow(tmp_path / "d.arrow", [row(i, 100) for i in range(20)])
    kw = {"split": "pretrain", "rules": Rules(), "max_points": 1000, "seed": 3}
    convert_dataset("ds", [p], tmp_path / "a.parquet", **kw)  # type: ignore[arg-type]
    ids = pq.read_table(tmp_path / "a.parquet").column("item_id").to_pylist()
    st2 = convert_dataset("ds", [p], tmp_path / "b.parquet", exclude=frozenset(ids[:2]), **kw)  # type: ignore[arg-type]
    ids2 = pq.read_table(tmp_path / "b.parquet").column("item_id").to_pylist()
    assert ids2 == ids[2:]
    assert st2.dropped["leakage"] == 2


def test_shift_start_calendar_freqs():
    assert shift_start(pd.Timestamp("2020-01-01"), "MS", 3) == pd.Timestamp("2020-04-01")
    assert shift_start(pd.Timestamp("2020-01-01"), "15T", 4) == pd.Timestamp("2020-01-01 01:00")


# ---------------------------------------------------------------- synthetic


def test_synthetic_deterministic_and_finite():
    a, da = sample_series(7, 123, 256, 5)
    b, db = sample_series(7, 123, 256, 5)
    np.testing.assert_array_equal(a, b)
    assert da == db
    assert np.isfinite(a).all()
    c, _ = sample_series(7, 124, 256, 5)
    assert not np.array_equal(a, c)


def test_synthetic_variety():
    descs = {sample_series(1, i, 128, 5)[1] for i in range(40)}
    assert len(descs) > 30
    assert any("periodic" in d for d in descs)


# ---------------------------------------------------------------- leakage


def test_window_hashes_affine_invariant():
    x = np.random.default_rng(0).normal(size=500)
    h1, s1 = leakage.window_hashes(x, 64, 1)
    h2, s2 = leakage.window_hashes(3.7 * x - 12.0, 64, 1)
    np.testing.assert_array_equal(s1, s2)
    assert (h1 == h2).mean() > 0.95  # rounding boundaries can flip a few windows


def test_low_information_windows_skipped():
    x = np.zeros(300)
    x[::50] = 1.0  # sparse spikes: < MIN_DISTINCT distinct values
    assert leakage.window_hashes(x, 64, 1)[0].size == 0
    assert leakage.window_hashes(np.full(200, np.nan), 64, 1)[0].size == 0


@settings(max_examples=40, deadline=None)
@given(offset_a=st.integers(0, 400), offset_b=st.integers(0, 400), extra=st.integers(0, 30))
def test_shared_segment_always_detected(offset_a: int, offset_b: int, extra: int):
    """The guarantee: a shared segment of MIN_SHARED_GUARANTEED points is always found,
    wherever it sits in either series and with any rescaling."""
    rng = np.random.default_rng(offset_a * 1000 + offset_b)
    seg = rng.normal(size=leakage.MIN_SHARED_GUARANTEED + extra)
    a = np.concatenate([rng.normal(size=offset_a), seg, rng.normal(size=50)])
    b = np.concatenate([rng.normal(size=offset_b), 5.0 * seg + 2.0, rng.normal(size=50)])
    ha, _ = leakage.window_hashes(a, leakage.W, leakage.G)
    idx = leakage.HashIndex(
        np.sort(ha), np.zeros(ha.size, np.int16), np.zeros(ha.size, bool), ["a"]
    )
    hb, _ = leakage.window_hashes(b, leakage.W, leakage.S)
    assert (idx.lookup(hb) >= 0).any()


def test_unrelated_series_do_not_match():
    rng = np.random.default_rng(1)
    ha, _ = leakage.window_hashes(rng.normal(size=20_000), leakage.W, leakage.G)
    idx = leakage.HashIndex(
        np.sort(ha), np.zeros(ha.size, np.int16), np.zeros(ha.size, bool), ["a"]
    )
    hb, _ = leakage.window_hashes(rng.normal(size=20_000), leakage.W, leakage.S)
    assert (idx.lookup(hb) >= 0).sum() == 0


def test_daily_means_complete_days_only():
    x = np.arange(24 * 3, dtype=np.float32)
    x[30] = np.nan  # day 2 incomplete
    d = leakage.daily_means(x, pd.Timestamp("2020-01-01 00:00"), "h")
    assert d is not None
    np.testing.assert_allclose(d[[0, 2]], [11.5, 59.5])
    assert np.isnan(d[1])
    # Starting mid-day: the first partial day is incomplete.
    d2 = leakage.daily_means(np.ones(48, np.float32), pd.Timestamp("2020-01-01 12:00"), "h")
    assert d2 is not None
    assert np.isnan(d2[0])
    assert d2[1] == 1.0
    assert leakage.daily_means(np.ones(10, np.float32), pd.Timestamp("2020-01-01"), "D") is None


# ---------------------------------------------------------------- committed results


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


@pytest.mark.skipif(
    not (repo_root() / "results/data/leakage.json").exists(), reason="leakage not run"
)
def test_committed_leakage_report_is_clean_and_matches_corpus():
    rep = _load(results_dir() / "data" / "leakage.json")
    man = _load(results_dir() / "data" / "corpus_manifest.json")
    assert rep["totals"]["native_hits"] == 0
    assert rep["totals"]["daily_hits"] == 0
    assert rep["leaking_series"] == []
    # The report was produced on exactly the corpus described by the manifest.
    assert {k: v["sha256"] for k, v in rep["files"].items()} == {
        k: v["sha256"] for k, v in man["files"].items() if "kernelsynth" not in k
    }
    assert rep["params"]["min_shared_points_guaranteed"] == leakage.MIN_SHARED_GUARANTEED


@pytest.mark.data
def test_local_corpus_matches_manifest():
    man = _load(results_dir() / "data" / "corpus_manifest.json")
    for rel, e in man["files"].items():
        assert sha256_file(processed_dir() / rel) == e["sha256"], rel


def test_linear_windows_skipped_but_noisy_kept():
    ramp = np.arange(300, dtype=np.float64)
    assert leakage.window_hashes(ramp, 64, 1)[0].size == 0
    noisy = ramp + np.random.default_rng(0).normal(scale=5.0, size=300)
    assert leakage.window_hashes(noisy, 64, 1)[0].size == 300 - 64 + 1
    # Piecewise: a gap filled by linear interpolation inside real data.
    x = np.random.default_rng(1).normal(size=400)
    x[100:180] = np.linspace(x[100], x[179], 80)
    _h, st = leakage.window_hashes(x, 64, 1)
    assert not any(s >= 100 and s + 64 <= 180 for s in st.tolist())
