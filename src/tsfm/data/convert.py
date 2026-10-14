"""Convert verified raw Arrow shards into validated univariate Parquet files.

Output layout (under ``$TSFM_HOME/processed``)::

    pretrain/<dataset>.parquet    training corpus
    val/<dataset>.parquet         held-out validation datasets (never trained on)
    synthetic/kernelsynth.parquet synthetic GP series (see synthetic.py)

Every file has the schema of :data:`SCHEMA`. Multivariate series are split into one
univariate series per variable (``item_id`` gets a ``_dim<k>`` suffix). The manifest
``results/data/corpus_manifest.json`` records per-file SHA-256, counts and drop reasons.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tsfm.data.sources import load_data_config, local_dir, read_lock
from tsfm.paths import processed_dir, results_dir
from tsfm.provenance import sha256_file, write_result

SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("dataset", pa.string()),
        ("freq", pa.string()),
        # Parquet has no second resolution; ms round-trips exactly.
        ("start", pa.timestamp("ms")),
        ("target", pa.list_(pa.field("element", pa.float32()))),
    ]
)


@dataclass
class Rules:
    min_observed: int = 12
    max_abs_value: float = 1e12
    drop_constant: bool = True
    max_nan_fraction: float = 0.9

    def __post_init__(self) -> None:
        # Guard against YAML quirks (e.g. "1.0e12" parses as a string).
        self.min_observed = int(self.min_observed)
        self.max_abs_value = float(self.max_abs_value)
        self.max_nan_fraction = float(self.max_nan_fraction)
        self.drop_constant = bool(self.drop_constant)

    def check(self, x: np.ndarray) -> str | None:
        """Return a drop reason, or None if the series is kept."""
        if x.size == 0:
            return "empty"
        obs = np.isfinite(x)
        if np.isinf(x).any():
            return "inf"
        n_obs = int(obs.sum())
        if n_obs < self.min_observed:
            return "too_short"
        if 1.0 - n_obs / x.size > self.max_nan_fraction:
            return "mostly_missing"
        xo = x[obs]
        if np.abs(xo).max() > self.max_abs_value:
            return "too_large"
        if self.drop_constant and xo.min() == xo.max():
            return "constant"
        return None


@dataclass
class DatasetStats:
    dataset: str
    split: str
    freqs: Counter[str] = field(default_factory=Counter)
    series_in: int = 0
    series_valid: int = 0
    series_kept: int = 0
    points_valid: int = 0
    points_kept: int = 0
    truncated: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "freqs": dict(self.freqs),
            "series_in": self.series_in,
            "series_valid": self.series_valid,
            "series_kept": self.series_kept,
            "points_valid": self.points_valid,
            "points_kept": self.points_kept,
            "series_truncated": self.truncated,
            "dropped": dict(self.dropped),
        }


def _list_values(arr: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a list<float> array into (values, offsets) with offsets starting at 0."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    lengths = pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False)
    values = pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return values, offsets


@dataclass(frozen=True)
class RawSeries:
    item_id: str
    freq: str
    start: Any
    values: np.ndarray


def iter_raw_series(paths: list[Path], dataset: str) -> Iterator[RawSeries]:
    """Yield univariate series from HF ``save_to_disk`` Arrow shards."""
    for path in paths:
        with pa.memory_map(str(path)) as src:
            reader = pa.ipc.open_stream(src)
            for batch in reader:
                ids = batch.column("item_id").to_pylist()
                freqs = batch.column("freq").to_pylist()
                starts = batch.column("start").cast(pa.timestamp("s")).to_pylist()
                tgt = batch.column("target")
                nested = pa.types.is_list(tgt.type.value_type) or pa.types.is_fixed_size_list(
                    tgt.type.value_type
                )
                if nested:
                    inner = pc.list_flatten(tgt)
                    n_dims = pc.list_value_length(tgt).to_numpy(zero_copy_only=False)
                    vals, offs = _list_values(inner)
                    row_dim_start = np.concatenate([[0], np.cumsum(n_dims)])
                    for i in range(len(ids)):
                        for d in range(int(n_dims[i])):
                            k = int(row_dim_start[i]) + d
                            yield RawSeries(
                                f"{dataset}/{ids[i]}_dim{d}",
                                freqs[i],
                                starts[i],
                                vals[offs[k] : offs[k + 1]],
                            )
                else:
                    vals, offs = _list_values(tgt)
                    for i in range(len(ids)):
                        yield RawSeries(
                            f"{dataset}/{ids[i]}", freqs[i], starts[i], vals[offs[i] : offs[i + 1]]
                        )


def _dataset_seed(base: int, dataset: str) -> int:
    return base ^ int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:4], "little")


def convert_dataset(
    dataset: str,
    paths: list[Path],
    out_path: Path,
    *,
    split: str,
    rules: Rules,
    max_points: int | None,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> DatasetStats:
    """Two passes: (1) validate every series and pick a seeded subset under the cap,
    (2) write the chosen series. Peak memory stays at one Arrow batch.

    ``exclude`` (item ids flagged by the leakage check) is applied *after* the subset is
    chosen, so excluding a series never pulls a new, unchecked series into the corpus."""
    stats = DatasetStats(dataset, split)
    candidates: list[tuple[int, int]] = []  # (ordinal, n_points)
    for ordinal, s in enumerate(iter_raw_series(paths, dataset)):
        stats.series_in += 1
        reason = rules.check(s.values)
        if reason:
            stats.dropped[reason] += 1
            continue
        stats.series_valid += 1
        stats.points_valid += int(s.values.size)
        candidates.append((ordinal, int(s.values.size)))

    rng = np.random.default_rng(_dataset_seed(seed, dataset))
    order = rng.permutation(len(candidates))
    keep: dict[int, int] = {}  # ordinal -> number of points to keep (from the end)
    budget = max_points if max_points is not None else np.iinfo(np.int64).max
    for j in order:
        if budget <= 0:
            break
        ordinal, n = candidates[j]
        take = min(n, int(budget))
        # Truncation keeps the most recent `take` points; it must still pass the rules.
        if take < n and take < rules.min_observed:
            break
        keep[ordinal] = take
        budget -= take

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.part")
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    buf: dict[str, list[Any]] = {k: [] for k in SCHEMA.names}
    buf_points = 0

    def flush() -> None:
        nonlocal buf_points
        if buf["item_id"]:
            writer.write_table(pa.table(buf, schema=SCHEMA))
            for v in buf.values():
                v.clear()
            buf_points = 0

    for ordinal, s in enumerate(iter_raw_series(paths, dataset)):
        if ordinal not in keep:
            continue
        if s.item_id in exclude:
            stats.dropped["leakage"] += 1
            continue
        take = keep[ordinal]
        x = s.values[-take:]
        start = s.start
        if take < s.values.size:
            stats.truncated += 1
            start = shift_start(s.start, s.freq, s.values.size - take)
            if rules.check(x):
                continue
        stats.series_kept += 1
        stats.points_kept += int(x.size)
        stats.freqs[s.freq] += 1
        buf["item_id"].append(s.item_id)
        buf["dataset"].append(dataset)
        buf["freq"].append(s.freq)
        buf["start"].append(start)
        buf["target"].append(x)
        buf_points += x.size
        if buf_points > 4_000_000:
            flush()
    flush()
    writer.close()
    tmp.replace(out_path)
    return stats


def shift_start(start: Any, freq: str, steps: int) -> Any:
    """Timestamp ``steps`` periods after ``start`` (used when truncating from the front)."""
    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # date_range reproduces the timestamps of the original series exactly, including
        # anchored frequencies (MS, W-SUN, A-DEC) that Period does not support.
        return pd.date_range(start, periods=steps + 1, freq=freq)[-1]


def dataset_shards(lock: dict[str, Any], root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for e in lock["files"]:
        if re.search(r"/data-\d+-of-\d+\.arrow$", e["path"]):
            out.setdefault(e["path"].split("/")[0], []).append(root / e["path"])
    return {k: sorted(v) for k, v in out.items()}


def build_corpus(only: set[str] | None = None) -> dict[str, Any]:
    cfg = load_data_config()
    pc_cfg = cfg["pretrain"]
    rules = Rules(**cfg["validation_rules"])
    shards = dataset_shards(read_lock("pretrain"), local_dir("pretrain"))
    val = set(cfg["validation_datasets"])
    missing = val - shards.keys()
    if missing:
        raise ValueError(f"validation datasets not in corpus: {sorted(missing)}")
    from tsfm.data.leakage import load_exclusions

    exclude = frozenset(load_exclusions())
    manifest_path = results_dir() / "data" / "corpus_manifest.json"
    entries: dict[str, Any] = {}
    if manifest_path.exists() and only:
        import json

        entries = json.loads(manifest_path.read_text()).get("files", {})
    for ds, paths in shards.items():
        if only and ds not in only:
            continue
        split = "val" if ds in val else "pretrain"
        out = processed_dir() / split / f"{ds}.parquet"
        st = convert_dataset(
            ds,
            paths,
            out,
            split=split,
            rules=rules,
            max_points=pc_cfg["max_points_per_dataset"],
            seed=pc_cfg["seed"],
            exclude=exclude,
        )
        entries[f"{split}/{ds}.parquet"] = {**st.as_dict(), "sha256": sha256_file(out)}
        print(f"{split:8s} {ds:36s} kept {st.series_kept:>8d} series {st.points_kept:>12,d} pts")
    write_manifest(entries)
    return entries


def write_manifest(entries: dict[str, Any]) -> None:
    import json

    p = results_dir() / "data" / "corpus_manifest.json"
    if p.exists():
        old = json.loads(p.read_text()).get("files", {})
        old.update(entries)
        entries = old
    totals: dict[str, dict[str, int]] = {}
    for e in entries.values():
        t = totals.setdefault(e["split"], {"datasets": 0, "series": 0, "points": 0})
        t["datasets"] += 1
        t["series"] += e["series_kept"]
        t["points"] += e["points_kept"]
    write_result(
        p,
        {"files": dict(sorted(entries.items())), "totals": totals, "schema": str(SCHEMA)},
        split="none",
        label="M1-corpus",
    )


def read_series(path: Path) -> pa.Table:
    return pq.read_table(path, schema=SCHEMA)


def validate_parquet(path: Path, rules: Rules) -> list[str]:
    """Re-check a written file against the schema and rules (used by tests)."""
    t = pq.read_table(path)
    errs = []
    if t.schema.remove_metadata() != SCHEMA:
        errs.append(f"schema mismatch: {t.schema}")
    vals, offs = _list_values(t.column("target"))
    ids = t.column("item_id").to_pylist()
    if len(set(ids)) != len(ids):
        errs.append("duplicate item_id")
    for i in range(t.num_rows):
        r = rules.check(vals[offs[i] : offs[i + 1]])
        if r:
            errs.append(f"{ids[i]}: {r}")
    return errs
