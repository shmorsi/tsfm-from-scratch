"""Packed corpus and deterministic window sampling for training.

``pack`` concatenates every series of a split into one float32 file
(``values.f32``, memory-mapped at training time) plus an index (``index.npz``). Random
windows are then cheap to read from local disk, which matters on Colab.

Every batch is a pure function of ``(seed, step, micro_step)``, so resuming a run at
step ``k`` reproduces exactly the batches an uninterrupted run would have seen; no
sampler state needs to be checkpointed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tsfm.data.convert import _list_values
from tsfm.provenance import sha256_file


def pack(parquet_files: list[Path], out_dir: Path) -> dict[str, Any]:
    """Pack Parquet files (one dataset each) into ``out_dir``. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = [0]
    ds_ids: list[int] = []
    names: list[str] = []
    freqs: list[str] = []
    tmp = out_dir / "values.f32.part"
    with open(tmp, "wb") as f:
        for i, path in enumerate(sorted(parquet_files)):
            t = pq.read_table(path, columns=["target", "dataset", "freq"])
            vals, offs = _list_values(t.column("target"))
            names.append(t.column("dataset")[0].as_py() if t.num_rows else path.stem)
            freqs.append(t.column("freq")[0].as_py() if t.num_rows else "")
            f.write(vals.astype(np.float32).tobytes())
            base = offsets[-1]
            offsets.extend((base + offs[1:]).tolist())
            ds_ids.extend([i] * (len(offs) - 1))
    tmp.replace(out_dir / "values.f32")
    np.savez(
        out_dir / "index.npz",
        offsets=np.asarray(offsets, np.int64),
        dataset=np.asarray(ds_ids, np.int32),
    )
    meta = {
        "datasets": names,
        "freqs": freqs,
        "n_series": len(ds_ids),
        "n_points": int(offsets[-1]),
        "sources": {p.name: sha256_file(p) for p in sorted(parquet_files)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    meta["values_sha256"] = sha256_file(out_dir / "values.f32")
    meta["index_sha256"] = sha256_file(out_dir / "index.npz")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


@dataclass
class Packed:
    """One memory-mapped corpus: values, offsets and dataset names."""

    values: np.ndarray  # memmap float32
    offsets: np.ndarray
    dataset: np.ndarray
    names: list[str]
    meta: dict[str, Any]

    @classmethod
    def open(cls, d: Path, in_memory: bool | None = None) -> Packed:
        """Memory-map the values, or (``in_memory`` / ``TSFM_IN_MEMORY=1``) read them into
        RAM: random window reads from a network disk dominate step time on Colab."""
        import os

        meta = json.loads((d / "meta.json").read_text())
        idx = np.load(d / "index.npz")
        if in_memory is None:
            in_memory = os.environ.get("TSFM_IN_MEMORY") == "1"
        if in_memory:
            vals = np.fromfile(d / "values.f32", dtype=np.float32)
        else:
            vals = np.memmap(d / "values.f32", dtype=np.float32, mode="r")
        if vals.size != meta["n_points"]:
            raise ValueError(
                f"{d}: values.f32 has {vals.size} points, meta says {meta['n_points']}"
            )
        return cls(vals, idx["offsets"], idx["dataset"], meta["datasets"], meta)

    @property
    def lengths(self) -> np.ndarray:
        """Length of each series."""
        return np.diff(self.offsets)

    def series(self, i: int) -> np.ndarray:
        """Values of series ``i``."""
        return np.asarray(self.values[self.offsets[i] : self.offsets[i + 1]])


@dataclass
class MixtureSampler:
    """Samples fixed-length windows from a mixture of packed corpora.

    Dataset weights are proportional to ``points ** temperature`` (temperature < 1
    up-weights small datasets); ``synthetic`` corpora get a fixed share of the samples.
    Within a dataset a window is drawn uniformly over all valid window end positions,
    so long series are sampled proportionally to their length."""

    corpora: list[Packed]
    synthetic: list[bool]
    synthetic_share: float = 0.1
    temperature: float = 0.5
    min_points: int = 16

    def __post_init__(self) -> None:
        self._groups: list[
            tuple[int, int, np.ndarray, np.ndarray]
        ] = []  # (corpus, ds, series ids, cum lengths)
        weights: list[float] = []
        is_syn: list[bool] = []
        for ci, (c, syn) in enumerate(zip(self.corpora, self.synthetic, strict=True)):
            lens = c.lengths
            for di in range(len(c.names)):
                ids = np.flatnonzero((c.dataset == di) & (lens >= self.min_points))
                if ids.size == 0:
                    continue
                cum = np.cumsum(lens[ids])
                self._groups.append((ci, di, ids, cum))
                weights.append(float(cum[-1]) ** self.temperature)
                is_syn.append(syn)
        w = np.asarray(weights)
        syn_mask = np.asarray(is_syn)
        if syn_mask.any() and (~syn_mask).any():
            w[syn_mask] *= self.synthetic_share / w[syn_mask].sum()
            w[~syn_mask] *= (1 - self.synthetic_share) / w[~syn_mask].sum()
        self.probs = w / w.sum()

    def group_names(self) -> list[str]:
        """Dataset name of each sampling group."""
        return [self.corpora[ci].names[di] for ci, di, _, _ in self._groups]

    def sample(self, rng: np.random.Generator, batch: int, length: int) -> np.ndarray:
        """(batch, length) float32 windows; NaN marks missing values and left padding."""
        out = np.full((batch, length), np.nan, np.float32)
        gs = rng.choice(len(self._groups), size=batch, p=self.probs)
        for b, g in enumerate(gs):
            ci, _, ids, cum = self._groups[g]
            j = int(np.searchsorted(cum, rng.integers(0, cum[-1]), side="right"))
            c = self.corpora[ci]
            sid = int(ids[j])
            off = int(c.offsets[sid])
            n = int(c.offsets[sid + 1]) - off
            # Slice the window straight from the memory map (never copy a whole series:
            # some are millions of points long).
            if n <= length:
                # Short series: a random-length prefix, so the end is not always the last point.
                end = int(rng.integers(min(self.min_points, n), n + 1))
                w = c.values[off : off + end]
            else:
                end = int(rng.integers(length, n + 1))
                w = c.values[off + end - length : off + end]
            out[b, length - w.size :] = w
        return out


def step_rng(seed: int, step: int, micro: int, stream: int = 0) -> np.random.Generator:
    """Generator unique to one seed, step, micro-batch and stream."""
    return np.random.default_rng([seed, step, micro, stream])
