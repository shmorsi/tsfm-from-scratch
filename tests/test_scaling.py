"""Scaling fits recover a known power law and give sensible intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsfm import scaling as S


def test_fit_recovers_exponent():
    c = np.geomspace(1e14, 1e18, 5)
    a, b = S.fit_power_law(c, 50 * c**-0.07)
    assert b == pytest.approx(0.07)
    assert a == pytest.approx(50)


def test_bootstrap_interval_covers_truth(tmp_path: Path):
    rng = np.random.default_rng(0)
    runs = []
    for i, c in enumerate(np.geomspace(1e14, 1e18, 5)):
        loss = 50 * c**-0.07
        runs.append(
            {
                "name": f"r{i}",
                "flops": float(c),
                "params_body": 1,
                "val_batches": (loss * (1 + 0.02 * rng.standard_normal(32))).tolist(),
            }
        )
    out = S.bootstrap_fit(runs, n=500)
    lo, hi = out["b_ci95"]
    assert lo < 0.07 < hi
    assert hi - lo < 0.02
    assert len(out["curve"]["fit"]) == 50


def test_load_runs_skips_unfinished(tmp_path: Path):
    for name, done in (("patch_s0", True), ("patch_s1", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "done": done,
                    "val_loss_batches": [1.0],
                    "config": {"model": {"tokenizer": "patch"}, "seed": 0},
                    "params_body": 1,
                    "train_tokens": 1,
                    "train_flops_6ND": 1,
                }
            )
        )
    assert [r["name"] for r in S.load_runs(tmp_path)] == ["patch_s0"]
