"""M3 training: sampler determinism, LR schedule, objective masking, exact resume."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tsfm.data.convert import SCHEMA
from tsfm.model import ModelConfig, TSModel
from tsfm.train.data import MixtureSampler, Packed, pack, step_rng
from tsfm.train.loop import RunConfig, TrainConfig, Trainer, lr_at
from tsfm.train.objective import draw_context, leading_pad, patch_targets


def write_parquet(path: Path, name: str, n: int, lo: int, hi: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    targets = [
        np.cumsum(rng.normal(size=int(rng.integers(lo, hi)))).astype(np.float32) + 10
        for _ in range(n)
    ]
    t = pa.table(
        {
            "item_id": [f"{name}/{i}" for i in range(n)],
            "dataset": [name] * n,
            "freq": ["h"] * n,
            "start": [np.datetime64("2020-01-01T00:00:00", "ms")] * n,
            "target": targets,
        },
        schema=SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path)
    return path


@pytest.fixture(scope="module")
def packed_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("packed")
    pq_dir = root / "pq"
    a = write_parquet(pq_dir / "a.parquet", "a", 30, 50, 400, 1)
    b = write_parquet(pq_dir / "b.parquet", "b", 5, 1000, 3000, 2)
    s = write_parquet(pq_dir / "s.parquet", "kernelsynth", 20, 256, 257, 3)
    v = write_parquet(pq_dir / "v.parquet", "v", 10, 100, 300, 4)
    pack([a, b], root / "pretrain")
    pack([s], root / "synthetic")
    pack([v], root / "val")
    return root


def test_pack_round_trip(packed_root: Path):
    p = Packed.open(packed_root / "pretrain")
    t = pq.read_table(packed_root / "pq" / "a.parquet")
    np.testing.assert_array_equal(
        p.series(3), np.asarray(t.column("target")[3].as_py(), np.float32)
    )
    assert p.names == ["a", "b"]
    assert p.meta["n_series"] == 35


def test_sampler_deterministic_and_mixture(packed_root: Path):
    s = MixtureSampler(
        [Packed.open(packed_root / "pretrain"), Packed.open(packed_root / "synthetic")],
        [False, True],
        synthetic_share=0.25,
    )
    np.testing.assert_allclose(s.probs[-1], 0.25)
    a = s.sample(step_rng(0, 5, 0), 16, 128)
    b = s.sample(step_rng(0, 5, 0), 16, 128)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, s.sample(step_rng(0, 6, 0), 16, 128))
    assert a.shape == (16, 128)
    # NaNs only appear as left padding (these series have no missing values).
    for row in a:
        fin = np.isfinite(row)
        assert fin[np.argmax(fin) :].all()


def test_windows_are_real_subsequences(packed_root: Path):
    p = Packed.open(packed_root / "pretrain")
    s = MixtureSampler([p], [False])
    w = s.sample(np.random.default_rng(1), 8, 64)
    allv = [p.series(i) for i in range(p.meta["n_series"])]
    for row in w:
        r = row[np.isfinite(row)]
        assert any(
            np.array_equal(x[k : k + r.size], r) for x in allv for k in np.flatnonzero(x == r[0])
        )


def test_lr_schedule():
    tc = TrainConfig(steps=100, lr=1.0, warmup=10, min_lr_ratio=0.1)
    assert lr_at(0, tc) == pytest.approx(0.1)
    assert lr_at(9, tc) == pytest.approx(1.0)
    assert lr_at(10, tc) == pytest.approx(1.0)
    assert lr_at(55, tc) == pytest.approx(0.1 + 0.45 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(100, tc) == pytest.approx(0.1)


def test_draw_context_respects_padding():
    x = np.full((3, 50), np.nan, np.float32)
    x[0, 10:] = 1
    x[1, :] = 1
    x[2, 45:] = 1
    c = draw_context(np.random.default_rng(0), x, min_ctx=4, max_ctx=20)
    assert 14 <= c[0] <= 30
    assert 4 <= c[1] <= 20
    assert c[2] <= 49


def test_patch_targets_alignment():
    x = torch.arange(12, dtype=torch.float32)[None]
    t = patch_targets(x, p=4, horizon=6)
    assert t.shape == (1, 3, 6)
    assert t[0, 0].tolist() == [4, 5, 6, 7, 8, 9]
    assert torch.isnan(t[0, 1, 4:]).all()
    assert torch.isnan(t[0, 2]).all()


def test_leading_pad():
    x = torch.tensor([[math.nan, math.nan, 1.0, math.nan, 2.0]])
    assert leading_pad(x).tolist() == [[True, True, False, False, False]]


def test_loss_ignores_future_scale(packed_root: Path):
    """Changing values after the context must not change the scale used, so the
    first prediction (of the first post-context patch) is unchanged."""
    from tsfm.train.objective import patch_loss

    torch.manual_seed(0)
    m = TSModel(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=8, horizon=8, max_seq_len=16)
    ).eval()
    x = torch.randn(1, 64).cumsum(-1) + 5
    ctx = torch.tensor([32])
    y = x.clone()
    y[:, 40:] *= 100  # beyond the first target patch [32, 40)
    with torch.no_grad():
        la, _ = patch_loss(m, x[:, :40], ctx)
        lb, _ = patch_loss(m, y[:, :40], ctx)
    assert la.item() == lb.item()


def rc(tok: str, root: Path, steps: int = 6) -> RunConfig:
    return RunConfig.from_dict(
        {
            "name": f"t-{tok}",
            "seed": 3,
            "model": {
                "tokenizer": tok,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "patch_len": 8,
                "horizon": 16,
                "max_seq_len": 64,
                "n_bins": 254,
                "dropout": 0.1,
            },
            "data": {
                "corpora": [{"path": "pretrain"}, {"path": "synthetic", "synthetic": True}],
                "val": [{"path": "val"}],
                "window": 64,
                "min_context": 8,
                "val_batches": 2,
                "val_batch_size": 8,
            },
            "train": {
                "steps": steps,
                "batch_size": 8,
                "micro_batch": 4,
                "lr": 3e-3,
                "warmup": 2,
                "device": "cpu",
                "eval_every": 3,
                "log_every": 1,
            },
        }
    )


@pytest.mark.parametrize("tok", ["patch", "bins"])
def test_resume_is_bit_exact(
    tok: str, packed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TSFM_PACKED", str(packed_root))
    torch.set_num_threads(1)
    full = Trainer(rc(tok, packed_root), tmp_path / "full")
    s_full = full.run()
    assert s_full["done"]
    part = Trainer(rc(tok, packed_root), tmp_path / "part")
    s1 = part.run(max_steps_this_session=2)
    assert not s1["done"]
    assert s1["step"] == 2
    del part  # "interrupted": a fresh process would rebuild everything from the checkpoint
    again = Trainer(rc(tok, packed_root), tmp_path / "part")
    s2 = again.run(max_steps_this_session=2)
    assert s2["resumed"]
    assert s2["step"] == 4
    final = Trainer(rc(tok, packed_root), tmp_path / "part").run()
    assert final["done"]
    a = torch.load(tmp_path / "full" / "model.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "part" / "model.pt", weights_only=False)["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k
    la = (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()
    lb = (tmp_path / "part" / "metrics.jsonl").read_text().splitlines()
    strip = lambda ls: [{k: v for k, v in json.loads(x).items() if k != "step_seconds"} for x in ls]  # noqa: E731
    assert strip(la) == strip(lb)


def test_resume_refuses_different_config(
    packed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TSFM_PACKED", str(packed_root))
    Trainer(rc("patch", packed_root, steps=2), tmp_path / "r").run(max_steps_this_session=1)
    with pytest.raises(ValueError, match="different config"):
        Trainer(rc("patch", packed_root, steps=3), tmp_path / "r").run()


def test_training_reduces_val_loss(
    packed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TSFM_PACKED", str(packed_root))
    tr = Trainer(rc("patch", packed_root, steps=60), tmp_path / "v")
    before = tr.val_loss()
    tr.run()
    after = tr.val_loss()
    assert before is not None
    assert after is not None
    assert after < 0.8 * before


def test_draw_context_short_series_gets_half_as_context():
    x = np.full((2, 64), np.nan, np.float32)
    x[0, 44:] = 1.0  # 20 real points
    x[1, 62:] = 1.0  # 2 real points
    for seed in range(20):
        c = draw_context(np.random.default_rng(seed), x, min_ctx=32, max_ctx=63)
        assert 44 + 10 <= c[0] <= 63
        assert c[1] == 63


def test_align_to_patches_keeps_values_and_ends_context_on_boundary():
    from tsfm.train.objective import align_to_patches

    x = torch.arange(32, dtype=torch.float32).repeat(2, 1)
    ctx = torch.tensor([13, 16])
    y, c = align_to_patches(x, ctx, 8)
    assert c.tolist() == [8, 16]
    assert y[0, :8].tolist() == list(range(5, 13))  # context = the 8 points before 13
    assert y[0, 8] == 13  # first target follows the context
    assert torch.isnan(y[0, -5:]).all()
    assert torch.equal(y[1], x[1])


def test_short_series_loss_is_on_the_scale_of_the_data():
    """Regression test: a 20-point series in a long window used to lose its whole context
    to patch rounding, fall back to scale 1 and produce a loss in raw units."""
    from tsfm.train.objective import patch_loss

    torch.manual_seed(0)
    m = TSModel(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=32, horizon=64, max_seq_len=64)
    ).eval()
    x = torch.full((4, 2048), float("nan"))
    x[:, -20:] = 5000.0 + torch.randn(4, 20) * 50
    ctx = torch.from_numpy(draw_context(np.random.default_rng(0), x.numpy(), 32, 2047))
    with torch.no_grad():
        loss, n = patch_loss(m, x, ctx)
    assert n > 0
    assert loss.item() < 10  # scaled units (~1), not raw units (~5000)


def test_fp16_forward_stays_finite_with_huge_future_values():
    """Regression test for the T4 divergence: a near-zero context followed by values ~1e6
    must not overflow half precision (inputs and targets are clipped in scaled units)."""
    from tsfm.train.objective import patch_loss

    torch.manual_seed(0)
    m = TSModel(
        ModelConfig(d_model=64, n_layers=2, n_heads=4, patch_len=32, horizon=64, max_seq_len=64)
    )
    x = torch.full((4, 2048), 1e-3)
    x[:, 1024:] = 1e6 * torch.rand(4, 1024)
    ctx = torch.full((4,), 1024)
    with torch.autocast("cpu", dtype=torch.float16):
        loss, n = patch_loss(m, x, ctx)
    assert torch.isfinite(loss)
    assert n > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_fresh_start_resets_metrics_log(
    packed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TSFM_PACKED", str(packed_root))
    d = tmp_path / "r"
    d.mkdir()
    (d / "metrics.jsonl").write_text('{"step": 99, "loss": 1}\n')  # stale log, no checkpoint
    Trainer(rc("patch", packed_root, steps=2), d).run()
    steps = [json.loads(x)["step"] for x in (d / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2]
