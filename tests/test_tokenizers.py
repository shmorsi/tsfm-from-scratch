"""M2: tokenizer round-trips and properties."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tsfm.tokenizers import (
    MISSING_ID,
    N_SPECIAL,
    PAD_ID,
    QUANTILES,
    BinTokenizer,
    PatchTokenizer,
    QuantileHead,
    mean_scale,
    pinball_loss,
)

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32)
series = arrays(np.float32, st.integers(1, 200), elements=finite)
tok = BinTokenizer()


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, np.float32))[None]


def test_vocab_is_4096():
    assert tok.vocab_size == 4096


@settings(max_examples=200, deadline=None)
@given(series)
def test_bin_round_trip_error_bounded(x: np.ndarray):
    xt = t(x)
    s = mean_scale(xt)
    y = tok.decode(tok.encode(xt, s), s)
    z = (xt.double() / s.double()).abs()
    inside = z <= tok.limit
    err = (y.double() - xt.double()).abs() / s.double()
    # Within range: at most half a bin (+ float slack); outside: clipped to the end bins.
    assert (err[inside] <= tok.bin_width / 2 + 1e-6).all()
    assert torch.allclose(
        y[~inside].abs().double(), (tok.limit * s.double()).expand_as(xt)[~inside], rtol=1e-6
    )


@settings(max_examples=100, deadline=None)
@given(series, st.floats(1e-3, 1e3))
def test_bin_scale_invariance(x: np.ndarray, c: float):
    xt = t(x)
    a = tok.encode(xt, mean_scale(xt))
    b = tok.encode(xt * c, mean_scale(xt * c))
    # Float rounding can move a value across a bin edge, but never by more than one bin.
    assert (a - b).abs().max() <= 1


@settings(max_examples=100, deadline=None)
@given(series)
def test_bin_monotone(x: np.ndarray):
    xt = t(np.sort(x))
    ids = tok.encode(xt, mean_scale(xt))
    assert (ids[0, 1:] >= ids[0, :-1]).all()


def test_missing_and_pad_tokens():
    x = torch.tensor([[1.0, math.nan, 2.0, 3.0]])
    pad = torch.tensor([[True, False, False, False]])
    ids = tok.encode(x, mean_scale(x), pad=pad)
    assert ids[0, 0] == PAD_ID
    assert ids[0, 1] == MISSING_ID
    assert (ids[0, 2:] >= N_SPECIAL).all()
    dec = tok.decode(ids, mean_scale(x))
    assert torch.isnan(dec[0, :2]).all()


def test_scale_of_constant_and_zero_series():
    assert mean_scale(torch.full((1, 10), -4.0)).item() == 4.0
    assert mean_scale(torch.zeros(1, 10)).item() == 1.0
    assert mean_scale(torch.full((1, 5), math.nan)).item() == 1.0
    x = torch.tensor([[2.0, math.nan, -4.0]])
    assert mean_scale(x).item() == 3.0


@settings(max_examples=100, deadline=None)
@given(series, st.sampled_from([1, 4, 16, 32]))
def test_patch_round_trip(x: np.ndarray, p: int):
    xt = t(x)
    s = mean_scale(xt)
    pt = PatchTokenizer(p)
    patches, obs, patch_pad = pt.encode(xt, s)
    n = -(-x.size // p)
    assert patches.shape == (1, n, p)
    back = pt.decode(patches, s)[:, -x.size :]
    torch.testing.assert_close(back, xt, rtol=1e-5, atol=1e-5 * float(s))
    assert obs[0].reshape(-1)[-x.size :].all()
    assert not obs[0].reshape(-1)[: n * p - x.size].any()
    assert not patch_pad.any()  # left padding never fills a whole patch


def test_patch_missing_values_masked():
    x = torch.tensor([[1.0, math.nan, 3.0, 4.0]])
    patches, obs, _ = PatchTokenizer(2).encode(x, mean_scale(x))
    assert obs.tolist() == [[[True, False], [True, True]]]
    assert patches[0, 0, 1] == 0.0


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 10_000))
def test_quantile_head_never_crosses(seed: int):
    torch.manual_seed(seed)
    head = QuantileHead(8, horizon=5)
    with torch.no_grad():
        head.proj.weight.mul_(torch.rand(()) * 100)
    q = head(torch.randn(3, 7, 8) * 10)
    assert q.shape == (3, 7, 5, len(QUANTILES))
    assert (q.diff(dim=-1) >= 0).all()


def test_pinball_is_minimised_at_true_quantile():
    torch.manual_seed(0)
    y = torch.randn(20_000)
    levels = torch.tensor([0.1, 0.5, 0.9])
    best = torch.quantile(y, levels)

    def loss(q: torch.Tensor) -> float:
        return float(
            pinball_loss(
                q.expand(y.numel(), 3)[None],
                y[None],
                levels,
                torch.ones(1, y.numel(), dtype=torch.bool),
            )
        )

    base = loss(best)
    for shift in (-0.1, 0.1):
        assert loss(best + shift) > base


def test_pinball_ignores_unobserved_nan_targets():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[1.0, math.nan]])
    obs = torch.tensor([[True, False]])
    out = pinball_loss(pred, target, torch.tensor([0.1, 0.5, 0.9]), obs)
    assert torch.isfinite(out)
    assert out.item() == pytest.approx((0.1 + 0.5 + 0.9) / 3)
