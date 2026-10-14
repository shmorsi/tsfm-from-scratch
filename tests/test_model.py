"""M3 model: causality, KV cache, padding, parameter count, overfitting."""

from __future__ import annotations

import itertools

import pytest
import torch

from tsfm.model import KVCache, ModelConfig, TSModel
from tsfm.tokenizers import pinball_loss

torch.set_num_threads(1)


def tiny(tokenizer: str = "patch", **kw: object) -> TSModel:
    cfg = ModelConfig(
        tokenizer=tokenizer,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        patch_len=4,
        horizon=8,
        n_bins=62,
        **kw,
    )  # type: ignore[arg-type]
    torch.manual_seed(0)
    return TSModel(cfg).eval()


def patch_inputs(
    b: int = 2, t: int = 10, p: int = 4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, p, generator=g), torch.ones(b, t, p, dtype=torch.bool)


def run(m: TSModel, x: torch.Tensor, obs: torch.Tensor | None = None, **kw: object) -> torch.Tensor:
    if m.cfg.tokenizer == "bins":
        return m(x, **kw)  # type: ignore[arg-type]
    return m(patches=x, observed=obs, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["bins", "patch"])
def test_causal_mask_future_does_not_leak(kind: str):
    m = tiny(kind)
    if kind == "bins":
        x = torch.randint(2, 64, (2, 12))
        y = x.clone()
        y[:, 7:] = torch.randint(2, 64, (2, 5))
        a, b = run(m, x), run(m, y)
    else:
        x, obs = patch_inputs(t=12)
        y = x.clone()
        y[:, 7:] += 5.0
        a, b = run(m, x, obs), run(m, y, obs)
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


@pytest.mark.parametrize("kind", ["bins", "patch"])
def test_kv_cache_matches_full_forward(kind: str):
    m = tiny(kind)
    if kind == "bins":
        x = torch.randint(2, 64, (3, 15))
        obs = None
        full = run(m, x)
    else:
        x, obs = patch_inputs(b=3, t=15)
        full = run(m, x, obs)
    cache = KVCache.empty(m.cfg, 3, torch.device("cpu"), torch.float32)
    outs = []
    # Prefill 6 positions, then 1, then 3, then 1 at a time.
    bounds = [0, 6, 7, 10, 11, 12, 13, 14, 15]
    for lo, hi in itertools.pairwise(bounds):
        if kind == "bins":
            outs.append(run(m, x[:, lo:hi], cache=cache))
        else:
            assert obs is not None
            outs.append(run(m, x[:, lo:hi], obs[:, lo:hi], cache=cache))
    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-5, atol=1e-5)
    assert cache.length == 15


def test_left_padding_does_not_change_real_positions():
    """RoPE is relative, and padded keys are masked: prepending padding must not change
    the outputs at real positions (up to float error)."""
    m = tiny("patch")
    x, obs = patch_inputs(b=1, t=8)
    ref = run(m, x, obs)
    xp = torch.cat([torch.zeros(1, 3, 4), x], dim=1)
    op = torch.cat([torch.zeros(1, 3, 4, dtype=torch.bool), obs], dim=1)
    pad = torch.zeros(1, 11, dtype=torch.bool)
    pad[:, :3] = True
    out = run(m, xp, op, pad=pad)
    torch.testing.assert_close(out[:, 3:], ref, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(out).all()


def test_param_count_formula():
    cfg = ModelConfig(d_model=384, n_layers=6, n_heads=6)
    m = TSModel(cfg)
    body = m.num_params(body_only=True)
    # 12 * L * d^2 up to the SwiGLU width rounding and the norm weights.
    assert abs(body - 12 * 6 * 384**2) / (12 * 6 * 384**2) < 0.03


def test_sequence_longer_than_max_is_rejected():
    m = tiny("patch")
    x, obs = patch_inputs(t=65)
    with pytest.raises(ValueError, match="max_seq_len"):
        run(m, x, obs)


@pytest.mark.parametrize("kind", ["bins", "patch"])
def test_overfits_a_tiny_batch(kind: str):
    torch.manual_seed(0)
    m = tiny(kind).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    if kind == "bins":
        x = torch.randint(2, 64, (4, 16))

        def loss_fn() -> torch.Tensor:
            logits = m(x[:, :-1])
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1)
            )
    else:
        x, obs = patch_inputs(b=4, t=6)
        target = torch.randn(4, 6, 8, generator=torch.Generator().manual_seed(1))
        levels = torch.tensor(m.cfg.quantiles)

        def loss_fn() -> torch.Tensor:
            return pinball_loss(
                m(patches=x, observed=obs),
                target,
                levels,
                torch.ones_like(target, dtype=torch.bool),
            )

    first = loss_fn().item()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    last = loss_fn().item()
    assert last < 0.05 * first, (first, last)
