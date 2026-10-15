"""Training losses for both tokenizers.

For every window a context length ``c`` is drawn at random. The scale is computed from
the first ``c`` points only and the loss is taken only on predictions of points after
``c``, so training never uses a statistic of the future it is asked to predict - exactly
the situation at forecast time - while the model still sees contexts of all lengths.
The prefix positions are still processed (they are the context); they just do not
contribute to the loss.

Short series: the minimum context adapts (at most half of the real points), and for the
patch model each window is shifted so that the context ends on a patch boundary, which
is also what happens at forecast time (the context is left-padded to whole patches).
Rows whose context has no observed value, or that have no observed target, are left out
of the loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tsfm.model import TSModel
from tsfm.tokenizers import (
    MISSING_ID,
    N_SPECIAL,
    SCALED_CLIP,
    PatchTokenizer,
    before_first,
    mean_scale,
    pinball_loss,
)


def leading_pad(x: Tensor) -> Tensor:
    """True for positions before the first finite value of each row."""
    return before_first(torch.isfinite(x))


def draw_context(rng: np.random.Generator, x: np.ndarray, min_ctx: int, max_ctx: int) -> np.ndarray:
    """Per-row context end position (in steps from the window start).

    The context holds at least ``min(min_ctx, n_real // 2)`` (and at least one) real
    points, at most ``max_ctx``, and at least one position follows it."""
    t = x.shape[1]
    first = np.argmax(np.isfinite(x), axis=1)
    n_real = t - first
    m = np.clip(np.minimum(min_ctx, n_real // 2), 1, None)
    lo = np.minimum(first + m, t - 1)
    hi = np.maximum(np.minimum(first + max_ctx, t - 1), lo)
    return rng.integers(lo, hi + 1)


def align_to_patches(x: Tensor, ctx: Tensor, p: int) -> tuple[Tensor, Tensor]:
    """Shift each row left by ``ctx % p`` so the context ends on a patch boundary.

    The ``ctx % p`` oldest positions are dropped and the end is filled with NaN; returns
    the shifted rows and the new (aligned) context ends."""
    t = x.shape[1]
    r = ctx % p
    idx = torch.arange(t, device=x.device)[None, :] + r[:, None]
    shifted = torch.gather(x, 1, idx.clamp(max=t - 1))
    shifted = torch.where(idx < t, shifted, torch.full_like(shifted, float("nan")))
    return shifted, ctx - r


def bins_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
    """x (B, T) values (NaN = missing/pad); ctx (B,) context lengths in steps."""
    _b, t = x.shape
    pos = torch.arange(t, device=x.device)
    pad = leading_pad(x)
    in_ctx = pos[None, :] < ctx[:, None]
    obs_ctx = torch.isfinite(x) & in_ctx
    scale = mean_scale(x, obs_ctx)
    ids = model.bins.encode(x, scale, pad=pad)
    logits = model(ids[:, :-1], pad=pad[:, :-1])
    target = ids[:, 1:]
    use = (pos[None, 1:] >= ctx[:, None]) & (target >= N_SPECIAL) & (target != MISSING_ID)
    use = use & obs_ctx.any(dim=1, keepdim=True)
    n = use.sum()
    loss = F.cross_entropy(logits[use].float(), target[use]) if bool(n) else logits.sum() * 0.0
    return loss, n


def patch_targets(x: Tensor, p: int, horizon: int) -> Tensor:
    """For token i (covering steps [iP, (i+1)P)), the next ``horizon`` steps. (B, N, H)"""
    b, t = x.shape
    ext = torch.cat([x, x.new_full((b, horizon), float("nan"))], dim=1)
    return ext[:, p:].unfold(1, horizon, p)[:, : t // p]


def patch_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
    """x (B, T) with T a multiple of patch_len; ctx (B,) context ends in steps."""
    cfg = model.cfg
    p = cfg.patch_len
    _b, t = x.shape
    if t % p:
        raise ValueError("window length must be a multiple of patch_len")
    x, ctx = align_to_patches(x, ctx, p)
    ctx_p = ctx // p  # context length in whole patches
    pos = torch.arange(t, device=x.device)
    obs_ctx = torch.isfinite(x) & (pos[None, :] < ctx[:, None])
    scale = mean_scale(x, obs_ctx)
    tok = PatchTokenizer(p)
    patches, observed, patch_pad = tok.encode(x, scale)
    pred = model(patches=patches, observed=observed, pad=patch_pad)  # (B, N, H, Q)
    tgt = patch_targets((x / scale).clamp(-SCALED_CLIP, SCALED_CLIP), p, cfg.horizon)
    tok_idx = torch.arange(t // p, device=x.device)
    use_tok = (tok_idx[None, :] >= (ctx_p - 1)[:, None]) & ~patch_pad
    use_tok = use_tok & (ctx_p >= 1)[:, None] & obs_ctx.any(dim=1, keepdim=True)
    obs = torch.isfinite(tgt) & use_tok[..., None]
    levels = torch.tensor(cfg.quantiles, device=x.device, dtype=pred.dtype)
    loss = pinball_loss(pred.float(), tgt.float(), levels.float(), obs)
    return loss, obs.sum()


def compute_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
    """(loss, number of target values in the loss). Both are tensors, so callers can avoid
    a GPU synchronisation per micro-batch."""
    if model.cfg.tokenizer == "bins":
        return bins_loss(model, x, ctx)
    return patch_loss(model, x, ctx)
