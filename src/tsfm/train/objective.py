"""Training losses for both tokenizers.

For every window a context length ``c`` is drawn at random. The scale is computed from
the first ``c`` points only and the loss is taken only on predictions of points after
``c``, so training never uses a statistic of the future it is asked to predict - exactly
the situation at forecast time - while the model still sees contexts of all lengths.
The prefix positions are still processed (they are the context); they just do not
contribute to the loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tsfm.model import TSModel
from tsfm.tokenizers import MISSING_ID, N_SPECIAL, PatchTokenizer, mean_scale, pinball_loss


def leading_pad(x: Tensor) -> Tensor:
    """True for positions before the first finite value of each row."""
    return torch.cumsum(torch.isfinite(x).int(), dim=-1) == 0


def draw_context(rng: np.random.Generator, x: np.ndarray, min_ctx: int, max_ctx: int) -> np.ndarray:
    """Per-row context length (in time steps), starting at the first real point."""
    first = np.argmax(np.isfinite(x), axis=1)
    lo = np.minimum(first + min_ctx, x.shape[1] - 1)
    hi = np.minimum(first + max_ctx, x.shape[1] - 1)
    return rng.integers(lo, np.maximum(hi, lo) + 1)


def bins_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, int]:
    """x (B, T) values (NaN = missing/pad); ctx (B,) context lengths in steps."""
    _b, t = x.shape
    pos = torch.arange(t, device=x.device)
    pad = leading_pad(x)
    in_ctx = pos[None, :] < ctx[:, None]
    scale = mean_scale(x, torch.isfinite(x) & in_ctx)
    ids = model.bins.encode(x, scale, pad=pad)
    logits = model(ids[:, :-1], pad=pad[:, :-1])
    target = ids[:, 1:]
    use = (pos[None, 1:] >= ctx[:, None]) & (target >= N_SPECIAL) & (target != MISSING_ID)
    loss = F.cross_entropy(logits[use].float(), target[use])
    return loss, int(use.sum())


def patch_targets(x: Tensor, p: int, horizon: int) -> Tensor:
    """For token i (covering steps [iP, (i+1)P)), the next ``horizon`` steps. (B, N, H)"""
    b, t = x.shape
    ext = torch.cat([x, x.new_full((b, horizon), float("nan"))], dim=1)
    return ext[:, p:].unfold(1, horizon, p)[:, : t // p]


def patch_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, int]:
    """x (B, T) with T a multiple of patch_len; ctx (B,) context lengths in steps,
    rounded down to whole patches here so the context ends on a patch boundary."""
    cfg = model.cfg
    p = cfg.patch_len
    _b, t = x.shape
    if t % p:
        raise ValueError("window length must be a multiple of patch_len")
    ctx_p = torch.clamp(ctx // p, min=1)  # context in patches
    pos = torch.arange(t, device=x.device)
    in_ctx = pos[None, :] < (ctx_p * p)[:, None]
    scale = mean_scale(x, torch.isfinite(x) & in_ctx)
    tok = PatchTokenizer(p)
    patches, observed, patch_pad = tok.encode(x, scale)
    pred = model(patches=patches, observed=observed, pad=patch_pad)  # (B, N, H, Q)
    tgt = patch_targets(x / scale, p, cfg.horizon)
    tok_idx = torch.arange(t // p, device=x.device)
    use_tok = (tok_idx[None, :] >= (ctx_p - 1)[:, None]) & ~patch_pad
    obs = torch.isfinite(tgt) & use_tok[..., None]
    levels = torch.tensor(cfg.quantiles, device=x.device, dtype=pred.dtype)
    loss = pinball_loss(pred.float(), tgt.float(), levels.float(), obs)
    return loss, int(obs.sum())


def compute_loss(model: TSModel, x: Tensor, ctx: Tensor) -> tuple[Tensor, int]:
    if model.cfg.tokenizer == "bins":
        return bins_loss(model, x, ctx)
    return patch_loss(model, x, ctx)
