"""Two ways to turn a real-valued series into transformer inputs.

(a) :class:`BinTokenizer` - mean scaling then uniform quantisation into ``n_bins``
    bins on ``[-limit, limit]`` (Chronos-style); one token per time step, trained with
    cross-entropy over the vocabulary.
(b) :class:`PatchTokenizer` - mean scaling then non-overlapping patches of ``patch_len``
    steps; one token per patch. The model's head (:class:`QuantileHead`) predicts
    quantiles of the next ``horizon`` steps, trained with the pinball loss.

Both use the same scale: the mean absolute value of the observed context, so a forecast
never looks at the future to choose its scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

PAD_ID = 0
MISSING_ID = 1
N_SPECIAL = 2

# Quantile levels predicted by the patch head. They contain the nine official GIFT-Eval
# levels (0.1..0.9) plus 0.05/0.25/0.75/0.95 so 50/80/90% intervals can be scored.
QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)


def mean_scale(x: Tensor, observed: Tensor | None = None, min_scale: float = 1e-10) -> Tensor:
    """Mean |x| over observed (finite, non-padded) points, per row. Shape (B,) -> (B, 1).

    Rows whose observed mean |x| is ~0 (all-zero or fully missing) get scale 1."""
    if observed is None:
        observed = torch.isfinite(x)
    xa = torch.where(observed, x.abs(), torch.zeros_like(x))
    n = observed.sum(dim=-1, keepdim=True).clamp(min=1)
    s = xa.sum(dim=-1, keepdim=True) / n
    return torch.where(s > min_scale, s, torch.ones_like(s))


@dataclass(frozen=True)
class BinTokenizer:
    n_bins: int = 4094
    limit: float = 15.0

    @property
    def vocab_size(self) -> int:
        return self.n_bins + N_SPECIAL

    @property
    def bin_width(self) -> float:
        return 2 * self.limit / (self.n_bins - 1)

    def centers(self, device: torch.device | None = None) -> Tensor:
        return torch.linspace(
            -self.limit, self.limit, self.n_bins, dtype=torch.float64, device=device
        )

    def edges(self, device: torch.device | None = None) -> Tensor:
        c = self.centers(device)
        return (c[1:] + c[:-1]) / 2

    def encode(self, x: Tensor, scale: Tensor, pad: Tensor | None = None) -> Tensor:
        """Values -> token ids. NaN -> MISSING_ID; ``pad`` positions -> PAD_ID."""
        z = (x.double() / scale.double()).nan_to_num(nan=0.0)
        ids = torch.bucketize(z, self.edges(x.device), right=True) + N_SPECIAL
        ids = torch.where(torch.isfinite(x), ids, torch.full_like(ids, MISSING_ID))
        if pad is not None:
            ids = torch.where(pad, torch.full_like(ids, PAD_ID), ids)
        return ids

    def decode(self, ids: Tensor, scale: Tensor) -> Tensor:
        """Token ids -> values (bin centres). Special tokens decode to NaN."""
        c = self.centers(ids.device)
        z = c[(ids - N_SPECIAL).clamp(0, self.n_bins - 1)]
        z = torch.where(ids >= N_SPECIAL, z, torch.full_like(z, math.nan))
        return (z * scale.double()).to(torch.float32)


@dataclass(frozen=True)
class PatchTokenizer:
    patch_len: int = 32

    def encode(self, x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(B, T) values -> (patches, observed, patch_pad), all (B, N, P) / (B, N).

        The series is left-padded to a multiple of ``patch_len``. Missing and padded
        points are 0 in ``patches`` and False in ``observed``. ``patch_pad`` marks patches
        before the first observed value (masked out of attention)."""
        b, t = x.shape
        p = self.patch_len
        n = -(-t // p)
        pad = n * p - t
        obs = torch.isfinite(x)
        z = torch.where(obs, x / scale, torch.zeros_like(x))
        if pad:
            z = torch.cat([z.new_zeros(b, pad), z], dim=1)
            obs = torch.cat([obs.new_zeros(b, pad), obs], dim=1)
        # Padding = everything before the first observed point (our own left padding and
        # any NaN left-padding of short series); a patch made only of it is masked.
        is_pad = torch.cumsum(obs.int(), dim=1) == 0
        patch_pad = is_pad.view(b, n, p).all(dim=-1)
        return z.view(b, n, p), obs.view(b, n, p), patch_pad

    @staticmethod
    def decode(patches: Tensor, scale: Tensor) -> Tensor:
        """(B, N, P) scaled patches -> (B, N * P) values."""
        b = patches.shape[0]
        return patches.reshape(b, -1) * scale


class QuantileHead(nn.Module):
    """Maps hidden states to non-crossing quantiles of the next ``horizon`` steps.

    The median is predicted directly; the other levels are the median plus/minus
    cumulative softplus increments, so quantiles are monotone by construction."""

    def __init__(self, d_model: int, horizon: int, quantiles: tuple[float, ...] = QUANTILES):
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError("quantile levels must include 0.5")
        self.horizon = horizon
        self.register_buffer("levels", torch.tensor(quantiles), persistent=False)
        self.mid = quantiles.index(0.5)
        self.n_q = len(quantiles)
        self.proj = nn.Linear(d_model, horizon * self.n_q)

    def forward(self, h: Tensor) -> Tensor:
        """(..., d_model) -> (..., horizon, n_quantiles), sorted along the last axis."""
        raw = self.proj(h).unflatten(-1, (self.horizon, self.n_q))
        med = raw[..., self.mid : self.mid + 1]
        up = torch.nn.functional.softplus(raw[..., self.mid + 1 :]).cumsum(-1)
        down = torch.nn.functional.softplus(raw[..., : self.mid]).flip(-1).cumsum(-1).flip(-1)
        return torch.cat([med - down, med, med + up], dim=-1)


def pinball_loss(pred: Tensor, target: Tensor, levels: Tensor, observed: Tensor) -> Tensor:
    """Mean quantile loss over observed targets.

    pred (..., H, Q), target (..., H), observed (..., H) -> scalar."""
    target = torch.where(observed, target, torch.zeros_like(target))  # NaN * 0 would be NaN
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(levels * diff, (levels - 1) * diff)
    mask = observed.unsqueeze(-1).expand_as(loss).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
