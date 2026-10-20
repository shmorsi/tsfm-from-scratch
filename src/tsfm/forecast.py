"""Forecasting with a trained model: contexts in, quantiles out.

patch model: one forward pass over the context; the last token's head gives quantiles
    for the next ``horizon`` steps. Longer horizons roll out autoregressively by feeding
    the predicted median back in, one patch at a time, with the KV cache. Feeding the
    median ignores the uncertainty of earlier steps, so later quantiles are likely too
    narrow; this is a known limitation of quantile-head models and is reported.
bins model: draws ``n_samples`` sample paths token by token (temperature 1, no top-k)
    with the KV cache, and reports empirical quantiles and the sample mean. Contexts are
    processed in chunks so that the expanded cache (one copy per sample path) stays
    within ``kv_budget_bytes``; the chunk size depends only on the inputs, never on the
    free memory of the device, so results are reproducible across machines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from tsfm.model import KVCache, TSModel
from tsfm.tokenizers import N_SPECIAL, PatchTokenizer, before_first, mean_scale


@dataclass
class Forecast:
    quantiles: np.ndarray  # (B, H, Q)
    mean: np.ndarray  # (B, H)
    levels: tuple[float, ...]


def _left_pad(contexts: Sequence[np.ndarray], length: int, multiple: int = 1) -> np.ndarray:
    """Most recent ``length`` points of each context, left-padded with NaN to a common
    length that is a multiple of ``multiple``."""
    n = min(length, max(len(c) for c in contexts))
    n = int(math.ceil(n / multiple) * multiple)
    out = np.full((len(contexts), n), np.nan, np.float32)
    for i, c in enumerate(contexts):
        c = np.asarray(c, np.float32)[-n:]
        out[i, n - c.size :] = c
    return out


def effective_max_context(model: TSModel, prediction_length: int, max_context: int) -> int:
    """Context length ``forecast_patch`` actually uses.

    Roll-out tokens have to fit in ``max_seq_len`` beside the context, so a long
    horizon shortens the context. The scale and the conformal scores must use this
    same length.
    """
    cfg = model.cfg
    room = cfg.max_seq_len - math.ceil(prediction_length / cfg.patch_len)
    if room < 1:
        raise ValueError(
            f"prediction_length {prediction_length} does not fit in max_seq_len {cfg.max_seq_len}"
        )
    return min(max_context, room * cfg.patch_len)


@torch.no_grad()
def forecast_patch(
    model: TSModel,
    contexts: Sequence[np.ndarray],
    prediction_length: int,
    max_context: int,
    adjustment: object | None = None,
) -> Forecast:
    cfg = model.cfg
    dev = next(model.parameters()).device
    p = cfg.patch_len
    max_context = effective_max_context(model, prediction_length, max_context)
    x = torch.from_numpy(_left_pad(contexts, max_context, p)).to(dev)
    b = x.shape[0]
    obs = torch.isfinite(x)
    scale = mean_scale(x, obs)
    tok = PatchTokenizer(p)
    patches, observed, patch_pad = tok.encode(x, scale)
    cache = KVCache.empty(cfg, b, dev, next(model.parameters()).dtype)
    q = model(patches=patches, observed=observed, pad=patch_pad, cache=cache)[:, -1]  # (B, H, Q)
    mid = cfg.quantiles.index(0.5)
    chunks = [q]
    have = cfg.horizon
    # Roll out: feed predicted medians back, one patch per step.
    fed = 0
    median_path = q[..., mid]
    while have < prediction_length:
        nxt = median_path[:, fed : fed + p]  # always available: fed + p <= have
        fed += p
        out = model(
            patches=nxt[:, None],
            observed=torch.ones_like(nxt[:, None], dtype=torch.bool),
            cache=cache,
        )[:, -1]
        # The new token's forecast starts right after the patch just fed in.
        offset = fed
        keep = out[:, have - offset :] if have - offset > 0 else out
        chunks.append(keep)
        have += keep.shape[1]
        median_path = torch.cat([median_path, keep[..., mid]], dim=1)
    qs = torch.cat(chunks, dim=1)[:, :prediction_length]
    qn = qs.float().cpu().numpy()
    if adjustment is not None:
        from tsfm.calibrate import CQR, apply_cqr

        if not isinstance(adjustment, CQR):
            raise TypeError("patch adjustment must be a CQR")
        # Expansions are in scaled units; the median column is left alone.
        qn = apply_cqr(qn, adjustment).astype(np.float32)
    scale_np = scale.float().cpu().numpy()[..., None]
    qn = qn * scale_np
    return Forecast(qn, qn[..., mid].copy(), tuple(cfg.quantiles))


@torch.no_grad()
def forecast_bins(
    model: TSModel,
    contexts: Sequence[np.ndarray],
    prediction_length: int,
    max_context: int,
    n_samples: int = 20,
    levels: tuple[float, ...] = (
        0.05,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        0.95,
    ),
    generator: torch.Generator | None = None,
    kv_budget_bytes: int = 3 * 2**30,
) -> Forecast:
    cfg = model.cfg
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    max_context = min(max_context, cfg.max_seq_len - prediction_length)
    if max_context < 1:
        raise ValueError(f"prediction_length {prediction_length} leaves no room for context")
    x_all = torch.from_numpy(_left_pad(contexts, max_context)).to(dev)
    length = x_all.shape[1] + prediction_length
    per_context = n_samples * KVCache.bytes_per_row(cfg, length, dtype)
    chunk = max(1, kv_budget_bytes // per_context)
    qs_parts, mean_parts = [], []
    for lo in range(0, x_all.shape[0], chunk):
        x = x_all[lo : lo + chunk]
        b = x.shape[0]
        scale = mean_scale(x, torch.isfinite(x))
        pad = before_first(torch.isfinite(x))
        ids = model.bins.encode(x, scale, pad=pad)
        cache = KVCache.empty(cfg, b, dev, dtype, max_len=length)
        logits = model(ids, pad=pad, cache=cache)[:, -1]
        cache = cache.repeat_interleave(n_samples)
        logits = logits.repeat_interleave(n_samples, 0)
        out = torch.empty(b * n_samples, prediction_length, dtype=torch.long, device=dev)
        for h in range(prediction_length):
            probs = torch.softmax(logits.float(), -1)
            probs[:, :N_SPECIAL] = 0  # never sample PAD/MISSING
            nxt = torch.multinomial(probs, 1, generator=generator)
            out[:, h] = nxt[:, 0]
            if h + 1 < prediction_length:
                logits = model(nxt, cache=cache)[:, -1]
        del cache
        vals = model.bins.decode(out, scale.repeat_interleave(n_samples, 0)).view(
            b, n_samples, prediction_length
        )
        q = torch.quantile(vals, torch.tensor(levels, device=dev), dim=1).permute(1, 2, 0)
        qs_parts.append(q.cpu().numpy())
        mean_parts.append(vals.mean(1).cpu().numpy())
    return Forecast(np.concatenate(qs_parts), np.concatenate(mean_parts), levels)


def forecast(
    model: TSModel,
    contexts: Sequence[np.ndarray],
    prediction_length: int,
    max_context: int,
    **kw: object,
) -> Forecast:
    if model.cfg.tokenizer == "patch":
        adjustment = kw.pop("adjustment", None)
        if kw:
            raise TypeError(f"unexpected forecast arguments: {sorted(kw)}")
        return forecast_patch(model, contexts, prediction_length, max_context, adjustment)
    return forecast_bins(model, contexts, prediction_length, max_context, **kw)  # type: ignore[arg-type]


def load_model(path: str, device: str | torch.device = "cpu") -> TSModel:
    from tsfm.model import ModelConfig

    st = torch.load(path, map_location=device, weights_only=False)
    m = TSModel(ModelConfig.from_dict(st["config"]["model"]))
    m.load_state_dict(st["model"])
    return m.to(device).eval()
