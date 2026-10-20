"""Attention readout for a patch model.

The training path uses fused attention and does not store weights. This module runs one
context through the model with the explicit attention implementation and summarises the
last query of the last layer: where a forecast looks in its own history. It is a
diagnostic, not a metric used to choose a model.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tsfm.forecast import _left_pad
from tsfm.model import TSModel
from tsfm.tokenizers import PatchTokenizer, mean_scale


@torch.no_grad()
def last_query_attention(
    model: TSModel, context: np.ndarray, max_context: int
) -> dict[str, object]:
    """Mean attention of the final query token, averaged over heads of the last layer.

    Padding patches are dropped. ``lag_steps`` counts time steps back from the last
    observed patch (0 = the most recent patch).
    """
    if model.cfg.tokenizer != "patch":
        raise ValueError("attention readout is implemented for the patch tokenizer")
    model.eval()
    cfg = model.cfg
    dev = next(model.parameters()).device
    p = cfg.patch_len
    x = torch.from_numpy(_left_pad([np.asarray(context, np.float32)], max_context, p)).to(dev)
    obs = torch.isfinite(x)
    scale = mean_scale(x, obs)
    patches, observed, patch_pad = PatchTokenizer(p).encode(x, scale)
    captured: list[torch.Tensor] = []
    model(patches=patches, observed=observed, pad=patch_pad, attn_out=captured)
    # (heads, keys) of the last query in the last layer, batch 0.
    weights = captured[-1][0, :, -1, :].mean(0)
    keep = ~patch_pad[0]
    w = weights[keep].float().cpu().numpy()
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        raise RuntimeError("attention over observed patches summed to 0")
    w = w / total
    # Position 0 in `w` is the oldest kept patch; lag 0 is the newest.
    lags = np.arange(w.size - 1, -1, -1)
    entropy = float(-(w * np.log(np.clip(w, 1e-12, None))).sum())
    peak = int(np.argmax(w))
    order = np.argsort(lags)  # increasing lag: 0 is the most recent patch
    return {
        "n_patches": int(w.size),
        "patch_len": int(p),
        "entropy_nats": entropy,
        "entropy_uniform_nats": float(math.log(w.size)),
        "peak_lag_patches": int(lags[peak]),
        "peak_lag_steps": int(lags[peak] * p),
        "peak_mass": float(w[peak]),
        "mass_last_patch": float(w[-1]),
        "mass_last_4_patches": float(w[-min(4, w.size) :].sum()),
        "lag_patches": [int(v) for v in lags[order]],
        "mass": [float(v) for v in w[order]],
    }
