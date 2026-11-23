"""One forecasting interface over two backends: PyTorch (``tsfm.forecast``) and ONNX Runtime.

The ONNX path does exactly what ``forecast_patch`` does before its first forward pass (keep
the last ``max_context`` points, left-pad with NaN to a whole number of patches, mean-scale,
patch) and then runs the exported graph. It covers horizons up to the model's own horizon;
a longer horizon needs the median roll-out with a KV cache, which stays in PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tsfm.forecast import Forecast, _left_pad, effective_max_context, forecast_patch, load_model
from tsfm.model import TSModel
from tsfm.tokenizers import PatchTokenizer, mean_scale


@dataclass
class Engine:
    model: TSModel
    max_context: int = 512
    onnx_session: Any | None = None
    onnx_file: str | None = None

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        onnx_path: Path | None = None,
        max_context: int = 512,
        threads: int | None = None,
    ) -> Engine:
        model = load_model(str(checkpoint))
        if threads:
            torch.set_num_threads(threads)
        sess = None
        if onnx_path is not None and onnx_path.exists():
            from tsfm.serve.export import session

            sess = session(onnx_path, threads)
        name = onnx_path.name if (sess is not None and onnx_path is not None) else None
        return cls(model, max_context, sess, name)

    @property
    def horizon(self) -> int:
        return self.model.cfg.horizon

    def backend_for(self, prediction_length: int) -> str:
        return (
            "onnx"
            if self.onnx_session is not None and prediction_length <= self.horizon
            else "torch"
        )

    def forecast(
        self, contexts: Sequence[np.ndarray], prediction_length: int, backend: str | None = None
    ) -> tuple[Forecast, str]:
        use = backend or self.backend_for(prediction_length)
        if use == "torch":
            return forecast_patch(
                self.model, contexts, prediction_length, self.max_context
            ), "torch"
        if self.onnx_session is None or prediction_length > self.horizon:
            raise ValueError(
                "the ONNX backend needs an exported model and a horizon within the model's own"
            )
        from tsfm.serve.export import run_onnx

        cfg = self.model.cfg
        p = cfg.patch_len
        ctx = effective_max_context(self.model, prediction_length, self.max_context)
        x = torch.from_numpy(_left_pad(contexts, ctx, p))
        obs = torch.isfinite(x)
        scale = mean_scale(x, obs)
        patches, observed, pad = PatchTokenizer(p).encode(x, scale)
        q = run_onnx(self.onnx_session, patches, observed, pad)[:, :prediction_length]
        q = q * scale.numpy()[..., None]
        mid = cfg.quantiles.index(0.5)
        return Forecast(q, q[..., mid].copy(), tuple(cfg.quantiles)), "onnx"
