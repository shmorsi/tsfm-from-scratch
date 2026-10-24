"""Fine-tuning a pretrained checkpoint on realized volatility (PREREGISTRATION_VOL.md).

Same objective as pretraining (random context length, scale from the context only),
windows drawn only from the fine-tuning train period. Every ``eval_every`` steps the
model forecasts 22 days ahead from origins every 5 days in the validation period; the
weights with the lowest pooled weighted quantile loss there are kept.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from typing import Any

import numpy as np
import torch

from tsfm.finance.study import TRAIN_END, VAL_END, H, Series, evaluate
from tsfm.forecast import forecast
from tsfm.model import TSModel
from tsfm.train.objective import compute_loss, draw_context


def model_forecaster(
    model: TSModel, max_context: int = 2048, n_samples: int = 20, seed: int = 0
) -> Any:
    """A callable that forecasts with a trained model, patch or bins."""
    gen = torch.Generator(next(model.parameters()).device).manual_seed(seed)

    def fn(hist: list[np.ndarray], h: int) -> np.ndarray:
        kw: dict[str, Any] = (
            {"n_samples": n_samples, "generator": gen} if model.cfg.tokenizer == "bins" else {}
        )
        return forecast(model, hist, h, max_context, **kw).quantiles

    return fn


def _window(model: TSModel) -> int:
    if model.cfg.tokenizer == "patch":
        return min(1024, model.cfg.patch_len * model.cfg.max_seq_len)
    return min(1024, model.cfg.max_seq_len + 1)


def pooled_wql(rows: list[dict[str, Any]]) -> float:
    """Pooled weighted quantile loss: total loss divided by total absolute target."""
    num = sum(r["wql_num"] for r in rows if np.isfinite(r["wql"]))
    den = sum(r["abs_y"] for r in rows if np.isfinite(r["wql"]))
    return float(num / den)


def finetune(
    model: TSModel,
    series: list[Series],
    *,
    steps: int = 300,
    lr: float = 1e-4,
    batch: int = 64,
    eval_every: int = 25,
    seed: int = 0,
) -> tuple[TSModel, list[dict[str, float]]]:
    """Continue training on realized volatility and return the run summary."""
    dev = next(model.parameters()).device
    w = _window(model)
    train = [s.values[: s.index(TRAIN_END) + 1] for s in series]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    def val_score() -> float:
        model.eval()
        # Origins whose 22 target days all fall inside the validation period.
        rows = evaluate(
            model_forecaster(model), series, TRAIN_END, VAL_END - timedelta(days=H), step=5
        )
        return pooled_wql(rows)

    log = [{"step": 0, "val_wql": val_score()}]
    best, best_state = log[0]["val_wql"], copy.deepcopy(model.state_dict())
    for step in range(1, steps + 1):
        model.train()
        x = np.full((batch, w), np.nan, np.float32)
        for b in range(batch):
            s = train[int(rng.integers(len(train)))]
            end = int(rng.integers(64, s.size + 1))
            seg = s[max(0, end - w) : end]
            x[b, w - seg.size :] = seg
        ctx = draw_context(rng, x, 32, w - 1)
        loss, _ = compute_loss(model, torch.from_numpy(x).to(dev), torch.from_numpy(ctx).to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0:
            v = val_score()
            log.append({"step": step, "train_loss": float(loss.detach()), "val_wql": v})
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, log
