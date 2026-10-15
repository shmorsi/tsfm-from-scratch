# Model card

## Model

{{ table:model_card_headline | default:*The headline model is chosen on the validation split by the pre-registered rule once the scaling grid has finished; until then this section is pending.* }}

Decoder-only transformer written in plain PyTorch: pre-norm RMSNorm, rotary position
embeddings, SwiGLU feed-forward layers, a key/value cache for incremental decoding.
Inputs are patches of 32 steps scaled by the mean absolute value of the context (values
clipped at 100 times that scale in magnitude); the head predicts 13 non-crossing quantiles of the next
64 steps. Longer horizons are produced by feeding the predicted median back in.

## Intended use

Research and demonstration: how far a small model pretrained from scratch on public,
leakage-checked data gets on zero-shot forecasting. **Not** intended for production
forecasting or for any financial or trading decision.

## Training data

`DATA_CARD.md`: a pinned subset of GiftEvalPretrain with every series that overlaps
GIFT-Eval removed, plus synthetic Gaussian-process series. GIFT-Eval's own train and
validation splits are never used, so the model is zero-shot in the leaderboard's sense.

## Evaluation

Validation (model selection): ten held-out datasets and held-out synthetic series.
Test: the 97 GIFT-Eval configurations, scored once per model after the pre-registration
tag (`PREREGISTRATION.md`). Results are in `README.md` and `RESEARCH_NOTE.md`.

## Known limitations

* Intervals are too narrow, most at long horizons where the median roll-out ignores the
  uncertainty of earlier steps (measured coverage is in the results tables).
* Short yearly and quarterly series are its weakest frequencies on the validation split.
* Values beyond 100 times the context's mean magnitude cannot be predicted (clipping).
* Univariate only; covariates and cross-series information are ignored.
