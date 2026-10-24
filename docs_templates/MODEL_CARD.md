# Model card

## The model with a test score

Weights for the three CPU seeds are `results/runs/cpu_s0_seed0/model.pt`, `results/runs/cpu_s0_seed1/model.pt` and `results/runs/cpu_s0_seed2/model.pt`.

| | |
|---|---|
| run | `cpu_s0`, seeds 0, 1 and 2 |
| parameters (total / transformer body) | {{ results/runs/cpu_s0_seed0/summary.json#params_total | int }} / {{ results/runs/cpu_s0_seed0/summary.json#params_body | int }} |
| layers x width x heads | {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_layers }} x {{ results/runs/cpu_s0_seed0/summary.json#config.model.d_model }} x {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_heads }} |
| patch length / horizon | {{ results/runs/cpu_s0_seed0/summary.json#config.model.patch_len }} / {{ results/runs/cpu_s0_seed0/summary.json#config.model.horizon }} |
| training steps x batch | {{ results/runs/cpu_s0_seed0/summary.json#config.train.steps | int }} x {{ results/runs/cpu_s0_seed0/summary.json#config.train.batch_size }} |
| precision / hardware | fp32 on CPU |
| test CRPS vs Seasonal Naive | {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }} [{{ results/analysis/cpu_study.json#test.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.0.ci95.1 | .3f }}] |
| test CRPS vs AutoARIMA | {{ results/analysis/cpu_study.json#test.comparisons.2.ratio | .3f }} [{{ results/analysis/cpu_study.json#test.comparisons.2.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.2.ci95.1 | .3f }}] |

Decoder-only transformer written in plain PyTorch: pre-norm RMSNorm, rotary position embeddings, SwiGLU, a key/value cache. Inputs are patches scaled by the mean absolute value of the context. The head predicts non-crossing quantiles. Longer horizons feed the predicted median back in. Design and the decision rule are fixed in `PREREGISTRATION_CPU.md`.

## The T4 validation headline

{{ table:model_card_headline }}

{{ if exists:results/analysis/headline.json }}
`patch_s2` was selected on the validation split (CRPS {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive, {{ results/analysis/val_summary.json#n_configs }} configurations, one seed). Its weights are not in this repository. It was scored once on the test split from its Drive checkpoint, one seed: CRPS {{ results/analysis/headline.json#comparisons.0.ratio | .3f }}× upstream AutoARIMA ({{ results/analysis/headline.json#comparisons.0.verdict }}; `results/analysis/headline.json`). The CPU test numbers above are a different model.
{{ else }}
`patch_s2` was selected on the validation split (CRPS {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive, {{ results/analysis/val_summary.json#n_configs }} configurations, one seed). Its weights are not in this repository. It has no test score. The CPU test numbers above are a different model.
{{ end }}

Inputs for that run are patches of {{ results/runs/patch_s2/summary.json#config.model.patch_len }} steps, clipped at 100 times the context scale in magnitude. The head predicts 13 non-crossing quantiles of the next {{ results/runs/patch_s2/summary.json#config.model.horizon }} steps.

## Intended use

Research and demonstration of a small model pretrained from scratch on public, leakage-checked data. Not for production forecasting, and not for any financial or trading decision.

## Training data

`DATA_CARD.md`. The CPU study trains on twelve leakage-exclusion-applied pretrain datasets that are disjoint from the ten validation datasets, plus synthetic Gaussian-process series (`PREREGISTRATION_CPU.md`). GIFT-Eval's own train and validation windows are not used for training.

## Evaluation

Test: the {{ results/analysis/cpu_study.json#test.n_configs }} GIFT-Eval configurations, three seeds, raw quantiles and split-conformal quantiles, scored once (`results/analysis/cpu_study.json`). Validation for this study: {{ results/analysis/cpu_study.json#validation.n_configs }} configurations, used to fit conformal expansions and for a positional-encoding ablation. The T4 grid's validation table is a separate study.

## Factors

One series at a time. Multivariate GIFT-Eval entries are split into univariate series before scoring. Datasets and frequencies are in `DATA_CARD.md`. The CPU test average covers the configurations and seeds above. The T4 headline was chosen on the validation split, one seed.

## Metrics

MASE and quantile-loss CRPS, each divided by Seasonal Naive, then a geometric mean over configurations. Coverage at the 50/80/90% levels. The T4 primary comparison is CRPS against upstream AutoARIMA, with a 95% paired bootstrap over configurations (`PREREGISTRATION.md`). The CPU study uses the same metrics (`PREREGISTRATION_CPU.md`).

## Scaling

{{ table:scaling }}

![Validation loss against training compute for the patch models. The line is the fitted power law and the band is its paired bootstrap, both read from the committed scaling fit.](docs/figures/scaling.svg)

## Compute

{{ table:bench }}

The trainer writes a checkpoint at least every 10 minutes (`ckpt_minutes` in the training loop). `DEVIATIONS.md` records which bench configurations were not trained.

## Ethical considerations

Research and demonstration only. Not a production forecaster, and not an input to a trading or other financial decision. The volatility numbers elsewhere in this repository are statistical scores. Training data is public, with the leakage exclusions in `DATA_CARD.md`.

## Citation

Apache-2.0 (`LICENSE`). Cite this software with `CITATION.cff`.

## Known limitations

* On the test split, CRPS against Seasonal Naive has a configuration interval that includes 1. MASE is worse, and both metrics are worse than upstream AutoARIMA.
* Raw 80% intervals cover {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} of the test outcomes. Pooled split-conformal adjustment, fit on validation only, moves that to {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }} and makes CRPS worse ({{ results/analysis/cpu_study.json#test_cqr.summary.crps_gmean | .3f }}× Seasonal Naive).
* The median roll-out ignores uncertainty from earlier steps, which is one reason the raw intervals are narrow.
* The no-position ablation was scored on validation only.
* No comparison against Chronos, TimesFM, or FlowState was run for this model. Upstream files for other models are context, not a claim that this model beat them.
* Univariate only. Values beyond the clipping range of the T4 patch head cannot be represented by that head.
{{ if exists:results/analysis/headline.json }}
* `patch_s2` has a test score from one seed only.
{{ else }}
* `patch_s2` still has no test score.
{{ end }}
