# tsfm: a time-series transformer, written from scratch

A decoder-only patch transformer (pre-norm RMSNorm, rotary positions, SwiGLU, a non-crossing quantile head) in plain PyTorch. It is pretrained on a pinned, leakage-checked public corpus plus synthetic series, and scored with the official GIFT-Eval metric code.

On the frozen test split, the committed CPU model ({{ results/runs/cpu_s0_seed0/summary.json#params_total | int }} parameters, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.n }} seeds, {{ results/analysis/cpu_study.json#test.n_configs }} configurations) has CRPS {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}× Seasonal Naive (95% CI over configurations [{{ results/analysis/cpu_study.json#test.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.0.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.0.verdict }}) and {{ results/analysis/cpu_study.json#test.comparisons.2.ratio | .3f }}× upstream AutoARIMA (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.2.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.2.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.2.verdict }}). MASE is {{ results/analysis/cpu_study.json#test.summary.mase_gmean | .3f }}× Seasonal Naive (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.1.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.1.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.1.verdict }}).

The {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive CRPS for `patch_s2` is a validation result: {{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed, {{ results/runs/patch_s2/summary.json#params_total | int }} parameters. Those weights are not in this repository, so that checkpoint has no test score.

![GIFT-Eval test coverage, raw quantiles against split-conformal](docs/figures/coverage.svg)

Nominal coverage against empirical coverage on the test split, after averaging the three seeds. Raw quantiles cover {{ results/analysis/cpu_study.json#test.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.90 | .3f }} at the 50/80/90% levels. Split-conformal adjustment, fit on the validation split only, covers {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.90 | .3f }}. A calibrated model would sit on the diagonal. The conformal 80% point is above the diagonal, and the CRPS comparison below is worse.

## Test results

{{ table:cpu_test }}

{{ table:cpu_vs_arima }}

The pre-registered comparison is CRPS against upstream AutoARIMA. Both families are worse on that comparison. Conformal adjustment leaves MASE unchanged, because the median is not moved, and it makes CRPS worse.

Reproduce the study on CPU. An eval that already has `all_results.csv` is skipped, so the committed test scores are not recomputed:

```bash
uv run tsfm cpu-study --config configs/cpu_study.yaml
```

## What I built and learned

I wrote the model, the training loop, and the GIFT-Eval scoring path in plain PyTorch, then trained a smaller model whose weights are in the repository. The {{ results/runs/patch_s2/summary.json#params_total | int }}-parameter run that produced the validation headline never had its checkpoint committed, so I did not invent a test number for it.

Seed-averaged validation CRPS is {{ results/analysis/cpu_study.json#validation.summary.crps_gmean | .3f }}× Seasonal Naive ({{ results/analysis/cpu_study.json#validation.n_configs }} configurations, 95% CI [{{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.1 | .3f }}]). The same statistic on the test split is {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}×. The three seed-level test ratios are {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.1 | .3f }} and {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.2 | .3f }}. Their mean is {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.mean | .3f }}, and the 95% Student-t interval on that mean is [{{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.1 | .3f }}]. That interval belongs to the mean of the three seed-level geometric means. It is not the interval on the seed-averaged ratio in the table, which is a paired bootstrap over configurations. I kept all three seeds.

Split conformal, one pooled expansion per seed fit only on validation, widens the intervals and pushes the test 80% coverage from {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }}, above the nominal 80% level. CRPS moves from {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.crps_gmean | .3f }} (worse than Seasonal Naive; the configuration interval is entirely above 1). Seed 0's expansions, in mean-absolute-scaled units, are {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.50 | .3f }} at 50%, {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.80 | .3f }} at 80% and {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.90 | .3f }} at 90%. The validation conformal row is scored on the same forecasts the expansions were fit on, so that coverage is not a held-out check. The test table is the held-out calibration result.

Turning rotary positions off, seed 0 only, on the same {{ results/analysis/cpu_study.json#validation_ablation.n_configs }} validation configurations, moves CRPS from {{ results/analysis/cpu_study.json#validation_ablation.rope.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.crps_gmean | .3f }} and MASE from {{ results/analysis/cpu_study.json#validation_ablation.rope.mase_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.mase_gmean | .3f }}. That run was not scored on the test split.

On one held-out hourly series (`bdg-2_bear`, context {{ results/analysis/attention_cpu_s0.json#context_length }} steps, patches of length {{ results/analysis/attention_cpu_s0.json#profile.patch_len }}), the last query puts {{ results/analysis/attention_cpu_s0.json#profile.mass_last_patch | .3f }} of its attention on the most recent patch and {{ results/analysis/attention_cpu_s0.json#profile.mass_last_4_patches | .3f }} on the last four. The peak is at lag {{ results/analysis/attention_cpu_s0.json#profile.peak_lag_patches }} patches. Entropy is {{ results/analysis/attention_cpu_s0.json#profile.entropy_nats | .3f }} nats against {{ results/analysis/attention_cpu_s0.json#profile.entropy_uniform_nats | .3f }} for a uniform distribution over the patches. The mass is local. The series is hourly and each patch is {{ results/analysis/attention_cpu_s0.json#profile.patch_len }} steps, so a day is not patch-aligned and I am not reading a daily cycle out of that peak.

## Validation study that selected `patch_s2`

Three sizes, pretrained from scratch on a free Colab T4, scored zero-shot on datasets held out of pretraining. This table is the validation split. It is how `patch_s2` was chosen. It is not a test-set result.

{{ table:val_models }}

The CPU study's own validation numbers, used to fit conformal intervals and to compare positional encodings, are separate:

{{ table:cpu_val }}

## Published reference points (other models)

Recomputed from upstream's committed GIFT-Eval result files, plus our Seasonal Naive run. These rows are not this repository's model.

{{ table:reported_reference }}

## Market volatility baselines

Daily realized volatility of BTCUSDT and ETHUSDT, 22-day forecasts, designed in `PREREGISTRATION_VOL.md`. Statistical forecast scores only. Our model was not run on this study.

{{ table:vol_overall }}

## Method in brief

* **Data.** A pinned subset of GiftEvalPretrain, whole datasets held out for validation, plus Gaussian-process series. A content check removed exact and rescaled copies of GIFT-Eval series (`DATA_CARD.md`).
* **Model.** Decoder-only transformer in plain PyTorch. The committed CPU model uses patch length {{ results/runs/cpu_s0_seed0/summary.json#config.model.patch_len }}, horizon {{ results/runs/cpu_s0_seed0/summary.json#config.model.horizon }}, width {{ results/runs/cpu_s0_seed0/summary.json#config.model.d_model }}, {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_layers }} layers and {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_heads }} heads (`MODEL_CARD.md`). The T4 headline used patch length 32 and a horizon of 64.
* **Training.** The CPU study is {{ results/runs/cpu_s0_seed0/summary.json#config.train.steps | int }} steps, batch {{ results/runs/cpu_s0_seed0/summary.json#config.train.batch_size }}, fp32. Seed 0 finished in {{ results/runs/cpu_s0_seed0/summary.json#session_seconds | .1f }} seconds. The T4 loop supports mixed precision and resumable checkpoints.
* **Evaluation.** GluonTS, the same way upstream scores GIFT-Eval: MASE and quantile-loss CRPS, each divided by Seasonal Naive, geometric mean over configurations, plus 50/80/90% coverage.

## Reproduce the rest

```bash
uv sync --extra cpu
uv run pytest
uv run tsfm pipeline --config configs/tiny_cpu.yaml
uv run tsfm docs build
```

`PREREGISTRATION_CPU.md` fixes the CPU study. `PREREGISTRATION.md` fixes the T4 study. `DEVIATIONS.md` records what the T4 budget cut, including that `patch_s2` was not test-scored.

## Limits

The CPU model does not beat AutoARIMA on the test split, and its CRPS against Seasonal Naive is not distinguishable from 1 under the pre-registered configuration interval. Raw intervals undercover. Pooled conformal overshoots and hurts CRPS. `patch_s2` still has no test score. Nothing here measures economic value.
