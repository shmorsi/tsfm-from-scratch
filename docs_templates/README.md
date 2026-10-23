# tsfm: a time-series transformer, written from scratch

[![ci](https://github.com/shmorsi/tsfm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/shmorsi/tsfm-from-scratch/actions/workflows/ci.yml)

A small forecasting model pretrained from scratch in plain PyTorch, then scored zero-shot on the GIFT-Eval benchmark under rules fixed in a git tag before any of its models saw the test split.

**Finding.** The public pretraining corpus, described by its authors as non-leaking, contains copies of benchmark series, some inside the test windows. A content audit flagged {{ results/data/leakage_round1.json#totals.series_with_hits | int }} of {{ results/data/leakage_round1.json#totals.pretrain_series_checked | int }} series. All were excluded; a re-run on the final corpus flags {{ results/data/leakage.json#totals.series_with_hits | int }}.

**Headline.**
{{ if exists:results/analysis/headline.json }}
`patch_s2` ({{ results/runs/patch_s2/summary.json#params_total | M }} parameters) on the test split, one seed: CRPS {{ results/analysis/headline.json#comparisons.0.ratio | .3f }}× upstream AutoARIMA (95% CI [{{ results/analysis/headline.json#comparisons.0.ci95.0 | .3f }}, {{ results/analysis/headline.json#comparisons.0.ci95.1 | .3f }}], {{ results/analysis/headline.json#comparisons.0.verdict }}). Its validation CRPS was {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive.
{{ else }}
`patch_s2` ({{ results/runs/patch_s2/summary.json#params_total | M }} parameters) reaches CRPS {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive on the validation split ({{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed). It has no test score yet. The table shows which model has which score.
{{ end }}

{{ table:score_map }}

Lower is better; 1 equals the reference. Each verdict uses the pre-registered rule: better if the interval is entirely below 1, worse if entirely above, otherwise no detectable difference.

**What's inside**

* Model, tokenizers, training loop and sampling written by hand: RMSNorm, RoPE, SwiGLU, KV cache, non-crossing quantile head.
* A SHA-256-pinned corpus, audited series by series for overlap with the benchmark (`DATA_CARD.md`).
* Scoring through GluonTS exactly as upstream does; our Seasonal Naive reproduces theirs on every configuration.
* Pre-registration (`PREREGISTRATION.md`, tag `prereg-v1`), a test split that `tsfm eval` scores once per model, and a test that fails if a doc shows a number not in `results/`.

[Case study](https://shmorsi.github.io/projects/tsfm.html) · [Research note](RESEARCH_NOTE.md) · [Data card](DATA_CARD.md) · [Model card](MODEL_CARD.md) · [Deviations](DEVIATIONS.md)

## Quickstart

Python 3.12 and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```bash
uv sync --locked --extra cpu
uv run pytest -m "not slow and not network and not data"
uv run tsfm pipeline --config configs/tiny_cpu.yaml
uv run tsfm docs build
```

`uv sync --locked` installs the pinned lockfile, including the CPU PyTorch build. The pipeline downloads a small pinned data subset (network required), then trains and scores a tiny model. It writes to a temporary directory, outside `results/`.

### Score `patch_s2` on the test split (Colab T4)

The validation headline's weights are not in this repository. Scoring them is the one remaining pre-registered test run, and it needs a Colab T4. Do not run `notebooks/colab_runner.ipynb` for it: that queue can resume a discarded partial test run, and it would also start the volatility-model stages that `DEVIATIONS.md` records as not run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmorsi/tsfm-from-scratch/blob/main/notebooks/score_patch_s2_test.ipynb)

1. Merge the commit that contains the notebook, or set `COMMIT` in the notebook to that branch. The notebook clones `COMMIT` (default `main`) and scores that revision.
2. Put the `patch_s2` checkpoint on Google Drive. The default `CHECKPOINT_PATH` is `/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt`, with `model.pt` beside it.
3. Open the notebook, choose a T4 GPU runtime, then Runtime -> Run all. The notebook checks that the GPU is a T4 and that the checkpoint is `patch_s2` before it touches the test split.
4. Download `harvest_patch_s2.txt` from Drive (`/content/drive/MyDrive/tsfm/score_patch_s2/`). A copied notebook print can truncate.
5. On a local clone of the same commit:

```bash
uv run python scripts/harvest_colab_output.py harvest_patch_s2.txt
uv run tsfm docs build
```

6. Commit the files the harvest script wrote and the docs `tsfm docs build` regenerated. Do not hand-edit numbers.

That harvest file is what the doc generators read. It contains `results/eval/test/patch_s2/` (`all_results.csv`, `run.json`, `config.json`), a rebuilt `results/analysis/test_summary.json`, and a new `results/analysis/headline.json`. The harvest script appends `results/sync_log.json`. `results/analysis/cpu_study.json` is not in the file. Ranks in the rebuilt summary include `patch_s2`. CRPS and MASE for the other models are recomputed from the same vendored files.

## Results

{{ if exists:results/analysis/headline.json }}
### Test split, `patch_s2`

Scored once from its Drive checkpoint (`notebooks/score_patch_s2_test.ipynb`), one seed, the settings fixed in `PREREGISTRATION.md`. The first row is the pre-registered primary comparison.

{{ table:headline }}

{{ end }}
### Test split, CPU model

On the frozen test split, the committed CPU model ({{ results/runs/cpu_s0_seed0/summary.json#params_total | int }} parameters, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.n }} seeds, {{ results/analysis/cpu_study.json#test.n_configs }} configurations) has CRPS {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}× Seasonal Naive (95% CI over configurations [{{ results/analysis/cpu_study.json#test.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.0.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.0.verdict }}) and {{ results/analysis/cpu_study.json#test.comparisons.2.ratio | .3f }}× upstream AutoARIMA (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.2.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.2.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.2.verdict }}). MASE is {{ results/analysis/cpu_study.json#test.summary.mase_gmean | .3f }}× Seasonal Naive (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.1.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.1.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.1.verdict }}).

![GIFT-Eval test coverage, raw quantiles against split-conformal](docs/figures/coverage.svg)

Nominal coverage against empirical coverage on the test split, after averaging the three seeds. Raw quantiles cover {{ results/analysis/cpu_study.json#test.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.90 | .3f }} at the 50/80/90% levels. Split-conformal adjustment, fit on the validation split only, covers {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.90 | .3f }}. A calibrated model would sit on the diagonal. The conformal 80% point is above the diagonal, and the CRPS comparison below is worse.

{{ table:cpu_test }}

{{ table:cpu_vs_arima }}

The pre-registered comparison is CRPS against upstream AutoARIMA. Both families are worse on that comparison. Conformal adjustment leaves MASE unchanged, because the median is not moved, and it makes CRPS worse.

Reproduce the study on CPU. An eval that already has `all_results.csv` is skipped, so the committed test scores are not recomputed:

```bash
uv run tsfm cpu-study --config configs/cpu_study.yaml
```

### Validation split, the study that selected `patch_s2`

{{ if exists:results/analysis/headline.json }}
The {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive CRPS for `patch_s2` is a validation result: {{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed, {{ results/runs/patch_s2/summary.json#params_total | int }} parameters. Its test score is in the table above and in `results/analysis/headline.json`. Its weights are not in this repository.
{{ else }}
The {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive CRPS for `patch_s2` is a validation result: {{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed, {{ results/runs/patch_s2/summary.json#params_total | int }} parameters. Those weights are not in this repository, so that checkpoint has no test score.
{{ end }}

Three sizes, pretrained from scratch on a free Colab T4, scored zero-shot on datasets held out of pretraining. This table is the validation split. It is how `patch_s2` was chosen. It is not a test-set result.

{{ table:val_models }}

The CPU study's own validation numbers, used to fit conformal intervals and to compare positional encodings, are separate:

{{ table:cpu_val }}

### Published reference points (other models)

{{ if exists:results/analysis/headline.json }}
Recomputed from upstream's committed GIFT-Eval result files, plus our Seasonal Naive and `patch_s2` runs. The other rows are other people's models.

{{ table:test_models }}
{{ else }}
Recomputed from upstream's committed GIFT-Eval result files, plus our Seasonal Naive run. These rows are not this repository's model.

{{ table:reported_reference }}
{{ end }}

## What I built and learned

{{ if exists:results/analysis/headline.json }}
I wrote the model, the training loop, and the GIFT-Eval scoring path in plain PyTorch, then trained a smaller model whose weights are in the repository. The {{ results/runs/patch_s2/summary.json#params_total | int }}-parameter run that produced the validation headline never had its checkpoint committed. It was scored on the test split later, from its checkpoint on Drive, with the pre-registered settings.
{{ else }}
I wrote the model, the training loop, and the GIFT-Eval scoring path in plain PyTorch, then trained a smaller model whose weights are in the repository. The {{ results/runs/patch_s2/summary.json#params_total | int }}-parameter run that produced the validation headline never had its checkpoint committed, so I did not invent a test number for it.
{{ end }}

Seed-averaged validation CRPS is {{ results/analysis/cpu_study.json#validation.summary.crps_gmean | .3f }}× Seasonal Naive ({{ results/analysis/cpu_study.json#validation.n_configs }} configurations, 95% CI [{{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.1 | .3f }}]). The same statistic on the test split is {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}×. The three seed-level test ratios are {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.1 | .3f }} and {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.2 | .3f }}. Their mean is {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.mean | .3f }}, and the 95% Student-t interval on that mean is [{{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.1 | .3f }}]. That interval belongs to the mean of the three seed-level geometric means. It is not the interval on the seed-averaged ratio in the table, which is a paired bootstrap over configurations. I kept all three seeds.

Split conformal, one pooled expansion per seed fit only on validation, widens the intervals and pushes the test 80% coverage from {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }}, above the nominal 80% level. CRPS moves from {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.crps_gmean | .3f }} (worse than Seasonal Naive; the configuration interval is entirely above 1). Seed 0's expansions, in mean-absolute-scaled units, are {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.50 | .3f }} at 50%, {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.80 | .3f }} at 80% and {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.90 | .3f }} at 90%. The validation conformal row is scored on the same forecasts the expansions were fit on, so that coverage is not a held-out check. The test table is the held-out calibration result.

Turning rotary positions off, seed 0 only, on the same {{ results/analysis/cpu_study.json#validation_ablation.n_configs }} validation configurations, moves CRPS from {{ results/analysis/cpu_study.json#validation_ablation.rope.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.crps_gmean | .3f }} and MASE from {{ results/analysis/cpu_study.json#validation_ablation.rope.mase_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.mase_gmean | .3f }}. That run was not scored on the test split.

On one held-out hourly series (`bdg-2_bear`, context {{ results/analysis/attention_cpu_s0.json#context_length }} steps, patches of length {{ results/analysis/attention_cpu_s0.json#profile.patch_len }}), the last query puts {{ results/analysis/attention_cpu_s0.json#profile.mass_last_patch | .3f }} of its attention on the most recent patch and {{ results/analysis/attention_cpu_s0.json#profile.mass_last_4_patches | .3f }} on the last four. The peak is at lag {{ results/analysis/attention_cpu_s0.json#profile.peak_lag_patches }} patches. Entropy is {{ results/analysis/attention_cpu_s0.json#profile.entropy_nats | .3f }} nats against {{ results/analysis/attention_cpu_s0.json#profile.entropy_uniform_nats | .3f }} for a uniform distribution over the patches. The mass is local. The series is hourly and each patch is {{ results/analysis/attention_cpu_s0.json#profile.patch_len }} steps, so a day is not patch-aligned and I am not reading a daily cycle out of that peak.

## Market volatility baselines

Daily realized volatility of BTCUSDT and ETHUSDT, 22-day forecasts, designed in `PREREGISTRATION_VOL.md`. Statistical forecast scores only. Our model was not run on this study.

{{ table:vol_overall }}

## Method in brief

* **Data.** A pinned subset of GiftEvalPretrain, whole datasets held out for validation, plus Gaussian-process series. A content check removed exact and rescaled copies of GIFT-Eval series (`DATA_CARD.md`).
* **Model.** Decoder-only transformer in plain PyTorch. The committed CPU model uses patch length {{ results/runs/cpu_s0_seed0/summary.json#config.model.patch_len }}, horizon {{ results/runs/cpu_s0_seed0/summary.json#config.model.horizon }}, width {{ results/runs/cpu_s0_seed0/summary.json#config.model.d_model }}, {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_layers }} layers and {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_heads }} heads (`MODEL_CARD.md`). The T4 headline used patch length {{ results/runs/patch_s2/summary.json#config.model.patch_len }} and a horizon of {{ results/runs/patch_s2/summary.json#config.model.horizon }}.
* **Training.** The CPU study is {{ results/runs/cpu_s0_seed0/summary.json#config.train.steps | int }} steps, batch {{ results/runs/cpu_s0_seed0/summary.json#config.train.batch_size }}, fp32. Seed 0 finished in {{ results/runs/cpu_s0_seed0/summary.json#session_seconds | .1f }} seconds. The T4 loop supports mixed precision and resumable checkpoints.
* **Evaluation.** GluonTS, the same way upstream scores GIFT-Eval: MASE and quantile-loss CRPS, each divided by Seasonal Naive, geometric mean over configurations, plus 50/80/90% coverage.

## Pre-registration

`PREREGISTRATION_CPU.md` fixes the CPU study. `PREREGISTRATION.md` fixes the T4 study. `DEVIATIONS.md` records what the T4 budget cut, including that `patch_s2` was not test-scored before the deadline.

## Limits

{{ if exists:results/analysis/headline.json }}
The CPU model does not beat AutoARIMA on the test split, and its CRPS against Seasonal Naive is not distinguishable from 1 under the pre-registered configuration interval. Raw intervals undercover. Pooled conformal overshoots and hurts CRPS. `patch_s2` has a test score from one seed only, so its interval covers configurations, not training seeds. Nothing here measures economic value.
{{ else }}
The CPU model does not beat AutoARIMA on the test split, and its CRPS against Seasonal Naive is not distinguishable from 1 under the pre-registered configuration interval. Raw intervals undercover. Pooled conformal overshoots and hurts CRPS. `patch_s2` still has no test score. Nothing here measures economic value.
{{ end }}

## License

Code in this repository is Apache-2.0 (`LICENSE`). Derived Binance volatility files under `results/finance/` carry the licence stated in `results/finance/README.md`.
