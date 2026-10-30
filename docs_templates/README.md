# tsfm

**A time-series foundation model, written and pretrained from scratch.** A decoder-only transformer in plain PyTorch, pretrained on a pinned corpus and scored zero-shot on GIFT-Eval under a protocol tagged in git before the test split was opened.

**[Case study](https://shmorsi.github.io/projects/tsfm.html)** · [Portfolio](https://shmorsi.github.io/) · [Research note](RESEARCH_NOTE.md) · [Model card](MODEL_CARD.md) · [Data card](DATA_CARD.md)

[![CI](https://img.shields.io/github/actions/workflow/status/shmorsi/tsfm-from-scratch/ci.yml?branch=main&label=ci&labelColor=000000&color=000000)](https://github.com/shmorsi/tsfm-from-scratch/actions/workflows/ci.yml)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-000000?labelColor=000000)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-000000?labelColor=000000)](pyproject.toml)
[![Open in Colab](https://img.shields.io/badge/colab-score%20patch__s2-000000?labelColor=000000&logo=googlecolab)](https://colab.research.google.com/github/shmorsi/tsfm-from-scratch/blob/main/notebooks/score_patch_s2_test.ipynb)

{{ if exists:results/analysis/headline.json }}
![Pipeline from a pinned corpus through a leakage audit and a decoder-only transformer. The headline is chosen on the validation split. The test split has a stored verdict.](docs/figures/hero.svg)
{{ else }}
![Pipeline from a pinned corpus through a leakage audit and a decoder-only transformer. The headline is chosen on the validation split. The test split is pending.](docs/figures/hero.svg)
{{ end }}

## Key results

{{ if exists:results/analysis/headline.json }}
* **Zero-shot forecasting.** `patch_s2` ({{ results/runs/patch_s2/summary.json#params_total | M }} parameters) scores test CRPS {{ results/analysis/headline.json#comparisons.0.ratio | .3f }}× upstream AutoARIMA (95% CI [{{ results/analysis/headline.json#comparisons.0.ci95.0 | .3f }}, {{ results/analysis/headline.json#comparisons.0.ci95.1 | .3f }}], {{ results/analysis/headline.json#comparisons.0.verdict }}), the pre-registered primary comparison. It was selected on validation, where its CRPS was {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive.
{{ else }}
* **Zero-shot forecasting.** `patch_s2` ({{ results/runs/patch_s2/summary.json#params_total | M }} parameters) reaches CRPS {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive on the validation split ({{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed). Its test score is pending. The weights are not in this repository. Steps are under [Blocked on owner](#blocked-on-owner).
{{ end }}
* **Benchmark leakage found.** The public pretraining corpus contained copies of GIFT-Eval series. A content audit flagged {{ results/data/leakage_round1.json#totals.series_with_hits | int }} of {{ results/data/leakage_round1.json#totals.pretrain_series_checked | int }} series; all were excluded, and a re-check flags {{ results/data/leakage.json#totals.series_with_hits | int }}.
* **A gated test split.** `tsfm eval` refuses a model test score unless the `prereg-v1` tag is in the commit's history, and refuses a second one.

{{ table:score_map }}

Lower is better; 1 equals the reference. Verdicts follow the pre-registered rule: better if the interval is entirely below 1, worse if entirely above, otherwise no detectable difference. **Caveat:** the `patch_s2` test score is one seed, so its interval covers configurations, not training seeds, and its weights are not in this repository. The small CPU model that is committed is worse than AutoARIMA.

## How it works

![One pre-norm decoder block: RMSNorm, causal attention with rotary positions, a residual add, RMSNorm, SwiGLU, a residual add, then a quantile head or token logits. The patch quantile head is the one that was selected.](docs/figures/architecture.svg)

Decoder-only, pre-norm, in plain PyTorch: RMSNorm, rotary positions, SwiGLU, a key/value cache, and a non-crossing quantile head. A second tokenizer quantises the scaled series into a fixed vocabulary and is trained with cross-entropy. The head that was selected is the patch head. Details are in `MODEL_CARD.md`.

* **Data.** A pinned subset of GiftEvalPretrain, whole datasets held out for validation, plus Gaussian-process series. A content check removed exact and rescaled copies of GIFT-Eval series (`DATA_CARD.md`).
* **Committed CPU model.** Patch length {{ results/runs/cpu_s0_seed0/summary.json#config.model.patch_len }}, horizon {{ results/runs/cpu_s0_seed0/summary.json#config.model.horizon }}, width {{ results/runs/cpu_s0_seed0/summary.json#config.model.d_model }}, {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_layers }} layers and {{ results/runs/cpu_s0_seed0/summary.json#config.model.n_heads }} heads. Training is {{ results/runs/cpu_s0_seed0/summary.json#config.train.steps | int }} steps, batch {{ results/runs/cpu_s0_seed0/summary.json#config.train.batch_size }}, fp32. Seed 0 finished in {{ results/runs/cpu_s0_seed0/summary.json#session_seconds | .1f }} seconds.
* **T4 headline.** `patch_s2` uses patch length {{ results/runs/patch_s2/summary.json#config.model.patch_len }} and a horizon of {{ results/runs/patch_s2/summary.json#config.model.horizon }}. The training loop supports mixed precision and resumable checkpoints.
* **Evaluation.** GluonTS, the same way upstream scores GIFT-Eval: MASE and quantile-loss CRPS, each divided by Seasonal Naive, geometric mean over configurations, plus 50/80/90% coverage. Our Seasonal Naive reproduces theirs on every configuration.

## Data integrity

![Leakage audit of the public pretraining corpus: series flagged on the first pass, removed, then re-checked. The check finds exact and rescaled copies.](docs/figures/integrity.svg)

The public pretraining corpus, described by its authors as non-leaking, contains copies of benchmark series, some inside the test windows. The audit finds exact and rescaled copies; `DATA_CARD.md` lists what was removed.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```bash
uv sync --locked --extra cpu
uv run pytest -m "not slow and not network and not data"
uv run tsfm pipeline --config configs/tiny_cpu.yaml
```

`uv sync --locked` installs the pinned lockfile, including the CPU PyTorch build. The pipeline downloads a small pinned data subset (network required), then trains and scores a tiny model. It writes to a temporary directory, outside `results/`.

## Repository map

```text
src/tsfm/model.py        decoder-only transformer: RMSNorm, RoPE, SwiGLU, KV cache, quantile head
src/tsfm/tokenizers.py   patch tokenizer and the fixed-vocabulary bins tokenizer
src/tsfm/train/          training loop, objectives, resumable checkpoints, GPU bench
src/tsfm/data/           pinned corpus download, conversion, synthetic series, leakage audit
src/tsfm/eval/           GIFT-Eval scoring (GluonTS), test-split gate, vendored upstream results
src/tsfm/calibrate.py    split-conformal intervals fit on validation only
src/tsfm/finance/        realized-volatility baselines (BTCUSDT, ETHUSDT)
src/tsfm/docs/           generator for this README, the cards, the note and the figures
configs/                 data, pilot, scaling and CPU-study configurations
results/                 every committed number the documents cite
locks/                   SHA-256 manifests for source data and the leakage exclusions
notebooks/               Colab notebooks (generated by scripts/make_colab_notebook.py)
tests/                   unit tests, including the guard against unbacked numbers
```

## Documents

| document | what it is |
|---|---|
| [RESEARCH_NOTE.md](RESEARCH_NOTE.md) | The study: question, data, method, results, scope |
| [MODEL_CARD.md](MODEL_CARD.md) | Models, intended use, metrics, compute |
| [DATA_CARD.md](DATA_CARD.md) | Corpus, held-out datasets, the leakage audit |
| [PREREGISTRATION.md](PREREGISTRATION.md) | T4 study protocol, tagged `prereg-v1` |
| [PREREGISTRATION_CPU.md](PREREGISTRATION_CPU.md) · [PREREGISTRATION_VOL.md](PREREGISTRATION_VOL.md) | CPU study and volatility-baseline protocols |
| [DEVIATIONS.md](DEVIATIONS.md) | What the compute budget cut from the plan |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Checks, generated files, frozen records |
| [docs/COMMIT_MAP.md](docs/COMMIT_MAP.md) | Lookup from commit hashes recorded in older `results/` files to current ones |

## Full results

Every table below is rendered from `results/` by `tsfm docs build`. Click a heading to expand it.

<details>
<summary><b>Scaling and compute</b></summary>

{{ table:scaling | default:*Scaling fit not available.* }}

![Validation loss against training compute for the patch models. The line is the fitted power law and the band is its paired bootstrap, both read from the committed scaling fit.](docs/figures/scaling.svg)

{{ table:bench }}

`DEVIATIONS.md` records which of these configurations were not trained. The bench is not a forecast of the test-split score.

</details>

{{ if exists:results/analysis/headline.json }}
<details>
<summary><b>Test split, <code>patch_s2</code></b></summary>

Scored once from its Drive checkpoint (`notebooks/score_patch_s2_test.ipynb`), one seed, the settings fixed in `PREREGISTRATION.md`. The first row is the pre-registered primary comparison.

{{ table:headline }}

</details>

{{ end }}
<details>
<summary><b>Test split, committed CPU model, and calibration</b></summary>

On the frozen test split, the committed CPU model ({{ results/runs/cpu_s0_seed0/summary.json#params_total | int }} parameters, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.n }} seeds, {{ results/analysis/cpu_study.json#test.n_configs }} configurations) has CRPS {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}× Seasonal Naive (95% CI over configurations [{{ results/analysis/cpu_study.json#test.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.0.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.0.verdict }}) and {{ results/analysis/cpu_study.json#test.comparisons.2.ratio | .3f }}× upstream AutoARIMA (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.2.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.2.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.2.verdict }}). MASE is {{ results/analysis/cpu_study.json#test.summary.mase_gmean | .3f }}× Seasonal Naive (95% CI [{{ results/analysis/cpu_study.json#test.comparisons.1.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.comparisons.1.ci95.1 | .3f }}]: {{ results/analysis/cpu_study.json#test.comparisons.1.verdict }}).

{{ if exists:results/analysis/headline.json }}
![Empirical coverage against nominal coverage for the committed CPU model on the GIFT-Eval test split. Raw quantiles and split-conformal adjustment, averaged across seeds. The dashed line is perfect calibration.](docs/figures/coverage.svg)
{{ else }}
![Empirical coverage against nominal coverage for the committed CPU model on the GIFT-Eval test split. Raw quantiles and split-conformal adjustment, averaged across seeds. The dashed line is perfect calibration. The headline test score is pending.](docs/figures/coverage.svg)
{{ end }}

Nominal coverage against empirical coverage on the test split, after averaging the three seeds. Raw quantiles cover {{ results/analysis/cpu_study.json#test.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test.summary.coverage_mean.90 | .3f }} at the 50/80/90% levels. Split-conformal adjustment, fit on the validation split only, covers {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.50 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }} / {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.90 | .3f }}. A calibrated model would sit on the diagonal. The conformal 80% point is above the diagonal, and the CRPS comparison below is worse.

{{ table:cpu_test }}

{{ table:cpu_vs_arima }}

Conformal adjustment leaves MASE unchanged, because the median is not moved, and it makes CRPS worse. Reproduce the study on CPU (an eval that already has `all_results.csv` is skipped, so committed test scores are not recomputed):

```bash
uv run tsfm cpu-study --config configs/cpu_study.yaml
```

</details>

<details>
<summary><b>Validation split, the study that selected <code>patch_s2</code></b></summary>

{{ if exists:results/analysis/headline.json }}
The {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive CRPS for `patch_s2` is a validation result: {{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed, {{ results/runs/patch_s2/summary.json#params_total | int }} parameters. Its test score is in `results/analysis/headline.json`.
{{ else }}
The {{ results/analysis/val_summary.json#models.patch_s2.crps_gmean | .3f }}× Seasonal Naive CRPS for `patch_s2` is a validation result: {{ results/analysis/val_summary.json#n_configs }} held-out configurations, one seed, {{ results/runs/patch_s2/summary.json#params_total | int }} parameters. Those weights are not in this repository, so that checkpoint has no test score.
{{ end }}

Three sizes, pretrained from scratch on a free Colab T4 and scored zero-shot on datasets held out of pretraining. This is how `patch_s2` was chosen; it is not a test-set result.

{{ table:val_models }}

The CPU study's own validation numbers, used to fit conformal intervals and to compare positional encodings:

{{ table:cpu_val }}

</details>

<details>
<summary><b>Published reference points (other models)</b></summary>

{{ if exists:results/analysis/headline.json }}
Recomputed from upstream's committed GIFT-Eval result files, plus our Seasonal Naive and `patch_s2` runs. The other rows are other people's models.

{{ table:test_models }}
{{ else }}
Recomputed from upstream's committed GIFT-Eval result files, plus our Seasonal Naive run. These rows are not this repository's model.

{{ table:reported_reference }}
{{ end }}

</details>

<details>
<summary><b>Seeds, conformal, positions, attention</b></summary>

Seed-averaged validation CRPS is {{ results/analysis/cpu_study.json#validation.summary.crps_gmean | .3f }}× Seasonal Naive ({{ results/analysis/cpu_study.json#validation.n_configs }} configurations, 95% CI [{{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#validation.comparisons.0.ci95.1 | .3f }}]). The same statistic on the test split is {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }}×. The three seed-level test ratios are {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.1 | .3f }} and {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.values.2 | .3f }}. Their mean is {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.mean | .3f }}, and the 95% Student-t interval on that mean is [{{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.0 | .3f }}, {{ results/analysis/cpu_study.json#test.crps_gmean_across_seeds.ci95.1 | .3f }}]. That interval belongs to the mean of the three seed-level geometric means. It is not the interval on the seed-averaged ratio in the table, which is a paired bootstrap over configurations. I kept all three seeds.

Split conformal, one pooled expansion per seed fit only on validation, widens the intervals and pushes the test 80% coverage from {{ results/analysis/cpu_study.json#test.summary.coverage_mean.80 | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.coverage_mean.80 | .3f }}, above the nominal 80% level. CRPS moves from {{ results/analysis/cpu_study.json#test.summary.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#test_cqr.summary.crps_gmean | .3f }} (worse than Seasonal Naive; the configuration interval is entirely above 1). Seed 0's expansions, in mean-absolute-scaled units, are {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.50 | .3f }} at 50%, {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.80 | .3f }} at 80% and {{ results/cpu_study/cpu_s0_seed0_cqr.json#cqr.expansion_scaled.90 | .3f }} at 90%.

Turning rotary positions off, seed 0 only, on the same {{ results/analysis/cpu_study.json#validation_ablation.n_configs }} validation configurations, moves CRPS from {{ results/analysis/cpu_study.json#validation_ablation.rope.crps_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.crps_gmean | .3f }} and MASE from {{ results/analysis/cpu_study.json#validation_ablation.rope.mase_gmean | .3f }} to {{ results/analysis/cpu_study.json#validation_ablation.none.mase_gmean | .3f }}. That run was not scored on the test split.

On one held-out hourly series (`bdg-2_bear`, context {{ results/analysis/attention_cpu_s0.json#context_length }} steps, patches of length {{ results/analysis/attention_cpu_s0.json#profile.patch_len }}), the last query puts {{ results/analysis/attention_cpu_s0.json#profile.mass_last_patch | .3f }} of its attention on the most recent patch and {{ results/analysis/attention_cpu_s0.json#profile.mass_last_4_patches | .3f }} on the last four. The peak is at lag {{ results/analysis/attention_cpu_s0.json#profile.peak_lag_patches }} patches. Entropy is {{ results/analysis/attention_cpu_s0.json#profile.entropy_nats | .3f }} nats against {{ results/analysis/attention_cpu_s0.json#profile.entropy_uniform_nats | .3f }} for a uniform distribution over the patches. The mass is local. The series is hourly and each patch is {{ results/analysis/attention_cpu_s0.json#profile.patch_len }} steps, so a day is not patch-aligned and I am not reading a daily cycle out of that peak.

</details>

<details>
<summary><b>Market volatility baselines</b></summary>

Daily realized volatility of BTCUSDT and ETHUSDT, 22-day forecasts, designed in `PREREGISTRATION_VOL.md`. Statistical forecast scores only; our model was not run on this study.

{{ table:vol_overall }}

</details>

## Reproducibility

* **Pre-registration.** `PREREGISTRATION.md` fixes the T4 study (tag `prereg-v1`), `PREREGISTRATION_CPU.md` the CPU study, `PREREGISTRATION_VOL.md` the volatility baselines.
* **Deviations.** `DEVIATIONS.md` records what the T4 budget cut: sizes and seeds not trained, and the bins tokenizer not scored.
* **Locks.** Manifests under `locks/` pin source files and the converted training data by SHA-256.
* **CI.** GitHub Actions runs ruff, the formatter, mypy, the unit tests, the tiny pipeline, and `tsfm docs check`.
* **Numbers.** `README.md`, the cards, the research note and `docs/figures/` are generated by `uv run tsfm docs build` from `docs_templates/` and `results/`. The check fails if a generated document shows a number that is not in `results/` or `docs_templates/allowed_numbers.txt`.

## Limitations

{{ if exists:results/analysis/headline.json }}
Beyond the caveat under Key results: the CPU model's CRPS against Seasonal Naive is not distinguishable from 1, raw intervals undercover, and pooled conformal overshoots and hurts CRPS. The leakage check finds exact and rescaled copies, not noisy or resampled ones. The scaling fit describes only the runs that were trained. Nothing here measures economic value.
{{ else }}
The CPU model does not beat AutoARIMA on the test split, and its CRPS against Seasonal Naive is not distinguishable from 1 under the pre-registered configuration interval. Raw intervals undercover. Pooled conformal overshoots and hurts CRPS. `patch_s2` still has no test score. Nothing here measures economic value.

The leakage check finds exact and rescaled copies, not noisy or resampled ones. The scaling fit describes the runs that were trained. The bins tokenizer was trained and not scored (`DEVIATIONS.md`).
{{ end }}

## Next steps

Not yet built or run; none of this is a result.

* **Tokenization.** The bins head is the language-model path: a fixed vocabulary, sampled token by token. It was trained at the headline width and not scored (`DEVIATIONS.md`). I would finish that comparison before treating the patch head as the architecture.
* **LLM-style scaling.** The pre-registration named more sizes than a free T4 could train. I would add those runs and only then refit, rather than extrapolate the current fit.
* **Language-conditioned forecasting.** Condition the same decoder on a short text description of the series (domain, frequency), scored only under a new pre-registration.
* **Agents.** Expose the forecaster as a tool a caller can ask for a context, a frequency and a baseline, with every test score behind the same pre-registered gate.

{{ if exists:results/analysis/headline.json }}
{{ else }}
## Blocked on owner

One item. The `patch_s2` test score is pending. The validation run's weights are not in this repository, and giving them the pre-registered test score needs a Colab T4. Do not run `notebooks/colab_runner.ipynb` for it: that queue can resume a discarded partial test run, and it would also start the volatility-model stages that `DEVIATIONS.md` records as not run.

The weights file recorded with the validation run has SHA-256 {{ results/eval/val/patch_s2/run.json#model_info.checkpoint_sha256 }}.

No test-split duration is stored. The validation score of this checkpoint stored a `seconds` field on each of its configurations in `results/eval/val/patch_s2/run.json`. That file is not a forecast of the test split, and the T4 training projection in `results/bench/Tesla_T4.json` is not a scoring time. Finished test configurations are written to the Drive work folder. Run all again after a disconnect and the score continues from those files. A completed `all_results.csv` is not scored a second time.

1. Merge this commit to `main`, or set `COMMIT` in the notebook to the branch that contains it. The notebook clones `COMMIT` (default `main`) and scores that revision. The badge at the top opens the notebook on `main`.
2. Put the checkpoint on Google Drive at `/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt`, with `model.pt` beside it. `ckpt.pt` or `model.pt` both work. The script scores the file with the SHA-256 above.
3. Open `notebooks/score_patch_s2_test.ipynb`, choose a T4 GPU runtime, then Runtime -> Run all. The notebook checks that the GPU is a T4 and that the checkpoint is `patch_s2` before it touches the test split.
4. Download `/content/drive/MyDrive/tsfm/score_patch_s2/harvest_patch_s2.txt`. A copied notebook print can truncate, and a truncated file fails the SHA-256 check.
5. On a local clone of the same commit:

```bash
uv run python scripts/harvest_colab_output.py harvest_patch_s2.txt
uv run tsfm docs build
```

6. Commit the files the harvest script wrote and the docs `tsfm docs build` regenerated. Do not hand-edit numbers.

That harvest file is what the doc generators read. It contains `results/eval/test/patch_s2/` (`all_results.csv`, `run.json`, `config.json`), a rebuilt `results/analysis/test_summary.json`, and a new `results/analysis/headline.json`. The harvest script appends `results/sync_log.json`. `results/analysis/cpu_study.json` is not in the file. Ranks in the rebuilt summary include `patch_s2`. CRPS and MASE for the other models are recomputed from the same vendored files.
{{ end }}

## License

Code in this repository is Apache-2.0 (`LICENSE`). Cite it with `CITATION.cff`. Derived Binance volatility files under `results/finance/` carry the licence stated in `results/finance/README.md`.
