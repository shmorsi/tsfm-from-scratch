# Pre-registration

Status: **binding** from the git tag `prereg-v1` (2026-10-14). Any later change is
listed under "Deviations" with a date and a reason; the tagged text is never edited.

Written before any model has been trained on the full corpus, and before any model
trained in this repository has been scored on the GIFT-Eval test split. The only
test-split result that exists at the tag is our Seasonal Naive
(`results/eval/test/seasonal_naive/`), which reproduces upstream's published Seasonal
Naive and serves as the normaliser; it cannot inform choices about our own models. The
published reference checkpoint (FlowState-9.1M) may also be scored before or after the
tag, for the same reason.

## 1. Question

How does a small decoder-only transformer, pretrained from scratch only on the
GiftEvalPretrain corpus subset and synthetic Gaussian-process series, compare with
classical statistical baselines and with a published foundation model of similar size on
the GIFT-Eval benchmark, zero-shot?

## 2. Data and splits

* **Training:** the corpus in `results/data/corpus_manifest.json` (pinned upstream
  revision, SHA-256 per file) plus the synthetic series in
  `results/data/synthetic_manifest.json`, after removal of every series flagged by the
  leakage check (`results/data/leakage.json` must report zero hits).
* **Validation (selection only):** the ten whole datasets held out of pretraining
  (`configs/data.yaml: validation_datasets`) and the held-out synthetic series, scored
  with GIFT-Eval's rules (`tsfm.eval.valsplit`). All model and hyper-parameter selection
  uses this split only.
* **Test:** the 97 official GIFT-Eval configurations at Hugging Face revision
  `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`, with upstream's split logic (vendored,
  hash-checked). GIFT-Eval's train/validation splits are never used for training or
  selection, so our models are "zero-shot" in the leaderboard's sense.

## 3. Metrics

Computed by GluonTS `evaluate_model` with exactly upstream's arguments and metric list:

* **MASE** = `MASE[0.5]` (median forecast).
* **CRPS** = `mean_weighted_sum_quantile_loss` over quantiles 0.1..0.9 (a quantile
  approximation of CRPS, as on the leaderboard).
* Each is divided by Seasonal Naive's value on the same configuration; the aggregate is
  the **geometric mean over the 97 configurations**. Our Seasonal Naive reproduces
  upstream's published Seasonal Naive on all 11 official metrics and all 97
  configurations (relative difference < 1e-9, `tests/test_eval.py`); if that test ever
  fails, no comparison is reported.
* The official metrics see only the "mean" key and the nine official quantile levels of
  each forecast (as for leaderboard models); coverage is computed from all levels.
* Secondary: average rank (by CRPS, among the models in our comparison), and empirical
  coverage of the central 50/80/90% intervals pooled over all forecast points.

## 4. Models

* **Ours:** the configs in `configs/scaling/` at this tag: patch-tokenizer sizes
  `patch_s0`..`patch_s4` (~0.8M to ~99M body parameters) and `bins_s2` (the 4,096-bin
  tokenizer at the `patch_s2` size and token budget). `patch_s4` is trained only if
  paid A100 compute is approved; the others are planned on a Colab T4.
* **Headline model:** the run with the lowest validation CRPS (geometric mean over the
  validation configurations of CRPS divided by validation Seasonal Naive, computed by
  `tsfm eval --split val` and `tsfm report --split val`) among the runs that were
  actually trained; ties (within 0.5%) go to the smaller model. Its config is then
  trained with seeds 1 and 2 as well (seed 0 is the original run). The choice is
  committed, with the validation table that justifies it, before any of our models is
  scored on the test split.
* **Baselines:**
  * Seasonal Naive (ours; equals upstream's, see above).
  * AutoARIMA and AutoETS: **upstream's published results** (`Auto_Arima`, `Auto_ETS`,
    vendored with SHA-256 pins). Upstream does not publish the settings behind them, so a
    rerun of ours would be a different AutoARIMA. If we also run our own statsforecast
    versions, they are reported as secondary, labelled as ours.
  * FlowState-9.1M (published checkpoint, zero-shot, no test leakage), scored by us with
    upstream's notebook settings through our pipeline (`tools/flowstate/`); upstream's
    published numbers are shown next to ours. If our run is not available, the
    comparison uses upstream's numbers and says so. TimesFM-2.5 is shown as "reported by
    upstream" only.

## 5. Primary comparison and decision rule

**Primary:** headline model vs upstream AutoARIMA on test **CRPS**.

For seed s, R_s = gmean_c(model_s,c / SN_c) / gmean_c(ARIMA_c / SN_c). We report the
mean of R_s over the three seeds, each R_s, and a 95% paired bootstrap interval over
configurations (10,000 resamples of the 97 configurations; per-configuration model
scores averaged over seeds first).

* **"better than AutoARIMA"** if the upper end of the interval is below 1;
* **"worse than AutoARIMA"** if the lower end is above 1;
* otherwise **"no detectable difference"**.

The same rule, labelled secondary, is applied to MASE and to the comparisons with
Seasonal Naive and FlowState-9.1M. Breakdowns by frequency, domain and term are
descriptive: no per-group claims of "better" or "worse" are made from them.

Inference settings for the test split are fixed here: context up to 2,048 points
(`--max-context 2048`); the bins model draws 20 sample paths with a fixed seed; the
patch model rolls out by feeding back medians.

## 6. Test-set discipline

* Each model is scored on the test split once (`tsfm eval` refuses a second run).
  Anything later is written to `results/eval/exploratory/` and labelled exploratory in
  every document.
* Models other than baselines cannot be test-scored unless `prereg-v1` is an ancestor of
  the commit being run.

## 7. What would change the plan

Only a bug that makes a reported number wrong. A fix is listed under "Deviations"; the
affected numbers are recomputed and the original ones are kept and shown alongside.

## Deviations

None yet.
