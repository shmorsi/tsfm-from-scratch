# Pre-registration: CPU study

Status: **binding for the models named here**, from the commit that adds this file.
The T4 study in `PREREGISTRATION.md` (tag `prereg-v1`) is unchanged. This file does not
re-open it and does not authorise a second test score of any model that already has one.

Written before `cpu_s0` is scored on the GIFT-Eval test split. The design choices below
are also stored, as data, in `results/cpu_study/protocol.json` when the study starts.

## Why this study exists

`patch_s2` was selected on the validation split by the rule in `PREREGISTRATION.md`.
Its checkpoint sha256 is `85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2`,
recorded in `results/eval/val/patch_s2/run.json`. The weights are not in this repository
(training checkpoints lived on a Colab Drive and `*.pt` was gitignored). That checkpoint
therefore cannot be scored on the test split from a clone. No test-set number is claimed
for `patch_s2`, `patch_s1`, `patch_s0`, or `pilot_patch_s0_cpu`. Their published figures
are validation figures.

This study trains a smaller model whose weights are committed, and scores the official
97-configuration test split once.

## Question

On the official GIFT-Eval test split, how does a small decoder-only patch transformer,
trained from scratch on a fixed subset of the leakage-checked pretrain corpus plus
synthetic Gaussian-process series, compare with Seasonal Naive and with upstream
AutoARIMA? Does split-conformal adjustment, fit only on the validation split, move
interval coverage toward the nominal 50/80/90% levels, and what does it do to CRPS?

## What is fixed

* **Code and config.** `configs/cpu_study.yaml`. Three seeds, 0, 1 and 2. The
  architecture is the repository's patch transformer (RMSNorm, RoPE, SwiGLU,
  non-crossing quantile head): width 128, 4 layers, 4 heads, patch length 16, horizon
  32, `max_seq_len` 128. Training is 2,000 steps, batch 32, fp32, CPU.
* **Data.** Training series: the twelve datasets named `TRAIN_DATASETS` in
  `src/tsfm/cpu_study.py`, which are disjoint from `configs/data.yaml`
  `validation_datasets`, converted with the same leakage-exclusion list as the main
  corpus, plus 8,000 synthetic series (length 384, seed 20261020) and 400 synthetic
  validation series used only for the training-loop loss. GIFT-Eval's own train and
  validation windows are not used for training.
* **Validation (not a leaderboard claim).** The ten held-out datasets, scored with
  `tsfm.eval.valsplit`, excluding the synthetic series. Used to fit conformal
  expansions and to report a positional-encoding ablation. Not used to change the
  architecture, the step count, or the learning rate after this file is committed.
* **Ablation.** One extra seed-0 run with rotary embeddings replaced by the identity
  (`cpu_s0_norope`). Validation only. It is not eligible for the test split, whatever
  its validation CRPS is.
* **Conformal.** For each of the three seeds, fit `tsfm.calibrate` on that seed's
  validation forecasts (scaled by the context mean absolute value; central 50/80/90%
  intervals; split-conformal order statistic; monotone repair away from the untouched
  median). Apply that seed's expansions to its test forecasts. No coefficient is
  chosen on the test split.
* **Test.** Score `cpu_s0_seed0`, `cpu_s0_seed1`, `cpu_s0_seed2` and the three
  conformalised names once each, through `tsfm eval` (GluonTS, the same metric list
  as the main protocol), `max_context` 512. A second run is refused.
* **Comparison.** Seed-average per-configuration MASE and CRPS, then the paired
  bootstrap of `PREREGISTRATION.md` section 5 (10,000 resamples of configurations,
  seed 0) against Seasonal Naive and against upstream AutoARIMA. Primary metric:
  CRPS versus AutoARIMA, same decision rule (better / worse / no detectable
  difference). Also report the three seed-level CRPS ratios and a 95% Student-t
  interval across those three numbers. With three seeds that interval is expected to
  be wide; it is not a reason to drop a seed.
* **Attention readout.** One held-out `bdg-2_bear` series, last-layer attention of the
  final query. Descriptive only.

## What this study will not claim

* That `cpu_s0` is the T4 headline model, or that its test score is the test score of
  `patch_s2`.
* That it beats TimesFM, FlowState, or any other published foundation model, unless a
  committed result file shows that comparison under the rule above. Those models are
  not opponents in this study.
* A test score for the no-position ablation.
