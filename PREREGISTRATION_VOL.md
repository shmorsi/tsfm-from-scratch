# Pre-registration addendum: realized-volatility study (milestones 8-9)

Binding from the git tag `prereg-vol-v1`. Committed before any model, including the
baselines, has been scored on the test period below. Changes after the tag go under
"Deviations" with a date and a reason.

## Data

* Binance Vision spot 5-minute klines, BTCUSDT and ETHUSDT, 2017-08 to 2026-09
  (monthly archives verified against Binance's published SHA-256; `locks/binance.lock.json`).
  Licence: CC BY-NC-SA 4.0 (Binance Vision dataset terms); used for non-commercial
  research, derived series in `results/finance/` carry the same licence and attribution.
* Target: daily realized volatility in percent, 100 * sqrt(sum of squared 5-minute log
  returns) per UTC day, using only returns between consecutive bars; days with fewer
  than 274 of 288 bars are missing.

## Splits (chronological)

* Fine-tuning train: first day to 2019-06-30. Fine-tuning validation (early stopping
  only): 2019-07-01 to 2019-12-31. Test: forecast origins from 2020-01-01 to the last
  day with 22 days of targets after it; every day is an origin; context = all data up to
  and including the origin.

## Task and scores

* Forecast the next 22 daily values (median and the 13 quantile levels used elsewhere).
* Per origin, pooled over the 22 steps: weighted quantile loss over the nine levels
  0.1..0.9 (as GIFT-Eval's "CRPS"), MAE of the median, and coverage of the central
  50/80/90% intervals. Aggregates pool loss over origins (sum of losses / sum of |y|).

## Models

1. Random walk: median = last value; quantiles from the trailing two-year empirical
   distribution of log(y[t+k] / y[t]) for each step k (the reference model).
2. HAR-RV on log volatility, one direct regression per step, refit every 22 days on
   all history up to the origin; quantiles from in-sample residuals.
3. Our headline model, zero-shot (the checkpoint chosen in `PREREGISTRATION.md`, seed 0).
4. The same model fine-tuned on the fine-tuning train period with the pretraining
   objective (AdamW, learning rate 1e-4, batch 64, at most 300 steps); the step with
   the lowest validation weighted quantile loss (origins every 5 days in the validation
   period) is kept.

## Comparisons and decision rule

* Primary: fine-tuned vs zero-shot, and each vs HAR-RV, on pooled weighted quantile loss
  over both symbols and all test origins. Each model's loss is reported as a ratio to
  the random walk with a 95% moving-block bootstrap interval (blocks of 22 origins).
  "Better than X" requires the interval of the ratio-to-X to lie below 1.
* Regime robustness (milestone 9), labelled by origin date: `covid_2020`
  (2020-02-20 to 2020-04-30), `terra_3ac_2022` (2022-05-01 to 2022-07-31), `ftx_2022`
  (2022-11-01 to 2022-12-31), and `calm` (origins whose 22-day target window touches no
  break). Origins near but outside a break are reported only in the overall numbers.
  Per-regime results (loss ratios and coverage) are descriptive.
* No trading or economic-value claim of any kind is made from these results.

## Deviations

None yet.
