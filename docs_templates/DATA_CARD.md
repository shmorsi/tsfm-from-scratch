# Data card

All numbers on this page are rendered from `results/data/*.json` by `tsfm docs build`.

## Sources

| Source | What we use | License (as stated upstream) | Pinned revision |
|---|---|---|---|
| [Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) | pretraining corpus (subset, below) and held-out validation datasets | Apache-2.0 tag on the collection; "for research purposes only" | `6830b624de7ed2b3d3e5b85bb6959d81dcc5d874` |
| [Salesforce/GiftEval](https://huggingface.co/datasets/Salesforce/GiftEval) | benchmark test data (evaluation only, never trained on) | Apache-2.0 tag on the collection; "for research purposes only" | `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0` |
| Synthetic (this repo) | Gaussian-process kernel mixtures, `tsfm.data.synthetic` | same as this repo | seed in `configs/data.yaml` |

**Licenses, honestly.** Both collections carry a single Apache-2.0 tag on Hugging Face,
but they are compilations of many third-party datasets (Monash archive, LibCity, LargeST,
BuildingsBench, CMIP6, ERA5, cloud traces, ...) that each have their own terms. The
upstream cards do not list per-dataset licenses, and we have **not** verified them one by
one. Treat this corpus as research-only, as upstream asks.

Every raw file is listed with its size and SHA-256 in `locks/pretrain.lock.json` and
`locks/gifteval.lock.json`; downloads that do not match are rejected.

## Selection

From the upstream corpus we take every dataset of at most 1.8 GB whole, and for the very
large families (CMIP6, ERA5, LargeST, BuildingsBench-900k, Borg, Azure, Alibaba) a few
shards only (`configs/data.yaml`). Multivariate series are split into univariate ones.
Each dataset is capped at a fixed number of observed points (series sampled with a fixed
seed; a series that crosses the cap keeps its most recent points), so no single dataset
dominates.

Series are dropped if they are empty, contain infinities, are constant, are mostly
missing, have too few observed points, or contain values above a magnitude limit.

| | datasets | series | points |
|---|---|---|---|
| pretraining | {{ results/data/corpus_manifest.json#totals.pretrain.datasets | int }} | {{ results/data/corpus_manifest.json#totals.pretrain.series | int }} | {{ results/data/corpus_manifest.json#totals.pretrain.points | int }} |
| validation (held out) | {{ results/data/corpus_manifest.json#totals.val.datasets | int }} | {{ results/data/corpus_manifest.json#totals.val.series | int }} | {{ results/data/corpus_manifest.json#totals.val.points | int }} |
| synthetic (train) | | {{ results/data/synthetic_manifest.json#files.[pretrain/kernelsynth.parquet].series_kept | int | default:pending }} | {{ results/data/synthetic_manifest.json#files.[pretrain/kernelsynth.parquet].points_kept | int | default:pending }} |
| synthetic (validation) | | {{ results/data/synthetic_manifest.json#files.[val/kernelsynth.parquet].series_kept | int | default:pending }} | {{ results/data/synthetic_manifest.json#files.[val/kernelsynth.parquet].points_kept | int | default:pending }} |

The validation datasets are whole datasets held out of pretraining. They are used for
model selection and scaling fits only. GIFT-Eval's own train/validation splits are never
used, so models trained here qualify as "zero-shot" under the leaderboard's definition.

## Leakage check

Upstream calls the pretraining corpus "non-leaking". We check that on the content of the
series (`tsfm.data.leakage`), comparing every pretraining and validation series against
every GIFT-Eval series, including the history before the test windows:

* **native frequency:** z-normalised windows, rounded and hashed. Any shared segment of
  at least {{ results/data/leakage.json#params.min_shared_points_guaranteed | int | default:pending }}
  points is found with certainty, also after rescaling or shifting.
* **daily aggregates:** sub-daily series on both sides averaged per calendar day; any
  shared stretch of at least {{ results/data/leakage.json#params.min_shared_days_guaranteed | int | default:pending }}
  complete days is found with certainty, which catches a GIFT-Eval series stored at
  another sampling rate in the corpus.

### Round 1: what the check found

The first run flagged {{ results/data/leakage_round1.json#totals.series_with_hits | int }} of
{{ results/data/leakage_round1.json#totals.pretrain_series_checked | int }} series
({{ results/data/leakage_round1.json#totals.native_hits | int }} native and
{{ results/data/leakage_round1.json#totals.daily_hits | int }} daily-aggregate window hits).
For each flagged series we located the matching GIFT-Eval window and measured how far the
two series keep agreeing under the same affine map (`scripts/inspect_leakage.py`):

{{ table:leakage_inspection | default:*pending* }}

A series that matched two GIFT-Eval datasets is listed under both. Test-window hits are
counted at native frequency only (daily-aggregate hits are not attributed to windows).

So the upstream corpus is **not** free of GIFT-Eval data: several datasets contain exact
(or rescaled) copies of GIFT-Eval series, some of them inside the test windows. The
"coincidence" rows are different energy-meter series that only share straight-line
stretches (linearly interpolated gaps) with SZ_TAXI; they are not copies. Every series
flagged in round 1 is excluded from our corpus whatever its verdict
(`locks/leakage_exclusions.json`), and straight-line windows are ignored from round 2 on.

### Final corpus

{{ results/data/leakage.json#totals.pretrain_windows_checked | int | default:pending }}
native windows and {{ results/data/leakage.json#totals.pretrain_daily_windows_checked | int | default:pending }}
daily windows checked against {{ results/data/leakage.json#totals.gifteval_windows | int | default:pending }}
and {{ results/data/leakage.json#totals.gifteval_daily_windows | int | default:pending }} GIFT-Eval
windows; native hits {{ results/data/leakage.json#totals.native_hits | int | default:pending }},
daily hits {{ results/data/leakage.json#totals.daily_hits | int | default:pending }}; series excluded in
total: {{ results/data/leakage.json#totals.excluded_series_total | int | default:pending }}.

{{ table:leakage_matches | default: }}

**Not detected by this check:** copies with added noise, warped or resampled time axes
other than whole days, and overlaps shorter than the guaranteed lengths above. A
zero-hit result therefore means "no exact or affine copies found", not "no related data".

## Per-dataset table

{{ table:corpus | default:*pending* }}
