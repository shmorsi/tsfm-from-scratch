# Vendored from SalesforceAIResearch/gift-eval

Upstream commit: `577a65cfa3acb812405c00d989230bf59f7f8a3d` (2026-10-09). Apache-2.0.

| File | Upstream path | Use |
|---|---|---|
| `gift_eval_data.py` | `src/gift_eval/data.py` | official split / horizon / window logic (unmodified) |
| `dataset_configurations.csv` | `scripts/dataset_configurations.csv` | the 97 official configs |
| `dataset_properties.json` | `notebooks/dataset_properties.json` | domain, frequency, num_variates |
| `statsforecast_predictor.py` | `notebooks/naive.ipynb` | the statsforecast wrapper behind upstream's Naive / Seasonal Naive rows (verbatim, below a header comment) |
| `upstream_results/*` | `results/*` | upstream baseline and checkpoint scores, for the reproduction check and as "reported" references |

`tests/test_eval.py` checks the SHA-256 of the vendored files listed in its
`UPSTREAM_SHA256` table, so local edits to them cannot slip in unnoticed.
