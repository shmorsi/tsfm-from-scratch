# Vendored from SalesforceAIResearch/gift-eval

Upstream commit: `577a65cfa3acb812405c00d989230bf59f7f8a3d` (2026-10-09). Apache-2.0.

| File | Upstream path | Use |
|---|---|---|
| `gift_eval_data.py` | `src/gift_eval/data.py` | official split / horizon / window logic (unmodified) |
| `dataset_configurations.csv` | `scripts/dataset_configurations.csv` | the 97 official configs |
| `dataset_properties.json` | `notebooks/dataset_properties.json` | domain, frequency, num_variates |
| `upstream_results/*` | `results/*` | upstream baseline and checkpoint scores, for the reproduction check and as "reported" references |

`tests/test_vendor.py` checks the SHA-256 of each vendored file against `UPSTREAM_SHA256`
so local edits cannot slip in unnoticed.
