# Deviations from the pre-registration

Decided on 2026-10-15, before any of our models was scored on the test split. The
pre-registered analysis (`PREREGISTRATION.md`, tag `prereg-v1`) is otherwise unchanged.

| Pre-registered | What was run | Reason |
|---|---|---|
| Scaling grid of 4–6 sizes (`patch_s0` … `patch_s4`) | Three sizes: `patch_s0`, `patch_s1`, `patch_s2` | Compute: the project was limited to a free T4 GPU and a fixed deadline. `patch_s4` needs a paid A100; `patch_s3` (about 5–7 T4 hours) was not trained. |
| Tokenizer comparison at the s2 size (patch vs bins) | `bins_s2` trained but not scored | Its validation scoring (20 sample paths per window) did not finish within the deadline. No tokenizer conclusion is drawn. |
| Headline scored with 3 seeds | Seed 0 only | Two extra seeds would have needed about 2.6 more T4 hours. With one seed, the paired bootstrap over configurations still runs, but it covers config-to-config variation only, not seed-to-seed variation. |
| Fine-tuned model in the volatility study | Zero-shot first; fine-tuned only if time allowed | Deadline. |

The headline was chosen by the pre-registered rule (lowest validation CRPS among the
runs that have validation results): see `results/analysis/headline_choice.json`.
