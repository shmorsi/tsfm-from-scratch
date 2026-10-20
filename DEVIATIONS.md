# Deviations from the pre-registration

Decided on 2026-10-15, before any of our models was scored on the test split. The
pre-registered analysis (`PREREGISTRATION.md`, tag `prereg-v1`) is otherwise unchanged.

| Pre-registered | What was run | Reason |
|---|---|---|
| Scaling grid of 4–6 sizes (`patch_s0` … `patch_s4`) | Three sizes: `patch_s0`, `patch_s1`, `patch_s2` | Compute: the project was limited to a free T4 GPU and a fixed deadline. `patch_s4` needs a paid A100; `patch_s3` (about 5–7 T4 hours) was not trained. |
| Tokenizer comparison at the s2 size (patch vs bins) | `bins_s2` trained but not scored | Its validation scoring (20 sample paths per window) did not finish within the deadline. No tokenizer conclusion is drawn. |
| Headline scored with 3 seeds | Seed 0 only | Two extra seeds would have needed about 2.6 more T4 hours. With one seed, the paired bootstrap over configurations still runs, but it covers config-to-config variation only, not seed-to-seed variation. |
| Headline scored once on the GIFT-Eval test split | Not completed | The test scoring started on Colab but the free runtime disconnected partway through, and the project deadline did not allow a restart. Partial per-configuration outputs were never inspected or used. No test-set result is claimed for our model; the test scoring remains pre-registered future work. |
| Our model in the volatility study (zero-shot and fine-tuned) | Not run | Same deadline; the study's baselines (random walk, HAR-RV) were run as designed. |

The headline was chosen by the pre-registered rule (lowest validation CRPS among the
runs that have validation results): see `results/analysis/headline_choice.json`.

## 2026-10-20 — the T4 checkpoint is not in the repository

`patch_s2`'s weights were never committed (sha256
`85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2` in
`results/eval/val/patch_s2/run.json`; the file lived on Colab Drive). A clone cannot
score that checkpoint. No test-set number is claimed for it.

A separate, smaller study is pre-registered in `PREREGISTRATION_CPU.md` and
`configs/cpu_study.yaml`. It trains its own model, commits the weights, and scores
the official test split once. It does not change the T4 validation tables and it is
not a second look at `patch_s2`.
