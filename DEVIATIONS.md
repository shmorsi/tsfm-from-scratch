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

That CPU study has now been scored once. The seed-averaged test comparison is in
`results/analysis/cpu_study.json`. It does not fill in a test score for `patch_s2`.

## 2026-10-23 — `patch_s2` was scored on the test split, once

The entries above are kept as they were written. Two statements in them describe the
state before this date and no longer hold:

* the table row "Headline scored once on the GIFT-Eval test split | Not completed", with
  its sentence "No test-set result is claimed for our model";
* the sentence "No test-set number is claimed for it" in the 2026-10-20 entry.

The last sentence of the 2026-10-20 entry, that the CPU study does not fill in a test
score for `patch_s2`, is still true: the score described here came from a separate run.
`PREREGISTRATION_CPU.md` also says that no test-set number is claimed for `patch_s2`.
That file is a pre-registration, it describes the state on its own date, and it is not
edited.

What happened: after the deadline, `patch_s2` received the one test score the
pre-registration allows.

* It was run from `notebooks/score_patch_s2_test.ipynb` on a Colab T4, at commit
  `89c8087`, with the settings fixed in `PREREGISTRATION.md` sections 5 and 6: the 97
  official configurations, context up to 2,048 points, and `prereg-v1` in the commit's
  history. The result files carry the time stamp 2026-10-24 02:30 UTC, which is the
  evening of 2026-10-23 in Montreal. They were committed in `1d5f7b4`.
* The weights that were scored are the ones the validation run recorded. Both
  `results/eval/val/patch_s2/run.json` and `results/eval/test/patch_s2/run.json` carry
  the SHA-256 `85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2`. The
  checkpoint is still not in the repository.
* `results/eval/test/patch_s2/run.json` records a complete run of all 97 configurations,
  not a partial one. The scoring script writes to its own Drive folder and does not read
  the partial outputs of the 2026-10-15 attempt.
* One deviation in the table above still holds: seed 0 only. The interval on the test
  score therefore covers the 97 configurations, not training seeds.

The score, its interval and its verdict are in `results/analysis/headline.json`. The
README, the model card and the research note are generated from that file and state the
one-seed caveat beside the number. `tsfm eval` refuses a second test score for this model.

Not changed by this entry: the volatility study row. Our model has still not been run on
that study. `notebooks/vol_study_patch_s2.ipynb` runs it, once, from the same checkpoint;
when it has been run, that is recorded here with its date.
