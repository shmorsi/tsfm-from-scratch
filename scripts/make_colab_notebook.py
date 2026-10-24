"""Generate the Colab notebooks (kept as code so they are reviewable and diffable).

* ``notebooks/colab_runner.ipynb``: the resumable GPU queue (training, validation scoring).
* ``notebooks/score_patch_s2_test.ipynb``: the one pre-registered test-split score of
  the T4 headline model, from its checkpoint on Google Drive.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


# Printed into the score notebook from committed results, so the parameter count and
# checkpoint hash cannot drift from the validation run.
_PATCH_S2 = _json("results/runs/patch_s2/summary.json")
_PATCH_S2_VAL = _json("results/eval/val/patch_s2/run.json")
PATCH_S2_PARAMS = f"{int(_PATCH_S2['params_total']):,}"
PATCH_S2_SHA = str(_PATCH_S2_VAL["model_info"]["checkpoint_sha256"])
# Training projection only. There is no stored test-split scoring duration.
_BENCH = _json("results/bench/Tesla_T4.json")
_PATCH_S2_BENCH = next(r for r in _BENCH["rows"] if r["config"] == "patch_s2.yaml")
BENCH_HOURS = f"{_PATCH_S2_BENCH['projected_hours']:.2f}"
BENCH_SPS = f"{_PATCH_S2_BENCH['seconds_per_step']:.3f}"
BENCH_STEPS = f"{int(_PATCH_S2_BENCH['configured_steps']):,}"
BENCH_MEM = f"{_PATCH_S2_BENCH['peak_memory_gb']:.2f}"
BENCH_TIMED = str(int(_BENCH["timed_steps"]))
BENCH_PREC = str(_PATCH_S2_BENCH["precision"])

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# tsfm Colab runner

Historical training queue (`scripts/colab_queue.py`). Checkpoints, logs and small result
files go to Google Drive; if the session disconnects, run all cells again and the queue
continues where it stopped (training resumes exactly from the last checkpoint, written
every <= 10 minutes).

This is not the remaining `patch_s2` test score. Use
`notebooks/score_patch_s2_test.ipynb` for that. Phase B of this queue can resume the
discarded partial test outputs from 2026-10-15, and it also starts the volatility-model
stages that `DEVIATIONS.md` records as not run. Run all does nothing until `CONFIRM`
below is set.

* Phase **A**: throughput benchmark, scaling grid and bins model (train + validation
  eval each), FlowState-9.1M on the test split.
* Phase **B**: after the headline model is chosen on the validation split and that
  choice is committed: seeds 1-2 of the headline config, then one test-split scoring
  of each seed.""",
    ),
    (
        "code",
        """# Parameters
# Set CONFIRM only when you mean to continue this historical queue.
CONFIRM = ""
assert CONFIRM == "run-queue", (
    "This queue is not the patch_s2 test score. "
    "Open notebooks/score_patch_s2_test.ipynb on a T4 runtime."
)
PHASE = "B"                # "A" or "B" (see above)
REPO = "shmorsi/tsfm-from-scratch"
COMMIT = "main"            # or a full commit SHA
DRIVE_ROOT = "/content/drive/MyDrive/tsfm"
HOURS = 11.0               # stop starting new work after this (Colab sessions are limited)""",
    ),
    (
        "code",
        """# 1. GPU check
import subprocess

import torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU"
cap = torch.cuda.get_device_capability(0)
print(torch.cuda.get_device_name(0), cap)
PRECISION = "bf16" if cap >= (8, 0) else "fp16"  # T4 (7.5) has no fast bf16
print("precision:", PRECISION)""",
    ),
    (
        "code",
        """# 2. Mount Drive (checkpoints, logs and results live here)
import os

from google.colab import drive

drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)""",
    ),
    (
        "code",
        """# 3. Clone the repo at COMMIT (with tags: test scoring checks the pre-registration tag)
import os
import subprocess

if not os.path.exists("/content/tsfm"):
    subprocess.run(["git", "clone", "-q", f"https://github.com/{REPO}.git", "/content/tsfm"], check=True)
subprocess.run(["git", "-C", "/content/tsfm", "fetch", "-q", "--force", "--tags", "origin"], check=True)
# A branch name follows the remote branch (even across a history rewrite); a SHA or tag is used as is.
remote = f"origin/{COMMIT}"
known = subprocess.run(["git", "-C", "/content/tsfm", "rev-parse", "-q", "--verify", remote], capture_output=True)
ref = remote if known.returncode == 0 else COMMIT
subprocess.run(["git", "-C", "/content/tsfm", "checkout", "-q", "--force", "--detach", ref], check=True)
print(subprocess.run(["git", "-C", "/content/tsfm", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)""",
    ),
    (
        "code",
        """# 4. Install (uv, locked dependencies, CUDA 12.6 PyTorch build)
!curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
import os

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
%cd /content/tsfm
!uv sync --locked --extra cu126 -q
!uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'""",
    ),
    (
        "code",
        """# 5. Data: pinned raw files from Hugging Face (SHA-256 checked), Parquet conversion,
#    synthetic series from the GitHub release (SHA-256 checked), then packing.
import json
import os

os.environ["TSFM_HOME"] = "/content/tsfm_home"
os.environ["TSFM_PACKED"] = "/content/tsfm_home/processed/packed"
# Phase B without extra seeds only scores the test split: GIFT-Eval data is enough.
choice_p = "results/analysis/headline_choice.json"
test_only = PHASE == "B" and os.path.exists(choice_p) and not json.load(open(choice_p)).get("extra_seeds", [1, 2])
if test_only:
    !uv run tsfm data download --which gifteval --workers 16
elif not os.path.exists("/content/tsfm_home/processed/packed/synthetic/meta.json"):
    !uv run tsfm data download --which both --workers 16
    !uv run tsfm data build
    !uv run python scripts/fetch_synthetic.py --repo {REPO} --tag data-v1
    !uv run tsfm pack
if not test_only:
    with open("results/data/corpus_manifest.json") as f:
        man = json.load(f)["files"]
    for n in ["pretrain", "val"]:
        with open(f"/content/tsfm_home/processed/packed/{n}/meta.json") as f:
            src = json.load(f)["sources"]
        expect = {k.split("/")[-1]: v["sha256"] for k, v in man.items() if k.startswith(n + "/") and "kernelsynth" not in k}
        bad = sorted(k for k in expect if src.get(k) != expect[k])
        print(n, "Parquet hashes match the committed manifest" if not bad else f"MISMATCH: {bad[:5]}")""",
    ),
    (
        "code",
        """# 6. Run the queue (re-run this cell, or Run all, after a disconnect)
!uv run python scripts/colab_queue.py --phase {PHASE} --drive {DRIVE_ROOT} --precision {PRECISION} --hours {HOURS}
!cat {DRIVE_ROOT}/queue/status.json""",
    ),
]


SCORE_INTRO = f"""# Score `patch_s2` on the GIFT-Eval test split (once)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmorsi/tsfm-from-scratch/blob/main/notebooks/score_patch_s2_test.ipynb)

**Time, before Run all.** A free Colab T4 can disconnect. Each finished test configuration
is written to the Drive `WORK` folder, and Run all again continues from those files. A
completed `all_results.csv` is not scored again.

No test-split duration is stored. `results/bench/Tesla_T4.json` measured {BENCH_TIMED}
steps and projects **{BENCH_HOURS} hours** to train `patch_s2` ({BENCH_STEPS} configured
steps, {BENCH_SPS} s/step, peak {BENCH_MEM} GB, {BENCH_PREC}). That is a training
projection, not a test-split scoring time. The validation score of this checkpoint
stored a `seconds` field on each configuration in `results/eval/val/patch_s2/run.json`.
Those values are not a forecast of the test split.

`patch_s2` ({PATCH_S2_PARAMS} parameters) was chosen on the validation split by the rule in
`PREREGISTRATION.md` section 4. Its checkpoint stayed on Google Drive, so it has no
test score yet. This notebook gives it the pre-registered one, on a Colab T4. Do not
paste in outputs from another machine. The parameter count and the checkpoint hash
below are read from the committed validation run when this notebook is generated.

The weights file recorded with that run has SHA-256 `{PATCH_S2_SHA}`.
`ckpt.pt` or the sibling `model.pt` both work; the script scores the file with that hash.

**Before Run all**

1. Merge the commit you want scored, or set `COMMIT` below to that branch. The notebook
   clones `COMMIT` (default `main`) and scores that revision.
2. Runtime -> Change runtime type -> T4 GPU.
3. Leave `CHECKPOINT_PATH` when the run folder is already at the default Drive path.
   It has to be that `patch_s2` run, not a neighbour such as `bins_s2`.

**What the pre-registration fixes, and this notebook follows**

* Test split: the 97 official configurations, at the pinned revision (section 2).
* Metrics: GluonTS MASE and CRPS over Seasonal Naive, geometric mean (section 3).
* Inference: context up to 2,048 points, patch roll-out by feeding back medians (section 5).
* Decision rule: CRPS against upstream AutoARIMA, 95% paired bootstrap over
  configurations; better, worse or no detectable difference (section 5).
* Scored once (section 6). `tsfm eval` refuses a second test score, and refuses any
  model test score unless the `prereg-v1` tag is in the commit's history.

**What this run is not.** It is one seed. `DEVIATIONS.md` (2026-10-15) already records
that seeds 1 and 2 were not trained, so the interval covers configurations, not seeds.
It is not a second look: the 2026-10-15 attempt disconnected partway and its partial
outputs were never inspected. This notebook writes to a new folder and never reads them.
Do not use `notebooks/colab_runner.ipynb` for this score. That queue can resume those
partial outputs, and it would also start the volatility-model stages that the same file
records as not run.
`PREREGISTRATION_CPU.md` does not authorise or forbid this run; the CPU study is a
separate model with its own test score.

**After it finishes, on a local clone of the same commit.** Download
`harvest_patch_s2.txt` from the Drive `WORK` folder. A copied notebook print can
truncate, and a truncated file fails the SHA-256 check. Then:

    uv run python scripts/harvest_colab_output.py harvest_patch_s2.txt
    uv run tsfm docs build

Commit the harvested result files and the docs that command regenerates. Do not
hand-edit numbers. The harvest file is what the doc generators read:

* `results/eval/test/patch_s2/all_results.csv`, `run.json` and `config.json`
* `results/analysis/test_summary.json` (rebuilt by `tsfm report`; this replaces the previous file)
* `results/analysis/headline.json` (new)

`scripts/harvest_colab_output.py` also appends `results/sync_log.json`.
`results/analysis/cpu_study.json` is not in the harvest. Ranks inside the rebuilt
`test_summary.json` include `patch_s2`. CRPS and MASE for the other models are
recomputed from the same vendored files.

Re-run the notebook after a disconnect. Finished configurations stay on Drive, and the
script does not score the test split a second time once `all_results.csv` is complete.
"""

SCORE_CELLS: list[tuple[str, str]] = [
    ("markdown", SCORE_INTRO),
    (
        "code",
        f"""# Parameters
# ============================================================================
print(
    "results/bench/Tesla_T4.json training projection for patch_s2: "
    "{BENCH_HOURS} hours ({BENCH_STEPS} configured steps, {BENCH_SPS} s/step, "
    "{BENCH_TIMED} timed steps, peak {BENCH_MEM} GB, {BENCH_PREC}). "
    "Not a test-split scoring time. No test-split duration is stored. "
    "Finished configurations resume from WORK on Drive."
)
# CHECKPOINT_PATH: the patch_s2 run folder on Drive. Set COMMIT as well when the
# revision to score is not on main yet.
# ckpt.pt (training checkpoint) or model.pt (inference weights) both work: the
# script checks the run config and the parameter count, then scores the file
# whose SHA-256 matches the validation run (printed in the cell above).
CHECKPOINT_PATH = "/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt"
# ============================================================================
REPO = "shmorsi/tsfm-from-scratch"
COMMIT = "main"  # or a full commit SHA, or a branch name
WORK = "/content/drive/MyDrive/tsfm/score_patch_s2"  # resumable outputs""",
    ),
    (
        "code",
        """# 1. Mount Drive (checkpoint in, resumable results out)
import os

from google.colab import drive

drive.mount("/content/drive")
assert os.path.isfile(CHECKPOINT_PATH), f"CHECKPOINT_PATH not found: {CHECKPOINT_PATH}"
os.makedirs(WORK, exist_ok=True)""",
    ),
    (
        "code",
        """# 2. Clone the repo at COMMIT, with tags (tsfm eval checks for prereg-v1)
import os
import subprocess

if not os.path.exists("/content/tsfm"):
    subprocess.run(["git", "clone", "-q", f"https://github.com/{REPO}.git", "/content/tsfm"], check=True)
subprocess.run(["git", "-C", "/content/tsfm", "fetch", "-q", "--force", "--tags", "origin"], check=True)
remote = f"origin/{COMMIT}"
known = subprocess.run(["git", "-C", "/content/tsfm", "rev-parse", "-q", "--verify", remote], capture_output=True)
ref = remote if known.returncode == 0 else COMMIT
subprocess.run(["git", "-C", "/content/tsfm", "checkout", "-q", "--force", "--detach", ref], check=True)
print(subprocess.run(["git", "-C", "/content/tsfm", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)""",
    ),
    (
        "code",
        """# 3. Install (uv, locked dependencies, CUDA 12.6 PyTorch build)
!curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
import os

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["TSFM_HOME"] = "/content/tsfm_home"
%cd /content/tsfm
!uv sync --locked --extra cu126 -q
""",
    ),
    (
        "code",
        """# 4. GPU check with the PyTorch this repo pins (not Colab's preinstalled build)
import subprocess

smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(smi.stdout)
probe = subprocess.run(
    [
        "uv",
        "run",
        "python",
        "-c",
        "import torch; name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''; print(torch.__version__); print(torch.cuda.is_available()); print(name)",
    ],
    check=True,
    capture_output=True,
    text=True,
)
print(probe.stdout)
lines = probe.stdout.splitlines()
assert len(lines) >= 3, "GPU probe did not print version, availability and device name"
assert lines[1] == "True", "No GPU: Runtime -> Change runtime type -> T4 GPU"
assert "T4" in lines[2], "Need a T4 GPU: Runtime -> Change runtime type -> T4 GPU. Got: " + lines[2]""",
    ),
    (
        "code",
        """# 5. Check the checkpoint is patch_s2 before anything touches the test split
!uv run python scripts/score_patch_s2.py --checkpoint "{CHECKPOINT_PATH}" --work "{WORK}" --dry-run""",
    ),
    (
        "code",
        """# 6. Score once: GIFT-Eval download (SHA-256 checked), tsfm eval --split test,
#    tsfm report, tsfm headline, tsfm docs build. Re-run after a disconnect.
!uv run python scripts/score_patch_s2.py --checkpoint "{CHECKPOINT_PATH}" --work "{WORK}" --device cuda""",
    ),
    (
        "code",
        """# 7. Harvest file on Drive. Download WORK/harvest_patch_s2.txt (do not copy a
#    truncated print). On a local clone of this COMMIT:
#    uv run python scripts/harvest_colab_output.py harvest_patch_s2.txt
#    uv run tsfm docs build
print("download this file from Drive:", f"{WORK}/harvest_patch_s2.txt")
print(open(f"{WORK}/harvest_patch_s2.txt").read())
!git -C /content/tsfm status --short""",
    ),
]


def build(
    cells_src: list[tuple[str, str]] | None = None, prefix: str = "cell"
) -> dict[str, object]:
    cells = []
    for i, (kind, src) in enumerate(CELLS if cells_src is None else cells_src):
        cell: dict[str, object] = {
            "cell_type": kind,
            "id": f"{prefix}-{i}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, tuple[list[tuple[str, str]], str]] = {
    "colab_runner.ipynb": (CELLS, "cell"),
    "score_patch_s2_test.ipynb": (SCORE_CELLS, "score"),
}


def build_all() -> dict[str, dict[str, object]]:
    return {name: build(cells, prefix) for name, (cells, prefix) in NOTEBOOKS.items()}


if __name__ == "__main__":
    for name, nb in build_all().items():
        out = ROOT / "notebooks" / name
        out.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", out)
