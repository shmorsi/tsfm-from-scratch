"""Generate the Colab notebooks (kept as code so they are reviewable and diffable).

* ``notebooks/colab_runner.ipynb``: the resumable GPU queue (training, validation scoring).
* ``notebooks/score_patch_s2_test.ipynb``: the one pre-registered test-split score of
  the T4 headline model, from its checkpoint on Google Drive.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# tsfm Colab runner

Runs the GPU work as a resumable queue (`scripts/colab_queue.py`). Checkpoints, logs and
small result files go to Google Drive; if the session disconnects, run all cells again
and the queue continues where it stopped (training resumes exactly from the last
checkpoint, written every <= 10 minutes).

* Phase **A**: throughput benchmark, scaling grid and bins model (train + validation
  eval each), FlowState-9.1M on the test split.
* Phase **B**: after the headline model is chosen on the validation split and that
  choice is committed: seeds 1-2 of the headline config, then one test-split scoring
  of each seed.""",
    ),
    (
        "code",
        """# Parameters
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
!uv sync --frozen --extra cu126 -q
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


SCORE_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Score `patch_s2` on the GIFT-Eval test split (once)

`patch_s2` (10,083,392 parameters) was chosen on the validation split by the rule in
`PREREGISTRATION.md` section 4. Its checkpoint stayed on Google Drive, so it has no
test score yet. This notebook gives it the pre-registered one.

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
`PREREGISTRATION_CPU.md` does not authorise or forbid this run; the CPU study is a
separate model with its own test score.

**Steps.** Set `CHECKPOINT_PATH` below, then Runtime -> Run all on a T4 GPU runtime.
Cell 5 checks the checkpoint and stops before the test split if it is not `patch_s2`.
After a disconnect, Run all again: finished configurations are kept on Drive. The last
cell prints the result files with SHA-256 for `scripts/harvest_colab_output.py`.""",
    ),
    (
        "code",
        """# Parameters
# ============================================================================
# CHECKPOINT_PATH: the only value to set. The patch_s2 run folder on Drive.
# ckpt.pt (training checkpoint) or model.pt (inference weights) both work: the
# script checks the run config and the 10,083,392 parameters, then scores the
# file whose SHA-256 matches the validation run.
CHECKPOINT_PATH = "/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt"
# ============================================================================
REPO = "shmorsi/tsfm-from-scratch"
COMMIT = "main"  # or a full commit SHA
WORK = "/content/drive/MyDrive/tsfm/score_patch_s2"  # resumable outputs""",
    ),
    (
        "code",
        """# 1. GPU check
import subprocess

import torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU"
print(torch.cuda.get_device_name(0))""",
    ),
    (
        "code",
        """# 2. Mount Drive (checkpoint in, resumable results out)
import os

from google.colab import drive

drive.mount("/content/drive")
assert os.path.isfile(CHECKPOINT_PATH), f"CHECKPOINT_PATH not found: {CHECKPOINT_PATH}"
os.makedirs(WORK, exist_ok=True)""",
    ),
    (
        "code",
        """# 3. Clone the repo at COMMIT, with tags (tsfm eval checks for prereg-v1)
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
        """# 4. Install (uv, locked dependencies, CUDA 12.6 PyTorch build)
!curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
import os

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["TSFM_HOME"] = "/content/tsfm_home"
%cd /content/tsfm
!uv sync --frozen --extra cu126 -q
!uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'""",
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
        """# 7. Result files with SHA-256. Copy this output into a text file, then, in the repo:
#    uv run python scripts/harvest_colab_output.py <that file> && uv run tsfm docs build
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
