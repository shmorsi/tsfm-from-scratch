"""Generate notebooks/colab_runner.ipynb (kept as code so it is reviewable and diffable)."""

from __future__ import annotations

import json
from pathlib import Path

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
if not os.path.exists("/content/tsfm_home/processed/packed/synthetic/meta.json"):
    !uv run tsfm data download --which both --workers 16
    !uv run tsfm data build
    !uv run python scripts/fetch_synthetic.py --repo {REPO} --tag data-v1
    !uv run tsfm pack
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


def build() -> dict[str, object]:
    cells = []
    for kind, src in CELLS:
        cell: dict[str, object] = {
            "cell_type": kind,
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


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_runner.ipynb"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print("wrote", out)
