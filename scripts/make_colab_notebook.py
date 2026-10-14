"""Generate notebooks/colab_runner.ipynb (kept as code so it is reviewable and diffable)."""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# tsfm Colab runner

Runs one config end to end on a Colab GPU. Re-running the notebook after a disconnect
**resumes exactly** from the last checkpoint on Drive (checkpoints every <= 10 min).

1. Set the parameters in the next cell (commit SHA, config, run name).
2. Runtime -> Run all.
3. When it finishes, run `scripts/sync_results.py` locally to copy results into the repo.""",
    ),
    (
        "code",
        """# Parameters
MODE = "train"             # "train" | "bench" (throughput of the scaling grid) | "flowstate"
REPO = "shmorsi/tsfm-from-scratch"
COMMIT = "main"            # pin a full commit SHA for real runs
CONFIG = "configs/scaling/patch_s2.yaml"
RUN_NAME = ""              # default: config name
DRIVE_ROOT = "/content/drive/MyDrive/tsfm"
TIME_BUDGET_MIN = 600      # stop + checkpoint before Colab's session limit
EVAL_SPLITS = ["val"]      # add "test" only after the pre-registration tag
EXTRA_TRAIN_ARGS = ''      # e.g. '--max-steps-this-session 100'""",
    ),
    (
        "code",
        """# 1. GPU check
import subprocess, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(name, cap, f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PRECISION = "bf16" if cap >= (8, 0) else "fp16"   # T4 (7.5) has no fast bf16
print("precision:", PRECISION)""",
    ),
    (
        "code",
        """# 2. Mount Drive (checkpoints, packed data and results live here)
from google.colab import drive
drive.mount("/content/drive")
import os
os.makedirs(DRIVE_ROOT, exist_ok=True)""",
    ),
    (
        "code",
        """# 3. Clone the repo at COMMIT. For a private repo add a Colab secret GITHUB_TOKEN
#    (read-only, repo scope) via the key icon in the left sidebar.
import os, subprocess
token = ""
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN") or ""
except Exception:
    pass
url = f"https://{token + '@' if token else ''}github.com/{REPO}.git"
if not os.path.exists("/content/tsfm"):
    subprocess.run(["git", "clone", "-q", url, "/content/tsfm"], check=True)
subprocess.run(["git", "-C", "/content/tsfm", "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", "/content/tsfm", "checkout", "-q", COMMIT], check=True)
SHA = subprocess.run(["git", "-C", "/content/tsfm", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("checked out", SHA)""",
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
        """# 5. Data. Packed training files are built once and cached on Drive; each run copies
#    them to local disk (memory-mapped random reads from Drive are slow).
import os, json, shutil, subprocess
os.environ["TSFM_HOME"] = "/content/tsfm_home"          # raw + parquet: local, rebuilt if needed
DRIVE_PACKED = f"{DRIVE_ROOT}/packed"
LOCAL_PACKED = "/content/packed"
need = ["pretrain", "val", "synthetic", "synthetic_val"]
if not all(os.path.exists(f"{DRIVE_PACKED}/{n}/meta.json") for n in need):
    !uv run tsfm data download --which both --workers 16
    !uv run tsfm data build
    # Synthetic series: the exact committed files from the GitHub release (SHA-256 checked);
    # regenerating them would take hours on Colab's CPUs.
    !GITHUB_TOKEN={token} uv run python scripts/fetch_synthetic.py --repo {REPO} --tag data-v1
    !uv run tsfm pack
    shutil.copytree("/content/tsfm_home/processed/packed", DRIVE_PACKED, dirs_exist_ok=True)
for n in need:
    if not os.path.exists(f"{LOCAL_PACKED}/{n}/meta.json"):
        shutil.copytree(f"{DRIVE_PACKED}/{n}", f"{LOCAL_PACKED}/{n}", dirs_exist_ok=True)
os.environ["TSFM_PACKED"] = LOCAL_PACKED
# Compare against the manifests committed in the repo.
man = json.load(open("results/data/corpus_manifest.json"))["files"]
for n in ["pretrain", "val"]:
    src = json.load(open(f"{LOCAL_PACKED}/{n}/meta.json"))["sources"]
    expect = {k.split("/")[-1]: v["sha256"] for k, v in man.items() if k.startswith(n + "/") and "kernelsynth" not in k}
    mismatch = {k for k in expect if src.get(k) != expect[k]}
    print(n, "parquet hashes match repo manifest" if not mismatch else f"MISMATCH: {sorted(mismatch)[:5]}")""",
    ),
    (
        "code",
        """# 6. Train (resumes automatically if a checkpoint exists on Drive)
import os, yaml
assert MODE in ("train", "bench", "flowstate"), MODE
cfg = yaml.safe_load(open(CONFIG))
run = RUN_NAME or cfg["name"]
os.environ["TSFM_RUNS"] = f"{DRIVE_ROOT}/runs"
RUN_DIR = f"{DRIVE_ROOT}/runs/{run}"
cfg["train"]["precision"] = PRECISION
cfg["train"]["device"] = "cuda"
os.makedirs(RUN_DIR, exist_ok=True)
CFG_USED = f"{RUN_DIR}/config.yaml"
if not os.path.exists(CFG_USED):
    yaml.safe_dump(cfg, open(CFG_USED, "w"), sort_keys=False)
if MODE == "train":
    !uv run tsfm train --config "{CFG_USED}" --run-dir "{RUN_DIR}" --time-budget-min {TIME_BUDGET_MIN} {EXTRA_TRAIN_ARGS}""",
    ),
    (
        "code",
        """# 7. Evaluate (val always; test only after the pre-registration tag, once per model)
import os
if MODE == "train" and os.path.exists(f"{RUN_DIR}/model.pt"):
    for split in EVAL_SPLITS:
        !uv run tsfm eval --model "{RUN_DIR}" --name {run} --split {split} --device cuda
    !mkdir -p "{RUN_DIR}/results" && cp -r results/eval "{RUN_DIR}/results/"
elif MODE == "train":
    print("training not finished yet - re-run the notebook to resume")""",
    ),
    (
        "code",
        """# 8. MODE == "bench": time 10 optimiser steps per scaling config on this GPU
if MODE == "bench":
    !uv run tsfm bench --configs configs/scaling/*.yaml --steps 10 --precision {PRECISION}
    !mkdir -p "{DRIVE_ROOT}/bench" && cp -r results/bench "{DRIVE_ROOT}/bench/"
    print("copy results/bench/*.json from Drive into the repo (scripts/sync_results.py --bench)")""",
    ),
    (
        "code",
        """# 9. MODE == "flowstate": score the published FlowState-9.1M checkpoint (test split)
if MODE == "flowstate":
    !bash tools/flowstate/setup.sh
    !GIFT_EVAL=/content/tsfm_home/raw/gifteval uv run tsfm data download --which gifteval
    !GIFT_EVAL=/content/tsfm_home/raw/gifteval FS_SRC=/content/granite-tsfm /content/fs-venv/bin/python tools/flowstate/run_flowstate.py --split test --device cuda
    !mkdir -p '{DRIVE_ROOT}/flowstate' && cp -r results/eval '{DRIVE_ROOT}/flowstate/eval'""",
    ),
    (
        "markdown",
        """Then, locally:

```bash
uv run python scripts/sync_results.py --src "<Drive>/tsfm/runs/<run>"
```

which copies `summary.json`, `metrics.jsonl` and evaluation results into `results/` and
records the SHA-256 of every copied file in `results/sync_log.json`.""",
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
