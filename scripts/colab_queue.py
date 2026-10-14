"""Run the GPU work as a resumable queue on Colab.

    python scripts/colab_queue.py --phase A --drive /content/drive/MyDrive/tsfm

Phase A: throughput benchmark, the scaling grid and the bins model (train + validation
eval each), and the published FlowState-9.1M checkpoint on the test split.
Phase B: needs ``results/analysis/headline_choice.json`` committed in the repo (the
pre-registered selection, made on the validation split). Trains seeds 1 and 2 of the
headline config, then scores the three seeds on the test split, once.

Every stage leaves ``<drive>/queue/<stage>.done``; re-running skips finished stages and
training resumes from its checkpoint, so a disconnected session just runs the cell again.
Small result files are mirrored to ``<drive>/export/`` with repo-relative paths, from
where they are copied into the repo (SHA-256 recorded in results/sync_log.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ["patch_s0", "patch_s1", "patch_s2", "bins_s2", "patch_s3"]


def sh(cmd: str, log: Path) -> None:
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")


def status(drive: Path, **kw: object) -> None:
    p = drive / "queue" / "status.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(kw, updated_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    p.write_text(json.dumps(old, indent=1, default=str) + "\n")


def export_run(drive: Path, name: str) -> None:
    run = drive / "runs" / name
    dst = drive / "export" / "results" / "runs" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "metrics.jsonl", "config.yaml"):
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)


def stage(drive: Path, name: str) -> bool:
    """True if the stage still has to run."""
    return not (drive / "queue" / f"{name}.done").exists()


def done(drive: Path, name: str) -> None:
    (drive / "queue" / f"{name}.done").write_text(dt.datetime.now(dt.UTC).isoformat() + "\n")


def run_config(
    drive: Path, cfg_path: Path, name: str, precision: str, log: Path, budget_min: float
) -> bool:
    """Train (or resume) one config. Returns True when finished."""
    import yaml

    run_dir = drive / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    used = run_dir / "config.yaml"
    if not used.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["name"] = name
        cfg["train"]["precision"] = precision
        cfg["train"]["device"] = "cuda"
        used.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sh(
        f'uv run tsfm train --config "{used}" --run-dir "{run_dir}" --time-budget-min {budget_min:.0f}',
        log,
    )
    return (run_dir / "model.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--hours", type=float, default=11.0, help="stop starting new work after this")
    a = ap.parse_args()
    drive = a.drive
    (drive / "queue").mkdir(parents=True, exist_ok=True)
    log = drive / "queue" / "log.txt"
    # Results (bench, evals) are written straight to Drive, so per-config eval progress
    # survives a disconnect and the files land in export/ with repo-relative paths.
    os.environ["TSFM_RESULTS"] = str(drive / "export" / "results")
    t0 = time.time()

    def left_min() -> float:
        return a.hours * 60 - (time.time() - t0) / 60

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    status(drive, phase=a.phase, git_sha=sha, state="running")

    if a.phase == "A":
        if stage(drive, "bench"):
            status(drive, stage="bench")
            sh(
                f"uv run tsfm bench --configs configs/scaling/*.yaml --steps 10 --precision {a.precision}",
                log,
            )
            done(drive, "bench")
        for name in GRID:
            if not stage(drive, f"train_{name}"):
                continue
            if left_min() < 20:
                status(drive, state="time budget reached; re-run to continue")
                return 0
            status(drive, stage=f"train_{name}")
            if not run_config(
                drive,
                ROOT / "configs" / "scaling" / f"{name}.yaml",
                name,
                a.precision,
                log,
                left_min() - 10,
            ):
                status(drive, state=f"{name} not finished; re-run to resume")
                return 0
            sh(
                f'uv run tsfm eval --model "{drive}/runs/{name}" --name {name} --split val --device cuda',
                log,
            )
            export_run(drive, name)
            done(drive, f"train_{name}")
        if stage(drive, "flowstate"):
            status(drive, stage="flowstate")
            sh("bash tools/flowstate/setup.sh", log)
            sh(
                "FS_SRC=/content/granite-tsfm /content/fs-venv/bin/python tools/flowstate/run_flowstate.py --split test --device cuda",
                log,
            )
            done(drive, "flowstate")
    else:
        choice_p = ROOT / "results" / "analysis" / "headline_choice.json"
        if not choice_p.exists():
            raise SystemExit(
                "phase B needs results/analysis/headline_choice.json in the checked-out commit"
            )
        choice = json.loads(choice_p.read_text())
        base = choice["headline"]
        cfg_path = ROOT / "configs" / "scaling" / f"{base}.yaml"
        names = [base]
        for seed in (1, 2):
            name = f"{base}_seed{seed}"
            names.append(name)
            if not stage(drive, f"train_{name}"):
                continue
            if left_min() < 20:
                status(drive, state="time budget reached; re-run to continue")
                return 0
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["seed"] = seed
            tmp = ROOT / "configs" / "scaling" / f"{name}.yaml"
            tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
            status(drive, stage=f"train_{name}")
            if not run_config(drive, tmp, name, a.precision, log, left_min() - 10):
                status(drive, state=f"{name} not finished; re-run to resume")
                return 0
            export_run(drive, name)
            done(drive, f"train_{name}")
        for name in names:
            if not stage(drive, f"test_{name}"):
                continue
            status(drive, stage=f"test_{name}")
            sh(
                f'uv run tsfm eval --model "{drive}/runs/{name}" --name {name} --split test --device cuda',
                log,
            )
            done(drive, f"test_{name}")
    status(drive, state=f"phase {a.phase} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
