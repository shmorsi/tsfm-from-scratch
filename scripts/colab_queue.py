"""Run the GPU work as a resumable queue on Colab.

    python scripts/colab_queue.py --phase A --drive /content/drive/MyDrive/tsfm

Phase A: throughput benchmark, the scaling grid and the bins model (train + validation
eval each), and the published FlowState-9.1M checkpoint on the test split.
Phase B: needs ``results/analysis/headline_choice.json`` committed in the repo (the
pre-registered selection, made on the validation split). Trains seeds 1 and 2 of the
headline config, scores the three seeds on the test split (once), then runs the
realized-volatility study for the seed-0 checkpoint (zero-shot and fine-tuned).

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


def say(msg: str) -> None:
    print(f"[queue {dt.datetime.now(dt.UTC):%H:%M:%S}] {msg}", flush=True)


def sh(cmd: str, log: Path) -> None:
    t0 = time.time()
    say(f"start: {cmd[:140]}")
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    say(f"exit {r.returncode} after {(time.time() - t0) / 60:.1f} min")
    if r.returncode:
        tail = log.read_text()[-3000:]
        print(tail, flush=True)
        raise SystemExit(f"failed ({r.returncode}): {cmd}")


def dump(drive: Path) -> None:
    """Print every new or changed file under <drive>/export between markers with its
    SHA-256, so results can be copied from the notebook output and verified."""
    import hashlib

    seen_p = drive / "queue" / "dumped.json"
    seen = json.loads(seen_p.read_text()) if seen_p.exists() else {}
    root = drive / "export"
    for p in sorted(root.rglob("*")) if root.exists() else []:
        if not p.is_file() or "/partial/" in str(p):
            continue
        rel = str(p.relative_to(root))
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if seen.get(rel) == sha:
            continue
        print(f"===FILE {rel} {sha}===", flush=True)
        print(data.decode(), end="" if data.endswith(b"\n") else "\n", flush=True)
        print("===END===", flush=True)
        seen[rel] = sha
    seen_p.write_text(json.dumps(seen, indent=1) + "\n")


def status(drive: Path, **kw: object) -> None:
    p = drive / "queue" / "status.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(kw, updated_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    p.write_text(json.dumps(old, indent=1, default=str) + "\n")


def export_run(drive: Path, name: str) -> None:
    """summary.json, config.yaml and a thinned training log (every validation point and
    every 500th step) for the repo; the full log stays on Drive with the checkpoint."""
    run = drive / "runs" / name
    dst = drive / "export" / "results" / "runs" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "config.yaml"):
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
    if (run / "metrics.jsonl").exists():
        keep = [
            ln
            for ln in (run / "metrics.jsonl").read_text().splitlines()
            if '"val_loss"' in ln or json.loads(ln)["step"] % 500 == 0
        ]
        (dst / "metrics_thin.jsonl").write_text("".join(ln + "\n" for ln in keep))


def stage(drive: Path, name: str) -> bool:
    """True if the stage still has to run."""
    return not (drive / "queue" / f"{name}.done").exists()


def done(drive: Path, name: str) -> None:
    (drive / "queue" / f"{name}.done").write_text(dt.datetime.now(dt.UTC).isoformat() + "\n")
    say(f"stage done: {name}")
    dump(drive)


def apply_resets(drive: Path) -> None:
    """Discard runs listed in scripts/queue_resets.json (each reset applied once):
    run directory, done marker and exported results."""
    applied_p = drive / "queue" / "resets_applied.json"
    applied = set(json.loads(applied_p.read_text())) if applied_p.exists() else set()
    for r in json.loads((ROOT / "scripts" / "queue_resets.json").read_text()):
        if r["id"] in applied:
            continue
        for name in r["runs"]:
            for p in (
                drive / "runs" / name,
                drive / "export" / "results" / "runs" / name,
                drive / "export" / "results" / "eval" / "val" / name,
            ):
                shutil.rmtree(p, ignore_errors=True)
            (drive / "queue" / f"train_{name}.done").unlink(missing_ok=True)
        say(f"applied reset {r['id']}: {r['runs']} ({r['reason'][:80]}...)")
        applied.add(r["id"])
    applied_p.write_text(json.dumps(sorted(applied)) + "\n")


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
    # Read the packed corpus into RAM: random window reads from Colab's disk were the
    # bottleneck (T4 benchmark, results/bench/Tesla_T4.json).
    os.environ.setdefault("TSFM_IN_MEMORY", "1")
    t0 = time.time()

    def left_min() -> float:
        return a.hours * 60 - (time.time() - t0) / 60

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    apply_resets(drive)
    status(drive, phase=a.phase, git_sha=sha, state="running")
    say(
        f"phase {a.phase} at {sha[:7]}; finished stages: {sorted(p.stem for p in (drive / 'queue').glob('*.done'))}"
    )
    dump(drive)

    if a.phase == "A":
        if stage(drive, "bench"):
            status(drive, stage="bench")
            sh(
                f"uv run tsfm bench --configs configs/scaling/*.yaml --steps 10 --precision {a.precision}",
                log,
            )
            done(drive, "bench")
        if stage(drive, "profile"):
            # Where does step time go on this GPU? (printed into the notebook output)
            status(drive, stage="profile")
            prof_log = drive / "queue" / "profile.txt"
            sh(
                f"uv run tsfm bench --profile --configs configs/scaling/patch_s3.yaml --steps 5 --precision {a.precision} > {prof_log}",
                log,
            )
            print(prof_log.read_text()[:12000], flush=True)
            done(drive, "profile")
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
        for seed in choice.get("extra_seeds", [1, 2]):
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
            dump(drive)
        # Milestones 8-9 (PREREGISTRATION_VOL.md): the seed-0 headline checkpoint,
        # zero-shot and fine-tuned, on the realized-volatility test period.
        for vname, flag in (("tsfm_zero_shot", ""), ("tsfm_finetuned", "--finetune")):
            if not stage(drive, f"vol_{vname}"):
                continue
            status(drive, stage=f"vol_{vname}")
            sh(
                f'uv run tsfm vol model --model "{drive}/runs/{base}" --name {vname} {flag} --device cuda',
                log,
            )
            done(drive, f"vol_{vname}")
            dump(drive)
    status(drive, state=f"phase {a.phase} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
