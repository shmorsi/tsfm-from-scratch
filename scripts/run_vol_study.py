"""Run the pre-registered realized-volatility study for the T4 headline model ``patch_s2``,
zero-shot and fine-tuned, once (``PREREGISTRATION_VOL.md``, models 3 and 4).

Run from ``notebooks/vol_study_patch_s2.ipynb`` on Colab:

    uv run python scripts/run_vol_study.py --checkpoint /content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt \\
        --work /content/drive/MyDrive/tsfm/vol_study

Steps:

1. Check that the checkpoint is ``patch_s2`` and pick the weights file whose SHA-256 equals
   the one recorded with the validation run (the checks of ``scripts/score_patch_s2.py``).
2. Check that the tag ``prereg-vol-v1`` is in HEAD's history.
3. Check the inputs: the two realized-volatility series in ``results/finance/`` must hash
   to the values recorded when they were built (``rvol_summary.json``). The baselines were
   scored on exactly these files, so the model is too.
4. Stop if the checkout already holds either model run: the test period is scored once.
5. Unless ``--skip-raw-check``: ``tsfm vol data`` downloads the Binance archives again,
   checks them against Binance's published checksums and ``locks/binance.lock.json``, and
   rebuilds the two series, which must come out byte-identical to the committed ones.
6. ``tsfm vol model`` twice, as the queue would have run it: ``tsfm_zero_shot``, then
   ``tsfm_finetuned`` with ``--finetune``. Each run's folder is linked to ``--work``, so a
   finished run survives a disconnect and is not run again.
7. ``tsfm vol report`` over the random walk, HAR-RV and the two runs.
8. Write ``<work>/harvest_vol_patch_s2.txt`` (and the same blocks split into
   ``<work>/parts/``) for ``scripts/harvest_colab_output.py``, and copy the weights that
   were scored to ``<work>/release/`` with their SHA-256, ready to attach to a release.
9. ``tsfm docs build``.

``--dry-run`` stops after step 4 and prints the commands it would run.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_colab_output as H
import score_patch_s2 as P

ROOT = Path(__file__).resolve().parents[1]
PREREG_TAG = "prereg-vol-v1"
RUNS = (("tsfm_zero_shot", ()), ("tsfm_finetuned", ("--finetune",)))
NAMES = tuple(name for name, _ in RUNS)
REPORT_MODELS = ("random_walk", "har", *NAMES)
REFERENCE_ROWS = "results/finance/random_walk/per_origin.csv"
HARVEST_NAME = "harvest_vol_patch_s2.txt"
RELEASE_NAME = "tsfm_patch_s2_seed0_model.pt"
TRACKED_INPUTS = ("results/finance/rvol_summary.json",)


def series_files(root: Path = ROOT) -> dict[str, str]:
    """``results/finance/rvol_<SYM>.csv`` -> the SHA-256 recorded when it was built."""
    summary = json.loads((root / "results" / "finance" / "rvol_summary.json").read_text())
    return {f"results/finance/rvol_{sym}.csv": v["sha256"] for sym, v in summary["symbols"].items()}


def input_problems(root: Path = ROOT) -> list[str]:
    """Reasons why the committed series are not the ones the baselines were scored on."""
    out = []
    for rel, want in series_files(root).items():
        p = root / rel
        if not p.is_file():
            out.append(f"{rel} is missing")
        elif P.sha256(p) != want:
            out.append(f"{rel} has SHA-256 {P.sha256(p)}, rvol_summary.json recorded {want}")
    if not (root / REFERENCE_ROWS).is_file():
        out.append(f"{REFERENCE_ROWS} is missing (the baselines' rows)")
    return out


def git_ok(root: Path = ROOT, tag: str = PREREG_TAG) -> str:
    """HEAD's SHA; stops if the pre-registration tag is not in HEAD's history."""
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=root, capture_output=True
    )
    if r.returncode != 0:
        raise SystemExit(
            f"{tag} is not an ancestor of HEAD (or tags were not fetched). The study is "
            "run only from a commit that contains its pre-registration. Clone with tags."
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def committed_runs(root: Path = ROOT) -> list[str]:
    """Model runs that are already files in the checkout (not links into ``--work``)."""
    out = []
    for name in NAMES:
        p = root / "results" / "finance" / name
        if p.exists() and not p.is_symlink():
            out.append(name)
    return out


def link_run(root: Path, work: Path, name: str) -> Path:
    """Make ``results/finance/<name>`` a link to ``<work>/results/<name>``."""
    out = root / "results" / "finance" / name
    target = work / "results" / name
    target.mkdir(parents=True, exist_ok=True)
    if out.is_symlink():
        if out.resolve() != target.resolve():
            raise SystemExit(f"{out} links to {out.resolve()}, expected {target}")
    elif out.exists():
        raise SystemExit(f"{out} already exists in the checkout; that run is already scored.")
    else:
        out.symlink_to(target, target_is_directory=True)
    return out


def run_done(path: Path) -> bool:
    """``study.save`` writes the rows first and ``run.json`` last."""
    return (path / "per_origin.csv").is_file() and (path / "run.json").is_file()


def tsfm(*args: str) -> list[str]:
    return [sys.executable, "-m", "tsfm.cli", *args]


def raw_check(root: Path, env: dict[str, str]) -> None:
    """Rebuild the series from Binance's archives; they must equal the committed files."""
    want = series_files(root)
    r = subprocess.run(tsfm("vol", "data"), cwd=root, env=env)
    got = {rel: P.sha256(root / rel) for rel in want}
    # ``tsfm vol data`` rewrites tracked files (the summary gets a new time stamp). Put the
    # committed ones back so the model runs are stamped from a clean checkout.
    subprocess.run(["git", "checkout", "--", *want, *TRACKED_INPUTS], cwd=root, check=True)
    if r.returncode != 0:
        raise SystemExit(
            "tsfm vol data failed (a download error, or an archive whose published checksum "
            "no longer matches locks/binance.lock.json). Nothing was scored. The committed "
            "series are untouched; to score on them without this re-check, set "
            "VERIFY_RAW = False in the notebook (--skip-raw-check). The harvest file "
            "records that the re-check was skipped."
        )
    changed = sorted(rel for rel in want if got[rel] != want[rel])
    if changed:
        raise SystemExit(
            f"The archives now give a different series for {changed}. The baselines were "
            "scored on the committed files, so nothing was scored. Report this before "
            "going further."
        )


def harvest_parts(root: Path = ROOT) -> dict[str, str]:
    """The harvest blocks, grouped so that each part can also be read on its own."""
    plain = ["results/finance/summary.json", *(f"results/finance/{n}/run.json" for n in NAMES)]
    parts = {"summary": P.harvest_blocks(root, tuple(plain))}
    ref = (root / REFERENCE_ROWS).read_bytes()
    for name in NAMES:
        rel = f"results/finance/{name}/per_origin.csv"
        parts[f"rows_{name}"] = H.packed_block(rel, (root / rel).read_bytes(), REFERENCE_ROWS, ref)
    return parts


def release_copy(weights: Path, work: Path) -> tuple[Path, str]:
    """Copy the scored weights to ``<work>/release/`` with a ``.sha256`` file beside them."""
    dest = work / "release" / RELEASE_NAME
    sha = P.sha256(weights)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or P.sha256(dest) != sha:
        shutil.copyfile(weights, dest)
    (dest.parent / (RELEASE_NAME + ".sha256")).write_text(f"{sha}  {RELEASE_NAME}\n")
    return dest, sha


LOCAL_FOLLOWUP = f"""
Local follow-up, from a clone of the same commit. Download {HARVEST_NAME} from Drive
(a copied notebook print can truncate, and a truncated file fails the SHA-256 check):

    uv run python scripts/harvest_colab_output.py {HARVEST_NAME}
    uv run tsfm docs build

Commit the files the harvest script wrote and the docs that command regenerated, and add a
dated entry to DEVIATIONS.md (its 2026-10-15 table says the model was not run on this
study). Do not hand-edit numbers.

The harvest file holds:
  results/finance/summary.json                 (rebuilt by tsfm vol report; replaces the previous file)
  results/finance/tsfm_zero_shot/run.json      (new)
  results/finance/tsfm_zero_shot/per_origin.csv (new, packed against the random-walk rows)
  results/finance/tsfm_finetuned/run.json      (new, with the fine-tuning log)
  results/finance/tsfm_finetuned/per_origin.csv (new, packed)
  results/sync_log.json                        (append-only log written by the harvest script)

The folder parts/ beside it holds the same blocks as three smaller files; applying all
three is the same as applying the one file.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="patch_s2 ckpt.pt or model.pt")
    ap.add_argument("--work", type=Path, required=True, help="folder that survives a disconnect")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="run the checks, print the plan")
    ap.add_argument(
        "--skip-raw-check",
        action="store_true",
        help="do not re-download the Binance archives; score on the committed series",
    )
    a = ap.parse_args(argv)

    exp = P.expected()
    weights, notes = P.pick_weights(a.checkpoint, exp)
    for n in notes:
        print("ok:", n, flush=True)
    sha = git_ok(ROOT)
    print(f"ok: {PREREG_TAG} is in the history of {sha[:12]}", flush=True)
    bad = input_problems(ROOT)
    if bad:
        raise SystemExit("The inputs are not the committed series:\n  - " + "\n  - ".join(bad))
    print("ok: both realized-volatility series match rvol_summary.json", flush=True)
    done = committed_runs(ROOT)
    if done:
        raise SystemExit(
            f"results/finance/ already holds {done} in this checkout. The test period is "
            "scored once per model; nothing was run."
        )
    steps = [
        tsfm("vol", "model", "--model", str(weights), "--name", name, *flags, "--device", a.device)
        for name, flags in RUNS
    ]
    report = tsfm("vol", "report", "--models", *REPORT_MODELS)
    docs = tsfm("docs", "build")
    if a.dry_run:
        print(
            f"dry run at {sha[:12]}; would link results/finance/<run> into {a.work / 'results'} and run:"
        )
        if not a.skip_raw_check:
            print(
                "   tsfm vol data   (then require the rebuilt series to equal the committed ones)"
            )
        for s in (*steps, report, docs):
            print("   tsfm", " ".join(s[3:]))
        print(LOCAL_FOLLOWUP)
        return 0

    # Results go to the checkout's results/ (the run folders are links to Drive). An
    # inherited TSFM_RESULTS would send them elsewhere, for example to the old queue export.
    env = {k: v for k, v in os.environ.items() if k != "TSFM_RESULTS"}
    a.work.mkdir(parents=True, exist_ok=True)
    outs = [link_run(ROOT, a.work, name) for name in NAMES]
    timings: dict[str, float] = {}
    raw = "skipped (--skip-raw-check)"
    if all(run_done(o) for o in outs):
        raw = "not repeated (both runs were already complete)"
    elif not a.skip_raw_check:
        t0 = time.time()
        print("$ tsfm vol data", flush=True)
        raw_check(ROOT, env)
        timings["raw re-check"] = time.time() - t0
        raw = "passed: archives match Binance's checksums and the lock, rebuilt series identical"
    print("raw re-check:", raw, flush=True)
    for (name, _), cmd, out in zip(RUNS, steps, outs, strict=True):
        if run_done(out):
            print(f"{name} is already complete in {out.resolve()}; not scoring again", flush=True)
            continue
        t0 = time.time()
        print("$", "tsfm", " ".join(cmd[3:]), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
        timings[name] = time.time() - t0
    print("$", "tsfm", " ".join(report[3:]), flush=True)
    subprocess.run(report, cwd=ROOT, check=True, env=env)

    parts = harvest_parts(ROOT)
    header = "\n".join(
        [
            "Realized-volatility study, patch_s2 seed 0 (PREREGISTRATION_VOL.md, models 3 and 4).",
            f"written_utc: {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}",
            f"commit: {sha}",
            f"weights_sha256: {exp.checkpoint_sha256}",
            f"device: {a.device}",
            f"raw re-check: {raw}",
            *(f"seconds, {k}: {v:.0f}" for k, v in timings.items()),
            "Apply with: uv run python scripts/harvest_colab_output.py " + HARVEST_NAME,
            "",
            "",
        ]
    )
    harvest = a.work / HARVEST_NAME
    harvest.write_text(header + "".join(parts.values()))
    (a.work / "parts").mkdir(exist_ok=True)
    for key, text in parts.items():
        (a.work / "parts" / f"harvest_vol_{key}.txt").write_text(header + text)
    print(
        f"wrote {harvest} ({harvest.stat().st_size:,} bytes) and {len(parts)} files in parts/",
        flush=True,
    )
    dest, wsha = release_copy(weights, a.work)
    print(f"weights for a release: {dest} (SHA-256 {wsha})", flush=True)
    print(LOCAL_FOLLOWUP, flush=True)
    print("$", "tsfm", " ".join(docs[3:]), flush=True)
    subprocess.run(docs, cwd=ROOT, check=True, env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
