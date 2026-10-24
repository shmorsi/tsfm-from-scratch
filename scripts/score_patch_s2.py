"""Score the T4 headline model ``patch_s2`` on the GIFT-Eval test split, once, with the
pre-registered settings (``PREREGISTRATION.md`` sections 5 and 6).

Run from ``notebooks/score_patch_s2_test.ipynb`` on a Colab GPU:

    uv run python scripts/score_patch_s2.py --checkpoint /content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt \\
        --work /content/drive/MyDrive/tsfm/score_patch_s2

Steps:

1. Check that the checkpoint is ``patch_s2``: same run config (hash), same model
   config, 10,083,392 parameters, final step. Sibling runs (``bins_s2`` and others)
   live in the same Drive folder, so a wrong path stops here.
2. Pick the weights file whose SHA-256 equals the one recorded with the validation run
   (``results/eval/val/patch_s2/run.json``). A training ``ckpt.pt`` holds the same
   weights plus optimizer state, so its hash differs; the ``model.pt`` next to it is used.
3. ``tsfm eval --split test`` with ``--max-context 2048``. ``tsfm eval`` itself refuses a
   second test score and any model test score without the ``prereg-v1`` tag. Results go
   to ``results/eval/test/patch_s2/``, which is linked to a folder under ``--work`` so a
   disconnected session resumes where it stopped.
4. ``tsfm report --split test`` and ``tsfm headline``.
5. Write the new results files, with SHA-256, in the format that
   ``scripts/harvest_colab_output.py`` reads, to ``<work>/harvest_patch_s2.txt``.
6. ``tsfm docs build``.

``--dry-run`` stops after step 2 and prints the commands it would run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NAME = "patch_s2"
PREREG_TAG = "prereg-v1"
# PREREGISTRATION.md section 5: "context up to 2,048 points (--max-context 2048)".
PREREG_MAX_CONTEXT = 2048
EXPORT = (
    f"results/eval/test/{NAME}/all_results.csv",
    f"results/eval/test/{NAME}/run.json",
    f"results/eval/test/{NAME}/config.json",
    "results/analysis/test_summary.json",
    "results/analysis/headline.json",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class Expected:
    config: dict[str, Any]
    config_hash: str
    params_total: int
    step: int
    checkpoint_sha256: str
    max_context: int


def expected(root: Path = ROOT) -> Expected:
    """What the validation run recorded about ``patch_s2``."""
    summary = json.loads((root / "results" / "runs" / NAME / "summary.json").read_text())
    val = json.loads((root / "results" / "eval" / "val" / NAME / "run.json").read_text())
    info = val["model_info"]
    return Expected(
        config=summary["config"],
        config_hash=summary["config_hash"],
        params_total=int(summary["params_total"]),
        step=int(summary["step"]),
        checkpoint_sha256=info["checkpoint_sha256"],
        max_context=int(info["max_context"]),
    )


def problems(path: Path, exp: Expected) -> list[str]:
    """Reasons why ``path`` is not the ``patch_s2`` checkpoint (empty if it is)."""
    import torch

    from tsfm.model import ModelConfig, TSModel
    from tsfm.provenance import config_hash

    try:
        st = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # unreadable, truncated, not a torch file
        return [f"cannot load {path}: {e}"]
    if not isinstance(st, dict) or "model" not in st or "config" not in st:
        return [f"{path} is not a tsfm checkpoint (expected 'model' and 'config' entries)"]
    cfg = st["config"]
    out = []
    want, got = exp.config.get("model", {}), cfg.get("model", {})
    for k in sorted(set(want) | set(got)):
        if want.get(k) != got.get(k):
            out.append(f"model.{k} is {got.get(k)!r}, patch_s2 has {want.get(k)!r}")
    if cfg.get("name") != exp.config.get("name"):
        out.append(f"run name is {cfg.get('name')!r}, expected {exp.config.get('name')!r}")
    if config_hash(cfg) != exp.config_hash:
        out.append(f"run config hash is {config_hash(cfg)}, patch_s2 has {exp.config_hash}")
    try:
        model = TSModel(ModelConfig.from_dict(got))
        model.load_state_dict(st["model"])
        n = model.num_params()
    except Exception as e:
        out.append(f"weights do not load into the patch_s2 architecture: {e}")
    else:
        if n != exp.params_total:
            out.append(f"{n:,} parameters, patch_s2 has {exp.params_total:,}")
    if "step" in st and int(st["step"]) != exp.step:
        out.append(
            f"training checkpoint at step {int(st['step']):,}, patch_s2 finished at {exp.step:,}"
        )
    return out


def pick_weights(checkpoint: Path, exp: Expected) -> tuple[Path, list[str]]:
    """The file to score and notes about the choice. Stops with a message if
    ``checkpoint`` is not ``patch_s2`` or no file matches the recorded SHA-256."""
    if not checkpoint.is_file():
        raise SystemExit(f"CHECKPOINT_PATH does not exist: {checkpoint}")
    bad = problems(checkpoint, exp)
    if bad:
        raise SystemExit(
            f"{checkpoint} is not the patch_s2 checkpoint, so nothing was scored:\n  - "
            + "\n  - ".join(bad)
            + "\nOther runs (bins_s2, patch_s0, ...) sit next to it in the same Drive folder."
        )
    notes = [f"{checkpoint}: run config, model config and parameter count match patch_s2"]
    if sha256(checkpoint) == exp.checkpoint_sha256:
        notes.append(f"SHA-256 matches the validation run ({exp.checkpoint_sha256[:12]})")
        return checkpoint, notes
    sibling = checkpoint.parent / "model.pt"
    if sibling != checkpoint and sibling.is_file() and sha256(sibling) == exp.checkpoint_sha256:
        bad = problems(sibling, exp)
        if bad:
            raise SystemExit(f"{sibling} matches the recorded SHA-256 but fails: {bad}")
        notes.append(
            f"its SHA-256 differs from the validation run's (a training checkpoint also holds "
            f"optimizer state); scoring {sibling}, whose SHA-256 matches ({exp.checkpoint_sha256[:12]})"
        )
        return sibling, notes
    raise SystemExit(
        f"No file matches the SHA-256 recorded for the patch_s2 validation run "
        f"({exp.checkpoint_sha256}). Checked {checkpoint} and {sibling}. The test split is "
        "scored once, so it is not spent on weights that cannot be tied to the validation "
        "numbers. Point CHECKPOINT_PATH at that run's model.pt."
    )


def git_ok(root: Path) -> str:
    """HEAD's SHA; stops if the pre-registration tag is not in HEAD's history."""
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_TAG, "HEAD"], cwd=root, capture_output=True
    )
    if r.returncode != 0:
        raise SystemExit(
            f"{PREREG_TAG} is not an ancestor of HEAD (or tags were not fetched); "
            "tsfm eval would refuse the test score. Clone with tags."
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def link_results(root: Path, work: Path) -> Path:
    """Make ``results/eval/test/patch_s2`` a link to ``<work>/eval_test_patch_s2`` so
    per-configuration progress survives a disconnect. Returns the results path."""
    out = root / "results" / "eval" / "test" / NAME
    target = work / f"eval_test_{NAME}"
    target.mkdir(parents=True, exist_ok=True)
    if out.is_symlink():
        if out.resolve() != target.resolve():
            raise SystemExit(f"{out} links to {out.resolve()}, expected {target}")
    elif out.exists():
        raise SystemExit(
            f"{out} already exists in the checkout; the test split may already be scored."
        )
    else:
        out.symlink_to(target, target_is_directory=True)
    return out


def harvest_blocks(root: Path, rels: tuple[str, ...] = EXPORT) -> str:
    """Files as ``===FILE <path> <sha256>===`` blocks for scripts/harvest_colab_output.py."""
    parts = []
    for rel in rels:
        data = (root / rel).read_bytes()
        parts.append(f"===FILE {rel} {hashlib.sha256(data).hexdigest()}===\n")
        text = data.decode()
        parts.append(text if text.endswith("\n") else text + "\n")
        parts.append("===END===\n")
    return "".join(parts)


def tsfm(*args: str) -> list[str]:
    return [sys.executable, "-m", "tsfm.cli", *args]


LOCAL_FOLLOWUP = """
Local follow-up, from a clone of the same commit. Download the harvest file from Drive
(a copied notebook print can truncate, and a truncated file fails the SHA-256 check):

    uv run python scripts/harvest_colab_output.py harvest_patch_s2.txt
    uv run tsfm docs build

Commit the files the harvest script wrote and the docs that command regenerated.
Do not hand-edit numbers.

The harvest file is what the doc generators read:
  results/eval/test/patch_s2/all_results.csv
  results/eval/test/patch_s2/run.json
  results/eval/test/patch_s2/config.json
  results/analysis/test_summary.json   (rebuilt by tsfm report; replaces the previous file)
  results/analysis/headline.json        (new)
  results/sync_log.json                 (append-only log written by the harvest script)

results/analysis/cpu_study.json is not in the harvest. Ranks in the rebuilt
test_summary.json include patch_s2. CRPS and MASE for the other models are recomputed
from the same vendored files.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="patch_s2 ckpt.pt or model.pt")
    ap.add_argument("--work", type=Path, required=True, help="folder that survives a disconnect")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="check the checkpoint, print the plan")
    a = ap.parse_args(argv)

    exp = expected()
    if exp.max_context != PREREG_MAX_CONTEXT:
        raise SystemExit(
            f"validation run used max_context {exp.max_context}, pre-registered {PREREG_MAX_CONTEXT}"
        )
    weights, notes = pick_weights(a.checkpoint, exp)
    for n in notes:
        print("ok:", n, flush=True)
    sha = git_ok(ROOT)
    out = ROOT / "results" / "eval" / "test" / NAME
    steps = [
        tsfm("data", "download", "--which", "gifteval", "--workers", "16"),
        tsfm(
            "eval",
            "--model",
            str(weights),
            "--name",
            NAME,
            "--split",
            "test",
            "--device",
            a.device,
            "--max-context",
            str(PREREG_MAX_CONTEXT),
        ),
        tsfm("report", "--split", "test"),
        tsfm("headline"),
        tsfm("docs", "build"),
    ]
    if a.dry_run:
        print(f"dry run at {sha[:12]}; would link {out} -> {a.work / f'eval_test_{NAME}'} and run:")
        for s in steps:
            print("   tsfm", " ".join(s[3:]))
        print(LOCAL_FOLLOWUP)
        return 0

    # Results go to the checkout's results/ (the test folder is a link to Drive). An
    # inherited TSFM_RESULTS would point elsewhere, for example at the export folder of
    # the 2026-10-15 attempt, whose partial outputs are not to be read or resumed.
    env = {k: v for k, v in os.environ.items() if k != "TSFM_RESULTS"}
    a.work.mkdir(parents=True, exist_ok=True)
    link_results(ROOT, a.work)
    *scoring, docs = steps
    for s in scoring:
        if s[3] == "eval" and (out / "all_results.csv").exists():
            print(
                f"{NAME} already has a complete test score in {out}; not scoring again", flush=True
            )
            continue
        print("$", "tsfm", " ".join(s[3:]), flush=True)
        subprocess.run(s, cwd=ROOT, check=True, env=env)
    harvest = a.work / "harvest_patch_s2.txt"
    harvest.write_text(harvest_blocks(ROOT))
    print(f"wrote {harvest} ({len(EXPORT)} files with SHA-256)", flush=True)
    print(LOCAL_FOLLOWUP, flush=True)
    print("$", "tsfm", " ".join(docs[3:]), flush=True)
    subprocess.run(docs, cwd=ROOT, check=True, env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
