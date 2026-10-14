"""Copy a finished Colab run's results from Drive (or an unzipped download) into the repo.

    uv run python scripts/sync_results.py --src "/path/to/My Drive/tsfm/runs/<run>"

Copies ``summary.json``, ``metrics.jsonl``, ``config.yaml`` and ``results/eval/**`` (not
checkpoints) into ``results/runs/<run>/`` and ``results/eval/``, refuses to overwrite a
committed test-split result, and appends every copied file with its SHA-256, source path
and the run's git SHA to ``results/sync_log.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ("summary.json", "metrics.jsonl", "config.yaml")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def sync(src: Path, dest_root: Path = ROOT) -> list[dict[str, str]]:
    if not (src / "summary.json").exists():
        raise SystemExit(f"{src} has no summary.json - is the run finished?")
    run = src.name
    summary = json.loads((src / "summary.json").read_text())
    copied: list[tuple[Path, Path]] = []
    for name in RUN_FILES:
        if (src / name).exists():
            copied.append((src / name, dest_root / "results" / "runs" / run / name))
    ev = src / "results" / "eval"
    if ev.exists():
        for p in sorted(ev.rglob("*")):
            if p.is_file():
                copied.append((p, dest_root / "results" / "eval" / p.relative_to(ev)))
    log: list[dict[str, str]] = []
    for s, d in copied:
        rel = d.relative_to(dest_root)
        if d.exists() and rel.parts[:3] == ("results", "eval", "test") and sha256(d) != sha256(s):
            raise SystemExit(f"refusing to overwrite an existing test result: {rel}")
    for s, d in copied:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        log.append(
            {
                "file": str(d.relative_to(dest_root)),
                "sha256": sha256(d),
                "source": str(s),
                "run": run,
                "run_git_sha": str(summary.get("git_sha")),
            }
        )
    lp = dest_root / "results" / "sync_log.json"
    old = json.loads(lp.read_text()) if lp.exists() else []
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    old += [{**e, "synced_utc": stamp} for e in log]
    lp.write_text(json.dumps(old, indent=1) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    a = ap.parse_args()
    for e in sync(a.src):
        print(e["sha256"][:12], e["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
