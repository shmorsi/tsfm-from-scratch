"""scripts/sync_results.py: copies results (not checkpoints) with hashes; never
overwrites a committed test result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_sync_results(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import sync_results as S

    src = tmp_path / "drive" / "run1"
    (src / "results" / "eval" / "val" / "run1").mkdir(parents=True)
    (src / "summary.json").write_text(json.dumps({"git_sha": "abc"}))
    (src / "metrics.jsonl").write_text("{}\n")
    (src / "ckpt.pt").write_bytes(b"x" * 10)
    (src / "results" / "eval" / "val" / "run1" / "all_results.csv").write_text("a\n")
    dest = tmp_path / "repo"
    log = S.sync(src, dest)
    files = {e["file"] for e in log}
    assert files == {
        "results/runs/run1/summary.json",
        "results/runs/run1/metrics.jsonl",
        "results/eval/val/run1/all_results.csv",
    }
    assert not (dest / "results" / "runs" / "run1" / "ckpt.pt").exists()
    assert json.loads((dest / "results" / "sync_log.json").read_text())[0]["run_git_sha"] == "abc"
    # A committed test result is never overwritten with different content.
    t = dest / "results" / "eval" / "test" / "run1"
    t.mkdir(parents=True)
    (t / "all_results.csv").write_text("old\n")
    (src / "results" / "eval" / "test" / "run1").mkdir(parents=True)
    (src / "results" / "eval" / "test" / "run1" / "all_results.csv").write_text("new\n")
    with pytest.raises(SystemExit, match="refusing"):
        S.sync(src, dest)
