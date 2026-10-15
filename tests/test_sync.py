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


def test_harvest_verifies_hashes_and_crlf(tmp_path: Path):
    import hashlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import harvest_colab_output as H

    csv_raw = b"a,b\r\n1,2\r\n"
    js = b'{"x": 1}\n'
    text = (
        f"noise\n===FILE results/eval/val/m/all_results.csv {hashlib.sha256(csv_raw).hexdigest()}===\n"
        + csv_raw.decode().replace("\r\n", "\n")  # carriage returns lost in copying
        + "===END===\n"
        + f"===FILE results/runs/m/summary.json {hashlib.sha256(js).hexdigest()}===\n{js.decode()}===END===\n"
        + f"===FILE results/runs/m/bad.json {'0' * 64}===\n{{}}\n===END===\n"
    )
    ok, bad = H.harvest(text, tmp_path)
    assert ok == ["results/eval/val/m/all_results.csv", "results/runs/m/summary.json"]
    assert bad == ["results/runs/m/bad.json: SHA-256 mismatch"]
    assert (tmp_path / "results/eval/val/m/all_results.csv").read_bytes() == csv_raw
