"""Copy result files printed by scripts/colab_queue.py (``===FILE <path> <sha256>===``
... ``===END===`` blocks in the notebook output) into the repo, verifying each SHA-256.

    uv run python scripts/harvest_colab_output.py notebook_output.txt

Copied text can lose carriage returns (csv files use CRLF), so both the text as given and
the CRLF form are checked; a file is written only if one of them matches the printed hash,
in the form that matches. Every copied file is logged in results/sync_log.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^===FILE (\S+) ([0-9a-f]{64})===\n(.*?)^===END===$", re.S | re.M)


def harvest(text: str, dest_root: Path = ROOT) -> tuple[list[str], list[str]]:
    ok, bad = [], []
    log = []
    for m in BLOCK.finditer(text.replace("\r\n", "\n")):
        rel, sha, body = m.group(1), m.group(2), m.group(3)
        if not rel.startswith("results/") or ".." in rel:
            bad.append(f"{rel}: not under results/")
            continue
        candidates = [body, body.replace("\n", "\r\n")]
        data = next(
            (c.encode() for c in candidates if hashlib.sha256(c.encode()).hexdigest() == sha), None
        )
        if data is None:
            bad.append(f"{rel}: SHA-256 mismatch")
            continue
        out = dest_root / rel
        if out.exists() and rel.startswith("results/eval/test/") and out.read_bytes() != data:
            bad.append(f"{rel}: refusing to overwrite a different test result")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        ok.append(rel)
        log.append(
            {
                "file": rel,
                "sha256": sha,
                "source": "colab notebook output",
                "synced_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        )
    lp = dest_root / "results" / "sync_log.json"
    old = json.loads(lp.read_text()) if lp.exists() else []
    lp.write_text(json.dumps(old + log, indent=1) + "\n")
    return ok, bad


def main() -> int:
    ok, bad = harvest(Path(sys.argv[1]).read_text())
    for r in ok:
        print("ok ", r)
    for r in bad:
        print("BAD", r)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
