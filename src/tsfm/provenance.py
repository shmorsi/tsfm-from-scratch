"""Hashing, git state and versioned results files.

Every results file written by this project goes through :func:`write_result`, which
stamps it with the code version (git SHA and dirty flag), library versions and a UTC
timestamp. Docs are generated only from these files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tsfm import __version__
from tsfm.paths import repo_root

CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing configs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(cfg).encode())[:16]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def environment() -> dict[str, str]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tsfm": __version__,
    }
    for mod in ("numpy", "pandas", "torch", "gluonts", "statsforecast"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional modules
            continue
    return env


def stamp(split: str, label: str) -> dict[str, Any]:
    """Metadata block attached to every results file.

    split: "val", "test", "exploratory" or "none" (e.g. data statistics).
    label: free-form run label, e.g. "M4-baselines".
    """
    if split not in {"val", "test", "exploratory", "none"}:
        raise ValueError(f"unknown split {split!r}")
    return {
        **git_state(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "split": split,
        "label": label,
        "env": environment(),
    }


def write_result(path: str | Path, payload: Mapping[str, Any], *, split: str, label: str) -> Path:
    """Write ``payload`` plus a ``_meta`` stamp as pretty, key-sorted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_meta": stamp(split, label), **payload}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
