"""Filesystem locations.

The repository holds code, configs, lockfiles and small results. Large data lives in
``$TSFM_HOME`` (default ``~/.cache/tsfm``); on Colab this points at Google Drive.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tsfm_home() -> Path:
    return Path(os.environ.get("TSFM_HOME", Path.home() / ".cache" / "tsfm")).expanduser()


def raw_dir() -> Path:
    return tsfm_home() / "raw"


def processed_dir() -> Path:
    return tsfm_home() / "processed"


def runs_dir() -> Path:
    return Path(os.environ.get("TSFM_RUNS", tsfm_home() / "runs")).expanduser()


def gifteval_dir() -> Path:
    return Path(os.environ.get("GIFT_EVAL", raw_dir() / "gifteval")).expanduser()


def results_dir() -> Path:
    """Committed results. ``TSFM_RESULTS`` redirects writes (used by the CI pipeline so a
    smoke run never touches the real results)."""
    return Path(os.environ.get("TSFM_RESULTS", repo_root() / "results")).expanduser()
