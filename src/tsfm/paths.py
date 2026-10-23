"""Filesystem locations.

The repository holds code, configs, lockfiles and small results. Large data lives in
``$TSFM_HOME`` (default ``~/.cache/tsfm``). The Colab notebooks set that to local disk
and keep checkpoints and resumable scores on Google Drive.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root (the directory that contains ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def tsfm_home() -> Path:
    """Large-data cache. ``TSFM_HOME``, or ``~/.cache/tsfm``."""
    return Path(os.environ.get("TSFM_HOME", Path.home() / ".cache" / "tsfm")).expanduser()


def raw_dir() -> Path:
    """Downloaded source files, under :func:`tsfm_home`."""
    return tsfm_home() / "raw"


def processed_dir() -> Path:
    """Converted and packed training data, under :func:`tsfm_home`."""
    return tsfm_home() / "processed"


def runs_dir() -> Path:
    """Training runs. ``TSFM_RUNS``, or ``<tsfm_home>/runs``."""
    return Path(os.environ.get("TSFM_RUNS", tsfm_home() / "runs")).expanduser()


def gifteval_dir() -> Path:
    """GIFT-Eval source data. ``GIFT_EVAL``, or ``<raw_dir>/gifteval``."""
    return Path(os.environ.get("GIFT_EVAL", raw_dir() / "gifteval")).expanduser()


def results_dir() -> Path:
    """Committed results. ``TSFM_RESULTS`` redirects writes (used by the CI pipeline so a
    smoke run never touches the real results)."""
    return Path(os.environ.get("TSFM_RESULTS", repo_root() / "results")).expanduser()
