"""Pinned raw sources: build the lockfile, download, verify SHA-256.

The lockfile (``locks/pretrain.lock.json`` and ``locks/gifteval.lock.json``) lists every
raw file we use with its size and SHA-256 at a fixed Hugging Face revision. Downloads
are verified against it; a mismatch is an error, never a warning.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from tsfm.paths import raw_dir, repo_root
from tsfm.provenance import sha256_file

HF = "https://huggingface.co"
GIFTEVAL_REPO = "Salesforce/GiftEval"
GIFTEVAL_REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
_SHARD_RE = re.compile(r"data-(\d+)-of-(\d+)\.arrow$")


def load_data_config(path: Path | None = None) -> dict[str, Any]:
    """Read ``configs/data.yaml``, or ``path`` when one is given."""
    cfg: dict[str, Any] = yaml.safe_load(
        (path or repo_root() / "configs" / "data.yaml").read_text()
    )
    return cfg


def lock_path(name: str) -> Path:
    """Path of the download lock named ``name``."""
    return repo_root() / "locks" / f"{name}.lock.json"


def list_repo_files(repo: str, revision: str) -> list[dict[str, Any]]:
    """All files in a HF dataset repo at ``revision`` (paginated API)."""
    url: str | None = f"{HF}/api/datasets/{repo}/tree/{revision}?recursive=true"
    files: list[dict[str, Any]] = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            files += json.load(r)
            link = r.headers.get("Link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return [f for f in files if f["type"] == "file"]


def _entry(f: dict[str, Any]) -> dict[str, Any]:
    sha = f["lfs"]["oid"] if "lfs" in f else None
    return {"path": f["path"], "size": f["size"], "sha256": sha, "git_oid": f["oid"]}


def select_pretrain_files(files: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the selection rules of ``configs/data.yaml`` to the repo file list."""
    pc = cfg["pretrain"]
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in files:
        if "/" in f["path"]:
            by_ds[f["path"].split("/")[0]].append(f)
    excluded = set(pc["exclude"])
    selected: list[dict[str, Any]] = []
    for ds, fs in sorted(by_ds.items()):
        # Only `data-XXXXX-of-YYYYY.arrow` shards hold data; some upstream directories also
        # contain stray HF `cache-*.arrow` files with a different schema, which we skip.
        arrow = [f for f in fs if _SHARD_RE.search(f["path"])]
        meta = [f for f in fs if f["path"].endswith(".json")]
        size_mb = sum(f["size"] for f in arrow) / 1e6
        if ds in pc["sampled_shards"]:
            want = set(pc["sampled_shards"][ds])
            arrow = [f for f in arrow if int(_SHARD_RE.search(f["path"]).group(1)) in want]  # type: ignore[union-attr]
            if len(arrow) != len(want):
                raise ValueError(f"{ds}: requested shards {sorted(want)} not all present")
        elif (
            ds in excluded
            or ds.startswith(tuple(pc["exclude_prefixes"]))
            or size_mb > pc["include_all_below_mb"]
        ):
            continue
        selected += [_entry(f) for f in meta + arrow]
    return selected


def build_locks() -> None:
    """Rewrite the pretrain and GIFT-Eval lock files from the data config."""
    cfg = load_data_config()
    pc = cfg["pretrain"]
    pre = select_pretrain_files(list_repo_files(pc["repo"], pc["revision"]), cfg)
    _write_lock("pretrain", pc["repo"], pc["revision"], pre)
    ge = [_entry(f) for f in list_repo_files(GIFTEVAL_REPO, GIFTEVAL_REVISION)]
    _write_lock("gifteval", GIFTEVAL_REPO, GIFTEVAL_REVISION, ge)


def _write_lock(name: str, repo: str, revision: str, files: list[dict[str, Any]]) -> None:
    p = lock_path(name)
    p.parent.mkdir(exist_ok=True)
    doc = {"repo": repo, "revision": revision, "files": sorted(files, key=lambda f: f["path"])}
    p.write_text(json.dumps(doc, indent=1) + "\n")


def read_lock(name: str) -> dict[str, Any]:
    """Load one download lock."""
    lock: dict[str, Any] = json.loads(lock_path(name).read_text())
    return lock


def local_dir(name: str) -> Path:
    """Directory where downloads for ``name`` are stored."""
    return raw_dir() / ("gifteval" if name == "gifteval" else "pretrain")


def _git_blob_sha1(path: Path) -> str:
    import hashlib

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_file(entry: dict[str, Any], root: Path) -> str | None:
    """Return an error string, or None if the local file matches the lock."""
    p = root / entry["path"]
    if not p.exists():
        return "missing"
    if p.stat().st_size != entry["size"]:
        return f"size {p.stat().st_size} != {entry['size']}"
    if entry["sha256"]:
        got = sha256_file(p)
        if got != entry["sha256"]:
            return f"sha256 {got} != {entry['sha256']}"
    elif _git_blob_sha1(p) != entry["git_oid"]:  # small non-LFS files: git blob hash
        return "git blob hash mismatch"
    return None


def download(name: str, *, workers: int = 8, only: set[str] | None = None) -> list[str]:
    """Download (if needed) and verify every file of lock ``name``. Returns errors."""
    lock = read_lock(name)
    root = local_dir(name)
    entries = [e for e in lock["files"] if only is None or e["path"].split("/")[0] in only]

    def fetch(e: dict[str, Any]) -> str | None:
        if verify_file(e, root) is None:
            return None
        dest = root / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/datasets/{lock['repo']}/resolve/{lock['revision']}/{e['path']}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.replace(dest)
        err = verify_file(e, root)
        return f"{e['path']}: {err}" if err else None

    with ThreadPoolExecutor(workers) as ex:
        return [err for err in ex.map(fetch, entries) if err]


def verify(name: str) -> list[str]:
    """Check every file in a lock against the local copy. Return the errors."""
    lock = read_lock(name)
    root = local_dir(name)
    out = []
    for e in lock["files"]:
        err = verify_file(e, root)
        if err:
            out.append(f"{e['path']}: {err}")
    return out
