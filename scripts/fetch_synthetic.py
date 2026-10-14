"""Download the synthetic Parquet files from the repo's GitHub release and verify them.

Regenerating 100k GP series takes hours on Colab's two vCPUs, and BLAS differences would
change the bits anyway, so runs use the exact files generated for the committed manifest:

    python scripts/fetch_synthetic.py --repo shmorsi/tsfm-from-scratch --tag data-v1

Set GITHUB_TOKEN for a private repository. Each file's SHA-256 must equal the one in
results/data/synthetic_manifest.json, otherwise the file is deleted and the script fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "pretrain/kernelsynth.parquet": "kernelsynth-pretrain.parquet",
    "val/kernelsynth.parquet": "kernelsynth-val.parquet",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def request(url: str, accept: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "tsfm"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", default="data-v1")
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("TSFM_HOME", Path.home() / ".cache" / "tsfm")) / "processed",
    )
    a = ap.parse_args()
    manifest = json.loads((ROOT / "results" / "data" / "synthetic_manifest.json").read_text())[
        "files"
    ]
    with urllib.request.urlopen(
        request(
            f"https://api.github.com/repos/{a.repo}/releases/tags/{a.tag}",
            "application/vnd.github+json",
        )
    ) as r:
        release = json.load(r)
    by_name = {x["name"]: x for x in release["assets"]}
    for rel, asset_name in ASSETS.items():
        dest = a.dest / rel
        want = manifest[rel]["sha256"]
        if dest.exists() and sha256(dest) == want:
            print("ok (cached)", rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(
                request(by_name[asset_name]["url"], "application/octet-stream")
            ) as r,
            open(dest, "wb") as f,
        ):
            while chunk := r.read(1 << 22):
                f.write(chunk)
        got = sha256(dest)
        if got != want:
            dest.unlink()
            print(f"SHA-256 mismatch for {rel}: {got} != {want}", file=sys.stderr)
            return 1
        print("ok", rel, got[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
