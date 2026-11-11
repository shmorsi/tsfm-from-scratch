"""Copy result files printed by scripts/colab_queue.py (``===FILE <path> <sha256>===``
... ``===END===`` blocks in the notebook output) into the repo, verifying each SHA-256.

    uv run python scripts/harvest_colab_output.py notebook_output.txt

Copied text can lose carriage returns (csv files use CRLF), so both the text as given and
the CRLF form are checked; a file is written only if one of them matches the printed hash,
in the form that matches. Every copied file is logged in results/sync_log.json.

Large files travel packed, so that a harvest file stays small enough to read through a
connector that returns file content inline::

    ===PACKED <path> <sha256 of the restored file> <codec> <reference path or ->===
    <base64>
    ===END===

Two codecs, both exact (the restored bytes must hash to the printed SHA-256, or nothing is
written):

* ``xz``: the file, xz-compressed.
* ``volrows1``: a volatility ``per_origin.csv`` (``src/tsfm/finance/study.py``), stored as
  the columns that differ from a reference run already in the repo. The symbol, origin,
  regime, target sum and count columns are the same for every model, and the loss ratio
  column follows from two others, so only the loss, the error and the coverage counts are
  carried.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import hashlib
import io
import json
import lzma
import re
import struct
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^===FILE (\S+) ([0-9a-f]{64})===\n(.*?)^===END===$", re.S | re.M)
PACKED = re.compile(
    r"^===PACKED (\S+) ([0-9a-f]{64}) (\S+) (\S+)===\n(.*?)^===END===$", re.S | re.M
)
# Scored once: an existing file under these prefixes is never replaced by different content.
ONCE_ONLY = ("results/eval/test/", "results/finance/tsfm_")

VOL_KEYS = [
    "symbol",
    "origin",
    "regime",
    "wql_num",
    "abs_y",
    "wql",
    "mae",
    "cov50",
    "cov80",
    "cov90",
    "n",
]
_SHARED = ("symbol", "origin", "regime", "abs_y", "n")
_COVS = ("cov50", "cov80", "cov90")
_MAGIC = b"VOLROWS1"


def _g6(x: float) -> str:
    """How ``study.save`` prints a float."""
    return f"{x:.6g}"


def _ulp6(text: str) -> Decimal:
    """One unit in the sixth significant digit of the decimal ``text``."""
    d = Decimal(text)
    return Decimal(1).scaleb(d.adjusted() - 5)


def _rows(data: bytes) -> list[dict[str, str]] | None:
    """The rows of a ``per_origin.csv``, or None if it does not have that layout."""
    try:
        reader = csv.reader(io.StringIO(data.decode(), newline=""))
        table = list(reader)
    except (UnicodeDecodeError, csv.Error):
        return None
    if not table or table[0] != VOL_KEYS or any(len(r) != len(VOL_KEYS) for r in table[1:]):
        return None
    return [dict(zip(VOL_KEYS, r, strict=True)) for r in table[1:]]


def _wql_candidate(wql_num: str, abs_y: str) -> str:
    """The loss ratio as it would print from the two printed columns it is made of."""
    return _g6(float(wql_num) / float(abs_y))


def _vol_csv(
    ref: list[dict[str, str]],
    wql_num: tuple[float, ...],
    mae: tuple[float, ...],
    counts: tuple[bytes, bytes, bytes],
    corr: tuple[int, ...],
) -> bytes:
    """Rebuild a ``per_origin.csv`` from the carried columns and the reference rows."""
    out = io.StringIO(newline="")
    w = csv.writer(out)  # CRLF line ends, as csv.DictWriter writes them in study.save
    w.writerow(VOL_KEYS)
    for i, r in enumerate(ref):
        num = _g6(wql_num[i])
        cand = _wql_candidate(num, r["abs_y"])
        wql = _g6(float(Decimal(cand) + corr[i] * _ulp6(cand))) if corr[i] else cand
        n = int(r["n"])
        covs = [_g6(c[i] / n) for c in counts]
        w.writerow(
            [
                r["symbol"],
                r["origin"],
                r["regime"],
                num,
                r["abs_y"],
                wql,
                _g6(mae[i]),
                *covs,
                r["n"],
            ]
        )
    return out.getvalue().encode()


def pack_vol_rows(data: bytes, ref_data: bytes) -> bytes | None:
    """``data`` (a model's ``per_origin.csv``) against ``ref_data`` (a run already in the
    repo), as xz-compressed columns; None if the file cannot be restored exactly that way."""
    rows, ref = _rows(data), _rows(ref_data)
    if rows is None or ref is None or len(rows) != len(ref) or not rows:
        return None
    if any(r[k] != q[k] for r, q in zip(rows, ref, strict=True) for k in _SHARED):
        return None
    try:
        nums = [float(r["wql_num"]) for r in rows]
        maes = [float(r["mae"]) for r in rows]
        counts = []
        for c in _COVS:
            ks = [round(float(r[c]) * int(r["n"])) for r in rows]
            if any(not 0 <= k <= 255 for k in ks):
                return None
            counts.append(bytes(ks))
        corr = []
        for r in rows:
            num32 = _g6(struct.unpack("<f", struct.pack("<f", float(r["wql_num"])))[0])
            cand = _wql_candidate(num32, r["abs_y"])
            steps = (Decimal(r["wql"]) - Decimal(cand)) / _ulp6(cand)
            if steps != steps.to_integral_value() or not -128 <= steps <= 127:
                return None
            corr.append(int(steps))
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError, struct.error):
        return None
    n = len(rows)
    blob = b"".join(
        [
            _MAGIC,
            struct.pack("<I", n),
            struct.pack(f"<{n}f", *nums),
            struct.pack(f"<{n}f", *maes),
            *counts,
            struct.pack(f"<{n}b", *corr),
        ]
    )
    packed = lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)
    try:
        if unpack_vol_rows(packed, ref_data) != data:
            return None
    except (ValueError, ZeroDivisionError, InvalidOperation, struct.error):
        return None
    return packed


def unpack_vol_rows(packed: bytes, ref_data: bytes) -> bytes:
    """Inverse of :func:`pack_vol_rows`."""
    ref = _rows(ref_data)
    if ref is None:
        raise ValueError("the reference file is not a volatility per_origin.csv")
    blob = lzma.decompress(packed)
    if not blob.startswith(_MAGIC):
        raise ValueError("not a volrows1 block")
    off = len(_MAGIC)
    (n,) = struct.unpack_from("<I", blob, off)
    off += 4
    if n != len(ref) or len(blob) != off + 12 * n:
        raise ValueError("volrows1 block does not match the reference rows")
    wql_num = struct.unpack_from(f"<{n}f", blob, off)
    mae = struct.unpack_from(f"<{n}f", blob, off + 4 * n)
    off += 8 * n
    counts = (blob[off : off + n], blob[off + n : off + 2 * n], blob[off + 2 * n : off + 3 * n])
    corr = struct.unpack_from(f"<{n}b", blob, off + 3 * n)
    return _vol_csv(ref, wql_num, mae, counts, corr)


def packed_block(
    rel: str, data: bytes, ref_rel: str | None = None, ref_data: bytes | None = None
) -> str:
    """A ``===PACKED===`` block for ``data``: ``volrows1`` against the reference when that
    restores the file exactly, plain ``xz`` otherwise."""
    codec, ref_name, body = "xz", "-", lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    if ref_rel is not None and ref_data is not None:
        vol = pack_vol_rows(data, ref_data)
        if vol is not None and len(vol) < len(body):
            codec, ref_name, body = "volrows1", ref_rel, vol
    text = base64.b64encode(body).decode()
    lines = "\n".join(text[i : i + 76] for i in range(0, len(text), 76))
    sha = hashlib.sha256(data).hexdigest()
    return f"===PACKED {rel} {sha} {codec} {ref_name}===\n{lines}\n===END===\n"


def _allowed(rel: str) -> bool:
    return rel.startswith("results/") and ".." not in rel


def harvest(text: str, dest_root: Path = ROOT) -> tuple[list[str], list[str]]:
    ok, bad = [], []
    log = []
    text = text.replace("\r\n", "\n")
    found: list[tuple[int, str, str, bytes | None, str]] = []
    for m in BLOCK.finditer(text):
        rel, sha, body = m.group(1), m.group(2), m.group(3)
        candidates = [body, body.replace("\n", "\r\n")]
        data = next(
            (c.encode() for c in candidates if hashlib.sha256(c.encode()).hexdigest() == sha), None
        )
        found.append((m.start(), rel, sha, data, "SHA-256 mismatch"))
    for m in PACKED.finditer(text):
        rel, sha, codec, ref_rel, body = (m.group(i) for i in range(1, 6))
        data, why = None, "SHA-256 mismatch"
        try:
            raw = base64.b64decode("".join(body.split()), validate=True)
            if codec == "xz":
                data = lzma.decompress(raw)
            elif codec == "volrows1" and _allowed(ref_rel) and (dest_root / ref_rel).is_file():
                data = unpack_vol_rows(raw, (dest_root / ref_rel).read_bytes())
            elif codec == "volrows1":
                why = f"reference {ref_rel} is not in the repo"
            else:
                why = f"unknown codec {codec}"
        except (ValueError, lzma.LZMAError, ZeroDivisionError, InvalidOperation, struct.error) as e:
            why = f"cannot unpack ({e})"
        if data is not None and hashlib.sha256(data).hexdigest() != sha:
            data = None
        found.append((m.start(), rel, sha, data, why))
    for _, rel, sha, data, why in sorted(found, key=lambda f: f[0]):
        if not _allowed(rel):
            bad.append(f"{rel}: not under results/")
            continue
        if data is None:
            bad.append(f"{rel}: {why}")
            continue
        out = dest_root / rel
        if out.exists() and rel.startswith(ONCE_ONLY) and out.read_bytes() != data:
            kind = "test result" if rel.startswith("results/eval/test/") else "scored-once result"
            bad.append(f"{rel}: refusing to overwrite a different {kind}")
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
    lp.parent.mkdir(parents=True, exist_ok=True)
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
