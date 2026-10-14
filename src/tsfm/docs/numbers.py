"""The "no invented numbers" guard.

Every number that appears in a generated document must either

* match a numeric value stored in a committed results file (at the precision shown), or
* be listed in ``docs_templates/allowed_numbers.txt`` with a reason (version numbers,
  years, structural constants such as "50/80/90% intervals").

Fenced and inline code, URLs and HTML comments are skipped: templates are not allowed to
put placeholders inside them (the renderer enforces this), so they only ever contain
static text written by hand.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_SUFFIX = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9}

# A number not glued to a preceding letter/digit/underscore/dot (so "M4", "T4", "sm_75",
# "v1.2" and "3.12.15" components are not split into numbers), with optional thousands
# separators, decimals, exponent, magnitude suffix and percent sign.
NUMBER_RE = re.compile(
    r"(?<![\w.\-/#])"
    r"(?P<sign>[-−])?"
    r"(?P<body>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<exp>[eE][-+]?\d+)?"
    r"(?P<suffix>[kKMBG](?![a-zA-Z]))?"
    r"(?P<pct>%)?"
    r"(?![\w.]\d|[a-zA-Z_])"
)

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"\]\([^)]*\)|https?://\S+|<!--.*?-->", re.S)


@dataclass(frozen=True)
class DocNumber:
    text: str
    value: float
    decimals: int
    scale: float  # suffix multiplier (1 if none); percent handled by `pct`
    pct: bool
    line: int


def strip_unchecked(text: str) -> str:
    """Blank out code, URLs and comments but keep line numbers stable."""

    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    for rx in (_FENCE_RE, _INLINE_CODE_RE, _URL_RE):
        text = rx.sub(blank, text)
    return text


def extract_numbers(text: str) -> list[DocNumber]:
    text = strip_unchecked(text)
    out: list[DocNumber] = []
    for m in NUMBER_RE.finditer(text):
        body = m.group("body").replace(",", "")
        decimals = len(body.split(".")[1]) if "." in body else 0
        val = float(body + (m.group("exp") or ""))
        if m.group("sign"):
            val = -val
        out.append(
            DocNumber(
                text=m.group(0),
                value=val,
                decimals=decimals if not m.group("exp") else -1,
                scale=_SUFFIX.get(m.group("suffix") or "", 1.0),
                pct=bool(m.group("pct")),
                line=text.count("\n", 0, m.start()) + 1,
            )
        )
    return out


def _numeric_leaves(obj: object) -> Iterator[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_meta":
                continue
            yield from _numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numeric_leaves(v)


def collect_result_values(paths: Iterable[Path]) -> list[float]:
    vals: list[float] = []
    for p in paths:
        if p.suffix == ".json":
            vals.extend(_numeric_leaves(json.loads(p.read_text())))
        elif p.suffix == ".csv":
            with open(p, newline="") as f:
                for row in csv.reader(f):
                    for cell in row:
                        try:
                            x = float(cell)
                        except ValueError:
                            continue
                        if math.isfinite(x):
                            vals.append(x)
    return vals


def _matches(n: DocNumber, v: float) -> bool:
    cands = [v / n.scale]
    if n.pct:
        cands = [v * 100.0, v]
    for c in cands:
        if n.decimals < 0:  # scientific notation: compare at the mantissa precision shown
            mant = n.text.lower().split("e")[0].lstrip("-−")
            digits = len(mant.replace(".", "").lstrip("0")) or 1
            if c != 0 and float(f"{c:.{digits - 1}e}") == n.value:
                return True
            continue
        if round(c, n.decimals) == n.value or (n.decimals == 0 and abs(c - n.value) < 1e-9):
            return True
    return False


def load_allowlist(path: Path) -> set[str]:
    allowed: set[str] = set()
    if not path.exists():
        return allowed
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            allowed.add(line)
    return allowed


def unverified_numbers(
    doc_text: str, result_values: list[float], allowlist: set[str]
) -> list[DocNumber]:
    """Numbers in ``doc_text`` that are neither allow-listed nor found in results."""
    bad: list[DocNumber] = []
    for n in extract_numbers(doc_text):
        if n.text in allowlist or n.text.rstrip("%") in allowlist:
            continue
        if not any(_matches(n, v) for v in result_values):
            bad.append(n)
    return bad
