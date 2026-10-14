"""Markdown tables rendered from results files. Each function takes the repo root."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

TABLES: dict[str, Callable[[Path], str]] = {}


def table(name: str) -> Callable[[Callable[[Path], str]], Callable[[Path], str]]:
    def deco(fn: Callable[[Path], str]) -> Callable[[Path], str]:
        TABLES[name] = fn
        return fn

    return deco


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)
