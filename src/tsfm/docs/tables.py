"""Markdown tables rendered from results files. Each function takes the repo root."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def _load(root: Path, rel: str) -> dict[str, Any]:
    from tsfm.docs.build import MissingResult

    p = root / rel
    if not p.exists():
        raise MissingResult(rel)
    d: dict[str, Any] = json.loads(p.read_text())
    return d


@table("corpus")
def corpus_table(root: Path) -> str:
    man = _load(root, "results/data/corpus_manifest.json")
    rows = []
    for _rel, e in man["files"].items():
        freqs = ", ".join(sorted(e["freqs"]))
        dropped = ", ".join(f"{k} {v:,}" for k, v in sorted(e["dropped"].items())) or "none"
        rows.append(
            [
                f"`{e['dataset']}`",
                e["split"],
                freqs,
                f"{e['series_kept']:,}",
                f"{e['points_kept']:,}",
                f"{e['series_truncated']:,}",
                dropped,
            ]
        )
    return md_table(
        ["dataset", "split", "freq", "series", "points", "truncated", "dropped (reason count)"],
        rows,
    )


@table("leakage_matches")
def leakage_matches(root: Path) -> str:
    rep = _load(root, "results/data/leakage.json")
    rows = [
        [
            f"`{k}`",
            f"{v['windows']:,}",
            f"{v['daily_windows']:,}",
            f"{v['hits']:,}",
            f"{v['daily_hits']:,}",
        ]
        for k, v in rep["files"].items()
        if v["hits"] or v["daily_hits"]
    ]
    if not rows:
        return "No matches in any file."
    return md_table(["file", "windows", "daily windows", "hits", "daily hits"], rows)


def _model_rows(summary: dict[str, Any], names: list[str]) -> list[list[str]]:
    rows = []
    for n in names:
        m = summary["models"].get(n)
        if m is None:
            continue
        cov = m["coverage_mean"].get("80")
        rows.append(
            [
                n.removeprefix("upstream:"),
                f"{m['mase_gmean']:.3f}",
                f"{m['crps_gmean']:.3f}",
                f"{m['rank_crps']:.2f}",
                f"{cov:.3f}" if cov is not None else "n/a",
            ]
        )
    return rows


@table("reported_reference")
def reported_reference(root: Path) -> str:
    s = _load(root, "results/analysis/test_summary.json")
    names = sorted(
        (n for n in s["models"] if n.startswith("upstream:")),
        key=lambda n: s["models"][n]["crps_gmean"],
    )
    note = (
        f"Geometric mean over the {s['n_configs']} configurations of each metric divided by "
        "Seasonal Naive; rank = mean CRPS rank among the models in this table. Computed by "
        "us from upstream's committed result files (`src/tsfm/eval/_vendor/upstream_results`)."
    )
    return (
        note
        + "\n\n"
        + md_table(["model", "MASE", "CRPS", "rank", "80% coverage"], _model_rows(s, names))
    )


@table("headline")
def headline(root: Path) -> str:
    h = _load(root, "results/analysis/headline.json")
    rows = [
        [
            c["a"],
            c["b"],
            c["metric"],
            f"{c['ratio']:.3f}",
            f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]",
            c["verdict"],
        ]
        for c in h["comparisons"]
    ]
    return md_table(["model", "vs", "metric", "ratio", "95% CI", "verdict"], rows)


@table("leakage_inspection")
def leakage_inspection(root: Path) -> str:
    s = _load(root, "results/data/leakage_inspection.json")
    rows = [
        [
            f"`{p['source']}`",
            f"`{p['gifteval']}`",
            f"{p['series']:,}",
            f"{p['series_with_test_window_hits']:,}",
            f"{p['agree_len_median']:,}" if "agree_len_median" in p else "n/a",
            f"{p['linear_fraction_median']:.2f}" if "linear_fraction_median" in p else "n/a",
            p["verdict"],
        ]
        for p in s["pairs"]
    ]
    return md_table(
        [
            "corpus dataset",
            "GIFT-Eval dataset",
            "flagged series",
            "of which with hits in GIFT-Eval test windows",
            "median agreement (points)",
            "median straight-line fraction",
            "verdict",
        ],
        rows,
    )
