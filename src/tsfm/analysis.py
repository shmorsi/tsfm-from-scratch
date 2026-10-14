"""Summaries from evaluation results: ``tsfm report``.

Reads ``results/eval/<split>/<model>/all_results.csv`` (ours) and the vendored upstream
CSVs ("reported" models), and writes ``results/analysis/<split>_summary.json`` with, for
every model: geometric means of Seasonal-Naive-normalised MASE and CRPS, average ranks,
mean interval coverage, breakdowns by frequency / domain / term, and the pre-registered
comparisons with paired bootstrap intervals. Docs render tables from that file only.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.eval.gifteval import VENDOR
from tsfm.eval.metrics import gmean, paired_bootstrap_ratio
from tsfm.paths import results_dir
from tsfm.provenance import write_result

MASE = "eval_metrics/MASE[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
REPORTED = {
    "upstream:Seasonal_Naive": "Seasonal_Naive",
    "upstream:Auto_Arima": "Auto_Arima",
    "upstream:Auto_ETS": "Auto_ETS",
    "upstream:Auto_Theta": "Auto_Theta",
    "upstream:FlowState-9.1M": "FlowState-9.1M",
    "upstream:TimesFM-2.5": "TimesFM-2.5",
}


def read_csv(p: Path) -> dict[str, dict[str, Any]]:
    with open(p, newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            out[r["dataset"]] = {**r, MASE: float(r[MASE]), CRPS: float(r[CRPS])}
        return out


def load_models(split: str) -> dict[str, dict[str, dict[str, Any]]]:
    models: dict[str, dict[str, dict[str, Any]]] = {}
    d = results_dir() / "eval" / split
    if d.exists():
        for sub in sorted(d.iterdir()):
            if (sub / "all_results.csv").exists():
                run = (
                    json.loads((sub / "run.json").read_text())
                    if (sub / "run.json").exists()
                    else {}
                )
                if run.get("_meta", {}).get("partial"):
                    continue
                rows = read_csv(sub / "all_results.csv")
                for k, v in run.get("per_config", {}).items():
                    if k in rows:
                        rows[k].update(v)
                models[sub.name] = rows
    if split == "test":
        for name, up in REPORTED.items():
            if name == "upstream:Seasonal_Naive" and "seasonal_naive" in models:
                continue  # identical to ours (tested); ranking both would split ties
            p = VENDOR / "upstream_results" / up / "all_results.csv"
            if p.exists():
                models[name] = read_csv(p)
    return models


FREQ_NAMES = {
    "S": "secondly",
    "T": "minutely",
    "H": "hourly",
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "A": "yearly",
}


def freq_group(cfg: str) -> str:
    """Frequency label without the multiplier, as upstream's leaderboard groups it
    (``15T`` and ``5T`` are both minutely)."""
    letters = re.sub(r"\d", "", cfg.split("/")[1]).upper().split("-")[0]  # A-DEC -> A
    letters = {"MS": "M", "QS": "Q", "AS": "A", "YS": "A", "Y": "A", "MIN": "T"}.get(
        letters, letters
    )
    return FREQ_NAMES.get(letters, letters)


def summarise(models: dict[str, dict[str, dict[str, Any]]], baseline: str) -> dict[str, Any]:
    base = models[baseline]
    keys = sorted(base)
    out: dict[str, Any] = {"n_configs": len(keys), "baseline": baseline, "models": {}}
    complete = {m: rows for m, rows in models.items() if set(rows) >= set(keys)}
    # Ranks per config by CRPS and MASE among complete models (method "first" like upstream).
    ranks: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"crps": [], "mase": []})
    for k in keys:
        for metric, tag in ((CRPS, "crps"), (MASE, "mase")):
            order = sorted(complete, key=lambda m: (complete[m][k][metric], m))
            for i, m in enumerate(order):
                ranks[m][tag].append(i + 1)
    for m, rows in complete.items():
        nm = {k: rows[k][MASE] / base[k][MASE] for k in keys}
        nc = {k: rows[k][CRPS] / base[k][CRPS] for k in keys}
        groups: dict[str, dict[str, Any]] = {}
        for gname, fn in (
            ("frequency", freq_group),
            ("domain", lambda k, r=rows: str(r[k].get("domain", ""))),
            ("term", lambda k: k.split("/")[2]),
        ):
            g: dict[str, list[str]] = defaultdict(list)
            for k in keys:
                g[fn(k)].append(k)
            groups[gname] = {
                gv: {
                    "n": len(ks),
                    "mase": gmean([nm[k] for k in ks]),
                    "crps": gmean([nc[k] for k in ks]),
                }
                for gv, ks in sorted(g.items())
            }
        cov: dict[str, float | None] = {}
        for lvl in (50, 80, 90):
            raw = [rows[k].get(f"coverage_{lvl}") for k in keys]
            vals = [float(v) for v in raw if v is not None and np.isfinite(float(v))]
            cov[str(lvl)] = float(np.mean(vals)) if len(vals) == len(keys) else None
        out["models"][m] = {
            "mase_gmean": gmean(list(nm.values())),
            "crps_gmean": gmean(list(nc.values())),
            "rank_crps": float(np.mean(ranks[m]["crps"])),
            "rank_mase": float(np.mean(ranks[m]["mase"])),
            "coverage_mean": cov,
            "by": groups,
            "worst_configs_crps": sorted(nc.items(), key=lambda kv: -kv[1])[:5],
        }
    return out


def compare(
    models: dict[str, dict[str, dict[str, Any]]], a: str, b: str, baseline: str, metric: str
) -> dict[str, Any]:
    keys = sorted(models[baseline])
    col = CRPS if metric == "crps" else MASE
    pa = {k: models[a][k][col] for k in keys}
    pb = {k: models[b][k][col] for k in keys}
    ref = {k: models[baseline][k][col] for k in keys}
    pt, lo, hi = paired_bootstrap_ratio(pa, pb, ref)
    verdict = "better" if hi < 1 else "worse" if lo > 1 else "no detectable difference"
    return {"a": a, "b": b, "metric": metric, "ratio": pt, "ci95": [lo, hi], "verdict": verdict}


def report(split: str) -> Path:
    models = load_models(split)
    baseline = "seasonal_naive" if "seasonal_naive" in models else "upstream:Seasonal_Naive"
    if baseline not in models:
        raise SystemExit(f"no Seasonal Naive results for split {split}")
    summary = summarise(models, baseline)
    return write_result(
        results_dir() / "analysis" / f"{split}_summary.json",
        summary,
        split="none" if split == "val" else split,
        label=f"report-{split}",
    )


def seed_average(
    models: dict[str, dict[str, dict[str, Any]]], names: list[str]
) -> dict[str, dict[str, Any]]:
    """Per-config arithmetic mean of MASE and CRPS over seed runs (pre-registered)."""
    keys = sorted(models[names[0]])
    out = {}
    for k in keys:
        out[k] = {
            MASE: float(np.mean([models[n][k][MASE] for n in names])),
            CRPS: float(np.mean([models[n][k][CRPS] for n in names])),
        }
    return out


def headline_report(seed_runs: list[str], label: str = "headline") -> Path:
    """The pre-registered comparisons (PREREGISTRATION.md section 5).

    ``seed_runs``: test-split result names of the headline model, one per seed."""
    models = load_models("test")
    missing = [n for n in seed_runs if n not in models]
    if missing:
        raise SystemExit(f"missing test results for {missing}")
    base = "seasonal_naive"
    if base not in models:
        raise SystemExit("our Seasonal Naive test results are required")
    models[label] = seed_average(models, seed_runs)
    keys = sorted(models[base])
    per_seed = {}
    for n in seed_runs:
        per_seed[n] = {
            metric: gmean([models[n][k][col] / models[base][k][col] for k in keys])
            / gmean([models["upstream:Auto_Arima"][k][col] / models[base][k][col] for k in keys])
            for metric, col in (("crps", CRPS), ("mase", MASE))
        }
    comparisons = [compare(models, label, "upstream:Auto_Arima", base, "crps")]  # primary
    for other in ("upstream:Auto_Arima", "seasonal_naive", "flowstate_9.1m"):
        if other not in models:
            continue
        for metric in ("crps", "mase"):
            if (other, metric) == ("upstream:Auto_Arima", "crps"):
                continue
            comparisons.append(compare(models, label, other, base, metric))
    for c in comparisons:
        c["primary"] = c["b"] == "upstream:Auto_Arima" and c["metric"] == "crps"
    out = {
        "seed_runs": seed_runs,
        "ratio_vs_auto_arima_per_seed": per_seed,
        "comparisons": comparisons,
    }
    return write_result(
        results_dir() / "analysis" / "headline.json", out, split="test", label="M7-headline"
    )
