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


def load_models(
    split: str, *, include_partial: bool = False
) -> dict[str, dict[str, dict[str, Any]]]:
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
                # ``partial`` means the run did not cover every config of the split.
                # The CPU study's validation run is partial relative to kernelsynth,
                # which that study excludes on purpose, so its report opts in.
                if run.get("_meta", {}).get("partial") and not include_partial:
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


def _in_reference_summary(split: str, name: str) -> bool:
    """Models from the CPU study set ``reference_summary`` false so they do not change
    the published-reference ranks in ``test_summary.json`` / ``val_summary.json``."""
    if name.startswith("upstream:"):
        return True
    path = results_dir() / "eval" / split / name / "run.json"
    if not path.exists():
        return True
    info = json.loads(path.read_text()).get("model_info") or {}
    flag = info.get("reference_summary", True)
    return bool(flag)


def reference_models(split: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {n: rows for n, rows in load_models(split).items() if _in_reference_summary(split, n)}


def report(split: str) -> Path:
    models = reference_models(split)
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


GRID_RUNS = ("patch_s0", "patch_s1", "patch_s2", "bins_s2", "patch_s3", "patch_s4")


def select_headline(tolerance: float = 0.005, extra_seeds: tuple[int, ...] = (1, 2)) -> Path:
    """PREREGISTRATION.md section 4: lowest validation CRPS (geometric mean normalised by
    validation Seasonal Naive) among the grid runs that were trained; candidates within
    ``tolerance`` (relative) of the best go to the smallest model. ``extra_seeds`` are the
    seeds trained on top of seed 0 for the test evaluation (recorded so the runner reads
    them from the committed choice)."""
    s = summarise(load_models("val"), "seasonal_naive")
    cands = []
    for name in GRID_RUNS:
        if name not in s["models"]:
            continue
        summ = json.loads((results_dir() / "runs" / name / "summary.json").read_text())
        cands.append(
            {
                "name": name,
                "crps_gmean": s["models"][name]["crps_gmean"],
                "mase_gmean": s["models"][name]["mase_gmean"],
                "params_total": summ["params_total"],
            }
        )
    if not cands:
        raise SystemExit("no trained grid runs with validation results")
    best = min(c["crps_gmean"] for c in cands)
    near = [c for c in cands if c["crps_gmean"] <= best * (1 + tolerance)]
    choice = min(near, key=lambda c: c["params_total"])
    out = {
        "rule": "PREREGISTRATION.md section 4",
        "tolerance": tolerance,
        "candidates": sorted(cands, key=lambda c: c["crps_gmean"]),
        "headline": choice["name"],
        "extra_seeds": list(extra_seeds),
    }
    return write_result(
        results_dir() / "analysis" / "headline_choice.json",
        out,
        split="val",
        label="M7-headline-choice",
    )


# Student-t critical value, two-sided 95%, df = 2 (three seeds).
_T_CRIT_DF2 = 4.302652729911275


def seed_t_interval(values: list[float]) -> dict[str, Any]:
    """Mean and a 95% Student-t interval across seeds.

    With three seeds the interval is an honest description of seed noise and is
    usually wide. It is not a substitute for the paired bootstrap over configurations.
    """
    a = np.asarray(values, np.float64)
    n = int(a.size)
    mean = float(a.mean())
    out: dict[str, Any] = {"n": n, "mean": mean, "values": [float(v) for v in a]}
    if n < 2:
        out["ci95"] = None
        return out
    sd = float(a.std(ddof=1))
    half = _T_CRIT_DF2 * sd / float(np.sqrt(n))
    out["sd"] = sd
    out["ci95"] = [mean - half, mean + half]
    return out


def _subset_models(
    models: dict[str, dict[str, dict[str, Any]]], names: list[str], baseline: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep configs present in every named model and in the baseline."""
    keys = set(models[baseline])
    for n in names:
        keys &= set(models[n])
    if not keys:
        raise SystemExit(f"no configs in common between {names} and {baseline}")
    keep = sorted(keys)

    def cut(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {k: rows[k] for k in keep}

    out = {baseline: cut(models[baseline])}
    for n in names:
        out[n] = cut(models[n])
    return out


def _family_block(
    models: dict[str, dict[str, dict[str, Any]]],
    seed_names: list[str],
    label: str,
    baseline: str,
    opponents: list[str],
) -> dict[str, Any]:
    """Seed-averaged scores, per-seed ratios to Seasonal Naive, config bootstrap, seed t interval."""
    present = [n for n in seed_names if n in models]
    missing = [n for n in seed_names if n not in models]
    if not present:
        raise SystemExit(f"no results for {seed_names}")
    sub = _subset_models(models, present, baseline)
    sub[label] = seed_average(sub, present)
    summary = summarise(sub, baseline)
    keys = sorted(sub[baseline])
    # The seed average only pools MASE and CRPS. Coverage is the mean across seeds
    # of each config, then the mean across configs, matching summarise's coverage.
    cov_rows: dict[str, dict[str, float]] = {}
    for k in keys:
        cov_rows[k] = {}
        for lvl in (50, 80, 90):
            raw = [sub[n][k].get(f"coverage_{lvl}") for n in present]
            vals = [float(v) for v in raw if v is not None]
            if len(vals) != len(raw) or not all(np.isfinite(v) for v in vals):
                continue
            cov_rows[k][f"coverage_{lvl}"] = float(np.mean(vals))
    summary["models"][label]["coverage_mean"] = {
        str(lvl): (
            float(np.mean([cov_rows[k][f"coverage_{lvl}"] for k in keys]))
            if all(f"coverage_{lvl}" in cov_rows[k] for k in keys)
            else None
        )
        for lvl in (50, 80, 90)
    }
    per_seed_crps = []
    per_seed = {}
    for n in present:
        crps = gmean([sub[n][k][CRPS] / sub[baseline][k][CRPS] for k in keys])
        mase = gmean([sub[n][k][MASE] / sub[baseline][k][MASE] for k in keys])
        per_seed[n] = {"crps_gmean": crps, "mase_gmean": mase}
        per_seed_crps.append(crps)
    comparisons = []
    for other in opponents:
        if other not in models or other == label:
            continue
        opp_keys = [k for k in keys if k in models[other]]
        if not opp_keys:
            continue
        both = {
            baseline: {k: sub[baseline][k] for k in opp_keys},
            label: {k: sub[label][k] for k in opp_keys},
            other: {k: models[other][k] for k in opp_keys},
        }
        for metric in ("crps", "mase"):
            comparisons.append(compare(both, label, other, baseline, metric))
    return {
        "label": label,
        "seed_runs": present,
        "missing_seed_runs": missing,
        "n_configs": summary["n_configs"],
        "summary": summary["models"][label],
        "per_seed": per_seed,
        "crps_gmean_across_seeds": seed_t_interval(per_seed_crps),
        "comparisons": comparisons,
    }


def cpu_study_report(prefix: str = "cpu_s0") -> Path:
    """Validation and test summaries for the CPU study.

    The test comparison follows PREREGISTRATION.md section 5: seed-average the
    per-configuration scores, then a 95% paired bootstrap over configurations.
    A Student-t interval across the three seed-level geometric means is reported
    beside it. Conformalised runs are a separate, pre-specified family.
    """
    # Validation runs are marked partial because they skip synthetic series.
    val = load_models("val", include_partial=True)
    test = load_models("test")
    seeds = [f"{prefix}_seed{i}" for i in (0, 1, 2)]
    cqr = [f"{n}_cqr" for n in seeds]
    ablation = f"{prefix}_norope"
    opponents = ["seasonal_naive", "upstream:Auto_Arima"]
    out: dict[str, Any] = {"prefix": prefix}
    if any(n in val for n in seeds):
        block = _family_block(val, seeds, prefix, "seasonal_naive", ["seasonal_naive"])
        out["validation"] = block
        if ablation in val:
            ab = _subset_models(val, [ablation, seeds[0]], "seasonal_naive")
            sm = summarise(ab, "seasonal_naive")
            out["validation_ablation"] = {
                "rope": {
                    "name": seeds[0],
                    "crps_gmean": sm["models"][seeds[0]]["crps_gmean"],
                    "mase_gmean": sm["models"][seeds[0]]["mase_gmean"],
                },
                "none": {
                    "name": ablation,
                    "crps_gmean": sm["models"][ablation]["crps_gmean"],
                    "mase_gmean": sm["models"][ablation]["mase_gmean"],
                },
                "n_configs": sm["n_configs"],
            }
        if any(n in val for n in cqr):
            out["validation_cqr"] = _family_block(
                val, cqr, f"{prefix}_cqr", "seasonal_naive", ["seasonal_naive"]
            )
    if any(n in test for n in seeds):
        out["test"] = _family_block(test, seeds, prefix, "seasonal_naive", opponents)
    if any(n in test for n in cqr):
        out["test_cqr"] = _family_block(test, cqr, f"{prefix}_cqr", "seasonal_naive", opponents)
    if "test" not in out and "validation" not in out:
        raise SystemExit(f"no {prefix} results to summarise")
    return write_result(
        results_dir() / "analysis" / "cpu_study.json",
        out,
        split="test" if "test" in out else "val",
        label="cpu-study",
    )
