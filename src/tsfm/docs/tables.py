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
    rows = _model_rows(s, names)
    if "seasonal_naive" in s["models"]:
        ours = _model_rows(s, ["seasonal_naive"])[0]
        ours[0] = "Seasonal Naive (run by us; equals upstream on all metrics)"
        rows.append(ours)
    note = (
        f"Geometric mean over the {s['n_configs']} configurations of each metric divided by "
        "Seasonal Naive; rank = mean CRPS rank among the models in this table. Upstream rows "
        "are computed by us from upstream's committed result files "
        "(`src/tsfm/eval/_vendor/upstream_results`); coverage needs quantiles that those files "
        "do not contain."
    )
    return note + "\n\n" + md_table(["model", "MASE", "CRPS", "rank", "80% coverage"], rows)


@table("test_models")
def test_models(root: Path) -> str:
    """Our headline model next to the reported references and our Seasonal Naive."""
    from tsfm.docs.build import MissingResult

    s = _load(root, "results/analysis/test_summary.json")
    hl = _load(root, "results/analysis/headline_choice.json")["headline"]
    if hl not in s["models"]:
        raise MissingResult("headline test results")
    names = sorted(
        (n for n in s["models"] if n.startswith("upstream:") or n in (hl, "seasonal_naive")),
        key=lambda n: s["models"][n]["crps_gmean"],
    )
    rows = _model_rows(s, names)
    for r, n in zip(rows, names, strict=True):
        if n == hl:
            r[0] = f"**{n} (ours, pretrained from scratch)**"
        elif n == "seasonal_naive":
            r[0] = "Seasonal Naive (run by us)"
    note = (
        f"GIFT-Eval test split, {s['n_configs']} configurations; geometric mean of each metric "
        "divided by Seasonal Naive (lower is better). All rows except ours and Seasonal Naive are "
        "computed from upstream's committed result files; coverage needs quantiles those files do not contain."
    )
    return note + "\n\n" + md_table(["model", "MASE", "CRPS", "rank", "80% coverage"], rows)


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


@table("scaling")
def scaling_table(root: Path) -> str:
    from tsfm.docs.build import MissingResult

    s = _load(root, "results/analysis/scaling.json")
    fit = s.get("fit")
    if not fit:
        raise MissingResult("scaling fit")
    rows = [
        [p["name"], f"{p['params_body']:,}", f"{p['flops']:.2e}", f"{p['val_loss']:.4f}"]
        for p in fit["points"]
    ]
    lo, hi = fit["b_ci95"]
    head = (
        f"Power-law fit L = a * C^(-b) over {fit['n_runs']} runs: b = {fit['b']:.4f} "
        f"(95% bootstrap interval {lo:.4f} to {hi:.4f}, paired over validation batches)."
    )
    return head + "\n\n" + md_table(["run", "body params", "train FLOPs (6ND)", "val loss"], rows)


@table("tokenizers")
def tokenizers_table(root: Path) -> str:
    from tsfm.docs.build import MissingResult

    s = _load(root, "results/analysis/val_summary.json")
    names = [n for n in ("patch_s2", "bins_s2") if n in s["models"]]
    if len(names) < 2:
        raise MissingResult("tokenizer comparison")
    return md_table(["model", "MASE", "CRPS", "rank", "80% coverage"], _model_rows(s, names))


@table("val_models")
def val_models(root: Path) -> str:
    s = _load(root, "results/analysis/val_summary.json")
    names = sorted(s["models"], key=lambda n: s["models"][n]["crps_gmean"])
    rows = _model_rows(s, names)
    for r, n in zip(rows, names, strict=True):
        cov = s["models"][n]["coverage_mean"]
        r.insert(4, f"{cov['50']:.3f}" if cov.get("50") is not None else "n/a")
        r.append(f"{cov['90']:.3f}" if cov.get("90") is not None else "n/a")
    note = f"Validation split ({s['n_configs']} configurations from datasets held out of pretraining); geometric means of scores divided by Seasonal Naive."
    return (
        note
        + "\n\n"
        + md_table(
            ["model", "MASE", "CRPS", "rank", "50% coverage", "80% coverage", "90% coverage"], rows
        )
    )


VOL_LABELS = {
    "random_walk": "Random walk (reference)",
    "har": "HAR-RV",
    "tsfm_zero_shot": "Our model, zero-shot",
    "tsfm_finetuned": "Our model, fine-tuned",
}


def _ratio_cell(g: dict[str, Any]) -> str:
    if "wql_ratio_vs_ref" not in g:
        return "1 (reference)"
    lo, hi = g["wql_ratio_ci95"]
    return f"{g['wql_ratio_vs_ref']:.3f} [{lo:.3f}, {hi:.3f}]"


@table("vol_overall")
def vol_overall(root: Path) -> str:
    s = _load(root, "results/finance/summary.json")
    rows = []
    for m in VOL_LABELS:
        g = s["models"].get(m, {}).get("both/all_test")
        if g:
            rows.append(
                [
                    VOL_LABELS[m],
                    f"{g['origins']:,}",
                    f"{g['wql']:.4f}",
                    _ratio_cell(g),
                    f"{g['cov50']:.3f}",
                    f"{g['cov80']:.3f}",
                    f"{g['cov90']:.3f}",
                ]
            )
    return md_table(
        [
            "model",
            "origins",
            "weighted quantile loss",
            "ratio to random walk [95% CI]",
            "50% cov.",
            "80% cov.",
            "90% cov.",
        ],
        rows,
    )


@table("vol_regimes")
def vol_regimes(root: Path) -> str:
    s = _load(root, "results/finance/summary.json")
    regs = ["calm", "covid_2020", "terra_3ac_2022", "ftx_2022"]
    rows = []
    for m in VOL_LABELS:
        res = s["models"].get(m)
        if not res:
            continue
        for r in regs:
            g = res.get(f"both/{r}")
            if g:
                rows.append(
                    [
                        VOL_LABELS[m],
                        r,
                        f"{g['origins']:,}",
                        f"{g['wql']:.4f}",
                        _ratio_cell(g),
                        f"{g['cov80']:.3f}",
                    ]
                )
    return md_table(
        [
            "model",
            "regime",
            "origins",
            "weighted quantile loss",
            "ratio to random walk [95% CI]",
            "80% coverage",
        ],
        rows,
    )


def _cpu_row(block: dict[str, Any], label: str) -> list[str]:
    summary = block["summary"]
    cov = summary["coverage_mean"].get("80")
    cmp = next(
        c for c in block["comparisons"] if c["b"] == "seasonal_naive" and c["metric"] == "crps"
    )
    lo, hi = cmp["ci95"]
    seeds = block["crps_gmean_across_seeds"]
    seed_ci = seeds.get("ci95")
    seed_txt = f"[{seed_ci[0]:.3f}, {seed_ci[1]:.3f}]" if seed_ci else "n/a"
    return [
        label,
        f"{summary['mase_gmean']:.3f}",
        f"{summary['crps_gmean']:.3f}",
        f"{cmp['ratio']:.3f} [{lo:.3f}, {hi:.3f}]",
        seed_txt,
        f"{cov:.3f}" if cov is not None else "n/a",
        cmp["verdict"],
    ]


@table("cpu_test")
def cpu_test(root: Path) -> str:
    """CPU study on the official test split. Missing until that study has been scored."""
    from tsfm.docs.build import MissingResult

    doc = _load(root, "results/analysis/cpu_study.json")
    if "test" not in doc:
        raise MissingResult("cpu test results")
    rows = [_cpu_row(doc["test"], "cpu_s0, raw quantiles")]
    if "test_cqr" in doc:
        rows.append(_cpu_row(doc["test_cqr"], "cpu_s0, split-conformal"))
    note = (
        f"Official GIFT-Eval test split, {doc['test']['n_configs']} configurations, "
        f"{doc['test']['crps_gmean_across_seeds']['n']} seeds. "
        "MASE and CRPS are geometric means of each score divided by Seasonal Naive. "
        "The interval next to the ratio is a 95% paired bootstrap over configurations "
        "of the seed-averaged scores (10,000 resamples). "
        "The other interval is a 95% Student-t interval across seed-level CRPS ratios; "
        "with three seeds it is wide on purpose. "
        "Conformal expansions were fit on the validation split only. "
        "A ratio below 1 is better than Seasonal Naive."
    )
    return (
        note
        + "\n\n"
        + md_table(
            [
                "model",
                "MASE",
                "CRPS",
                "CRPS ratio vs Seasonal Naive [95% config CI]",
                "CRPS ratio across seeds [95% t CI]",
                "80% coverage",
                "vs Seasonal Naive",
            ],
            rows,
        )
    )


@table("cpu_val")
def cpu_val(root: Path) -> str:
    from tsfm.docs.build import MissingResult

    doc = _load(root, "results/analysis/cpu_study.json")
    if "validation" not in doc:
        raise MissingResult("cpu validation results")
    rows = [_cpu_row(doc["validation"], "cpu_s0, raw")]
    if "validation_cqr" in doc:
        rows.append(_cpu_row(doc["validation_cqr"], "cpu_s0, split-conformal"))
    extra = ""
    ab = doc.get("validation_ablation")
    if ab:
        extra = (
            f"\n\nPositional encoding, seed 0, same {ab['n_configs']} validation configurations: "
            f"RoPE CRPS {ab['rope']['crps_gmean']:.3f}, no positions CRPS {ab['none']['crps_gmean']:.3f}. "
            "The no-position run was not scored on the test split."
        )
    note = (
        f"Validation split used only for the conformal fit and this ablation "
        f"({doc['validation']['n_configs']} configurations from the ten held-out datasets; "
        "synthetic validation series are not included). "
        "The split-conformal row is scored on the same forecasts the expansions were fit on, "
        "so its coverage is not a held-out check. The test table is."
    )
    return (
        note
        + "\n\n"
        + md_table(
            [
                "model",
                "MASE",
                "CRPS",
                "CRPS ratio vs Seasonal Naive [95% config CI]",
                "CRPS ratio across seeds [95% t CI]",
                "80% coverage",
                "vs Seasonal Naive",
            ],
            rows,
        )
        + extra
    )


@table("cpu_vs_arima")
def cpu_vs_arima(root: Path) -> str:
    from tsfm.docs.build import MissingResult

    doc = _load(root, "results/analysis/cpu_study.json")
    if "test" not in doc:
        raise MissingResult("cpu test results")
    rows = []
    for key, label in (("test", "cpu_s0 raw"), ("test_cqr", "cpu_s0 split-conformal")):
        block = doc.get(key)
        if not block:
            continue
        for metric in ("crps", "mase"):
            cmp = next(
                (
                    c
                    for c in block["comparisons"]
                    if c["b"] == "upstream:Auto_Arima" and c["metric"] == metric
                ),
                None,
            )
            if cmp is None:
                continue
            lo, hi = cmp["ci95"]
            rows.append(
                [
                    label,
                    metric,
                    f"{cmp['ratio']:.3f}",
                    f"[{lo:.3f}, {hi:.3f}]",
                    cmp["verdict"],
                ]
            )
    if not rows:
        raise MissingResult("cpu vs AutoARIMA")
    note = (
        "Ratio of seed-averaged geometric means, model divided by upstream AutoARIMA, "
        "both already divided by Seasonal Naive. "
        "Below 1 would be better than AutoARIMA. "
        "The pre-registered decision rule: better if the interval is entirely below 1, "
        "worse if it is entirely above 1, otherwise no detectable difference."
    )
    return (
        note
        + "\n\n"
        + md_table(["model", "metric", "ratio to AutoARIMA", "95% CI", "verdict"], rows)
    )


@table("model_card_headline")
def model_card_headline(root: Path) -> str:
    h = _load(root, "results/analysis/headline_choice.json")
    name = h["headline"]
    s = _load(root, f"results/runs/{name}/summary.json")
    m = s["config"]["model"]
    t = s["config"]["train"]
    rows = [
        ["run", f"`{name}`"],
        ["parameters (total / transformer body)", f"{s['params_total']:,} / {s['params_body']:,}"],
        ["layers x width x heads", f"{m['n_layers']} x {m['d_model']} x {m['n_heads']}"],
        ["training steps x batch", f"{t['steps']:,} x {t['batch_size']}"],
        ["training tokens (patches)", f"{s['train_tokens']:,}"],
        ["training compute (6ND)", f"{s['train_flops_6ND']:.2e} FLOPs"],
        ["precision / hardware", f"{t['precision']} on one Colab T4"],
    ]
    return md_table(["", ""], rows)
