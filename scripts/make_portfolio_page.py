"""Render the portfolio case-study page (shmorsi.github.io/projects/tsfm.html) from the
committed results, so the page can only show numbers that are in results/.

    uv run python scripts/make_portfolio_page.py --out ../portfolio/projects/tsfm.html

The page uses the portfolio site's own stylesheet and layout classes.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/shmorsi/tsfm-from-scratch"


def load(rel: str) -> dict[str, Any]:
    d: dict[str, Any] = json.loads((ROOT / rel).read_text())
    return d


def esc(s: object) -> str:
    return html.escape(str(s))


def table(caption: str, head: list[str], rows: list[list[str]], best: int | None = None) -> str:
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    body = []
    for i, r in enumerate(rows):
        cells = f'<th scope="row">{esc(r[0])}</th>' + "".join(
            f"<td{' class="best"' if best == i else ''}>{esc(c)}</td>" for c in r[1:]
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap" tabindex="0" role="region" aria-label="{esc(caption)}"><table>'
        f"<caption>{esc(caption)}</caption><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def build() -> str:
    leak = load("results/data/leakage.json")["totals"]
    r1 = load("results/data/leakage_round1.json")["totals"]
    insp = load("results/data/leakage_inspection.json")["pairs"]
    man = load("results/data/corpus_manifest.json")["totals"]
    test = load("results/analysis/test_summary.json")
    val = load("results/analysis/val_summary.json")
    choice = load("results/analysis/headline_choice.json")
    head = load("results/analysis/headline.json")
    vol = load("results/finance/summary.json")["models"]
    scaling = load("results/analysis/scaling.json").get("fit")
    hl = choice["headline"]
    hsum = load(f"results/runs/{hl}/summary.json")

    prim = next(c for c in head["comparisons"] if c.get("primary"))
    copies = [p for p in insp if p["verdict"] == "copy"]

    # --- headline test table: our seed-averaged model, references
    tm = test["models"]
    order = [
        "headline",
        "upstream:TimesFM-2.5",
        "upstream:FlowState-9.1M",
        "flowstate_9.1m",
        "upstream:Auto_Arima",
        "seasonal_naive",
        "upstream:Auto_ETS",
    ]
    n_seeds = len(head["seed_runs"])
    seed_txt = (
        "mean of 3 seeds"
        if n_seeds == 3
        else ("1 seed" if n_seeds == 1 else f"mean of {n_seeds} seeds")
    )
    labels = {
        "headline": f"Ours ({hl}, {hsum['params_total'] / 1e6:.0f}M, {seed_txt})",
        "upstream:TimesFM-2.5": "TimesFM-2.5, 231M (reported)",
        "upstream:FlowState-9.1M": "FlowState, 9.1M (reported)",
        "flowstate_9.1m": "FlowState, 9.1M (run by us)",
        "upstream:Auto_Arima": "AutoARIMA (reported)",
        "seasonal_naive": "Seasonal Naive (run by us)",
        "upstream:Auto_ETS": "AutoETS (reported)",
    }
    trows = []
    for k in order:
        m = tm.get(k)
        if m is None:
            continue
        cov = m["coverage_mean"].get("80")
        trows.append(
            [
                labels[k],
                f"{m['crps_gmean']:.3f}",
                f"{m['mase_gmean']:.3f}",
                f"{cov:.2f}" if cov is not None else "n/a",
            ]
        )
    test_table = table(
        "GIFT-Eval test, 97 configurations · geometric mean of score / Seasonal Naive, lower is better",
        ["Model", "CRPS", "MASE", "80% coverage"],
        trows,
    )

    # --- scaling (validation)
    vrows = []
    for name in ("patch_s0", "patch_s1", "patch_s2", "patch_s3", "bins_s2"):
        m = val["models"].get(name)
        if m is None:
            continue
        p = load(f"results/runs/{name}/summary.json")["params_total"]
        vrows.append([name, f"{p / 1e6:.1f}M", f"{m['crps_gmean']:.3f}", f"{m['mase_gmean']:.3f}"])
    val_table = table(
        "Validation split (held-out datasets) · score / Seasonal Naive",
        ["Run", "Parameters", "CRPS", "MASE"],
        vrows,
    )
    scaling_text = ""
    if scaling:
        lo, hi = scaling["b_ci95"]
        scaling_text = f"<p>Validation loss falls with training compute as a power law with exponent {scaling['b']:.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f}) across {scaling['n_runs']} patch-model sizes.</p>"

    # --- volatility
    vlab = {
        "random_walk": "Random walk",
        "har": "HAR-RV",
        "tsfm_zero_shot": "Ours, zero-shot",
        "tsfm_finetuned": "Ours, fine-tuned",
    }
    volrows = []
    for k, lab in vlab.items():
        g = vol.get(k, {}).get("both/all_test")
        if not g:
            continue
        ratio = (
            "1 (reference)"
            if "wql_ratio_vs_ref" not in g
            else f"{g['wql_ratio_vs_ref']:.3f} [{g['wql_ratio_ci95'][0]:.3f}, {g['wql_ratio_ci95'][1]:.3f}]"
        )
        volrows.append([lab, ratio, f"{g['cov80']:.2f}"])
    vol_table = table(
        "Daily realized volatility, BTC and ETH, 2020-2026 · weighted quantile loss vs random walk [95% CI]",
        ["Model", "Loss ratio", "80% coverage"],
        volrows,
    )

    ours = tm["headline"]["crps_gmean"]
    ahead = [
        labels[k].split(" (")[0].split(",")[0]
        for k in ("upstream:TimesFM-2.5", "upstream:FlowState-9.1M")
        if k in tm and tm[k]["crps_gmean"] < ours
    ]
    behind = [
        labels[k].split(" (")[0].split(",")[0]
        for k in ("upstream:TimesFM-2.5", "upstream:FlowState-9.1M")
        if k in tm and tm[k]["crps_gmean"] >= ours
    ]
    published_txt = ""
    if ahead:
        published_txt += f"Published foundation models trained with far more data and compute score better on CRPS ({', '.join(ahead)}). "
    if behind:
        published_txt += f"It matches or beats the reported CRPS of {', '.join(behind)}. "
    has_ft = "both/all_test" in vol.get("tsfm_finetuned", {})
    glance = [
        (
            f"{prim['ratio']:.3f}",
            f"CRPS ratio to AutoARIMA on the GIFT-Eval test set (95% CI {prim['ci95'][0]:.3f} to {prim['ci95'][1]:.3f}): pre-registered verdict <em>{esc(prim['verdict'])}</em>",
        ),
        (
            f"{r1['series_with_hits']:,}",
            "series in the upstream &ldquo;non-leaking&rdquo; pretraining corpus found to overlap GIFT-Eval, removed before training",
        ),
        (
            f"{hsum['params_total'] / 1e6:.1f}M",
            f"parameters, trained from scratch on one free Colab T4 ({hsum['train_flops_6ND']:.1e} FLOPs)",
        ),
        (
            "1e-9",
            "agreement with upstream's Seasonal Naive on all 11 official metrics and 97 configurations",
        ),
    ]
    glance_html = "".join(
        f'<div class="readout"><b>{b}</b><span>{s}</span></div>' for b, s in glance
    )
    copy_list = "".join(
        f"<li><code>{esc(p['source'])}</code> &harr; <code>{esc(p['gifteval'])}</code>: {p['series']} series</li>"
        for p in copies
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <title>Time-series foundation model from scratch · Sherif Morsi</title>
  <meta name="description" content="A small time-series foundation model pretrained from scratch and evaluated honestly on GIFT-Eval: leakage audit, pre-registration, scaling runs and a realized-volatility study.">
  <meta name="theme-color" content="#06080c">
  <meta name="color-scheme" content="dark">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://shmorsi.github.io/projects/tsfm.html">
  <meta property="og:title" content="Time-series foundation model from scratch · Sherif Morsi">
  <meta property="og:image" content="https://shmorsi.github.io/assets/og.png">
  <link rel="canonical" href="https://shmorsi.github.io/projects/tsfm.html">
  <link rel="icon" href="../assets/favicon.svg" type="image/svg+xml">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Unbounded:wght@500;600&display=swap">
  <link rel="stylesheet" href="../assets/site.css">
  <script src="../assets/site.js" defer></script>
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header class="site-header">
    <div class="wrap header-inner">
      <a class="brand" href="../" aria-label="Sherif Morsi, home"><span class="brand-mark" aria-hidden="true">SM</span><span class="brand-name">Sherif Morsi</span></a>
      <button class="menu-button" type="button" aria-expanded="false" aria-controls="site-nav" data-menu-button>Menu</button>
      <nav class="site-nav" id="site-nav" aria-label="Main" data-site-nav>
        <a href="../#work">Work</a>
        <a href="../#experience">Experience</a>
        <a href="../#activities">Activities</a>
        <a href="../#about">About</a>
        <a class="nav-cta" href="../#contact">Contact</a>
      </nav>
    </div>
  </header>

  <main id="main">
    <section class="cs-hero" aria-labelledby="cs-title">
      <div class="graticule" aria-hidden="true"></div>
      <div class="wrap cs-hero-inner">
        <a class="back" href="../#tsfm">← All projects</a>
        <span class="label signal">01 · Deep learning · Forecasting · Research engineering</span>
        <h1 id="cs-title">A time-series foundation model, from scratch</h1>
        <p class="cs-question">How far does a small transformer get on a public zero-shot forecasting benchmark when it is pretrained from scratch, on data checked for overlap with the benchmark, and judged by a test fixed in advance?</p>
        <p class="cs-summary">A decoder-only transformer, its tokenizers, training loop and sampling written in plain PyTorch; pretrained on a pinned, leakage-audited corpus plus synthetic Gaussian-process series; scored once on the 97 GIFT-Eval configurations against classical baselines and published models; then {"zero-shot and fine-tuned" if has_ft else "zero-shot"} on daily realized volatility of crypto markets, tested across the 2020 and 2022 market breaks.</p>
        <div class="project-links">
          <a class="button primary" href="{REPO}" target="_blank" rel="noopener noreferrer">Code and results <span aria-hidden="true">↗</span></a>
          <a class="button ghost" href="{REPO}/blob/main/RESEARCH_NOTE.md" target="_blank" rel="noopener noreferrer">Read the research note <span aria-hidden="true">↗</span></a>
        </div>
      </div>
    </section>

    <div class="wrap cs-body">
      <section class="glance" aria-label="At a glance">{glance_html}</section>

      <section class="cs-section" aria-labelledby="s-problem">
        <h2 id="s-problem">Problem</h2>
        <div class="prose">
          <p>Foundation models for time series are pretrained on large collections of series and asked to forecast new ones with no further training. Benchmarks such as GIFT-Eval make them comparable, but results are only meaningful if no benchmark data leaks into pretraining and if the evaluation is fixed before the scores are seen. This project builds the whole pipeline itself and holds it to both standards.</p>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-data">
        <h2 id="s-data">Data, and a leakage finding</h2>
        <div class="prose">
          <p>Pretraining uses {man["pretrain"]["datasets"]} datasets ({man["pretrain"]["points"]:,} points) from GiftEvalPretrain, the corpus its authors describe as non-leaking, plus synthetic Gaussian-process series. Every raw file is pinned by its SHA-256 fingerprint. A content-based check compared every pretraining series with every GIFT-Eval series, at native frequency and after daily aggregation, and found genuine copies:</p>
          <ul>{copy_list}</ul>
          <p>All {leak["excluded_series_total"]:,} flagged series were removed, and the re-run on the final corpus finds {leak["native_hits"]} matches. A first version of the check also produced false matches from linearly interpolated gaps, which look identical in every series once normalised; inspecting each match separated those from real copies.</p>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-method">
        <h2 id="s-method">Method</h2>
        <div class="prose">
          <ul>
            <li><strong>Model:</strong> decoder-only transformer with pre-norm RMSNorm, rotary position embeddings, SwiGLU and a key/value cache, written without model libraries. Two tokenizers: mean-scaled 4,096-bin quantisation with cross-entropy, and patches with a non-crossing quantile head.</li>
            <li><strong>Training:</strong> mixed precision, gradient accumulation, cosine schedule, checkpoints to Google Drive every 10 minutes; a resumed run is bit-identical to an uninterrupted one (tested). Trained on a free Colab T4; profiling showed half of each step went to cumulative-sum kernels, which were replaced by equivalent cheaper operations.</li>
            <li><strong>Evaluation:</strong> GluonTS exactly as GIFT-Eval scores the leaderboard, verified by reproducing upstream's Seasonal Naive on every metric. The split, metrics, baselines, primary comparison and decision rule were committed and tagged in git before any of our models saw the test set, which each model faced once.</li>
          </ul>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-results">
        <h2 id="s-results">Results</h2>
        <div class="prose">
          {test_table}
          <p>The headline model was chosen on held-out validation datasets by the pre-registered rule. Against AutoARIMA, its test CRPS ratio is {prim["ratio"]:.3f} (95% bootstrap interval {prim["ci95"][0]:.3f} to {prim["ci95"][1]:.3f} over configurations), so the verdict fixed in advance is <strong>{esc(prim["verdict"])}</strong>. {published_txt}</p>
          {val_table}
          {scaling_text}
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-vol">
        <h2 id="s-vol">{"Fine-tuning on" if has_ft else "Transfer to"} market volatility</h2>
        <div class="prose">
          <p>Daily realized volatility of BTCUSDT and ETHUSDT, computed from Binance Vision 5-minute data, forecast 22 days ahead from every day of 2020 to 2026, with the 2020 crash and the 2022 collapses as stress regimes. The design was pre-registered separately.</p>
          {vol_table}
          <p>These are statistical forecast scores only; no trading performance is claimed.</p>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-miss">
        <h2 id="s-miss">What didn't work, and limits</h2>
        <div class="prose">
          <div class="callout">
            <span class="label">Reported as found</span>
            <ul>
              <li>The first GPU run diverged: future patches reach about a million in scaled units when the context is near zero, overflowing half precision. Clipping fixed it; the broken runs are discarded and recorded in the repository.</li>
              <li>Prediction intervals are too narrow, worst at long horizons, because longer forecasts feed back medians.</li>
              <li>Scope was cut before test scoring to fit a free GPU and a deadline, and the cuts are recorded in the repository: three model sizes instead of five, the second tokenizer trained but not scored, and the headline scored with {seed_txt} instead of three, so seed-to-seed variation is not measured.</li>
              <li>The leakage check finds exact and rescaled copies only, not noisy or re-sampled ones.</li>
            </ul>
          </div>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="s-eng">
        <h2 id="s-eng">Engineering</h2>
        <div class="prose">
          <ul>
            <li>Every document in the repository is generated from committed result files, and a test fails if one contains a number that is not in them. This page is generated the same way.</li>
            <li>A tiny configuration runs the whole pipeline, from verified download to evaluation, in continuous integration on every push.</li>
          </ul>
          <p class="stack">Python 3.12 · PyTorch · NumPy · pandas · Parquet · GluonTS · statsforecast · pytest · Hypothesis · ruff · mypy · uv · GitHub Actions · Google Colab</p>
        </div>
      </section>

      <nav class="next-project" aria-label="Next project">
        <a class="next" href="world-shock-lab.html"><span class="label">Next project · 02</span><strong>World Shock Lab →</strong></a>
        <a class="text-link" href="../#work">All projects</a>
      </nav>
    </div>
  </main>

  <footer class="wrap site-footer">
    <span>© 2026 Sherif Morsi · Montréal</span>
    <a href="mailto:shmorsi@outlook.com">shmorsi@outlook.com</a>
    <a href="#main">Back to top ↑</a>
  </footer>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(build())
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
