"""Render the portfolio card figure (an SVG) from the committed results: validation CRPS
against model size, and test CRPS of the headline model next to the baselines and
published models. Every value drawn is read from results/.

    uv run python scripts/make_portfolio_figure.py --out ../portfolio/assets/img/tsfm-results.svg
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
W, H = 1600, 1000
# Same ground, grid and accent as docs/figures. One warm series; comparisons stay neutral.
BG, PANEL, GRID = "#07090d", "#10141b", "#5a626a"
INK, SECOND, MUTED, FAINT = "#f4f7fb", "#d5dde6", "#c5ced8", "#8fa0b3"
SIGNAL = "#e4c48a"
FONT = "'Inter', 'Space Grotesk', 'Helvetica Neue', Helvetica, ui-sans-serif, system-ui, sans-serif"
MONO = FONT


def load(rel: str) -> dict[str, Any]:
    d: dict[str, Any] = json.loads((ROOT / rel).read_text())
    return d


def text(
    x: float,
    y: float,
    s: str,
    size: int = 26,
    fill: str = SECOND,
    anchor: str = "start",
    weight: int = 400,
    font: str = FONT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(s)}</text>'
    )


def scaling_panel(x0: float, y0: float, w: float, h: float) -> str:
    val = load("results/analysis/val_summary.json")["models"]
    pts = []
    for name in ("patch_s0", "patch_s1", "patch_s2", "patch_s3"):
        if name in val:
            p = load(f"results/runs/{name}/summary.json")["params_total"]
            pts.append((name, p, val[name]["crps_gmean"]))
    out = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="2" fill="{PANEL}" stroke="{GRID}"/>',
        text(x0 + 40, y0 + 62, "Validation: bigger model, lower error", 30, INK, weight=600),
        text(
            x0 + 40,
            y0 + 100,
            "CRPS vs Seasonal Naive (lower is better), held-out datasets",
            22,
            MUTED,
        ),
    ]
    px0, px1, py0, py1 = x0 + 110, x0 + w - 60, y0 + h - 110, y0 + 160
    lx = [math.log10(p) for _, p, _ in pts]
    xmin, xmax = min(lx) - 0.15, max(lx) + 0.15
    ys = [c for *_, c in pts]
    ymin, ymax = min(ys) - 0.05, max(ys) + 0.05
    fx = lambda v: px0 + (math.log10(v) - xmin) / (xmax - xmin) * (px1 - px0)  # noqa: E731
    fy = lambda v: py0 - (v - ymin) / (ymax - ymin) * (py0 - py1)  # noqa: E731
    for k in range(5):
        yv = ymin + k * (ymax - ymin) / 4
        out.append(
            f'<line x1="{px0}" x2="{px1}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}" stroke="{GRID}"/>'
        )
        out.append(text(px0 - 14, fy(yv) + 8, f"{yv:.2f}", 20, MUTED, "end", font=MONO))
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{fx(p):.1f},{fy(c):.1f}" for i, (_, p, c) in enumerate(pts)
    )
    out.append(f'<path d="{path}" fill="none" stroke="{SIGNAL}" stroke-width="1.75"/>')
    for _name, p, c in pts:
        out.append(
            f'<circle cx="{fx(p):.1f}" cy="{fy(c):.1f}" r="7" fill="{SIGNAL}" '
            f'stroke="{BG}" stroke-width="2"/>'
        )
        out.append(
            text(fx(p) + 22, fy(c) - 22, f"{c:.3f}", 24, INK, "start", 600, MONO)
            if _name != pts[-1][0]
            else text(fx(p), fy(c) - 26, f"{c:.3f}", 24, INK, "middle", 600, MONO)
        )
        out.append(text(fx(p), py0 + 44, f"{p / 1e6:.1f}M params", 21, SECOND, "middle", font=MONO))
    return "".join(out)


def test_panel(x0: float, y0: float, w: float, h: float) -> str:
    summary = load("results/analysis/test_summary.json")
    tm = summary["models"]
    head = load("results/analysis/headline.json")
    hl = load("results/analysis/headline_choice.json")["headline"]
    n = len(head["seed_runs"])
    rows = [
        ("headline", f"Ours, {hl} ({'1 seed' if n == 1 else f'{n} seeds'})", SIGNAL),
        ("upstream:TimesFM-2.5", "TimesFM-2.5 (reported)", SECOND),
        ("upstream:FlowState-9.1M", "FlowState-9.1M (reported)", SECOND),
        ("upstream:Auto_Arima", "AutoARIMA (reported)", SECOND),
        ("upstream:Auto_ETS", "AutoETS (reported)", SECOND),
        ("seasonal_naive", "Seasonal Naive", MUTED),
    ]
    if "headline" not in tm:
        tm = dict(tm, headline=tm[hl])
    rows = [(k, lab, col) for k, lab, col in rows if k in tm]
    rows.sort(key=lambda r: tm[r[0]]["crps_gmean"])
    out = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="2" fill="{PANEL}" stroke="{GRID}"/>',
        text(x0 + 40, y0 + 62, "GIFT-Eval test, scored once", 30, INK, weight=600),
        text(
            x0 + 40,
            y0 + 100,
            f"{summary['n_configs']} configurations · CRPS vs Seasonal Naive, lower is better",
            22,
            MUTED,
        ),
    ]
    top, bh, gap = y0 + 150, 64, 40
    xmax = max(tm[k]["crps_gmean"] for k, *_ in rows) * 1.08
    bx0, bx1 = x0 + 40, x0 + w - 150
    for i, (k, lab, col) in enumerate(rows):
        v = tm[k]["crps_gmean"]
        y = top + i * (bh + gap)
        out.append(
            text(
                bx0,
                y + 2,
                lab,
                23,
                INK if k == "headline" else SECOND,
                weight=600 if k == "headline" else 400,
            )
        )
        bw = (bx1 - bx0) * v / xmax
        out.append(
            f'<rect x="{bx0}" y="{y + 14}" width="{bw:.1f}" height="{bh - 30}" rx="2" fill="{col}"/>'
        )
        out.append(text(bx0 + bw + 14, y + 42, f"{v:.3f}", 24, INK, weight=600, font=MONO))
    return "".join(out)


def val_panel(x0: float, y0: float, w: float, h: float) -> str:
    """Zero-shot CRPS and MASE of each size next to Seasonal Naive (= 1)."""
    val = load("results/analysis/val_summary.json")
    rows = [(n, n, SIGNAL) for n in ("patch_s2", "patch_s1", "patch_s0") if n in val["models"]]
    rows.append(("seasonal_naive", "Seasonal Naive (baseline)", MUTED))
    out = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="2" fill="{PANEL}" stroke="{GRID}"/>',
        text(x0 + 40, y0 + 62, "Zero-shot vs the naive baseline", 30, INK, weight=600),
        text(
            x0 + 40,
            y0 + 100,
            f"{val['n_configs']} held-out configurations · relative to Seasonal Naive",
            22,
            MUTED,
        ),
    ]
    top, gap = y0 + 160, 140
    bx0, bx1 = x0 + 40, x0 + w - 150
    for i, (k, lab, col) in enumerate(rows):
        m = val["models"][k]
        y = top + i * gap
        out.append(
            text(
                bx0,
                y,
                lab,
                23,
                INK if k == "patch_s2" else SECOND,
                weight=600 if k == "patch_s2" else 400,
                font=MONO if k.startswith("patch") else FONT,
            )
        )
        for j, (metric, key, c2) in enumerate(
            (
                ("CRPS", "crps_gmean", col),
                ("MASE", "mase_gmean", SECOND if col == SIGNAL else MUTED),
            )
        ):
            v = m[key]
            yy = y + 18 + j * 42
            bw = (bx1 - bx0 - 70) * v / 1.2
            out.append(text(bx0, yy + 26, metric, 18, MUTED, font=MONO))
            out.append(
                f'<rect x="{bx0 + 70}" y="{yy + 6}" width="{bw:.1f}" height="28" rx="2" fill="{c2}"/>'
            )
            out.append(
                text(bx0 + 70 + bw + 12, yy + 28, f"{v:.3f}", 22, INK, weight=600, font=MONO)
            )
    return "".join(out)


def build() -> str:
    body = [
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        text(
            60, 84, "A time-series foundation model, pretrained from scratch", 40, INK, weight=600
        ),
        text(
            60,
            130,
            "Plain PyTorch · leakage-audited corpus · pre-registered evaluation · free T4 GPU",
            24,
            MUTED,
        ),
        scaling_panel(60, 180, 700, 760),
    ]
    has_headline = (ROOT / "results/analysis/headline.json").exists()
    if has_headline:
        body.append(test_panel(800, 180, 740, 760))
        label = "Validation CRPS by model size and GIFT-Eval test CRPS against baselines"
    else:
        body.append(val_panel(800, 180, 740, 760))
        label = "Validation CRPS by model size, and zero-shot scores against Seasonal Naive"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="fig-title fig-desc">'
        f'<title id="fig-title">{html.escape(label)}</title>'
        f'<desc id="fig-desc">{html.escape(label)}</desc>'
        + "".join(body)
        + f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" fill="none" stroke="{GRID}"/>'
        + "</svg>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(build())
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
