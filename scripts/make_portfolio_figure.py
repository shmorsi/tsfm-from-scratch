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
BG, PANEL, LINE, INK, INK2, MUTED = "#06080c", "#0d121a", "#1f2a3a", "#eaf0f8", "#b4c0d1", "#8a96a9"
SIGNAL, ICE = "#ffb547", "#8fd3ff"
FONT = "'Helvetica Neue', 'Segoe UI', system-ui, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"


def load(rel: str) -> dict[str, Any]:
    d: dict[str, Any] = json.loads((ROOT / rel).read_text())
    return d


def text(
    x: float,
    y: float,
    s: str,
    size: int = 26,
    fill: str = INK2,
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
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="18" fill="{PANEL}" stroke="{LINE}"/>',
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
            f'<line x1="{px0}" x2="{px1}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}" stroke="{LINE}"/>'
        )
        out.append(text(px0 - 14, fy(yv) + 8, f"{yv:.2f}", 20, MUTED, "end", font=MONO))
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{fx(p):.1f},{fy(c):.1f}" for i, (_, p, c) in enumerate(pts)
    )
    out.append(f'<path d="{path}" fill="none" stroke="{SIGNAL}" stroke-width="4"/>')
    for _name, p, c in pts:
        out.append(f'<circle cx="{fx(p):.1f}" cy="{fy(c):.1f}" r="11" fill="{SIGNAL}"/>')
        out.append(text(fx(p), fy(c) - 26, f"{c:.3f}", 24, INK, "middle", 600, MONO))
        out.append(text(fx(p), py0 + 44, f"{p / 1e6:.1f}M params", 21, INK2, "middle", font=MONO))
    return "".join(out)


def test_panel(x0: float, y0: float, w: float, h: float) -> str:
    tm = load("results/analysis/test_summary.json")["models"]
    head = load("results/analysis/headline.json")
    hl = load("results/analysis/headline_choice.json")["headline"]
    n = len(head["seed_runs"])
    rows = [
        ("headline", f"Ours, {hl} ({'1 seed' if n == 1 else f'{n} seeds'})", SIGNAL),
        ("upstream:TimesFM-2.5", "TimesFM-2.5, 231M (reported)", ICE),
        ("upstream:FlowState-9.1M", "FlowState, 9.1M (reported)", ICE),
        ("upstream:Auto_Arima", "AutoARIMA (reported)", INK2),
        ("upstream:Auto_ETS", "AutoETS (reported)", INK2),
        ("seasonal_naive", "Seasonal Naive", MUTED),
    ]
    if "headline" not in tm:
        tm = dict(tm, headline=tm[hl])
    rows = [(k, lab, col) for k, lab, col in rows if k in tm]
    rows.sort(key=lambda r: tm[r[0]]["crps_gmean"])
    out = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="18" fill="{PANEL}" stroke="{LINE}"/>',
        text(x0 + 40, y0 + 62, "GIFT-Eval test, scored once", 30, INK, weight=600),
        text(
            x0 + 40,
            y0 + 100,
            "97 configurations · CRPS vs Seasonal Naive, lower is better",
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
                INK if k == "headline" else INK2,
                weight=600 if k == "headline" else 400,
            )
        )
        bw = (bx1 - bx0) * v / xmax
        out.append(
            f'<rect x="{bx0}" y="{y + 14}" width="{bw:.1f}" height="{bh - 30}" rx="6" fill="{col}" opacity="{1 if k == "headline" else 0.75}"/>'
        )
        out.append(text(bx0 + bw + 14, y + 42, f"{v:.3f}", 24, INK, weight=600, font=MONO))
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
    if (ROOT / "results/analysis/headline.json").exists():
        body.append(test_panel(800, 180, 740, 760))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="Validation CRPS by model size and GIFT-Eval test CRPS against baselines">'
        + "".join(body)
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
