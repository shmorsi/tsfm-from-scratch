"""One SVG figure for the README, drawn from committed results.

No plotting dependency: the file is a small hand-built SVG so a clone can rebuild it
with ``tsfm cpu-study --stage report``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tsfm.paths import repo_root


def _svg(series: list[tuple[str, list[tuple[float, float]]]], title: str) -> str:
    """series: (name, [(nominal, empirical), ...]) with both axes in 0..1."""
    width, height = 640, 420
    left, right, top, bottom = 70, 20, 48, 56
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(v: float) -> float:
        return left + v * plot_w

    def y(v: float) -> float:
        return top + (1.0 - v) * plot_h

    colors = ("#b45309", "#1d4ed8", "#047857")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#fafaf9"/>',
        f'<text x="{left}" y="28" font-family="ui-sans-serif, sans-serif" font-size="16" '
        f'fill="#1c1917">{title}</text>',
    ]
    # Diagonal: perfect calibration.
    parts.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" '
        'stroke="#a8a29e" stroke-dasharray="5 4" stroke-width="1.5"/>'
    )
    for tick in (0.0, 0.5, 0.8, 0.9, 1.0):
        parts.append(
            f'<line x1="{x(tick):.1f}" y1="{y(0):.1f}" x2="{x(tick):.1f}" y2="{y(0) + 6:.1f}" '
            'stroke="#78716c"/>'
        )
        parts.append(
            f'<text x="{x(tick):.1f}" y="{height - 28}" text-anchor="middle" '
            f'font-family="ui-sans-serif, sans-serif" font-size="12" fill="#44403c">'
            f"{tick:.1f}</text>"
        )
        parts.append(
            f'<text x="{left - 10}" y="{y(tick) + 4:.1f}" text-anchor="end" '
            f'font-family="ui-sans-serif, sans-serif" font-size="12" fill="#44403c">'
            f"{tick:.1f}</text>"
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" '
        'font-family="ui-sans-serif, sans-serif" font-size="13" fill="#1c1917">'
        "nominal coverage</text>"
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.0f})" '
        'font-family="ui-sans-serif, sans-serif" font-size="13" fill="#1c1917">'
        "empirical coverage</text>"
    )
    for i, (name, pts) in enumerate(series):
        color = colors[i % len(colors)]
        coords = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}"/>'
        )
        for a, b in pts:
            parts.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="{color}"/>')
        lx = left + 12
        ly = top + 18 + i * 18
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="14" height="4" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 20}" y="{ly}" font-family="ui-sans-serif, sans-serif" '
            f'font-size="13" fill="#1c1917">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def coverage_series(
    root: Path | None = None,
) -> tuple[str, list[tuple[str, list[tuple[float, float]]]]]:
    """Prefer the CPU study's test coverage; otherwise the T4 validation result."""
    root = root or repo_root()
    cpu = root / "results" / "analysis" / "cpu_study.json"
    if cpu.exists():
        doc = json.loads(cpu.read_text())
        series: list[tuple[str, list[tuple[float, float]]]] = []

        def points(cov: dict[str, float | None]) -> list[tuple[float, float]] | None:
            out: list[tuple[float, float]] = []
            for nominal, key in ((0.5, "50"), (0.8, "80"), (0.9, "90")):
                value = cov.get(key)
                if value is None:
                    return None
                out.append((nominal, float(value)))
            return out

        if "test" in doc:
            pts = points(doc["test"]["summary"]["coverage_mean"])
            if pts:
                series.append(("test, raw quantiles", pts))
        if "test_cqr" in doc:
            pts = points(doc["test_cqr"]["summary"]["coverage_mean"])
            if pts:
                series.append(("test, split-conformal", pts))
        if series:
            return "GIFT-Eval test coverage", series
    val = json.loads((root / "results" / "analysis" / "val_summary.json").read_text())
    cov = val["models"]["patch_s2"]["coverage_mean"]
    return (
        "Validation coverage, patch_s2 (one seed)",
        [
            (
                "patch_s2 validation",
                [(0.5, float(cov["50"])), (0.8, float(cov["80"])), (0.9, float(cov["90"]))],
            )
        ],
    )


def write_coverage_figure(root: Path | None = None) -> Path:
    root = root or repo_root()
    title, series = coverage_series(root)
    path = root / "docs" / "figures" / "coverage.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg(series, title))
    return path
