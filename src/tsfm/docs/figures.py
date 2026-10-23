"""SVG figures for the generated documents, drawn from committed results.

No plotting dependency. ``tsfm docs build`` rewrites these files, and ``tsfm docs check``
fails if they drift from the results they plot. Both charts share one palette and one
type scale.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

from tsfm.paths import repo_root

# Warm paper, one accent for this repository's series, one ink blue for a comparison.
BG = "#f7f6f3"
INK = "#1c1917"
MUTED = "#57534e"
FAINT = "#a8a29e"
GRID = "#e7e5e4"
SIGNAL = "#9a3412"
SECOND = "#1e3a5f"
FONT = "ui-sans-serif, system-ui, sans-serif"
SERIES_COLORS = (SIGNAL, SECOND, "#3f6212")

COVERAGE_REL = "docs/figures/coverage.svg"
SCALING_REL = "docs/figures/scaling.svg"


@dataclass(frozen=True)
class _Box:
    width: int = 720
    height: int = 420
    left: int = 64
    right: int = 24
    top: int = 52
    bottom: int = 58

    @property
    def plot_w(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> int:
        return self.height - self.top - self.bottom


def _text(
    x: float,
    y: float,
    body: str,
    size: int = 13,
    fill: str = INK,
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(body)}</text>'
    )


def _open(box: _Box, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{box.width}" height="{box.height}" '
        f'viewBox="0 0 {box.width} {box.height}" role="img" aria-labelledby="fig-title fig-desc">',
        f'<title id="fig-title">{html.escape(title)}</title>',
        f'<desc id="fig-desc">{html.escape(desc)}</desc>',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        _text(box.left, 28, title, size=16, weight=600),
    ]


def _nice_step(span: float, count: int) -> float:
    if span <= 0:
        return 1.0
    raw = span / count
    exp = math.floor(math.log10(raw))
    base = raw / (10.0**exp)
    nice = 1.0 if base <= 1 else 2.0 if base <= 2 else 5.0 if base <= 5 else 10.0
    return nice * (10.0**exp)


def _ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    step = _nice_step(hi - lo, count)
    start = math.ceil(lo / step) * step
    out: list[float] = []
    v = start
    # A small epsilon keeps the last tick when it lands on `hi`.
    while v <= hi + step * 1e-6:
        out.append(v)
        v += step
    return out


def _fmt(v: float, step: float) -> str:
    if step >= 1:
        return f"{v:.0f}"
    decimals = max(0, -math.floor(math.log10(step)))
    return f"{v:.{decimals}f}"


def _sup(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


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


def coverage_svg(root: Path | None = None) -> str:
    """Nominal coverage against empirical coverage. The dashed line is perfect calibration."""
    title, series = coverage_series(root)
    box = _Box()
    parts = _open(
        box,
        title,
        "Empirical coverage plotted against nominal coverage. "
        "The dashed line is perfect calibration.",
    )

    def x(v: float) -> float:
        return box.left + v * box.plot_w

    def y(v: float) -> float:
        return box.top + (1.0 - v) * box.plot_h

    parts.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="{FAINT}" stroke-dasharray="4 4" stroke-width="1.5"/>'
    )
    for tick in (0.0, 0.5, 0.8, 0.9, 1.0):
        parts.append(
            f'<line x1="{box.left}" y1="{y(tick):.1f}" x2="{box.left + box.plot_w}" '
            f'y2="{y(tick):.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<line x1="{x(tick):.1f}" y1="{y(0):.1f}" x2="{x(tick):.1f}" y2="{y(0) + 5:.1f}" '
            f'stroke="{MUTED}"/>'
        )
        parts.append(
            _text(x(tick), box.height - 28, f"{tick:.1f}", size=12, fill=MUTED, anchor="middle")
        )
        parts.append(
            _text(box.left - 8, y(tick) + 4, f"{tick:.1f}", size=12, fill=MUTED, anchor="end")
        )
    parts.append(
        _text(
            box.left + box.plot_w / 2, box.height - 8, "nominal coverage", size=13, anchor="middle"
        )
    )
    parts.append(
        f'<text x="16" y="{box.top + box.plot_h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {box.top + box.plot_h / 2:.0f})" font-family="{FONT}" '
        f'font-size="13" fill="{INK}">empirical coverage</text>'
    )
    for i, (name, pts) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        coords = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.25" points="{coords}"/>'
        )
        for a, b in pts:
            parts.append(
                f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="{color}" '
                f'stroke="{BG}" stroke-width="1.5"/>'
            )
        ly = box.top + 16 + i * 18
        parts.append(
            f'<line x1="{box.left + 8}" y1="{ly - 4}" x2="{box.left + 26}" y2="{ly - 4}" stroke="{color}" stroke-width="2.25"/>'
        )
        parts.append(_text(box.left + 32, ly, name, size=13))
    parts.append("</svg>\n")
    return "\n".join(parts)


def _scaling_fit(root: Path) -> dict[str, object] | None:
    path = root / "results" / "analysis" / "scaling.json"
    if not path.exists():
        return None
    fit = json.loads(path.read_text()).get("fit")
    if not isinstance(fit, dict) or not fit.get("points") or not fit.get("curve"):
        return None
    return fit


def scaling_svg(root: Path | None = None) -> str | None:
    """Validation loss against training compute, with the fitted power law and its band.

    Returns ``None`` when ``results/analysis/scaling.json`` has no fit yet.
    """
    root = root or repo_root()
    fit = _scaling_fit(root)
    if fit is None:
        return None
    points = fit["points"]
    curve = fit["curve"]
    assert isinstance(points, list)
    assert isinstance(curve, dict)
    xs = [float(p["flops"]) for p in points]
    ys = [float(p["val_loss"]) for p in points]
    cflops = [float(v) for v in curve["flops"]]
    clo = [float(v) for v in curve["lo"]]
    chi = [float(v) for v in curve["hi"]]
    cfit = [float(v) for v in curve["fit"]]
    log_lo = math.log10(min([*xs, *cflops]))
    log_hi = math.log10(max([*xs, *cflops]))
    y_lo = min([*ys, *clo, *cfit])
    y_hi = max([*ys, *chi, *cfit])
    log_pad = (log_hi - log_lo) * 0.06 or 0.1
    y_pad = (y_hi - y_lo) * 0.12 or 0.01
    log_lo -= log_pad
    log_hi += log_pad
    y_lo -= y_pad
    y_hi += y_pad
    box = _Box(right=28)
    parts = _open(
        box,
        "Validation loss vs training compute",
        "Validation loss plotted against training compute on a logarithmic axis. "
        "The line is the fitted power law and the shaded band is the paired bootstrap.",
    )

    def x(flops: float) -> float:
        return box.left + (math.log10(flops) - log_lo) / (log_hi - log_lo) * box.plot_w

    def y(loss: float) -> float:
        return box.top + (y_hi - loss) / (y_hi - y_lo) * box.plot_h

    y_step = _nice_step(y_hi - y_lo, 5)
    for tick in _ticks(y_lo, y_hi, 5):
        parts.append(
            f'<line x1="{box.left}" y1="{y(tick):.1f}" x2="{box.left + box.plot_w}" '
            f'y2="{y(tick):.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            _text(box.left - 8, y(tick) + 4, _fmt(tick, y_step), size=12, fill=MUTED, anchor="end")
        )
    for exp in range(math.ceil(log_lo), math.floor(log_hi) + 1):
        fx = x(10**exp)
        parts.append(
            f'<line x1="{fx:.1f}" y1="{y(y_lo):.1f}" x2="{fx:.1f}" y2="{y(y_lo) + 5:.1f}" stroke="{MUTED}"/>'
        )
        parts.append(
            _text(fx, box.height - 28, f"10{_sup(exp)}", size=12, fill=MUTED, anchor="middle")
        )
    parts.append(
        _text(
            box.left + box.plot_w / 2,
            box.height - 8,
            "training compute, FLOPs (6ND)",
            size=13,
            anchor="middle",
        )
    )
    parts.append(
        f'<text x="16" y="{box.top + box.plot_h / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {box.top + box.plot_h / 2:.0f})" font-family="{FONT}" '
        f'font-size="13" fill="{INK}">validation loss</text>'
    )
    band = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in zip(cflops, chi, strict=True))
    band += " " + " ".join(
        f"{x(f):.1f},{y(v):.1f}" for f, v in zip(reversed(cflops), reversed(clo), strict=True)
    )
    parts.append(f'<polygon points="{band}" fill="{SIGNAL}" fill-opacity="0.12"/>')
    fit_line = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in zip(cflops, cfit, strict=True))
    parts.append(
        f'<polyline fill="none" stroke="{SIGNAL}" stroke-width="2.25" points="{fit_line}"/>'
    )
    for p in points:
        cx, cy = x(float(p["flops"])), y(float(p["val_loss"]))
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{SIGNAL}" stroke="{BG}" stroke-width="1.5"/>'
        )
        label_y = cy - 10 if cy - 10 > box.top + 14 else cy + 16
        label = str(p["name"])
        # Keep the name inside the frame. The largest run sits against the right edge.
        if cx + 8 + len(label) * 7 > box.width - 8:
            parts.append(_text(cx - 8, label_y, label, size=12, anchor="end"))
        else:
            parts.append(_text(cx + 8, label_y, label, size=12))
    # Legend sits in the empty corner: high compute, high loss.
    lx, ly = box.left + box.plot_w - 188, box.top + 8
    parts.append(
        f'<rect x="{lx}" y="{ly}" width="180" height="62" rx="4" fill="{BG}" stroke="{GRID}"/>'
    )
    parts.append(
        f'<line x1="{lx + 12}" y1="{ly + 16}" x2="{lx + 32}" y2="{ly + 16}" stroke="{SIGNAL}" stroke-width="2.25"/>'
    )
    parts.append(_text(lx + 40, ly + 20, "power-law fit", size=12))
    parts.append(
        f'<rect x="{lx + 12}" y="{ly + 30}" width="20" height="8" fill="{SIGNAL}" fill-opacity="0.25"/>'
    )
    parts.append(_text(lx + 40, ly + 38, "95% paired bootstrap", size=12))
    parts.append(f'<circle cx="{lx + 18}" cy="{ly + 50}" r="3.5" fill="{SIGNAL}"/>')
    parts.append(_text(lx + 40, ly + 54, "committed runs", size=12))
    parts.append("</svg>\n")
    return "\n".join(parts)


def figure_svgs(root: Path | None = None) -> dict[str, str]:
    """Relative path to SVG text, for every figure the committed results can draw."""
    root = root or repo_root()
    out = {COVERAGE_REL: coverage_svg(root)}
    scaling = scaling_svg(root)
    if scaling is not None:
        out[SCALING_REL] = scaling
    return out


def write_figures(root: Path | None = None) -> list[Path]:
    """Write every figure and return the paths."""
    root = root or repo_root()
    written: list[Path] = []
    for rel, text in figure_svgs(root).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written


def write_coverage_figure(root: Path | None = None) -> Path:
    root = root or repo_root()
    path = root / COVERAGE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(coverage_svg(root))
    return path
