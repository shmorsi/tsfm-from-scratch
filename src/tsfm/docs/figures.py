"""SVG figures for the generated documents, drawn from committed results.

No plotting dependency. ``tsfm docs build`` rewrites these files, and ``tsfm docs check``
fails if they drift from the results they plot. Every figure uses the portfolio palette:
black ground, condensed display type, one accent, and hairline rules. Text is at least
WCAG AA against the ground; rules that carry information meet the non-text 3:1 bar.

Measured numbers in the diagrams are read from ``results/``. Stage names are labels.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsfm.paths import repo_root

# Same tokens as https://shmorsi.github.io/ (Inter Tight, black, one blue accent).
BG = "#000000"
PANEL = "#0a0a0a"
INK = "#f5f5f7"
MUTED = "#a8a8ad"
FAINT = "#8e8e93"
GRID = "#5a626a"  # 3.4:1 on black, so a 1px rule still reads
SIGNAL = "#8ec5ff"
SECOND = "#d5dde6"
FONT = "'Inter Tight', 'Helvetica Neue', Helvetica, Arial, sans-serif"
SERIES_COLORS = (SIGNAL, SECOND, FAINT)

HERO_REL = "docs/figures/hero.svg"
ARCHITECTURE_REL = "docs/figures/architecture.svg"
INTEGRITY_REL = "docs/figures/integrity.svg"
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
    tracking: str | None = None,
) -> str:
    """One SVG text node. ``tracking`` is an SVG letter-spacing value, such as ``-0.04em``."""
    track = f' letter-spacing="{tracking}"' if tracking else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}"{track}>{html.escape(body)}</text>'
    )


def _open(box: _Box, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{box.width}" height="{box.height}" '
        f'viewBox="0 0 {box.width} {box.height}" role="img" aria-labelledby="fig-title fig-desc">',
        f'<title id="fig-title">{html.escape(title)}</title>',
        f'<desc id="fig-desc">{html.escape(desc)}</desc>',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<rect x="0.5" y="0.5" width="{box.width - 1}" height="{box.height - 1}" '
        f'fill="none" stroke="{GRID}" stroke-width="1"/>',
        _text(box.left, 28, title, size=15, weight=600, tracking="-0.03em"),
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
            # Name the model. This chart is the committed CPU study, not the pending headline.
            return f"GIFT-Eval test coverage, {doc['prefix']}", series
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
            f'<polyline fill="none" stroke="{color}" stroke-width="1.75" points="{coords}"/>'
        )
        for a, b in pts:
            parts.append(
                f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4.5" fill="{color}" '
                f'stroke="{BG}" stroke-width="1.5"/>'
            )
        ly = box.top + 16 + i * 18
        parts.append(
            f'<line x1="{box.left + 8}" y1="{ly - 4}" x2="{box.left + 26}" y2="{ly - 4}" stroke="{color}" stroke-width="1.75"/>'
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
    parts.append(f'<polygon points="{band}" fill="{SIGNAL}" fill-opacity="0.22"/>')
    fit_line = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in zip(cflops, cfit, strict=True))
    parts.append(
        f'<polyline fill="none" stroke="{SIGNAL}" stroke-width="1.75" points="{fit_line}"/>'
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
        f'<rect x="{lx}" y="{ly}" width="180" height="62" rx="2" fill="{PANEL}" stroke="{GRID}"/>'
    )
    parts.append(
        f'<line x1="{lx + 12}" y1="{ly + 16}" x2="{lx + 32}" y2="{ly + 16}" stroke="{SIGNAL}" stroke-width="1.75"/>'
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


def _load(root: Path, rel: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / rel).read_text())
    return data


def _svg_open(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="fig-title fig-desc">',
        f'<title id="fig-title">{html.escape(title)}</title>',
        f'<desc id="fig-desc">{html.escape(desc)}</desc>',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
        f'fill="none" stroke="{GRID}" stroke-width="1"/>',
    ]


def _panel(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{PANEL}" stroke="{GRID}" stroke-width="1"/>'
    )


def _rule(x1: float, x2: float, y: float) -> str:
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{GRID}"/>'


def _arrow(x1: float, x2: float, y: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 7:.1f}" y2="{y:.1f}" stroke="{GRID}"/>'
        f'<path d="M{x2 - 7:.1f} {y - 3.5:.1f} L{x2:.1f} {y:.1f} L{x2 - 7:.1f} {y + 3.5:.1f}" '
        f'fill="none" stroke="{GRID}"/>'
    )


def _verdict(comparisons: list[dict[str, Any]], baseline: str, metric: str) -> str:
    found = next(c for c in comparisons if c["b"] == baseline and c["metric"] == metric)
    verdict = str(found["verdict"])
    # The figure prints the stored verdict. It does not editorialize it.
    return verdict


def hero_svg(root: Path | None = None) -> str:
    """Pipeline strip for the README. Readouts are formatted from committed results.

    The test-split slot says ``pending`` until ``results/analysis/headline.json`` exists.
    After that it shows the stored primary verdict, never a claim that the model beat
    a baseline.
    """
    root = root or repo_root()
    val = _load(root, "results/analysis/val_summary.json")
    choice = _load(root, "results/analysis/headline_choice.json")
    name = str(choice["headline"])
    crps = float(val["models"][name]["crps_gmean"])
    n_configs = int(val["n_configs"])
    params = int(_load(root, f"results/runs/{name}/summary.json")["params_total"])
    cpu = _load(root, "results/analysis/cpu_study.json")["test"]
    vs_naive = _verdict(cpu["comparisons"], "seasonal_naive", "crps")
    vs_arima = _verdict(cpu["comparisons"], "upstream:Auto_Arima", "crps")
    headline_path = root / "results" / "analysis" / "headline.json"
    pending = not headline_path.exists()
    if pending:
        test_value, test_note, test_fill = "pending", "scored at most once", SIGNAL
        test_desc = "The test split has not been scored."
    else:
        head = _load(root, "results/analysis/headline.json")
        primary = next(c for c in head["comparisons"] if c.get("primary"))
        test_value = str(primary["verdict"])
        test_note = "stored verdict, CRPS vs AutoARIMA"
        test_fill = INK
        test_desc = "The test split has a stored verdict."
    width, height = 1100, 328
    parts = _svg_open(
        width,
        height,
        "From a pinned corpus to a single test score",
        "Six stages: pinned corpus, leakage audit, training windows, decoder, "
        "validation (where the headline is chosen), and the test split. "
        f"{test_desc} The validation readout and the CPU-model verdicts are read from results.",
    )
    parts.append(_text(36, 40, "TSFM", size=26, weight=600, tracking="-0.04em"))
    parts.append(
        _text(
            118,
            40,
            "FROM SCRATCH",
            size=13,
            fill=MUTED,
            weight=600,
            tracking="0.16em",
        )
    )
    parts.append(
        _text(
            width - 36,
            40,
            "ZERO-SHOT  ·  GIFT-EVAL",
            size=12,
            fill=FAINT,
            anchor="end",
            weight=600,
            tracking="0.14em",
        )
    )
    parts.append(_rule(36, width - 36, 58))
    stages = (
        ("CORPUS", "Pinned sources"),
        ("AUDIT", "Copies removed"),
        ("WINDOWS", "Training only"),
        ("DECODER", "Plain PyTorch"),
        ("VALIDATE", "Headline chosen"),
        ("TEST", "At most once"),
    )
    left, right = 36.0, float(width - 36)
    span = right - left
    y_line = 108.0
    parts.append(_rule(left, right, y_line))
    for i, (label, caption) in enumerate(stages):
        x = left + span * (i + 0.5) / len(stages)
        chosen = label == "VALIDATE"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y_line:.1f}" r="4.5" fill="{SIGNAL if chosen else BG}" '
            f'stroke="{SIGNAL if chosen else GRID}" stroke-width="1"/>'
        )
        parts.append(
            _text(
                x,
                86,
                label,
                size=13,
                anchor="middle",
                weight=600,
                fill=SIGNAL if chosen else INK,
                tracking="0.12em",
            )
        )
        parts.append(_text(x, 136, caption, size=13, fill=MUTED, anchor="middle"))
    parts.append(_rule(36, width - 36, 162))
    columns = (
        (
            "VALIDATION",
            f"{crps:.3f}×",
            "CRPS vs Seasonal Naive",
            f"{name} · {params / 1e6:.1f}M · {n_configs} configurations",
            SIGNAL,
        ),
        ("TEST SPLIT", test_value, test_note, "weights not in this repository", test_fill),
        ("CPU ON TEST", vs_arima, "vs AutoARIMA", f"{vs_naive} · Seasonal Naive", INK),
    )
    col_w = (width - 72) / 3
    for i, (label, value, note, note2, fill) in enumerate(columns):
        x = 36 + i * col_w
        if i:
            parts.append(
                f'<line x1="{x:.1f}" y1="178" x2="{x:.1f}" y2="{height - 24}" stroke="{GRID}"/>'
            )
        parts.append(_text(x + 20, 198, label, size=12, fill=FAINT, weight=600, tracking="0.16em"))
        value_size = 28 if len(value) <= 16 else 16
        parts.append(
            _text(x + 20, 238, value, size=value_size, fill=fill, weight=600, tracking="-0.03em")
        )
        parts.append(_text(x + 20, 266, note, size=14, fill=MUTED))
        parts.append(_text(x + 20, 288, note2, size=13, fill=FAINT))
    parts.append("</svg>\n")
    return "\n".join(parts)


def architecture_svg(root: Path | None = None) -> str:
    """One pre-norm decoder block. No measured quantities."""
    del root
    width, height = 1100, 300
    parts = _svg_open(
        width,
        height,
        "Decoder block",
        "A pre-norm decoder block: RMSNorm, causal attention with rotary positions, "
        "a residual add, RMSNorm, SwiGLU, a residual add, then a quantile head or token logits. "
        "The patch quantile head is the one that was selected.",
    )
    parts.append(_text(36, 40, "DECODER", size=22, weight=600, tracking="-0.04em"))
    parts.append(
        _text(
            168,
            40,
            "PRE-NORM  ·  PLAIN PYTORCH",
            size=12,
            fill=MUTED,
            weight=600,
            tracking="0.14em",
        )
    )
    parts.append(_rule(36, width - 36, 58))
    panels = ((36, 240, "INPUT"), (300, 480, "BLOCK"), (804, 260, "HEAD"))
    for x, w, _label in panels:
        parts.append(_panel(x, 80, w, 188))
    parts.append(_arrow(276, 300, 174))
    parts.append(_arrow(780, 804, 174))
    parts.append(_text(52, 108, "INPUT", size=12, fill=FAINT, weight=600, tracking="0.16em"))
    parts.append(_text(52, 156, "Patch or bin token", size=18, weight=600, tracking="-0.03em"))
    parts.append(_text(52, 184, "Scaled by the context only.", size=14, fill=MUTED))
    parts.append(_text(52, 212, "Mean absolute value.", size=14, fill=MUTED))
    parts.append(_text(316, 108, "ONE BLOCK", size=12, fill=FAINT, weight=600, tracking="0.16em"))
    rows = (
        "RMSNorm",
        "Causal attention with RoPE",
        "Residual add",
        "RMSNorm",
        "SwiGLU",
        "Residual add",
    )
    for i, row in enumerate(rows):
        y = 136 + i * 20
        parts.append(_text(316, y, row, size=15, weight=500))
    parts.append(_text(820, 108, "HEAD", size=12, fill=FAINT, weight=600, tracking="0.16em"))
    parts.append(
        _text(820, 150, "Quantile head", size=18, fill=SIGNAL, weight=600, tracking="-0.03em")
    )
    parts.append(_text(820, 174, "Selected. Non-crossing.", size=14, fill=MUTED))
    parts.append(_text(820, 214, "Token logits", size=16, weight=500))
    parts.append(_text(820, 236, "Bins path. Not scored.", size=14, fill=FAINT))
    parts.append("</svg>\n")
    return "\n".join(parts)


def integrity_svg(root: Path | None = None) -> str:
    """Leakage audit strip. Counts are the committed round-1 and final totals."""
    root = root or repo_root()
    first = _load(root, "results/data/leakage_round1.json")["totals"]
    final = _load(root, "results/data/leakage.json")["totals"]
    flagged = int(first["series_with_hits"])
    checked = int(first["pretrain_series_checked"])
    removed = int(final["excluded_series_total"])
    hits = int(final["series_with_hits"])
    rechecked = int(final["pretrain_series_checked"])
    width, height = 1100, 292
    parts = _svg_open(
        width,
        height,
        "Leakage audit",
        "Round 1 flagged series in the public pretraining corpus. Those series were "
        "removed. A re-run on the final corpus reports how many hits remain. "
        "Counts are read from the committed leakage reports. The check finds exact "
        "and rescaled copies, not noisy or resampled ones.",
    )
    parts.append(_text(36, 40, "AUDIT", size=22, weight=600, tracking="-0.04em"))
    parts.append(
        _text(
            140,
            40,
            "EXACT AND RESCALED COPIES",
            size=12,
            fill=MUTED,
            weight=600,
            tracking="0.14em",
        )
    )
    parts.append(_rule(36, width - 36, 58))
    columns = (
        ("ROUND 1", f"{flagged:,}", f"of {checked:,} series", "flagged", INK),
        ("EXCLUDED", f"{removed:,}", "series removed", "before training", INK),
        ("RE-RUN", f"{hits:,}", f"hits in {rechecked:,} series", "on the final corpus", SIGNAL),
    )
    col_w = (width - 72) / 3
    for i, (label, value, note, note2, fill) in enumerate(columns):
        x = 36 + i * col_w
        parts.append(_panel(x, 80, col_w - 16, 176))
        parts.append(_text(x + 20, 112, label, size=12, fill=FAINT, weight=600, tracking="0.16em"))
        parts.append(_text(x + 20, 168, value, size=40, fill=fill, weight=600, tracking="-0.04em"))
        parts.append(_text(x + 20, 204, note, size=15, fill=MUTED))
        parts.append(_text(x + 20, 228, note2, size=14, fill=FAINT))
    parts.append("</svg>\n")
    return "\n".join(parts)


def figure_svgs(root: Path | None = None) -> dict[str, str]:
    """Relative path to SVG text, for every figure the committed results can draw."""
    root = root or repo_root()
    out = {
        HERO_REL: hero_svg(root),
        ARCHITECTURE_REL: architecture_svg(root),
        INTEGRITY_REL: integrity_svg(root),
        COVERAGE_REL: coverage_svg(root),
    }
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
    """Write the coverage chart and return its path."""
    root = root or repo_root()
    path = root / COVERAGE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(coverage_svg(root))
    return path
