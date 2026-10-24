"""The portfolio SVG and case-study page use the same dark palette as the docs figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_portfolio_figure as fig  # noqa: E402
import make_portfolio_page as page  # noqa: E402


def _contrast(a: str, b: str) -> float:
    def lin(hex_color: str) -> float:
        c = int(hex_color[1:], 16)
        channels = (((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255)

        def f(x: float) -> float:
            return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

        r, g, bl = (f(x) for x in channels)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl

    hi, lo = sorted((lin(a), lin(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_portfolio_figure_matches_the_doc_palette():
    svg = fig.build()
    assert fig.BG in svg
    assert fig.SIGNAL in svg
    assert fig.GRID in svg
    assert 'role="img"' in svg
    assert "<title" in svg
    assert "#ffb547" not in svg
    assert "#8fd3ff" not in svg
    assert _contrast(fig.INK, fig.BG) >= 4.5
    assert _contrast(fig.MUTED, fig.BG) >= 4.5
    assert _contrast(fig.FAINT, fig.BG) >= 4.5
    assert _contrast(fig.GRID, fig.BG) >= 3.0
    assert _contrast(fig.SIGNAL, fig.BG) >= 3.0
    # Plotted validation CRPS values are still the committed ones.
    val = fig.load("results/analysis/val_summary.json")["models"]
    for name in ("patch_s0", "patch_s1", "patch_s2"):
        assert f"{val[name]['crps_gmean']:.3f}" in svg


def test_portfolio_page_carries_the_dark_theme():
    html = page.build()
    assert "--bg: #07090d" in html
    assert "Inter" in html
    assert "Instrument Sans" not in html
    assert "Unbounded" not in html
    assert html.count("<html") == 1
