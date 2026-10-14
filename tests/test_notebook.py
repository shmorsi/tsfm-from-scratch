"""The generated Colab notebook is current and every code cell is valid Python
(IPython `!`/`%` lines are replaced by `pass` before compiling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_matches_generator():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_colab_notebook as M

    committed = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text())
    assert committed == M.build(), "run scripts/make_colab_notebook.py"


def test_code_cells_compile():
    nb = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for ln in "".join(cell["source"]).splitlines():
            stripped = ln.lstrip()
            indent = ln[: len(ln) - len(stripped)]
            lines.append(indent + "pass" if stripped.startswith(("!", "%")) else ln)
        compile("\n".join(lines), f"cell {i}", "exec")
