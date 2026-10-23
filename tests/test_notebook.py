"""The generated Colab notebooks are current and every code cell is valid Python
(IPython `!`/`%` lines are replaced by `pass` before compiling)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_colab_notebook as M  # noqa: E402

NAMES = sorted(M.NOTEBOOKS)


def _load(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / name).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_notebook_matches_generator(name: str):
    assert _load(name) == M.build_all()[name], "run scripts/make_colab_notebook.py"


@pytest.mark.parametrize("name", NAMES)
def test_code_cells_compile(name: str):
    nb = _load(name)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for ln in "".join(cell["source"]).splitlines():
            stripped = ln.lstrip()
            indent = ln[: len(ln) - len(stripped)]
            lines.append(indent + "pass" if stripped.startswith(("!", "%")) else ln)
        compile("\n".join(lines), f"{name} cell {i}", "exec")


@pytest.mark.parametrize("name", NAMES)
def test_nbformat_structure(name: str):
    """The fields nbformat 4.5 requires: unique cell ids, outputs on code cells."""
    nb = _load(name)
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    ids = [c["id"] for c in nb["cells"]]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-zA-Z0-9-_]{1,64}", i) for i in ids)
    for c in nb["cells"]:
        assert c["cell_type"] in {"markdown", "code"}
        if c["cell_type"] == "code":
            assert c["outputs"] == []
            assert c["execution_count"] is None


def test_score_notebook_has_one_marked_parameter():
    """Sherif sets one value; it defaults to the patch_s2 run folder on Drive, and the
    scoring cell passes it to the script that checks the checkpoint first."""
    nb = _load("score_patch_s2_test.ipynb")
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    params = code[0]
    assert params.startswith("# Parameters")
    assert params.count("CHECKPOINT_PATH =") == 1
    assert 'CHECKPOINT_PATH = "/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt"' in params
    joined = "\n".join(code)
    assert "drive.mount" in joined
    assert joined.index("--dry-run") < joined.index("--device cuda")
    assert "scripts/score_patch_s2.py" in joined
