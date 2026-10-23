"""The no-invented-numbers guard: unit tests for the extractor and the repo-wide check."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tsfm.docs.build import OUTPUTS, TEMPLATE_DIR, build, render
from tsfm.docs.numbers import (
    collect_result_values,
    extract_numbers,
    load_allowlist,
    unverified_numbers,
)
from tsfm.paths import repo_root


def texts(s: str) -> list[str]:
    return [n.text for n in extract_numbers(s)]


def test_extracts_plain_decimal_pct_suffix_sci():
    assert texts("CRPS 0.912, coverage 81.5%, 10M params, 1.2e16 FLOPs, 4,096 bins") == [
        "0.912",
        "81.5%",
        "10M",
        "1.2e16",
        "4,096",
    ]


def test_ignores_identifiers_code_and_urls():
    s = "M4 on a T4 or A100, sm_75, `--seed 3`, [link](http://x/1.5) and\n```\nlr=3e-4\n```\n"
    assert texts(s) == []


def test_negative_and_unicode_minus():
    assert [n.value for n in extract_numbers("delta -0.05 and −1.5")] == [-0.05, -1.5]


def test_matching_respects_displayed_precision():
    vals = [0.91234, 1.2e16, 0.815, 31_457_280]
    assert unverified_numbers("0.912 0.9123 81.5% 1.2e16 31M", vals, set()) == []
    bad = unverified_numbers("0.913 83% 82%", vals, set())
    assert [b.text for b in bad] == ["0.913", "83%"]  # 81.5% -> 82% is a valid rounding


def test_allowlist_exempts():
    assert unverified_numbers("Python 3.12 since 2020", [], {"3.12", "2020"}) == []


def test_render_rejects_placeholder_in_code(tmp_path: Path):
    with pytest.raises(ValueError, match="placeholder inside code"):
        render("```\n{{ results/x.json#a }}\n```", tmp_path)


def test_render_and_default(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text(json.dumps({"m": {"crps": 0.48712}}))
    out = render(
        "a {{ results/r.json#m.crps | .3f }} b {{ results/none.json#x | default:pending }}",
        tmp_path,
    )
    assert out == "a 0.487 b pending"


def test_render_if_exists_block(tmp_path: Path):
    (tmp_path / "results").mkdir()
    tpl = (
        "a\n{{ if exists:results/r.json }}\nyes {{ results/r.json#v | .1f }}\n"
        "{{ else }}\nno\n{{ end }}\nb\n{{ if exists:results/none.json }}\nhidden\n{{ end }}\nc\n"
    )
    assert render(tpl, tmp_path) == "a\nno\nb\nc\n"
    (tmp_path / "results" / "r.json").write_text(json.dumps({"v": 0.25}))
    assert render(tpl, tmp_path) == "a\nyes 0.2\nb\nc\n"
    # A tag inside a line is not a block tag, so it is left for the placeholder pass.
    with pytest.raises(KeyError):
        render("x {{ if exists:results/r.json }} y", tmp_path)


def test_millions_format_is_checked_by_the_guard(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text(json.dumps({"n": 10_083_392}))
    out = render("{{ results/r.json#n | M }} and {{ results/r.json#n | M2 }}", tmp_path)
    assert out == "10.1M and 10.08M"
    assert unverified_numbers(out, [10_083_392.0], set()) == []
    assert [b.text for b in unverified_numbers("10.2M", [10_083_392.0], set())] == ["10.2M"]


def _result_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / "results").rglob("*") if p.suffix in {".json", ".csv"})


def test_generated_docs_are_up_to_date():
    assert build(check=True) == []


def test_generated_docs_contain_only_backed_numbers():
    root = repo_root()
    values = collect_result_values(_result_files(root))
    allow = load_allowlist(root / TEMPLATE_DIR / "allowed_numbers.txt")
    problems = []
    for out in OUTPUTS.values():
        p = root / out
        if p.exists():
            problems += [
                f"{out}:{n.line}: {n.text}"
                for n in unverified_numbers(p.read_text(), values, allow)
            ]
    assert problems == [], "numbers not found in results files:\n" + "\n".join(problems)


def test_docs_only_cite_committed_results():
    """Every results file a template references must be tracked by git."""
    root = repo_root()
    try:
        tracked = set(
            subprocess.run(
                ["git", "ls-files", "results"], cwd=root, capture_output=True, text=True, check=True
            ).stdout.split()
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    import re

    for tpl in (root / TEMPLATE_DIR).glob("*.md"):
        for ref in re.findall(r"\{\{\s*(results/[^#|}\s]+)", tpl.read_text()):
            if (root / ref).exists():
                assert ref in tracked, f"{tpl.name} cites untracked {ref}"


def test_bracket_keys(tmp_path: Path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text(json.dumps({"files": {"a/b.parquet": {"n": 7}}}))
    assert render("{{ results/r.json#files.[a/b.parquet].n | int }}", tmp_path) == "7"


def test_templates_have_no_numbered_headings():
    """Section numbers are not results; keep them out of generated docs entirely (a
    number like 4.1 could otherwise slip through by matching an unrelated value)."""
    import re

    for tpl in (repo_root() / TEMPLATE_DIR).glob("*.md"):
        for ln in tpl.read_text().splitlines():
            assert not re.match(r"^#+\s+\d", ln), f"{tpl.name}: {ln}"
