"""The volatility study's model runs: the pre-registered comparisons, the documents that
show them, the packed harvest, and scripts/run_vol_study.py's guards.

No model run exists in the repository until the notebook has been run, so every test here
builds its own rows (the committed HAR-RV rows, scaled) in a temporary results folder.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tsfm.docs.build import TEMPLATE_DIR, MissingResult, render
from tsfm.docs.numbers import collect_result_values, load_allowlist, unverified_numbers
from tsfm.finance import study as S
from tsfm.paths import repo_root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import harvest_colab_output as H  # noqa: E402
import run_vol_study as R  # noqa: E402

REF = "results/finance/random_walk/per_origin.csv"
HAR = "results/finance/har/per_origin.csv"


# ------------------------------------------------------------------ comparisons


def test_verdict_follows_the_pre_registered_rule():
    assert S.verdict([0.80, 0.95]) == "better"
    assert S.verdict([1.02, 1.20]) == "worse"
    assert S.verdict([0.95, 1.05]) == "no detectable difference"
    assert S.verdict([0.90, 1.00]) == "no detectable difference"  # the interval must lie below 1


def _toy_rows(scale: float, n: int = 66) -> list[dict]:
    return [
        {
            "symbol": "A",
            "origin": f"2021-{1 + i // 28:02d}-{1 + i % 28:02d}",
            "regime": "calm",
            "wql_num": scale * (1.0 + 0.01 * (i % 7)),
            "abs_y": 10.0,
            "wql": scale * (1.0 + 0.01 * (i % 7)) / 10.0,
            "mae": 1.0,
            "cov50": 0.5,
            "cov80": 0.8,
            "cov90": 0.9,
        }
        for i in range(n)
    ]


def test_compare_is_the_ratio_to_another_model():
    a, b = _toy_rows(0.8), _toy_rows(1.0)
    c = S.compare(a, b, n_boot=200)
    assert c["wql_ratio"] == pytest.approx(0.8)
    assert c["ci95"] == [pytest.approx(0.8), pytest.approx(0.8)]
    assert c["verdict"] == "better"
    assert c["origins"] == 66
    same = S.compare(a, a, n_boot=200)
    assert same["wql_ratio"] == 1.0
    assert same["verdict"] == "no detectable difference"
    assert S.compare(b, a, n_boot=200)["verdict"] == "worse"
    with pytest.raises(ValueError, match="paired origins"):
        S.compare(a[:10], b[:10])


def _scaled(rows: list[dict], scale: float) -> list[dict]:
    return [{**r, "wql_num": r["wql_num"] * scale, "wql": r["wql"] * scale} for r in rows]


@pytest.fixture
def model_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo-shaped root: the committed results as links, plus two synthetic model runs
    and the summary ``tsfm vol report`` writes for them. Nothing real is touched."""
    root = tmp_path / "repo"
    src = repo_root() / "results"
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if rel.as_posix() == "finance/summary.json" or rel.as_posix().startswith("finance/tsfm_"):
            continue  # written below; a link here would write through to the real file
        dest = root / "results" / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(path)
    monkeypatch.setenv("TSFM_RESULTS", str(root / "results"))
    har = S.load_rows("har")
    S.save("tsfm_zero_shot", _scaled(har, 0.97), {"checkpoint_sha256": "0" * 64, "finetune": False})
    log = [
        {"step": 0, "val_wql": 0.31},
        {"step": 25, "train_loss": 0.4, "val_wql": 0.27},
        {"step": 50, "train_loss": 0.4, "val_wql": 0.29},
    ]
    S.save(
        "tsfm_finetuned",
        _scaled(har, 0.90),
        {
            "checkpoint_sha256": "0" * 64,
            "finetune": True,
            "finetune_log": log,
            "finetune_kept_step": 25,
            "finetune_kept_val_wql": 0.27,
        },
    )
    S.report(list(R.REPORT_MODELS))
    return root


def test_report_adds_the_comparisons_only_for_model_runs(
    model_results: Path, monkeypatch: pytest.MonkeyPatch
):
    s = json.loads((model_results / "results/finance/summary.json").read_text())
    assert [(c["model"], c["versus"]) for c in s["comparisons"]] == list(S.COMPARISONS)
    by = {(c["model"], c["versus"]): c for c in s["comparisons"]}
    ft_zs = by[("tsfm_finetuned", "tsfm_zero_shot")]
    assert ft_zs["wql_ratio"] == pytest.approx(0.90 / 0.97, rel=1e-4)
    assert ft_zs["verdict"] == "better"
    assert by[("tsfm_zero_shot", "har")]["wql_ratio"] == pytest.approx(0.97, rel=1e-4)
    assert s["models"]["tsfm_zero_shot"]["both/all_test"]["origins"] == 4886
    # A ratio to HAR-RV is computed the way the ratios to the random walk are.
    direct = S.summarise(S.load_rows("tsfm_zero_shot"), S.load_rows("har"))["both/all_test"]
    assert by[("tsfm_zero_shot", "har")]["ci95"] == direct["wql_ratio_ci95"]
    # The baselines alone give the file its committed shape: no comparisons.
    S.report(["random_walk", "har"])
    again = json.loads((model_results / "results/finance/summary.json").read_text())
    committed = json.loads((repo_root() / "results/finance/summary.json").read_text())
    assert "comparisons" not in again
    assert sorted(again["models"]) == ["har", "random_walk"]
    for m in again["models"]:
        assert again["models"][m] == committed["models"][m]


def test_documents_show_the_model_runs_once_they_exist(model_results: Path):
    from tsfm.docs.tables import TABLES

    readme = render((repo_root() / TEMPLATE_DIR / "README.md").read_text(), model_results)
    note = render((repo_root() / TEMPLATE_DIR / "RESEARCH_NOTE.md").read_text(), model_results)
    for doc in (readme, note):
        assert "Our model, zero-shot" in doc
        assert "Our model, fine-tuned" in doc
        assert "Fine-tuned against zero-shot" in doc
        assert "Zero-shot against HAR-RV" in doc
        assert "Fine-tuned against HAR-RV" in doc
        assert "kept step 25" in doc.replace("\n", " ")
        assert "was not run on this study" not in doc
    assert "Market volatility: the baselines and our model" in readme
    assert "Market volatility baselines" not in readme
    assert "| Our model, fine-tuned | calm |" in note  # the regime table takes the runs too
    table = TABLES["vol_comparisons"](model_results)
    assert table.count("\n") == 1 + len(S.COMPARISONS)
    assert "| better |" in table
    # Every number the two documents show is in the results files of that root.
    files = sorted(
        p for p in (model_results / "results").rglob("*") if p.suffix in {".json", ".csv"}
    )
    values = collect_result_values(files)
    allow = load_allowlist(repo_root() / TEMPLATE_DIR / "allowed_numbers.txt")
    bad = [
        f"{n.line}: {n.text}"
        for doc in (readme, note)
        for n in unverified_numbers(doc, values, allow)
    ]
    assert bad == []


def test_documents_are_unchanged_until_the_model_runs_exist():
    from tsfm.docs.tables import TABLES

    root = repo_root()
    if (root / "results/finance/tsfm_finetuned/run.json").exists():
        pytest.skip("the model runs are in the repository")
    readme = (root / "README.md").read_text()
    assert "Market volatility baselines" in readme
    assert "our model was not run on this study" in readme
    assert "Our model, zero-shot" not in readme
    with pytest.raises(MissingResult):
        TABLES["vol_comparisons"](root)


def test_kept_is_the_first_lowest_validation_step():
    from tsfm.finance.finetune import kept

    log = [
        {"step": 0, "val_wql": 0.30},
        {"step": 25, "val_wql": 0.25},
        {"step": 50, "val_wql": 0.25},
        {"step": 75, "val_wql": 0.28},
    ]
    assert kept(log)["step"] == 25
    assert kept(log[:1])["step"] == 0  # nothing beat the starting weights


# ------------------------------------------------------------------ packed harvest


def test_volrows_packs_a_real_file_exactly_and_small():
    data, ref = (ROOT / HAR).read_bytes(), (ROOT / REF).read_bytes()
    packed = H.pack_vol_rows(data, ref)
    assert packed is not None
    assert H.unpack_vol_rows(packed, ref) == data
    block = H.packed_block("results/finance/tsfm_zero_shot/per_origin.csv", data, REF, ref)
    assert block.startswith("===PACKED results/finance/tsfm_zero_shot/per_origin.csv ")
    assert f" volrows1 {REF}===" in block.splitlines()[0]
    plain = H.packed_block("results/finance/tsfm_zero_shot/per_origin.csv", data)
    assert " xz -===" in plain.splitlines()[0]
    assert len(block) < 0.75 * len(plain)
    assert len(block) < 60_000  # two runs and the summary stay well under 150 KB


def test_volrows_refuses_rows_that_do_not_line_up_and_falls_back_to_xz(tmp_path: Path):
    data, ref = (ROOT / HAR).read_bytes(), (ROOT / REF).read_bytes()
    shifted = data.replace(b"BTCUSDT,2020-01-01,", b"BTCUSDT,2020-01-02,", 1)
    assert H.pack_vol_rows(shifted, ref) is None
    assert H.pack_vol_rows(b"not,a,rows,file\r\n", ref) is None
    rel = "results/finance/tsfm_zero_shot/per_origin.csv"
    block = H.packed_block(rel, shifted, REF, ref)
    assert " xz -===" in block.splitlines()[0]
    ok, bad = H.harvest(block, tmp_path)
    assert (ok, bad) == ([rel], [])
    assert (tmp_path / rel).read_bytes() == shifted


def test_harvest_restores_packed_rows_and_keeps_scored_once_results(tmp_path: Path):
    data, ref = (ROOT / HAR).read_bytes(), (ROOT / REF).read_bytes()
    (tmp_path / REF).parent.mkdir(parents=True)
    (tmp_path / REF).write_bytes(ref)
    rel = "results/finance/tsfm_finetuned/per_origin.csv"
    js = b'{\n  "x": 1\n}\n'
    text = (
        "header line, not a block\n"
        + f"===FILE results/finance/summary.json {hashlib.sha256(js).hexdigest()}===\n{js.decode()}===END===\n"
        + H.packed_block(rel, data, REF, ref)
    )
    ok, bad = H.harvest(text, tmp_path)
    assert bad == []
    assert ok == ["results/finance/summary.json", rel]
    assert (tmp_path / rel).read_bytes() == data
    log = json.loads((tmp_path / "results/sync_log.json").read_text())
    assert [e["file"] for e in log] == ok
    assert log[1]["sha256"] == hashlib.sha256(data).hexdigest()
    # The same rows again are fine; different rows for a scored-once run are refused.
    assert H.harvest(H.packed_block(rel, data, REF, ref), tmp_path)[1] == []
    other = (ROOT / REF).read_bytes()
    ok, bad = H.harvest(H.packed_block(rel, other, REF, ref), tmp_path)
    assert ok == []
    assert bad == [f"{rel}: refusing to overwrite a different scored-once result"]
    assert (tmp_path / rel).read_bytes() == data
    # The summary is rebuilt by tsfm vol report, so it may be replaced.
    js2 = b'{\n  "x": 2\n}\n'
    text2 = f"===FILE results/finance/summary.json {hashlib.sha256(js2).hexdigest()}===\n{js2.decode()}===END===\n"
    assert H.harvest(text2, tmp_path) == (["results/finance/summary.json"], [])


def test_harvest_rejects_damaged_or_orphaned_packed_blocks(tmp_path: Path):
    data, ref = (ROOT / HAR).read_bytes(), (ROOT / REF).read_bytes()
    rel = "results/finance/tsfm_zero_shot/per_origin.csv"
    block = H.packed_block(rel, data, REF, ref)
    # No reference rows in the destination.
    ok, bad = H.harvest(block, tmp_path)
    assert ok == []
    assert bad == [f"{rel}: reference {REF} is not in the repo"]
    (tmp_path / REF).parent.mkdir(parents=True)
    (tmp_path / REF).write_bytes(ref)
    # A truncated copy.
    lines = block.splitlines()
    cut = "\n".join([lines[0], *lines[1 : len(lines) // 2], lines[-1]]) + "\n"
    ok, bad = H.harvest(cut, tmp_path)
    assert ok == []
    assert len(bad) == 1
    assert bad[0].startswith(f"{rel}: ")
    # A wrong hash, an unknown codec and a path outside results/.
    wrong = block.replace(hashlib.sha256(data).hexdigest(), "0" * 64)
    assert H.harvest(wrong, tmp_path) == ([], [f"{rel}: SHA-256 mismatch"])
    odd = block.replace(f" volrows1 {REF}===", " zip -===")
    assert H.harvest(odd, tmp_path) == ([], [f"{rel}: unknown codec zip"])
    outside = block.replace(rel, "src/tsfm/model.py")
    assert H.harvest(outside, tmp_path) == ([], ["src/tsfm/model.py: not under results/"])
    assert not (tmp_path / rel).exists()


# ------------------------------------------------------------------ scripts/run_vol_study.py


def test_inputs_are_the_series_the_baselines_were_scored_on(tmp_path: Path):
    assert R.input_problems(ROOT) == []
    files = R.series_files(ROOT)
    assert sorted(files) == [
        "results/finance/rvol_BTCUSDT.csv",
        "results/finance/rvol_ETHUSDT.csv",
    ]
    fin = tmp_path / "results" / "finance"
    fin.mkdir(parents=True)
    (fin / "rvol_summary.json").write_bytes(
        (ROOT / "results/finance/rvol_summary.json").read_bytes()
    )
    (fin / "rvol_BTCUSDT.csv").write_bytes((ROOT / "results/finance/rvol_BTCUSDT.csv").read_bytes())
    (fin / "rvol_ETHUSDT.csv").write_text("date,rvol_pct,bars\n2017-08-17,1.0,288\n")
    bad = R.input_problems(tmp_path)
    assert len(bad) == 2
    assert "rvol_ETHUSDT.csv has SHA-256" in bad[0]
    assert "random_walk/per_origin.csv is missing" in bad[1]


def test_a_committed_run_stops_the_script_and_links_resume(tmp_path: Path):
    fin = tmp_path / "results" / "finance"
    fin.mkdir(parents=True)
    work = tmp_path / "work"
    assert R.committed_runs(tmp_path) == []
    out = R.link_run(tmp_path, work, "tsfm_zero_shot")
    assert out.is_symlink()
    assert out.resolve() == (work / "results" / "tsfm_zero_shot").resolve()
    assert R.link_run(tmp_path, work, "tsfm_zero_shot") == out  # idempotent on resume
    assert R.committed_runs(tmp_path) == []  # a link into --work is not a committed run
    assert not R.run_done(out)
    (out / "per_origin.csv").write_text("x\n")
    assert not R.run_done(out)  # run.json is written last
    (out / "run.json").write_text("{}\n")
    assert R.run_done(out)
    (fin / "tsfm_finetuned").mkdir()
    assert R.committed_runs(tmp_path) == ["tsfm_finetuned"]
    with pytest.raises(SystemExit, match="already exists"):
        R.link_run(tmp_path, work, "tsfm_finetuned")


def _git(cwd: Path, *args: str) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(cwd),
    }
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


def test_the_tag_must_be_in_history(tmp_path: Path):
    try:
        _git(tmp_path, "init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    (tmp_path / "a").write_text("1\n")
    _git(tmp_path, "add", "a")
    _git(tmp_path, "commit", "-q", "-m", "one")
    with pytest.raises(SystemExit, match="prereg-vol-v1 is not an ancestor"):
        R.git_ok(tmp_path)
    _git(tmp_path, "tag", "prereg-vol-v1")
    (tmp_path / "a").write_text("2\n")
    _git(tmp_path, "commit", "-q", "-am", "two")
    assert len(R.git_ok(tmp_path)) == 40


def test_harvest_parts_round_trip_and_release_copy(model_results: Path, tmp_path: Path):
    parts = R.harvest_parts(model_results)
    assert sorted(parts) == ["rows_tsfm_finetuned", "rows_tsfm_zero_shot", "summary"]
    dest = tmp_path / "clone"
    (dest / REF).parent.mkdir(parents=True)
    (dest / REF).write_bytes((ROOT / REF).read_bytes())
    ok, bad = H.harvest("free text before the blocks\n" + "".join(parts.values()), dest)
    assert bad == []
    assert ok == [
        "results/finance/summary.json",
        "results/finance/tsfm_zero_shot/run.json",
        "results/finance/tsfm_finetuned/run.json",
        "results/finance/tsfm_zero_shot/per_origin.csv",
        "results/finance/tsfm_finetuned/per_origin.csv",
    ]
    for rel in ok:
        assert (dest / rel).read_bytes() == (model_results / rel).read_bytes()
    assert sum(len(t) for t in parts.values()) < 150_000
    # Each part can be applied on its own, in any order.
    alone = tmp_path / "alone"
    (alone / REF).parent.mkdir(parents=True)
    (alone / REF).write_bytes((ROOT / REF).read_bytes())
    for key in ("rows_tsfm_finetuned", "summary", "rows_tsfm_zero_shot"):
        assert H.harvest(parts[key], alone)[1] == []
    for rel in ok:
        assert (alone / rel).read_bytes() == (model_results / rel).read_bytes()

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    path, sha = R.release_copy(weights, tmp_path / "work")
    assert path.read_bytes() == b"weights"
    assert sha == hashlib.sha256(b"weights").hexdigest()
    assert (path.parent / (R.RELEASE_NAME + ".sha256")).read_text() == f"{sha}  {R.RELEASE_NAME}\n"


def test_the_queue_skips_results_the_commit_already_holds():
    import colab_queue as Q

    assert Q.in_checkout("results/eval/test/patch_s2/all_results.csv")
    assert not Q.in_checkout("results/eval/test/patch_s2_seed1/all_results.csv")
    assert (
        Q.in_checkout("results/finance/tsfm_zero_shot/run.json")
        == (ROOT / "results/finance/tsfm_zero_shot/run.json").is_file()
    )


def test_vol_notebook_checks_before_it_runs():
    nb = json.loads((ROOT / "notebooks" / "vol_study_patch_s2.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    params = code[0]
    assert params.startswith("# Parameters")
    assert 'CHECKPOINT_PATH = "/content/drive/MyDrive/tsfm/runs/patch_s2/ckpt.pt"' in params
    assert "VERIFY_RAW = True" in params
    assert 'WORK = "/content/drive/MyDrive/tsfm/vol_study"' in params
    joined = "\n".join(code)
    assert "drive.mount" in joined
    assert '"--is-ancestor", "prereg-vol-v1", "HEAD"' in joined
    assert "uv sync --locked --extra cu126" in joined
    assert joined.index("--dry-run") < joined.index("--device cuda")
    assert joined.count("scripts/run_vol_study.py") == 2
    assert "--skip-raw-check" in joined
    md = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert (
        "https://colab.research.google.com/github/shmorsi/tsfm-from-scratch/blob/main/"
        "notebooks/vol_study_patch_s2.ipynb"
    ) in md
    val = json.loads((ROOT / "results/eval/val/patch_s2/run.json").read_text())
    assert val["model_info"]["checkpoint_sha256"] in md
    assert "harvest_vol_patch_s2.txt" in md
    assert "No duration is stored for this study" in md
    assert "No trading or economic-value claim" in md
    n = json.loads((ROOT / "results/finance/har/run.json").read_text())["n_rows"]
    assert f"{n:,} forecast origins" in md
    assert R.HARVEST_NAME == "harvest_vol_patch_s2.txt"
    assert R.NAMES == ("tsfm_zero_shot", "tsfm_finetuned")
