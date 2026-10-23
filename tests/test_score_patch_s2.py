"""scripts/score_patch_s2.py: refuses any checkpoint that is not patch_s2, scores only
weights whose SHA-256 matches the validation run, and uses the pre-registered settings."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_patch_s2 as S  # noqa: E402


def test_expected_reads_the_committed_validation_run():
    e = S.expected()
    summary = json.loads((ROOT / "results/runs/patch_s2/summary.json").read_text())
    assert e.params_total == summary["params_total"]
    assert e.config_hash == summary["config_hash"]
    assert e.max_context == S.PREREG_MAX_CONTEXT
    assert len(e.checkpoint_sha256) == 64


def test_harvest_blocks_round_trip(tmp_path: Path):
    import harvest_colab_output as H

    # As written by the pipeline: JSON with a final newline, CSV with CRLF line ends.
    files = {
        "results/eval/test/patch_s2/run.json": b'{\n  "x": 1\n}\n',
        "results/eval/test/patch_s2/all_results.csv": b"dataset,model\r\nm4_daily/D/short,patch_s2\r\n",
    }
    for rel, data in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    text = S.harvest_blocks(tmp_path, tuple(files))
    dest = tmp_path / "repo"
    (dest / "results").mkdir(parents=True)
    ok, bad = H.harvest(text, dest)
    assert bad == []
    assert ok == list(files)
    for rel, data in files.items():
        assert (dest / rel).read_bytes() == data


def test_link_results_refuses_existing_test_result(tmp_path: Path):
    out = tmp_path / "results" / "eval" / "test" / "patch_s2"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match="already exists"):
        S.link_results(tmp_path, tmp_path / "work")


def test_link_results_links_into_work(tmp_path: Path):
    (tmp_path / "results" / "eval" / "test").mkdir(parents=True)
    out = S.link_results(tmp_path, tmp_path / "work")
    assert out.is_symlink()
    assert out.resolve() == (tmp_path / "work" / "eval_test_patch_s2").resolve()
    assert S.link_results(tmp_path, tmp_path / "work") == out  # idempotent on resume


# ------------------------------------------------------------------ needs torch


def _fake_run(tmp_path: Path, *, n_layers: int = 1, step: int = 5):
    """A tiny run in the training-checkpoint format, plus expectations for it."""
    torch = pytest.importorskip("torch")
    from tsfm.model import ModelConfig, TSModel
    from tsfm.provenance import config_hash

    torch.manual_seed(0)
    mcfg = ModelConfig(d_model=32, n_layers=1, n_heads=2, patch_len=8, horizon=16, max_seq_len=32)
    model = TSModel(mcfg)
    cfg = {"name": "patch_s2", "seed": 0, "model": mcfg.to_dict(), "data": {}, "train": {}}
    run = tmp_path / "runs" / "patch_s2"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": cfg}, run / "model.pt")
    torch.save(
        {"step": 5, "model": model.state_dict(), "opt": {}, "config": cfg, "config_hash": "x"},
        run / "ckpt.pt",
    )
    exp = S.Expected(
        config=cfg,
        config_hash=config_hash(cfg),
        params_total=model.num_params(),
        step=step,
        checkpoint_sha256=hashlib.sha256((run / "model.pt").read_bytes()).hexdigest(),
        max_context=2048,
    )
    if n_layers != 1:  # a sibling run with a different architecture
        other = ModelConfig(**{**mcfg.to_dict(), "n_layers": n_layers, "quantiles": mcfg.quantiles})
        ocfg = {**cfg, "name": "bins_s2", "model": other.to_dict()}
        sib = tmp_path / "runs" / "bins_s2"
        sib.mkdir(parents=True)
        torch.save({"model": TSModel(other).state_dict(), "config": ocfg}, sib / "ckpt.pt")
    return run, exp


def test_training_checkpoint_resolves_to_matching_model_pt(tmp_path: Path):
    run, exp = _fake_run(tmp_path)
    path, notes = S.pick_weights(run / "ckpt.pt", exp)
    assert path == run / "model.pt"
    assert any("SHA-256 matches" in n for n in notes)
    path, _ = S.pick_weights(run / "model.pt", exp)
    assert path == run / "model.pt"


def test_wrong_run_is_refused_before_scoring(tmp_path: Path):
    _, exp = _fake_run(tmp_path, n_layers=2)
    with pytest.raises(SystemExit, match="not the patch_s2 checkpoint") as e:
        S.pick_weights(tmp_path / "runs" / "bins_s2" / "ckpt.pt", exp)
    assert "parameters" in str(e.value)
    assert "n_layers" in str(e.value)


def test_unfinished_training_checkpoint_is_refused(tmp_path: Path):
    run, exp = _fake_run(tmp_path, step=12_000)
    with pytest.raises(SystemExit, match="step"):
        S.pick_weights(run / "ckpt.pt", exp)


def test_no_matching_sha_is_refused(tmp_path: Path):
    run, exp = _fake_run(tmp_path)
    exp = S.Expected(**{**exp.__dict__, "checkpoint_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="No file matches"):
        S.pick_weights(run / "ckpt.pt", exp)


def test_missing_path_is_refused(tmp_path: Path):
    _, exp = _fake_run(tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        S.pick_weights(tmp_path / "nope.pt", exp)
