"""Serving: the ONNX graph matches PyTorch, the engine's two backends agree, and the HTTP API validates input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from tsfm.forecast import forecast_patch
from tsfm.model import ModelConfig, TSModel
from tsfm.serve.engine import Engine
from tsfm.serve.export import export, parity, quantise


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, TSModel]:
    torch.manual_seed(0)
    cfg = ModelConfig(
        tokenizer="patch",
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        patch_len=8,
        horizon=16,
    )
    model = TSModel(cfg).eval()
    d = tmp_path_factory.mktemp("serve")
    ckpt = d / "model.pt"
    torch.save({"config": {"model": cfg.to_dict()}, "model": model.state_dict()}, ckpt)
    onnx = export(model, d / "model.onnx")
    return ckpt, onnx, model


def test_onnx_matches_torch(tiny: tuple[Path, Path, TSModel]) -> None:
    _, onnx, model = tiny
    p = parity(model, onnx, lengths=(1, 3, 12, 32), batch=3)
    assert p["max_abs"] < 1e-4


def test_int8_runs_and_is_smaller(tiny: tuple[Path, Path, TSModel]) -> None:
    _, onnx, model = tiny
    q = quantise(onnx, onnx.with_name("model.int8.onnx"))
    assert q.stat().st_size < onnx.stat().st_size
    assert np.isfinite(parity(model, q)["max_abs"])


def test_engine_backends_agree_with_missing_values(tiny: tuple[Path, Path, TSModel]) -> None:
    ckpt, onnx, _ = tiny
    eng = Engine.load(ckpt, onnx, max_context=64)
    rng = np.random.default_rng(1)
    ctx = [rng.normal(10, 2, 50).astype(np.float32), rng.normal(0, 1, 7).astype(np.float32)]
    ctx[0][[3, 20]] = np.nan
    a, used_a = eng.forecast(ctx, 12, "onnx")
    b, used_b = eng.forecast(ctx, 12, "torch")
    ref = forecast_patch(eng.model, ctx, 12, 64)
    assert (used_a, used_b) == ("onnx", "torch")
    assert np.allclose(a.quantiles, ref.quantiles, atol=1e-4)
    assert np.allclose(b.quantiles, ref.quantiles)
    assert eng.backend_for(40) == "torch"  # past the model's horizon: roll-out in PyTorch


def test_http_api(tiny: tuple[Path, Path, TSModel], monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    ckpt, onnx, _ = tiny
    import tsfm.serve.app as app_mod

    monkeypatch.setenv("TSFM_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("TSFM_ONNX", str(onnx))
    monkeypatch.setattr(app_mod, "ENGINE", None)
    c = TestClient(app_mod.app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/metadata").json()["horizon"] == 16
    r = c.post("/forecast", json={"series": [[1.0, 2.0, None, 4.0, 5.0] * 6], "horizon": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "onnx"
    assert len(body["quantiles"][0]) == 8
    assert len(body["levels"]) == 13
    long = c.post("/forecast", json={"series": [[1.0] * 30], "horizon": 40})
    assert long.status_code == 200
    assert long.json()["backend"] == "torch"
    assert c.post("/forecast", json={"series": [[None, None]], "horizon": 4}).status_code == 422
    assert c.post("/forecast", json={"series": [[1.0]], "horizon": 0}).status_code == 422


def test_psi_and_drift_monitor() -> None:
    from tsfm.serve.monitor import Monitor, features, psi

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.0, 1, 5000)) > 0.25
    f = features(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 7.0, 6.0, 8.0]))
    assert f[1] == pytest.approx(1 / 8)
    reference = np.stack([features(rng.normal(10, 1, 200)) for _ in range(300)])
    m = Monitor(reference)
    m.observe([rng.normal(10, 1, 200) for _ in range(800)], "onnx", 3.0)
    assert m.drift()["status"] == "stable"
    m2 = Monitor(reference)
    m2.observe([rng.normal(1e4, 1, 40) for _ in range(300)], "onnx", 3.0)
    assert m2.drift()["status"].startswith("shift")
    text = m.prometheus()
    assert 'tsfm_requests_total{backend="onnx"} 1' in text
    assert 'tsfm_latency_ms_bucket{backend="onnx",le="5.0"} 1' in text


def test_metrics_and_drift_endpoints(
    tiny: tuple[Path, Path, TSModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    import tsfm.serve.app as app_mod
    from tsfm.serve.monitor import Monitor, features

    ckpt, onnx, _ = tiny
    monkeypatch.setenv("TSFM_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("TSFM_ONNX", str(onnx))
    monkeypatch.setattr(app_mod, "ENGINE", None)
    rng = np.random.default_rng(3)
    monkeypatch.setattr(
        app_mod, "MONITOR", Monitor(np.stack([features(rng.normal(5, 1, 64)) for _ in range(100)]))
    )
    c = TestClient(app_mod.app)
    c.post("/forecast", json={"series": [list(rng.normal(5, 1, 64))], "horizon": 8})
    c.post("/forecast", json={"series": [[None]], "horizon": 8})
    body = c.get("/metrics").text
    assert 'tsfm_requests_total{backend="onnx"} 1' in body
    assert "tsfm_request_errors_total 1" in body
    assert c.get("/drift").json()["status"] == "warming up"
