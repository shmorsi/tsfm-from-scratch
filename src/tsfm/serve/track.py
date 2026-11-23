"""Experiment tracking and a model registry with MLflow: ``tsfm serve register``.

Logs one run with the model's configuration as parameters, the export parity and the serving
benchmark as metrics, and the checkpoint, ONNX graphs and result files as artifacts; then
registers the fp32 ONNX graph as a new version of the model ``tsfm-cpu`` and tags the version
with the git commit. The store is a local SQLite file (``mlruns.db``, artifacts in ``mlruns/``, both git-ignored); the same calls
work against an MLflow tracking server through ``MLFLOW_TRACKING_URI``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def register(
    checkpoint: Path, artifacts: Path, results: Path, name: str = "tsfm-cpu"
) -> dict[str, Any]:
    import mlflow
    import torch

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlruns.db').resolve()}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("tsfm-serving")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    with mlflow.start_run(run_name=f"serve-{sha[:8]}") as run:
        mlflow.log_params(
            {f"model.{k}": v for k, v in cfg["model"].items() if not isinstance(v, list | tuple)}
        )
        mlflow.set_tags({"git_commit": sha, "checkpoint": str(checkpoint)})
        for f in sorted(results.glob("*.json")):
            metrics = _flatten(json.loads(f.read_text()), f.stem + ".")
            mlflow.log_metrics({k.replace("/", "_"): v for k, v in metrics.items()})
            mlflow.log_artifact(str(f), "results")
        mlflow.log_artifact(str(checkpoint), "checkpoint")
        import mlflow.onnx
        import onnx

        mlflow.log_artifact(str(artifacts / "model.int8.onnx"), "onnx")
        info = mlflow.onnx.log_model(
            onnx.load(str(artifacts / "model.onnx")), name="model", registered_model_name=name
        )
        version = mlflow.MlflowClient().get_model_version(name, str(info.registered_model_version))
        client = mlflow.MlflowClient()
        client.set_model_version_tag(name, version.version, "git_commit", sha)
        client.set_registered_model_alias(name, "candidate", version.version)
        return {
            "tracking_uri": uri,
            "run_id": run.info.run_id,
            "model": name,
            "version": int(version.version),
            "alias": "candidate",
            "metrics_logged": len(mlflow.get_run(run.info.run_id).data.metrics),
        }
