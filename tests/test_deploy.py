"""The deployment files agree with the service: port, probes, metrics path, names (no cluster needed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"


def _load(name: str) -> dict:
    data: dict = yaml.safe_load((K8S / name).read_text())
    return data


def test_manifests_are_consistent_with_the_image_and_each_other() -> None:
    dep, svc, hpa = _load("deployment.yaml"), _load("service.yaml"), _load("hpa.yaml")
    kust = _load("kustomization.yaml")
    assert sorted(kust["resources"]) == ["deployment.yaml", "hpa.yaml", "service.yaml"]
    assert dep["kind"] == "Deployment"
    assert svc["kind"] == "Service"
    assert hpa["kind"] == "HorizontalPodAutoscaler"
    labels = dep["spec"]["template"]["metadata"]["labels"]
    assert dep["spec"]["selector"]["matchLabels"] == labels == svc["spec"]["selector"]
    c = dep["spec"]["template"]["spec"]["containers"][0]
    port = c["ports"][0]["containerPort"]
    expose = re.search(r"^EXPOSE (\d+)", (ROOT / "Dockerfile").read_text(), re.M)
    assert expose
    assert int(expose.group(1)) == port
    assert dep["spec"]["template"]["metadata"]["annotations"]["prometheus.io/port"] == str(port)
    assert svc["spec"]["ports"][0]["targetPort"] == c["ports"][0]["name"]
    assert hpa["spec"]["scaleTargetRef"]["name"] == dep["metadata"]["name"]
    assert hpa["spec"]["minReplicas"] <= dep["spec"]["replicas"] <= hpa["spec"]["maxReplicas"]
    uid = re.search(r"useradd .*--uid (\d+)", (ROOT / "Dockerfile").read_text())
    assert uid
    assert int(uid.group(1)) == dep["spec"]["template"]["spec"]["securityContext"]["runAsUser"]


def test_probe_and_metrics_paths_exist_in_the_app() -> None:
    pytest.importorskip("fastapi")
    from tsfm.serve.app import app

    routes = {r.path for r in app.routes}
    dep = _load("deployment.yaml")
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] in routes
    assert c["livenessProbe"]["httpGet"]["path"] in routes
    assert dep["spec"]["template"]["metadata"]["annotations"]["prometheus.io/path"] in routes
