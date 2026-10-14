"""Summaries, comparisons and the results sync script."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tsfm import analysis as A
from tsfm.eval.run import CSV_HEADER


def write_model(root: Path, split: str, name: str, scale: dict[str, float]) -> None:
    d = root / "eval" / split / name
    d.mkdir(parents=True)
    with open(d / "all_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for cfg, s in scale.items():
            row = dict.fromkeys(CSV_HEADER, "1.0")
            row.update(
                {
                    "dataset": cfg,
                    "model": name,
                    "domain": "Energy" if "elec" in cfg else "Sales",
                    "num_variates": "1",
                }
            )
            row[A.MASE] = str(s)
            row[A.CRPS] = str(s / 2)
            w.writerow([row[k] for k in CSV_HEADER])
    (d / "run.json").write_text(
        json.dumps(
            {
                "_meta": {"partial": False},
                "per_config": {
                    c: {"coverage_50": 0.5, "coverage_80": 0.8, "coverage_90": 0.9} for c in scale
                },
            }
        )
    )


def test_summary_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(A, "results_dir", lambda: tmp_path)
    cfgs = ["elec/H/short", "elec/15T/long", "sales/D/short", "sales/W/short"]
    write_model(tmp_path, "val", "seasonal_naive", dict.fromkeys(cfgs, 2.0))
    write_model(tmp_path, "val", "m", {c: 1.0 for c in cfgs})
    write_model(tmp_path, "val", "partial_run", {cfgs[0]: 1.0})
    models = A.load_models("val")
    s = A.summarise(models, "seasonal_naive")
    assert set(s["models"]) == {"seasonal_naive", "m"}  # incomplete model excluded
    m = s["models"]["m"]
    assert m["mase_gmean"] == pytest.approx(0.5)
    assert m["crps_gmean"] == pytest.approx(0.5)
    assert m["rank_crps"] == 1.0
    assert m["coverage_mean"]["80"] == pytest.approx(0.8)
    assert set(m["by"]["frequency"]) == {"hourly", "minutely", "daily", "weekly"}
    assert m["by"]["term"]["long"]["n"] == 1
    c = A.compare(models, "m", "seasonal_naive", "seasonal_naive", "crps")
    assert c["ratio"] == pytest.approx(0.5)
    assert c["verdict"] == "better"


@pytest.mark.parametrize(
    ("cfg", "group"),
    [
        ("a/15T/short", "minutely"),
        ("a/10S/long", "secondly"),
        ("a/W/short", "weekly"),
        ("a/A/short", "yearly"),
        ("a/M/short", "monthly"),
    ],
)
def test_freq_group(cfg: str, group: str):
    assert A.freq_group(cfg) == group


def test_headline_report_uses_upstream_arima_and_seed_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tsfm.eval.gifteval import configs

    monkeypatch.setattr(A, "results_dir", lambda: tmp_path)
    names = [c.name for c in configs()]
    up = A.read_csv(A.VENDOR / "upstream_results" / "Auto_Arima" / "all_results.csv")
    sn = A.read_csv(A.VENDOR / "upstream_results" / "Seasonal_Naive" / "all_results.csv")
    # Our SN = upstream SN; two "seeds" that are exactly 0.5x and 0.7x AutoARIMA's CRPS.
    write_model(tmp_path, "test", "seasonal_naive", {k: sn[k][A.MASE] for k in names})
    for seed, f in ((0, 0.5), (1, 0.7)):
        d = tmp_path / "eval" / "test" / f"m-seed{seed}"
        write_model(tmp_path, "test", f"m-seed{seed}", {k: up[k][A.MASE] for k in names})
        # write_model sets CRPS = MASE/2; overwrite CRPS to f * AutoARIMA CRPS
        rows = (d / "all_results.csv").read_text().splitlines()
        hdr = rows[0].split(",")
        ci = hdr.index(A.CRPS)
        out = [rows[0]]
        for r in rows[1:]:
            cells = r.split(",")
            cells[ci] = str(f * up[cells[0]][A.CRPS])
            out.append(",".join(cells))
        (d / "all_results.csv").write_text("\n".join(out) + "\n")
    # Fix SN CRPS to upstream too
    d = tmp_path / "eval" / "test" / "seasonal_naive"
    rows = (d / "all_results.csv").read_text().splitlines()
    ci = rows[0].split(",").index(A.CRPS)
    out = [rows[0]]
    for r in rows[1:]:
        cells = r.split(",")
        cells[ci] = str(sn[cells[0]][A.CRPS])
        out.append(",".join(cells))
    (d / "all_results.csv").write_text("\n".join(out) + "\n")
    p = A.headline_report(["m-seed0", "m-seed1"])
    h = json.loads(p.read_text())
    assert h["ratio_vs_auto_arima_per_seed"]["m-seed0"]["crps"] == pytest.approx(0.5)
    assert h["ratio_vs_auto_arima_per_seed"]["m-seed1"]["crps"] == pytest.approx(0.7)
    primary = next(c for c in h["comparisons"] if c["primary"])
    assert primary["b"] == "upstream:Auto_Arima"
    assert primary["ratio"] == pytest.approx(0.6)  # seed-averaged per config
    assert primary["verdict"] == "better"
