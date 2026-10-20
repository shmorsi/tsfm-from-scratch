"""CPU study: a small from-scratch patch model, three seeds, test scored once.

The T4 ``patch_s2`` checkpoint (sha256 recorded in its validation ``run.json``) is not
in this repository, so the pre-registered test evaluation of that checkpoint cannot be
run from a clone. This study is a separate, smaller experiment that *can* be run here.
Its choices are fixed in ``PREREGISTRATION_CPU.md`` before any of its test scores exist.

``tsfm cpu-study --stage all`` is resumable: a finished train run, a finished eval
directory, or an existing conformal file is left untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tsfm.cli import register

# Held out of training. Same list as configs/data.yaml; the runner checks they match.
VAL_DATASETS = (
    "m1_monthly",
    "tourism_quarterly",
    "monash_m3_yearly",
    "nn5_daily_with_missing",
    "uber_tlc_daily",
    "traffic_weekly",
    "pedestrian_counts",
    "bdg-2_bear",
    "australian_electricity_demand",
    "kdd2022",
)

TRAIN_DATASETS = (
    "tourism_monthly",
    "tourism_yearly",
    "m1_quarterly",
    "m1_yearly",
    "nn5_weekly",
    "monash_m3_quarterly",
    "monash_m3_other",
    "fred_md",
    "cif_2016_12",
    "vehicle_trips_with_missing",
    "bitcoin_with_missing",
    "favorita_transactions",
)


def load_study(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return raw


def run_name(prefix: str, seed: int, *, position: str = "rope", cqr: bool = False) -> str:
    if position == "none":
        return f"{prefix}_norope"
    name = f"{prefix}_seed{seed}"
    return f"{name}_cqr" if cqr else name


def _run_config(study: dict[str, Any], seed: int, position: str) -> dict[str, Any]:
    prefix = str(study["name"])
    model = dict(study["model"])
    model["position"] = position
    data = dict(study["data"])
    return {
        "name": run_name(prefix, seed, position=position),
        "seed": seed,
        "model": model,
        "data": data,
        "train": dict(study["train"]),
    }


def write_protocol(study: dict[str, Any], study_path: Path) -> Path:
    """Record the design choices as data, before any test number exists."""
    from tsfm.paths import repo_root, results_dir
    from tsfm.provenance import write_result

    overlap = set(TRAIN_DATASETS) & set(VAL_DATASETS)
    if overlap:
        raise SystemExit(f"train/val overlap: {sorted(overlap)}")
    from tsfm.data.sources import load_data_config

    declared = tuple(load_data_config()["validation_datasets"])
    if declared != VAL_DATASETS:
        raise SystemExit(
            f"VAL_DATASETS does not match configs/data.yaml: {declared} vs {VAL_DATASETS}"
        )
    payload = {
        "study": study["name"],
        "config_file": str(
            study_path.relative_to(repo_root()) if study_path.is_absolute() else study_path
        ),
        "preregistration": "PREREGISTRATION_CPU.md",
        "seeds": list(study["seeds"]),
        "ablation": "position=none, seed 0, validation only, not eligible for the test split",
        "max_context": int(study["max_context"]),
        "eval_batch_size": int(study["eval_batch_size"]),
        "train_datasets": list(TRAIN_DATASETS),
        "val_datasets": list(VAL_DATASETS),
        "n_train_datasets": len(TRAIN_DATASETS),
        "n_val_datasets": len(VAL_DATASETS),
        "synthetic": study["synthetic"],
        "model": study["model"],
        "train": study["train"],
        "data": study["data"],
        "test_models": [run_name(study["name"], int(s)) for s in study["seeds"]]
        + [run_name(study["name"], int(s), cqr=True) for s in study["seeds"]],
        "not_test_scored": [run_name(study["name"], 0, position="none")],
        "t4_checkpoint_note": (
            "patch_s2 weights are not in the repository; this study does not score that checkpoint"
        ),
    }
    return write_result(
        results_dir() / "cpu_study" / "protocol.json",
        payload,
        split="none",
        label="cpu-protocol",
    )


def stage_data(study: dict[str, Any]) -> None:
    """Download the declared datasets and GIFT-Eval, convert, synthesise, pack.

    Corpus and synthetic manifests for this study are not the full-corpus manifests:
    conversion writes its manifest to a scratch directory so ``results/data/`` stays
    the audited full-corpus record.
    """
    import tempfile

    from tsfm.data.convert import build_corpus
    from tsfm.data.sources import download
    from tsfm.data.synthetic import build_synthetic
    from tsfm.paths import processed_dir, results_dir
    from tsfm.train.data import pack

    wanted = set(TRAIN_DATASETS) | set(VAL_DATASETS)
    print("downloading pretrain subset", flush=True)
    errs = download("pretrain", only=wanted, workers=8)
    print("downloading GIFT-Eval", flush=True)
    errs += download("gifteval", workers=8)
    if errs:
        raise SystemExit("\n".join(errs))
    scratch = Path(tempfile.mkdtemp(prefix="tsfm-cpu-manifest-"))
    previous = os.environ.get("TSFM_RESULTS")
    os.environ["TSFM_RESULTS"] = str(scratch)
    try:
        build_corpus(only=wanted)
    finally:
        if previous is None:
            os.environ.pop("TSFM_RESULTS", None)
        else:
            os.environ["TSFM_RESULTS"] = previous
    syn = study["synthetic"]
    build_synthetic(
        n_series=int(syn["n_series"]),
        length=int(syn["length"]),
        seed=int(syn["seed"]),
        max_kernels=int(syn["max_kernels"]),
        val_series=int(syn["val_series"]),
        workers=4,
        manifest_path=results_dir() / "cpu_study" / "synthetic_manifest.json",
    )
    root = processed_dir()
    real = [p for p in (root / "pretrain").glob("*.parquet") if p.stem != "kernelsynth"]
    # Only pack datasets this study declared, in case the cache already holds others.
    real = [p for p in real if p.stem in TRAIN_DATASETS]
    pack(real, root / "packed" / "pretrain")
    pack([root / "pretrain" / "kernelsynth.parquet"], root / "packed" / "synthetic")
    pack([root / "val" / "kernelsynth.parquet"], root / "packed" / "synthetic_val")
    print("data ready", flush=True)


def _write_run_yaml(rc: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(rc, sort_keys=False))
    return dest


def stage_train(study: dict[str, Any], only_seed: int | None) -> None:
    from tsfm.paths import results_dir
    from tsfm.train.loop import train

    jobs: list[tuple[int, str]] = [(int(s), "rope") for s in study["seeds"]]
    jobs.append((0, "none"))
    for seed, position in jobs:
        if only_seed is not None and seed != only_seed:
            continue
        if position == "none" and only_seed not in (None, 0):
            continue
        rc = _run_config(study, seed, position)
        run_dir = results_dir() / "runs" / rc["name"]
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("done"):
                print(f"train {rc['name']} already done", flush=True)
                continue
        cfg_path = _write_run_yaml(rc, run_dir / "config.yaml")
        print(f"training {rc['name']}", flush=True)
        train(cfg_path, run_dir)


def _factory(model_path: Path, study: dict[str, Any], cqr_path: Path | None) -> Any:
    from tsfm.calibrate import CQR
    from tsfm.eval.run import TSFMPredictor
    from tsfm.forecast import load_model

    model = load_model(str(model_path), "cpu")
    adjustment = None
    if cqr_path is not None:
        adjustment = CQR.from_dict(json.loads(cqr_path.read_text())["cqr"])
    max_context = int(study["max_context"])
    batch = int(study["eval_batch_size"])
    fkw = {"adjustment": adjustment} if adjustment is not None else {}

    def factory(ds: Any, h: int, s: int, f: str) -> TSFMPredictor:
        del ds, s, f
        return TSFMPredictor(model, h, max_context, batch, fkw or None)

    return factory, model


def _model_info(path: Path, model: Any, study: dict[str, Any], *, cqr: bool) -> dict[str, Any]:
    from tsfm.provenance import sha256_file

    return {
        "kind": "tsfm",
        "study": "cpu_prereg",
        "reference_summary": False,
        "checkpoint_sha256": sha256_file(path),
        "model_config": model.cfg.to_dict(),
        "params_total": model.num_params(),
        "max_context": int(study["max_context"]),
        "cqr": cqr,
        "leaderboard": {
            "model": path.parent.name,
            "model_type": "zero-shot",
            "model_dtype": "float32",
            "testdata_leakage": "No",
            "replication_code_available": "Yes",
        },
    }


def _eval_one(
    study: dict[str, Any],
    name: str,
    split: str,
    *,
    cqr: bool,
    only: list[str] | None,
) -> None:
    from tsfm.eval.run import output_dir, run_eval
    from tsfm.paths import results_dir

    out = output_dir(name, split, False)
    if (out / "all_results.csv").exists():
        print(f"eval {split} {name} already done", flush=True)
        return
    base = name.removesuffix("_cqr") if cqr else name
    model_path = results_dir() / "runs" / base / "model.pt"
    cqr_path = results_dir() / "cpu_study" / f"{base}_cqr.json" if cqr else None
    if cqr_path is not None and not cqr_path.exists():
        raise SystemExit(f"missing conformal fit {cqr_path}")
    factory, model = _factory(model_path, study, cqr_path)
    print(f"eval {split} {name}", flush=True)
    run_eval(
        name,
        factory,
        split=split,
        model_info=_model_info(model_path, model, study, cqr=cqr),
        only=only,
    )


def stage_val(study: dict[str, Any], only_seed: int | None) -> None:
    prefix = str(study["name"])
    names = [run_name(prefix, int(s)) for s in study["seeds"]]
    names.append(run_name(prefix, 0, position="none"))
    for name in names:
        seed = 0 if name.endswith("norope") else int(name.rsplit("seed", 1)[1])
        if only_seed is not None and seed != only_seed:
            continue
        _eval_one(study, name, "val", cqr=False, only=list(VAL_DATASETS))


def _windows(test_data: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if hasattr(test_data, "input") and hasattr(test_data, "label"):
        return list(zip(test_data.input, test_data.label, strict=True))
    return [(a, b) for a, b in test_data]


def fit_cqr_for_model(study: dict[str, Any], name: str) -> Path:
    """Conformity scores on the real validation datasets only. Synthetic val is excluded."""
    from tsfm.calibrate import fit_cqr, scaled_interval_scores
    from tsfm.eval.valsplit import load_val, val_configs
    from tsfm.forecast import effective_max_context, forecast
    from tsfm.paths import results_dir
    from tsfm.provenance import sha256_file, write_result
    from tsfm.tokenizers import QUANTILES, mean_scale

    out = results_dir() / "cpu_study" / f"{name}_cqr.json"
    if out.exists():
        print(f"cqr {name} already fit", flush=True)
        return out
    import torch

    from tsfm.forecast import _left_pad, load_model

    model_path = results_dir() / "runs" / name / "model.pt"
    model = load_model(str(model_path), "cpu")
    max_context = int(study["max_context"])
    batch = int(study["eval_batch_size"])
    buckets: dict[int, list[np.ndarray]] = {50: [], 80: [], 90: []}
    n_points = 0
    cfgs = [c for c in val_configs() if c.dataset in VAL_DATASETS]
    if not cfgs:
        raise SystemExit("no validation configs; run the data stage first")
    for cfg in cfgs:
        ds = load_val(cfg)
        windows = _windows(ds.test_data)
        print(f"cqr scores {cfg.name} ({len(windows)} windows)", flush=True)
        for start in range(0, len(windows), batch):
            chunk = windows[start : start + batch]
            contexts = [np.asarray(inp["target"], np.float32) for inp, _lab in chunk]
            labels = [np.asarray(lab["target"], np.float32) for _inp, lab in chunk]
            pred_len = int(labels[0].shape[0])
            used = effective_max_context(model, pred_len, max_context)
            x = torch.from_numpy(_left_pad(contexts, used, model.cfg.patch_len))
            scales = mean_scale(x, torch.isfinite(x)).numpy().reshape(-1)
            fc = forecast(model, contexts, pred_len, max_context)
            for y, q, scale in zip(labels, fc.quantiles, scales, strict=True):
                scored = scaled_interval_scores(y, q, float(scale), tuple(model.cfg.quantiles))
                for key, arr in scored.items():
                    buckets[key].append(arr)
                    if key == 80:
                        n_points += int(arr.size)
    scores = {k: np.concatenate(v) if v else np.empty(0) for k, v in buckets.items()}
    cqr = fit_cqr(scores, tuple(model.cfg.quantiles) or QUANTILES)
    payload = {
        "model": name,
        "checkpoint_sha256": sha256_file(model_path),
        "n_scores": {str(k): int(v.size) for k, v in scores.items()},
        "n_80_points": n_points,
        "cqr": cqr.to_dict(),
        "val_datasets": list(VAL_DATASETS),
    }
    print(f"cqr {name} expansion {cqr.expansion}", flush=True)
    return write_result(out, payload, split="val", label=f"cqr-{name}")


def stage_calibrate(study: dict[str, Any], only_seed: int | None) -> None:
    prefix = str(study["name"])
    for seed in study["seeds"]:
        if only_seed is not None and int(seed) != only_seed:
            continue
        fit_cqr_for_model(study, run_name(prefix, int(seed)))


def stage_val_cqr(study: dict[str, Any], only_seed: int | None) -> None:
    prefix = str(study["name"])
    for seed in study["seeds"]:
        if only_seed is not None and int(seed) != only_seed:
            continue
        _eval_one(
            study,
            run_name(prefix, int(seed), cqr=True),
            "val",
            cqr=True,
            only=list(VAL_DATASETS),
        )


def stage_test(study: dict[str, Any], only_seed: int | None) -> None:
    """Score each seed once raw and once conformalised. No other model is test-scored."""
    prefix = str(study["name"])
    os.environ.setdefault("GIFT_EVAL", str(_gifteval()))
    for seed in study["seeds"]:
        if only_seed is not None and int(seed) != only_seed:
            continue
        base = run_name(prefix, int(seed))
        _eval_one(study, base, "test", cqr=False, only=None)
        _eval_one(study, run_name(prefix, int(seed), cqr=True), "test", cqr=True, only=None)


def _gifteval() -> Path:
    from tsfm.paths import gifteval_dir

    return gifteval_dir()


def stage_attention(study: dict[str, Any]) -> None:
    """Last-layer attention on one held-out series. Not used for selection."""
    import pyarrow.parquet as pq

    from tsfm.forecast import load_model
    from tsfm.paths import processed_dir, results_dir
    from tsfm.provenance import write_result
    from tsfm.repr import last_query_attention

    out = results_dir() / "analysis" / "attention_cpu_s0.json"
    if out.exists():
        print("attention readout already written", flush=True)
        return
    name = run_name(str(study["name"]), 0)
    model = load_model(str(results_dir() / "runs" / name / "model.pt"), "cpu")
    # Hourly building load: a strong daily period, held out of training.
    path = processed_dir() / "val" / "bdg-2_bear.parquet"
    table = pq.read_table(path, columns=["item_id", "target", "freq"])
    target = table.column("target")[0].as_py()
    context = np.asarray(target, np.float32)[: int(study["data"]["window"])]
    profile = last_query_attention(model, context, int(study["max_context"]))
    profile_small = {k: profile[k] for k in profile if k not in {"lag_patches", "mass"}}
    # Keep the curve, but it is long; store it so the figure can be redrawn.
    profile_small["lag_patches"] = profile["lag_patches"]
    profile_small["mass"] = profile["mass"]
    write_result(
        out,
        {
            "model": name,
            "dataset": "bdg-2_bear",
            "item_id": table.column("item_id")[0].as_py(),
            "freq": table.column("freq")[0].as_py(),
            "context_length": int(context.size),
            "profile": profile_small,
        },
        split="val",
        label="attention-cpu",
    )
    print("wrote", out, flush=True)


def stage_report() -> None:
    from tsfm.analysis import cpu_study_report
    from tsfm.docs.figures import write_coverage_figure

    path = cpu_study_report()
    print("wrote", path, flush=True)
    fig = write_coverage_figure()
    if fig:
        print("wrote", fig, flush=True)


def run_stages(study_path: Path, stage: str, only_seed: int | None) -> None:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    study = load_study(study_path)
    if stage in {"protocol", "all"}:
        print(write_protocol(study, study_path), flush=True)
    if stage in {"data", "all"}:
        stage_data(study)
    if stage in {"train", "all"}:
        stage_train(study, only_seed)
    if stage in {"val", "all"}:
        stage_val(study, only_seed)
    if stage in {"calibrate", "all"}:
        stage_calibrate(study, only_seed)
    if stage in {"val-cqr", "all"}:
        stage_val_cqr(study, only_seed)
    if stage in {"test", "all"}:
        stage_test(study, only_seed)
    if stage in {"attention", "all"}:
        stage_attention(study)
    if stage in {"report", "all"}:
        stage_report()


def _cmd(sub: Any) -> None:
    import argparse

    p = sub.add_parser("cpu-study", help="train and score the pre-registered CPU study")
    p.add_argument("--config", type=Path, default=Path("configs/cpu_study.yaml"))
    p.add_argument(
        "--stage",
        default="all",
        choices=[
            "protocol",
            "data",
            "train",
            "val",
            "calibrate",
            "val-cqr",
            "test",
            "attention",
            "report",
            "all",
        ],
    )
    p.add_argument("--seed", type=int, default=None, help="restrict train/eval to one seed")

    def run(a: argparse.Namespace) -> int:
        run_stages(a.config, a.stage, a.seed)
        return 0

    p.set_defaults(func=run)


register(_cmd)
