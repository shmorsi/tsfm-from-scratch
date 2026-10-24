"""``tsfm pipeline --config configs/tiny_cpu.yaml``: the whole pipeline, small, on CPU.

download (verified) -> convert -> synthetic -> pack -> train -> eval (val + a few
GIFT-Eval configs, labelled exploratory). Everything is written under a temporary
TSFM_HOME / TSFM_RESULTS, so the committed results are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from tsfm.cli import register


def run_pipeline(config: Path, workdir: Path) -> dict[str, Any]:
    """Download, convert, train and score the tiny config under ``workdir``."""
    raw = yaml.safe_load(config.read_text())
    pc = raw["pipeline"]
    home = workdir / "home"
    os.environ["TSFM_HOME"] = str(home)
    os.environ["TSFM_RESULTS"] = str(workdir / "results")
    os.environ["TSFM_PACKED"] = str(home / "processed" / "packed")
    from tsfm.paths import gifteval_dir

    gifteval = Path(os.environ.get("GIFT_EVAL_CACHE", gifteval_dir()))
    t0 = time.time()
    timings: dict[str, float] = {}

    from tsfm.data import sources
    from tsfm.data.convert import Rules, convert_dataset, dataset_shards

    wanted = set(pc["pretrain_datasets"]) | set(pc["val_datasets"])
    errs = sources.download("pretrain", only=wanted)
    os.environ["GIFT_EVAL"] = str(gifteval)
    keys = {c.split("/")[0] for c in pc["eval_configs"]}
    from tsfm.eval.gifteval import PRETTY_NAMES

    inv = {v: k for k, v in PRETTY_NAMES.items()}
    ge_dirs = {inv.get(k, k) for k in keys}
    lock = sources.read_lock("gifteval")
    only = [e for e in lock["files"] if e["path"].split("/")[0] in ge_dirs]
    errs += _download_entries(lock, only, gifteval)
    if errs:
        raise SystemExit("\n".join(errs))
    timings["download"] = time.time() - t0

    shards = dataset_shards(sources.read_lock("pretrain"), sources.local_dir("pretrain"))
    for ds in sorted(wanted):
        split = "val" if ds in pc["val_datasets"] else "pretrain"
        convert_dataset(
            ds,
            shards[ds],
            home / "processed" / split / f"{ds}.parquet",
            split=split,
            rules=Rules(),
            max_points=None,
            seed=0,
        )
    from tsfm.data.synthetic import build_synthetic

    build_synthetic(
        n_series=pc["synthetic_series"],
        length=pc["synthetic_length"],
        val_series=pc["synthetic_val_series"],
        workers=2,
    )
    timings["convert"] = time.time() - t0

    from tsfm.train.data import pack

    root = home / "processed"
    pack(
        [p for p in (root / "pretrain").glob("*.parquet") if p.stem != "kernelsynth"],
        root / "packed" / "pretrain",
    )
    pack(
        [p for p in (root / "val").glob("*.parquet") if p.stem != "kernelsynth"],
        root / "packed" / "val",
    )
    pack([root / "pretrain" / "kernelsynth.parquet"], root / "packed" / "synthetic")
    pack([root / "val" / "kernelsynth.parquet"], root / "packed" / "synthetic_val")

    from tsfm.train.loop import RunConfig, Trainer

    rc = RunConfig.from_dict({k: v for k, v in raw.items() if k != "pipeline"})
    run_dir = workdir / "runs" / rc.name
    summary = Trainer(rc, run_dir).run()
    timings["train"] = time.time() - t0

    from tsfm.eval.run import TSFMPredictor, run_eval
    from tsfm.forecast import load_model

    model = load_model(str(run_dir / "model.pt"))
    factory = lambda ds, h, s, f: TSFMPredictor(model, h, 512, 256)  # noqa: E731
    info = {"kind": "tsfm", "pipeline": True}
    val_out = run_eval(rc.name, factory, split="val", model_info=info)
    test_out = run_eval(
        rc.name, factory, split="test", model_info=info, exploratory=True, only=pc["eval_configs"]
    )
    timings["eval"] = time.time() - t0
    return {
        "train": summary,
        "val_results": str(val_out),
        "test_results": str(test_out),
        "timings": timings,
    }


def _download_entries(lock: dict[str, Any], entries: list[dict[str, Any]], root: Path) -> list[str]:
    import urllib.request

    from tsfm.data.sources import HF, verify_file

    errs = []
    for e in entries:
        if verify_file(e, root) is None:
            continue
        dest = root / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/datasets/{lock['repo']}/resolve/{lock['revision']}/{e['path']}"
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        err = verify_file(e, root)
        if err:
            errs.append(f"{e['path']}: {err}")
    return errs


def _cmd(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("pipeline", help="run the whole pipeline small, on CPU (CI)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workdir", type=Path)

    def run(a: argparse.Namespace) -> int:
        wd = a.workdir or Path(tempfile.mkdtemp(prefix="tsfm-pipeline-"))
        out = run_pipeline(a.config, wd)
        print(json.dumps(out, indent=1, default=str))
        return 0

    p.set_defaults(func=run)


register(_cmd)
