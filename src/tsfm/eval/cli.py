"""``tsfm eval --model NAME --split val|test``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tsfm.cli import register

STAT_INFO = {
    "seasonal_naive": "Seasonal_Naive",
    "naive": "Naive",
    "auto_ets": "Auto_ETS",
    "auto_arima": "Auto_Arima",
    "auto_theta": "Auto_Theta",
}


def _cmd(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("eval", help="score a model on the validation or test split")
    p.add_argument(
        "--model", required=True, help="baseline name or path to a trained run dir / model.pt"
    )
    p.add_argument("--name", help="results name for a trained model (default: run dir name)")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument(
        "--exploratory", action="store_true", help="re-score; results labelled exploratory"
    )
    p.add_argument(
        "--only", nargs="*", help="restrict to these configs or dataset keys (partial run)"
    )
    p.add_argument(
        "--max-length", type=int, default=None, help="statsforecast: fit on the last N points"
    )
    p.add_argument("--parallel", action="store_true", help="statsforecast: use all cores")
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=20, help="bins model: sample paths")
    p.add_argument("--device", default="auto")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (baselines only)")

    def run(a: argparse.Namespace) -> int:
        from tsfm.eval.run import run_eval

        factory: Any
        if a.model in STAT_INFO:
            import functools

            from tsfm.eval.run import statsforecast_factory

            kw: dict[str, Any] = {"parallel": a.parallel}
            if a.max_length:
                kw["max_length"] = a.max_length
            factory = functools.partial(statsforecast_factory, a.model, kw)
            name = a.model
            info = {
                "kind": "statsforecast",
                "model": a.model,
                "max_length": a.max_length,
                "fallback": "SeasonalNaive",
                "leaderboard": {
                    "model": STAT_INFO[a.model],
                    "model_type": "statistical",
                    "testdata_leakage": "No",
                    "replication_code_available": "Yes",
                },
            }
        else:
            import torch

            from tsfm.eval.run import TSFMPredictor
            from tsfm.forecast import load_model
            from tsfm.provenance import sha256_file
            from tsfm.train.loop import pick_device

            path = Path(a.model)
            mp = path / "model.pt" if path.is_dir() else path
            dev = pick_device(a.device)
            model = load_model(str(mp), dev)
            fkw = (
                {"n_samples": a.n_samples, "generator": torch.Generator(dev).manual_seed(0)}
                if model.cfg.tokenizer == "bins"
                else {}
            )
            factory = lambda ds, h, s, f: TSFMPredictor(model, h, a.max_context, a.batch_size, fkw)  # noqa: E731
            name = a.name or (path.name if path.is_dir() else path.parent.name)
            info = {
                "kind": "tsfm",
                "checkpoint_sha256": sha256_file(mp),
                "model_config": model.cfg.to_dict(),
                "params_total": model.num_params(),
                "max_context": a.max_context,
                "n_samples": a.n_samples if model.cfg.tokenizer == "bins" else None,
                "leaderboard": {
                    "model": name,
                    "model_type": "zero-shot",
                    "model_dtype": "float32",
                    "testdata_leakage": "No",
                    "replication_code_available": "Yes",
                },
            }
        out = run_eval(
            name,
            factory,
            split=a.split,
            model_info=info,
            exploratory=a.exploratory,
            only=a.only,
            jobs=a.jobs if a.model in STAT_INFO else 1,
        )
        print("results written to", out)
        return 0

    p.set_defaults(func=run)


register(_cmd)


def _report(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("report", help="summarise evaluation results into results/analysis/")
    p.add_argument("--split", choices=["val", "test"], required=True)

    def run(a: argparse.Namespace) -> int:
        from tsfm.analysis import report

        print("wrote", report(a.split))
        return 0

    p.set_defaults(func=run)


register(_report)


def _scaling(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("scaling", help="fit validation loss vs compute over finished runs")
    p.add_argument("--prefix", default="patch_s")

    def run(a: argparse.Namespace) -> int:
        from tsfm.scaling import scaling_report

        print("wrote", scaling_report(a.prefix))
        return 0

    p.set_defaults(func=run)


register(_scaling)
