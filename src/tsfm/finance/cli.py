"""``tsfm vol ...``: the realized-volatility study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm.cli import register


def _cmd(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("vol", help="realized-volatility fine-tuning study (milestones 8-9)")
    p.add_argument("action", choices=["data", "baselines", "model", "report"])
    p.add_argument("--model", type=Path, help="run dir or model.pt (action=model)")
    p.add_argument("--name", help="results name (action=model)")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--models", nargs="*", default=["random_walk", "har"])

    def run(a: argparse.Namespace) -> int:
        from tsfm.finance import study

        if a.action == "data":
            from tsfm.finance import data

            errs = data.download()
            if errs:
                print("\n".join(errs))
                return 1
            print(data.build())
        elif a.action == "baselines":
            study.baseline_runs()
        elif a.action == "model":
            from tsfm.finance.finetune import finetune, model_forecaster
            from tsfm.forecast import load_model
            from tsfm.provenance import sha256_file
            from tsfm.train.loop import pick_device

            mp = a.model / "model.pt" if a.model.is_dir() else a.model
            model = load_model(str(mp), pick_device(a.device))
            series = study.load_all()
            info: dict[str, object] = {"checkpoint_sha256": sha256_file(mp), "finetune": a.finetune}
            if a.finetune:
                model, log = finetune(model, series)
                info["finetune_log"] = log
            rows = study.evaluate(
                model_forecaster(model), series, study.TEST_START, series[0].dates[-1]
            )
            print("wrote", study.save(a.name, rows, info))
        else:
            print("wrote", study.report(a.models))
        return 0

    p.set_defaults(func=run)


register(_cmd)
