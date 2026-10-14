"""Score the published FlowState-9.1M checkpoint on GIFT-Eval with our evaluation code.

Run inside the environment from ``tools/flowstate/setup.sh``::

    $FS_VENV/bin/python tools/flowstate/run_flowstate.py --split test --device cuda

Follows upstream's notebooks/flowstate.ipynb (model ``ibm-research/FlowState``, revision
pinned below; batch size 16; seed 0 per dataset; ``no_daily`` for bizitobs_l2c; the
dataset's domain passed to the wrapper), but scores through ``tsfm.eval.run`` so the
metrics, coverage and result files are produced exactly as for every other model.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, os.environ.get("FS_SRC", "/content/granite-tsfm"))

MODEL = "ibm-research/FlowState"
REVISION = "ec815e416006b6a74130e1c79c6c67004c12c0c8"  # "main" on 2026-10-14 = FlowState-9.1M
NAME = "flowstate_9.1m"


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002 - replicate upstream seeding exactly
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test"], default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()

    from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper
    from tsfm_public import FlowStateForPrediction

    from tsfm.eval import run as R
    from tsfm.eval.gifteval import configs, load_dataset

    R.check_test_discipline(NAME, a.split, a.exploratory)
    cfgs = [c for c in configs() if not a.only or c.name in a.only]
    out = R.output_dir(NAME, a.split, a.exploratory)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cfgs:
        set_seed(0)
        ds = load_dataset(c.ds_name, c.term)
        first = next(iter(ds.test_data))[0]["target"]
        n_ch = 1 if first.ndim == 1 else first.shape[0]

        def factory(dset, h, s, f, c=c, n_ch=n_ch):
            m = FlowStateForPrediction.from_pretrained(MODEL, revision=REVISION).to(a.device)
            m.config.min_context = 0
            m.config.device = a.device
            return FlowState_Gift_Wrapper(
                m,
                h,
                n_ch=n_ch,
                batch_size=16,
                f=f,
                device=a.device,
                domain=c.domain,
                no_daily="l2c" in c.ds_name,
            )

        import torch

        with torch.no_grad():
            row = R.evaluate_one(c.name, ds, ds.test_data, factory)
        row.update(
            {"dataset": c.name, "model": NAME, "domain": c.domain, "num_variates": c.num_variates}
        )
        rows.append(row)
        print(
            f"{c.name:45s} MASE {row['eval_metrics/MASE[0.5]']:.4f} CRPS {row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f}",
            flush=True,
        )
    info = {
        "kind": "published-checkpoint",
        "hf_model": MODEL,
        "hf_revision": REVISION,
        "granite_tsfm_commit": "640f7c80a21a4e2253b6d6140d067452bb93e7d4",
        "leaderboard": {
            "model": "FlowState-9.1M (reproduced)",
            "model_type": "zero-shot",
            "testdata_leakage": "No",
            "replication_code_available": "Yes",
        },
    }
    R._write(
        out,
        NAME,
        rows,
        split=a.split,
        exploratory=a.exploratory,
        model_info=info,
        partial=bool(a.only),
    )
    print("results written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
