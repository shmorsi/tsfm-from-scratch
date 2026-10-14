"""``tsfm data ...`` commands."""

from __future__ import annotations

import argparse

from tsfm.cli import register


def _cmd(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("data", help="pinned raw data, Parquet corpus, synthetic data, leakage")
    p.add_argument(
        "action",
        choices=["lock", "download", "verify", "build", "synth", "leakage", "all"],
    )
    p.add_argument("--which", choices=["pretrain", "gifteval", "both"], default="both")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", nargs="*", help="restrict to these dataset names")

    def run(a: argparse.Namespace) -> int:
        from tsfm.data import sources

        names = ["pretrain", "gifteval"] if a.which == "both" else [a.which]
        only = set(a.only) if a.only else None
        if a.action == "lock":
            sources.build_locks()
            return 0
        errors: list[str] = []
        if a.action in ("download", "all"):
            for n in names:
                errors += sources.download(n, workers=a.workers, only=only)
        if a.action == "verify":
            for n in names:
                errors += sources.verify(n)
        if errors:
            print("\n".join(errors))
            return 1
        if a.action in ("build", "all"):
            from tsfm.data.convert import build_corpus

            build_corpus(only=only)
        if a.action in ("synth", "all"):
            from tsfm.data.synthetic import build_synthetic

            build_synthetic()
        if a.action in ("leakage", "all"):
            from tsfm.data.leakage import run_leakage_check

            return 0 if run_leakage_check() else 1
        return 0

    p.set_defaults(func=run)


register(_cmd)
