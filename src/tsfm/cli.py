"""Command-line entry point: ``tsfm <group> <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]
_REGISTRY: list[Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None]] = []


def register(fn: Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None]) -> None:
    """Add a subcommand registrar. Each ``tsfm.<area>.cli`` module calls this on import."""
    _REGISTRY.append(fn)


def _docs(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("docs", help="generate README and cards from results")
    p.add_argument("action", choices=["build", "check"])

    def run(a: argparse.Namespace) -> int:
        from tsfm.docs.build import build

        stale = build(check=a.action == "check")
        if stale:
            print("stale generated docs:", ", ".join(stale), "- run `tsfm docs build`")
            return 1
        return 0

    p.set_defaults(func=run)


def _load_commands() -> None:
    # Imported lazily so `tsfm docs` works without torch installed.
    import importlib

    for mod in (
        "tsfm.data.cli",
        "tsfm.train.cli",
        "tsfm.eval.cli",
        "tsfm.pipeline",
        "tsfm.finance.cli",
        "tsfm.cpu_study",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if not e.name or not e.name.startswith("tsfm"):
                raise


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default: the process arguments) and run the selected subcommand."""
    parser = argparse.ArgumentParser(prog="tsfm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _docs(sub)
    _load_commands()
    # Use the registry of the importable module even when run as `python -m tsfm.cli`.
    import tsfm.cli as registry

    for fn in registry._REGISTRY:
        fn(sub)
    args = parser.parse_args(argv)
    func: Handler = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
