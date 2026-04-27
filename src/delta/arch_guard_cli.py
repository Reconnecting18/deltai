"""Console script entry: adds project/ to path and runs arch-update-guard CLI."""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    project = repo / "project"
    sys.path.insert(0, str(project))
    from core.arch_update_guard.cli import main as inner

    return inner()


if __name__ == "__main__":
    raise SystemExit(main())
