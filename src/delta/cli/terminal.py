"""Console script ``deltai`` — rich terminal client (``project/cli.py``)."""

from __future__ import annotations

import runpy
import sys

from delta.cli.repo_root import project_dir


def main() -> None:
    cli = project_dir() / "cli.py"
    if not cli.is_file():
        print(
            "deltai: expected project/cli.py next to this install (clone the full repo).",
            file=sys.stderr,
        )
        sys.exit(1)
    argv0 = sys.argv[0]
    sys.argv[0] = str(cli)
    try:
        runpy.run_path(str(cli), run_name="__main__")
    finally:
        sys.argv[0] = argv0
