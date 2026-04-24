"""Console script ``deltai-server`` — uvicorn for ``project/main.py`` on Linux."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from delta.cli.repo_root import project_dir


def main() -> None:
    proj = project_dir()
    if not proj.is_dir():
        print(
            "deltai-server: expected a project/ directory in the repo checkout.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="deltai-server",
        description="Run the deltai FastAPI app (project/main.py) via uvicorn.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes (development only).",
    )
    args = parser.parse_args()

    os.chdir(proj)
    project_str = str(proj)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
