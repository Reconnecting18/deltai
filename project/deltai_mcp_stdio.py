#!/usr/bin/env python3
"""
MCP stdio server for deltai — run from the project/ directory (same as uvicorn).

  cd project && python deltai_mcp_stdio.py

Or use the installed launcher: deltai-mcp (requires pip install -e ".[mcp]").
"""
from __future__ import annotations

import sys


def main() -> None:
    try:
        import anyio
    except ImportError:
        print("Install dependencies: pip install -e '.[dev]'", file=sys.stderr)
        raise SystemExit(1) from None
    try:
        from mcp_bridge import run_stdio_async
    except ImportError as e:
        print(
            f"MCP extra required: pip install -e '.[mcp]'  ({e!s})",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    anyio.run(run_stdio_async)


if __name__ == "__main__":
    main()
