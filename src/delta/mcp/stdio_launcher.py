"""
Console script `deltai-mcp`: run project/deltai_mcp_stdio.py from a git checkout.

Resolves paths in this order:
1. DELTAI_PROJECT_ROOT environment variable
2. Parent of src/delta/mcp (repo root when installed editable from layout repo/src/delta/mcp)
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _resolve_project_dir() -> Path:
    env = os.environ.get("DELTAI_PROJECT_ROOT", "").strip()
    if env:
        root = Path(os.path.expanduser(env)).resolve()
    else:
        # stdio_launcher.py -> mcp -> delta -> src -> repo root
        root = Path(__file__).resolve().parents[3]
    proj = root / "project"
    if not (proj / "deltai_mcp_stdio.py").is_file():
        print(
            "deltai-mcp: expected project/deltai_mcp_stdio.py under DELTAI_PROJECT_ROOT "
            f"or repo root; got root={root}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return proj


def main() -> None:
    proj = _resolve_project_dir()
    script = proj / "deltai_mcp_stdio.py"
    sys.path.insert(0, str(proj))
    os.chdir(proj)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
