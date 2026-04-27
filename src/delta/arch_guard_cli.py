"""Console script entry: resolves the deltai ``project/`` tree and runs arch-update-guard CLI."""

from __future__ import annotations

import os
import pathlib
import sys

_ARCH_GUARD_REL = ("core", "arch_update_guard", "__init__.py")
_ERR_HINT = (
    "arch-update-guard needs the deltai repo ``project/`` package on disk.\n"
    "  • Set DELTAI_PROJECT_ROOT or DELTA_PROJECT_ROOT to the repo root "
    "(parent of project/), or\n"
    "  • Run from inside the clone (we search upward for "
    "project/core/arch_update_guard), or\n"
    "  • Use an editable install from the clone so paths resolve next to src/.\n"
)


def _is_valid_project_dir(project: pathlib.Path) -> bool:
    return project.joinpath(*_ARCH_GUARD_REL).is_file()


def _resolve_project_dir() -> pathlib.Path | None:
    for key in ("DELTAI_PROJECT_ROOT", "DELTA_PROJECT_ROOT"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        root = pathlib.Path(raw).expanduser().resolve()
        candidate = root / "project"
        if _is_valid_project_dir(candidate):
            return candidate

    here = pathlib.Path.cwd().resolve()
    for base in (here, *here.parents):
        candidate = base / "project"
        if _is_valid_project_dir(candidate):
            return candidate
        nested = base / "deltai" / "project"
        if _is_valid_project_dir(nested):
            return nested

    this = pathlib.Path(__file__).resolve()
    for parent in this.parents:
        candidate = parent / "project"
        if _is_valid_project_dir(candidate):
            return candidate

    return None


def main() -> int:
    project = _resolve_project_dir()
    if project is None:
        print(_ERR_HINT, file=sys.stderr)
        return 1
    project_str = str(project.resolve())
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
    from core.arch_update_guard.cli import main as inner

    return inner()


if __name__ == "__main__":
    raise SystemExit(main())
