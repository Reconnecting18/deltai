"""Launcher for web training data collection (implementation under project/extensions/training/)."""

from __future__ import annotations

import sys
from pathlib import Path

_project = Path(__file__).resolve().parents[1] / "project"
_project_str = str(_project)
if _project_str not in sys.path:
    sys.path.insert(0, _project_str)

from extensions.training import collect_training_data as _collect  # noqa: E402

if __name__ == "__main__":
    sys.exit(_collect.main())
