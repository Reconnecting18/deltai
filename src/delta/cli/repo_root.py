"""Resolve the repository root for sibling paths like ``project/``."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the deltai git checkout root (directory that contains ``project/``)."""
    here = Path(__file__).resolve()
    # src/delta/cli/repo_root.py -> parents[3] == repo root
    return here.parents[3]


def project_dir() -> Path:
    return repo_root() / "project"
