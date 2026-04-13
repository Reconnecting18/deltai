"""
Filesystem path helpers for training code — constrains paths under known roots
to mitigate path-injection / tainted-path issues (CodeQL py/path-injection).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import IO, Literal

# Export formats allowed by export_dataset()
_EXPORT_FORMATS = frozenset({"alpaca", "sharegpt", "chatml"})

# Basenames only from os.listdir(DATASETS_PATH) — conservative allowlist
_JSONL_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl$")


def safe_dataset_basename(name: str) -> str:
    """Dataset stem safe as a filename segment (same rules as legacy ``_dataset_path``)."""
    if not isinstance(name, str):
        raise ValueError("Dataset name must be a string.")
    safe = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe:
        raise ValueError("Invalid dataset name")
    return safe


def resolve_under(root: str, *segments: str) -> str:
    """
    Resolve ``root`` (expanduser + realpath) then append each segment (must be a
    single path component, no ``..`` or separators). Ensures the final path stays
    under ``root`` (after realpath), independent of process CWD.
    """
    root_resolved = Path(os.path.realpath(os.path.expanduser(root)))
    cur = root_resolved
    for seg in segments:
        if not seg or seg in (".", ".."):
            raise ValueError("invalid path segment")
        part = Path(seg)
        if part.is_absolute():
            raise ValueError("path segment must not be absolute")
        if part.name != seg or ".." in part.parts:
            raise ValueError("path segment must be a single file name")
        nxt = cur / seg
        cur = Path(os.path.realpath(nxt))
        try:
            cur.relative_to(root_resolved)
        except ValueError as e:
            raise ValueError("path escapes root directory") from e
    return os.fspath(cur)


def safe_export_filename(dataset_name: str, fmt: str) -> str:
    """Return a safe basename ``{stem}_{fmt}.json`` for files under EXPORTS_PATH."""
    if fmt not in _EXPORT_FORMATS:
        raise ValueError(f"Unknown export format: {fmt!r}")
    stem = safe_dataset_basename(dataset_name)
    return f"{stem}_{fmt}.json"


def safe_jsonl_basename(fname: str) -> str | None:
    """If ``fname`` is a safe dataset listdir basename, return it; else None."""
    if not fname or "/" in fname or "\\" in fname or fname in (".", ".."):
        return None
    if not _JSONL_BASENAME.match(fname):
        return None
    return fname


def require_path_under(path: str, root: str) -> str:
    """Resolve ``path`` and ensure it lies under ``root`` (realpath + expanduser)."""
    p = Path(os.path.realpath(os.path.expanduser(path)))
    r = Path(os.path.realpath(os.path.expanduser(root)))
    try:
        p.relative_to(r)
    except ValueError as e:
        raise ValueError("path is outside allowed directory") from e
    return os.fspath(p)


def open_text(
    path: str,
    root: str,
    mode: Literal["r", "w", "a"] = "r",
    *,
    encoding: str = "utf-8",
) -> IO[str]:
    """
    Open a text file only after verifying ``path`` resolves under ``root``.
    Helps CodeQL py/path-injection recognize ``require_path_under`` before ``open``.
    """
    safe = require_path_under(path, root)
    return open(safe, mode, encoding=encoding)
