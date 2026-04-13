"""Resolve user-influenced paths under trusted directory anchors (CodeQL path-injection)."""

from __future__ import annotations

import os


def realpath_under(anchor: str, user_path: str) -> str:
    """
    Return the real path of user_path if it lies under anchor (after expanduser).
    Raises ValueError if the path escapes the anchor.
    """
    anchor_r = os.path.realpath(os.path.expanduser(anchor))
    full = os.path.realpath(os.path.expanduser(os.path.normpath(user_path)))
    if full == anchor_r:
        return full
    prefix = anchor_r + os.sep
    if full.startswith(prefix):
        return full
    raise ValueError("path outside allowed root")


def tool_filesystem_roots() -> list[str]:
    """Allowed roots for agent tool file read/write/list."""
    roots: list[str] = []
    for env in ("DELTAI_WORKSPACE", "DELTAI_REPO_ROOT"):
        v = os.getenv(env)
        if v:
            roots.append(v)
    roots.append(os.path.expanduser("~/deltai"))
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        key = os.path.normcase(os.path.realpath(os.path.expanduser(r)))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def resolve_tool_path(user_path: str) -> str:
    """
    Resolve a tool path under allowed roots.
    Absolute paths must lie under a root; relative paths are resolved under each root
    until one matches.
    """
    roots_r = [os.path.realpath(os.path.expanduser(r)) for r in tool_filesystem_roots() if r]
    if not roots_r:
        raise ValueError("no valid anchor roots configured")
    norm = os.path.normpath(os.path.expanduser(user_path.strip()))
    trials: list[str] = []
    if os.path.isabs(norm):
        trials.append(os.path.realpath(norm))
    else:
        for root in roots_r:
            trials.append(os.path.realpath(os.path.join(root, norm)))
    for full in trials:
        for anchor_r in roots_r:
            if full == anchor_r or full.startswith(anchor_r + os.sep):
                return full
    raise ValueError("path outside allowed roots")


def export_dir_default() -> str:
    return os.path.expanduser(os.getenv("DELTAI_EXPORT_DIR", "~/deltai/exports"))
