"""
Gather system + deltai state for snapshots (read-only, bounded).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from typing import Any

import httpx

from .pacman_audit import capture_pacman_q, get_pending_updates

# Optional key configs under /etc (no traversal)
_DEFAULT_ETC_FILES = (
    "/etc/pacman.conf",
    "/etc/makepkg.conf",
    "/etc/mkinitcpio.conf",
)

_MAX_FILE_BYTES = 512 * 1024


def _safe_etc_path(path: str) -> str | None:
    p = os.path.realpath(os.path.expanduser(path.strip()))
    if not p.startswith("/etc/") or ".." in path:
        return None
    return p


def _read_bounded(path: str) -> tuple[str | None, str | None]:
    try:
        st = os.stat(path)
        if st.st_size > _MAX_FILE_BYTES:
            return None, f"file too large ({st.st_size} bytes)"
        with open(path, "rb") as f:
            raw = f.read()
        return raw.decode("utf-8", errors="replace"), None
    except OSError as e:
        return None, str(e)


def collect_config_snapshots(
    extra_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return { path: {content, sha256, error} } for allowlisted /etc files."""
    paths = list(_DEFAULT_ETC_FILES)
    if extra_paths:
        for p in extra_paths:
            sp = _safe_etc_path(p)
            if sp and sp not in paths:
                paths.append(sp)

    out: dict[str, Any] = {}
    for p in paths:
        safe = _safe_etc_path(p)
        if not safe:
            out[p] = {"error": "path not under /etc or invalid"}
            continue
        content, err = _read_bounded(safe)
        if err:
            out[safe] = {"error": err}
            continue
        assert content is not None
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        out[safe] = {"content": content, "sha256": h, "byte_size": len(content.encode("utf-8"))}
    return out


def parse_pacman_q_list(stdout: str) -> dict[str, str]:
    """Parse `pacman -Q` lines into package -> version."""
    pkg_ver: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # "pkg ver-rel" — version may contain spaces rarely; pacman uses space split from right
        parts = line.split()
        if len(parts) >= 2:
            pkg = parts[0]
            ver = " ".join(parts[1:])
            pkg_ver[pkg] = ver
    return pkg_ver


def _ollama_tags() -> dict[str, Any]:
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{base}/api/tags")
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}", "models": []}
            data = r.json()
    except Exception as e:
        return {"ok": False, "error": str(e), "models": []}

    models: list[dict[str, str]] = []
    for m in data.get("models") or []:
        name = (m.get("name") or "").strip()
        if name:
            models.append({"name": name, "digest": (m.get("digest") or "")[:20]})
    return {"ok": True, "models": models}


def collect_deltai_state() -> dict[str, Any]:
    try:
        from persistence import get_sqlite_path

        db_path = get_sqlite_path()
    except Exception:
        db_path = os.path.expanduser("~/.local/share/deltai/delta.db")

    db_info: dict[str, Any] = {"path": db_path}
    try:
        st = os.stat(os.path.expanduser(db_path))
        db_info["size_bytes"] = st.st_size
        db_info["mtime"] = st.st_mtime
    except OSError as e:
        db_info["error"] = str(e)

    ollama = _ollama_tags()
    data_dir = os.path.expanduser(os.getenv("DELTA_DATA_DIR", ""))
    dd: dict[str, Any] = {}
    if data_dir and os.path.isdir(data_dir):
        try:
            dd["path"] = data_dir
            dd["writable"] = os.access(data_dir, os.W_OK)
        except Exception as e:
            dd["error"] = str(e)

    return {
        "sqlite": db_info,
        "ollama": ollama,
        "delta_data_dir": dd or None,
        "env_model": os.getenv("DELTAI_MODEL"),
        "env_small_model": os.getenv("DELTAI_SMALL_MODEL"),
    }


def build_snapshot_payload(
    *,
    include_reverse_deps: bool = False,
    extra_config_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Full in-memory payload before DB insert (blobs registered separately)."""
    pending = get_pending_updates(include_reverse_deps=include_reverse_deps, reverse_deps_limit=5)
    pq_code, pq_out, pq_err = capture_pacman_q()
    configs = collect_config_snapshots(extra_paths=extra_config_paths)
    deltai = collect_deltai_state()

    uname = platform.uname()
    system_summary = {
        "pending_updates": pending,
        "pacman_Q": {
            "exit_code": pq_code,
            "stderr_tail": (pq_err or "")[-2000:],
            "line_count": len([x for x in pq_out.splitlines() if x.strip()]),
        },
        "config_paths_captured": list(configs.keys()),
    }

    flags = _derive_flags(pending, configs, pq_out)

    return {
        "hostname": uname.node,
        "kernel_release": uname.release,
        "system_summary": system_summary,
        "deltai_summary": deltai,
        "flags": flags,
        "blobs": {
            "pacman_Q": pq_out if pq_code in (0, 1) else "",
            "configs_json": json.dumps(configs, ensure_ascii=False),
        },
        "errors": {
            "pacman_Q": None if pq_code in (0, 1) else (pq_err or f"exit {pq_code}")[:2000],
        },
    }


_KERNELISH = re.compile(r"^linux($|-)", re.IGNORECASE)
_GLIBC = re.compile(r"^glibc$", re.IGNORECASE)


def _derive_flags(
    pending: dict[str, Any],
    configs: dict[str, Any],
    pacman_q_stdout: str,
) -> list[str]:
    flags: list[str] = []
    for rec in pending.get("pending") or []:
        pkg = rec.get("package") or ""
        if _KERNELISH.match(pkg):
            flags.append("pending_kernel_upgrade")
        if _GLIBC.match(pkg):
            flags.append("pending_glibc_upgrade")
    if pending.get("pending_count", 0) >= 50:
        flags.append("large_update_batch")
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
