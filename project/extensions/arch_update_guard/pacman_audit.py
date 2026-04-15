"""
Allowlisted read-only pacman / checkupdates helpers for evidence-based update reports.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

_PKG_RE = re.compile(r"^[a-z0-9@._+-]+$", re.IGNORECASE)

# Maximum lines to parse from pactree to avoid huge payloads
_PACTREE_LINE_CAP = 500


def _validate_pkg(name: str) -> bool:
    return bool(name and _PKG_RE.fullmatch(name.strip()))


def _run_allowlisted(argv: list[str], timeout: float = 120.0) -> tuple[int, str, str]:
    """Run only explicitly allowlisted argv. Returns (returncode, stdout, stderr)."""
    if not argv:
        raise ValueError("empty argv")
    exe = argv[0]
    if exe not in ("checkupdates", "pacman", "pactree"):
        raise ValueError(f"executable not allowlisted: {exe}")

    if argv == ["checkupdates"]:
        pass
    elif argv[:2] == ["pacman", "-Qu"] and len(argv) == 2:
        pass
    elif argv[:2] == ["pacman", "-Q"] and len(argv) == 2:
        pass
    elif argv[:2] == ["pacman", "-Qi"] and len(argv) == 3 and _validate_pkg(argv[2]):
        pass
    elif argv[:3] == ["pactree", "-ru"] and len(argv) == 4 and _validate_pkg(argv[3]):
        pass
    else:
        raise ValueError(f"argv not allowlisted: {argv}")

    resolved = shutil.which(exe)
    if not resolved:
        return 127, "", f"{exe}: not found in PATH"

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"{exe}: timed out after {timeout}s"
    except OSError as _:
        return 1, "", "os error"


def _parse_checkupdates_line(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # "pkg  1.0-1 -> 1.1-1" (one or more spaces)
    m = re.match(r"^(\S+)\s+(\S+)\s+->\s+(\S+)$", line)
    if not m:
        return None
    return {"package": m.group(1), "from_version": m.group(2), "to_version": m.group(3)}


def _parse_pending_from_output(stdout: str, source: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in stdout.splitlines():
        rec = _parse_checkupdates_line(line)
        if rec:
            rec["source"] = source
            out.append(rec)
    return out


def _reverse_depends(pkg: str) -> tuple[list[str], str | None]:
    """Return (local packages depending on pkg) via pactree -ru, or ([], error)."""
    code, out, err = _run_allowlisted(["pactree", "-ru", pkg])
    if code != 0:
        return [], (err or out or f"pactree exited {code}").strip() or None
    names: list[str] = []
    for line in out.splitlines()[:_PACTREE_LINE_CAP]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # pactree prints package names, possibly with provides
        tok = line.split()[0] if line.split() else ""
        if tok and _validate_pkg(tok) and tok not in names:
            names.append(tok)
    return names, None


def get_pending_updates(
    *,
    include_reverse_deps: bool = False,
    reverse_deps_limit: int = 5,
) -> dict[str, Any]:
    """
    List pending sync updates using checkupdates (preferred) or pacman -Qu.

    include_reverse_deps: run pactree -ru for the first N pending packages (can be slow).
    """
    errors: list[str] = []
    pending: list[dict[str, Any]] = []
    method = "none"

    if shutil.which("checkupdates"):
        method = "checkupdates"
        code, out, err = _run_allowlisted(["checkupdates"])
        # checkupdates: 0 = upgrades available, 1 = already up to date (both success)
        if code not in (0, 1):
            errors.append(f"checkupdates failed (exit {code}): {(err or out).strip()}")
        else:
            for rec in _parse_pending_from_output(out, "checkupdates"):
                pending.append(rec)
            if err.strip():
                errors.append(err.strip())
    elif shutil.which("pacman"):
        method = "pacman_-Qu"
        code, out, err = _run_allowlisted(["pacman", "-Qu"])
        combined = (err or "").strip() or (out or "").strip()
        if code == 0:
            for rec in _parse_pending_from_output(out, "pacman_-Qu"):
                pending.append(rec)
        elif code == 1 and not combined:
            # pacman -Qu: exit 1 with no stdout/stderr means nothing to upgrade
            # (see Arch forums e.g. bbs.archlinux.org/viewtopic.php?id=301276).
            pass
        elif code != 0:
            msg = combined or f"exit {code}"
            if "unable to lock" in msg.lower() or "permission denied" in msg.lower():
                errors.append(
                    "pacman -Qu failed (database lock or permissions). "
                    "Install pacman-contrib and use `checkupdates` as an unprivileged user, "
                    "or run sync checks only when pacman is idle."
                )
            else:
                errors.append(f"pacman -Qu: {msg}")
    else:
        errors.append("Neither checkupdates nor pacman found in PATH (not Arch/pacman?)")

    if include_reverse_deps and pending and shutil.which("pactree"):
        limit = max(0, min(reverse_deps_limit, 20))
        for rec in pending[:limit]:
            pkg = rec["package"]
            rdeps, rerr = _reverse_depends(pkg)
            rec["reverse_depends_local"] = rdeps
            if rerr:
                rec["reverse_depends_error"] = rerr
    elif include_reverse_deps and not shutil.which("pactree"):
        errors.append("pactree not in PATH; install pacman-contrib for reverse dependency hints")

    return {
        "method": method,
        "pending_count": len(pending),
        "pending": pending,
        "errors": errors,
    }
