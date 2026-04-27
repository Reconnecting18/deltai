"""Fastfetch-style terminal health panel for deltai."""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx

from delta import __version__
from delta.config import Settings

# Default catalog (core + computation + diagnostic + adapter; extensions add more at runtime).
_CORE_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "write_file",
    "list_directory",
    "run_shell",
    "get_system_info",
    "search_knowledge",
    "memory_stats",
    "calculate",
    "summarize_data",
    "lookup_reference",
    "self_diagnostics",
    "manage_models",
    "repair_subsystem",
    "resource_status",
    "manage_adapters",
)

_HTTP = "http://127.0.0.1"


def _uds_client(socket_path: str) -> httpx.Client:
    transport = httpx.HTTPTransport(uds=socket_path)
    return httpx.Client(transport=transport, base_url=_HTTP, timeout=5.0)


def _is_socket_path(path: str) -> bool:
    try:
        st = os.stat(path)
        return stat.S_ISSOCK(st.st_mode)
    except OSError:
        return False


def _probe_daemon(socket_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": None, "payload": None}
    try:
        with _uds_client(socket_path) as client:
            r = client.get("/health")
    except Exception as exc:
        out["error"] = str(exc)
        return out
    if r.status_code != 200:
        out["error"] = f"HTTP {r.status_code}"
        return out
    try:
        out["payload"] = r.json()
    except json.JSONDecodeError:
        out["error"] = "invalid JSON from /health"
        return out
    data = out["payload"] or {}
    if data.get("status") == "ok":
        out["ok"] = True
    else:
        out["error"] = f"unexpected: {data!r}"
    return out


def _probe_ollama(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"{base}/api/tags")
    except (OSError, httpx.HTTPError) as exc:
        return {"ok": False, "detail": str(exc)[:120]}
    if r.status_code != 200:
        return {"ok": False, "detail": f"HTTP {r.status_code}"}
    try:
        data = r.json()
        n = len(data.get("models", [])) if isinstance(data, dict) else 0
        return {"ok": True, "detail": f"{n} model(s) listed"}
    except json.JSONDecodeError:
        return {"ok": True, "detail": "reachable"}


def _probe_ipc(path: str) -> str:
    if not hasattr(socket, "AF_UNIX"):
        return "skipped (no AF_UNIX — use Linux for IPC socket checks)"
    if not _is_socket_path(path):
        if os.path.exists(path):
            return "path exists (not a socket)"
        return "no socket (daemon not listening here)"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(path)
    except OSError as exc:
        return f"connect failed: {exc}"
    return "accepting connections"


def _probe_project_stack(base: str) -> dict[str, Any] | None:
    url = f"{base.rstrip('/')}/api/health"
    try:
        with httpx.Client(timeout=2.5) as c:
            r = c.get(url)
    except (OSError, httpx.HTTPError):
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def _sqlite_info(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False, "size_bytes": 0}
    return {"path": str(p), "exists": True, "size_bytes": p.stat().st_size}


def _format_human_panel(
    settings: Settings,
    daemon_path: str,
    project_url: str,
    daemon: dict[str, Any],
    ollama: dict[str, Any],
    ipc: str,
    sql: dict[str, Any],
    project: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    w = 52
    top = "╔" + "═" * (w - 2) + "╗"
    mid = "║" + " deltai".ljust(w - 2) + "║"
    sub = f"║  v{__version__} — local AI layer".ljust(w - 1) + "║"
    bot = "╚" + "═" * (w - 2) + "╝"
    lines.extend(["", top, mid, sub, bot, ""])

    d_ok = daemon.get("ok")
    d_label = "OK" if d_ok else "!!"
    svc = (daemon.get("payload") or {}).get("service", "?")
    lines.append(f"  [{d_label:2}]  delta-daemon (UDS)  {svc}")
    lines.append(f"         {daemon_path}")
    if not d_ok and daemon.get("error"):
        lines.append(f"         ! {daemon['error'][: w + 20]}")

    o_ok = ollama.get("ok")
    o_label = "OK" if o_ok else "!!"
    od = ollama.get("detail", "")
    lines.append(f"  [{o_label:2}]  Ollama  {settings.ollama_url}")
    lines.append(f"         {od}")

    ipc_ok = "accepting" in ipc or "skipped" in ipc
    i_label = "OK" if ipc_ok else "!!"
    lines.append(f"  [{i_label:2}]  IPC  {settings.ipc_socket_path}")
    lines.append(f"         {ipc}")

    sz = sql.get("size_bytes", 0)
    ex = "ready" if sql.get("exists") else "not created yet"
    s_ok = sql.get("exists")
    s_label = "OK" if s_ok else ".."
    lines.append(f"  [{s_label:2}]  SQLite  {ex}  ({sz} bytes)")
    lines.append(f"         {sql.get('path', '')}")

    lines.append("  [..]  Core tool names (catalog; extensions add more at runtime):")
    wrapped = textwrap.fill(", ".join(_CORE_TOOL_NAMES), width=56)
    for part in wrapped.splitlines():
        lines.append("         " + part)

    if project is not None:
        lines.append(f"  [ok]  Project stack  {project_url}/api/health")
        for key in sorted(project.keys()):
            val = project[key]
            lines.append(f"         · {key}: {val}")
    else:
        lines.append(
            f"  [--]  Project dev app  (not reachable at {project_url} — start "
            f"uvicorn in project/ for full RAG/tools dashboard)"
        )

    lines.append("")
    lines.append("  data_dir   " + str(settings.data_dir))
    lines.append("  config_dir " + str(settings.config_dir))
    if settings.anthropic_api_key:
        lines.append("  cloud      ANTHROPIC_API_KEY is set (optional cloud path armed)")
    else:
        lines.append("  cloud      ANTHROPIC_API_KEY unset (local-first)")
    lines.append("")
    return "\n".join(lines)


def run_status(
    settings: Settings,
    daemon_socket: str,
    project_url: str,
    as_json: bool,
) -> int:
    """Print health panel or JSON. Returns process exit code (0 = looks alive)."""
    daemon = _probe_daemon(daemon_socket)
    ollama = _probe_ollama(settings.ollama_url)
    ipc = _probe_ipc(settings.ipc_socket_path)
    sql = _sqlite_info(str(settings.sqlite_path))
    project = _probe_project_stack(project_url)

    blob: dict[str, Any] = {
        "version": __version__,
        "daemon_socket": daemon_socket,
        "daemon": daemon,
        "ollama": ollama,
        "ipc": {"path": settings.ipc_socket_path, "summary": ipc},
        "sqlite": sql,
        "project_url": project_url,
        "project_subsystems": project,
        "core_tool_names": list(_CORE_TOOL_NAMES),
    }

    if as_json:
        print(json.dumps(blob, indent=2))
    else:
        out = _format_human_panel(
            settings,
            daemon_socket,
            project_url,
            daemon,
            ollama,
            ipc,
            sql,
            project,
        )
        print(out, file=sys.stdout)

    okish = bool(daemon.get("ok")) and bool(ollama.get("ok"))
    return 0 if okish else 1
