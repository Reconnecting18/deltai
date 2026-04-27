"""CLI interface entrypoint.

User-facing commands that talk to delta-daemon (HTTP over UDS) or raw IPC.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from typing import Any

import httpx

from delta import __version__
from delta.config import Settings, load_settings
from delta.core import set_plugin_enabled, upsert_plugin_enabled
from delta.interfaces.cli_reference import TOPIC_CHOICES, render_reference
from delta.interfaces.status_panel import run_status

_EXIT_OK = 0
_EXIT_ERR = 1
_EXIT_USAGE = 2

_HTTP_BASE = "http://127.0.0.1"

logger = logging.getLogger("delta.cli")


def _epilog() -> str:
    return """Environment:
  DELTA_DAEMON_SOCKET   Unix socket for HTTP API (default under $XDG_RUNTIME_DIR/deltai/).
  DELTA_IPC_SOCKET      Line-delimited JSON IPC socket (orchestrator).
  DELTA_DATA_DIR, DELTA_CONFIG_DIR, DELTA_CACHE_DIR, DELTA_SQLITE_PATH
  DELTAI_STATUS_PROJECT_URL   Optional: project dev app for /api/health (default http://127.0.0.1:8000)

Start the daemon (systemd user unit):
  systemctl --user enable --now delta-daemon

Legacy full-stack REPL (TCP :8000, /chat) remains: python project/cli.py

With no subcommand, `deltai` runs `status` (system health panel).
Plain status (pipes, scripts): NO_COLOR=1, TERM=dumb, or DELTAI_STATUS_MINIMAL=1.
Command lists: deltai reference [--topic TOPIC]
"""


def daemon_http_client(socket_path: str) -> httpx.Client:
    """HTTP client that speaks to uvicorn bound to a Unix domain socket."""
    transport = httpx.HTTPTransport(uds=socket_path)
    return httpx.Client(transport=transport, base_url=_HTTP_BASE, timeout=120.0)


def cmd_version() -> int:
    print(__version__)
    return _EXIT_OK


def cmd_paths(settings: Settings) -> int:
    print(f"data_dir={settings.data_dir}")
    print(f"config_dir={settings.config_dir}")
    print(f"cache_dir={settings.cache_dir}")
    print(f"sqlite_path={settings.sqlite_path}")
    print(f"reports_dir={settings.reports_dir}")
    print(f"daemon_socket_path={settings.daemon_socket_path}")
    print(f"ipc_socket_path={settings.ipc_socket_path}")
    print(f"ollama_url={settings.ollama_url}")
    return _EXIT_OK


def cmd_health(socket_path: str) -> int:
    try:
        with daemon_http_client(socket_path) as client:
            r = client.get("/health")
    except (OSError, httpx.HTTPError) as exc:
        print(f"health: cannot reach daemon at {socket_path}: {exc}", file=sys.stderr)
        return _EXIT_ERR
    if r.status_code != 200:
        print(f"health: HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return _EXIT_ERR
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("health: invalid JSON response", file=sys.stderr)
        return _EXIT_ERR
    if data.get("status") != "ok":
        print(f"health: unexpected payload: {data!r}", file=sys.stderr)
        return _EXIT_ERR
    print(json.dumps(data))
    return _EXIT_OK


def _resolve_query(query_parts: list[str]) -> str | None:
    if query_parts:
        return " ".join(query_parts).strip() or None
    if sys.stdin.isatty():
        return None
    return sys.stdin.read().strip() or None


def cmd_execute(
    socket_path: str,
    query_parts: list[str],
    session_id: str | None,
    as_json: bool,
) -> int:
    query = _resolve_query(query_parts)
    if not query:
        print("execute: provide QUERY words or pipe query on stdin", file=sys.stderr)
        return _EXIT_USAGE
    payload: dict[str, Any] = {"query": query, "source": "cli"}
    if session_id:
        payload["session_id"] = session_id
    try:
        with daemon_http_client(socket_path) as client:
            r = client.post("/v1/execute", json=payload)
    except (OSError, httpx.HTTPError) as exc:
        print(f"execute: cannot reach daemon at {socket_path}: {exc}", file=sys.stderr)
        return _EXIT_ERR
    if r.status_code != 200:
        print(f"execute: HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return _EXIT_ERR
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("execute: invalid JSON response", file=sys.stderr)
        return _EXIT_ERR
    if as_json:
        print(json.dumps(data))
    else:
        print(data.get("output", ""))
    if data.get("status") == "error":
        return _EXIT_ERR
    return _EXIT_OK


def cmd_ipc(
    ipc_socket_path: str,
    query_parts: list[str],
    session_id: str | None,
    as_json: bool,
) -> int:
    query = _resolve_query(query_parts)
    if not query:
        print("ipc: provide QUERY words or pipe query on stdin", file=sys.stderr)
        return _EXIT_USAGE
    payload: dict[str, Any] = {"query": query, "source": "cli-ipc"}
    if session_id:
        payload["session_id"] = session_id
    line = (json.dumps(payload) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(ipc_socket_path)
            sock.sendall(line)
            buf = bytearray()
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in buf:
                    break
    except OSError as exc:
        print(f"ipc: cannot connect to {ipc_socket_path}: {exc}", file=sys.stderr)
        return _EXIT_ERR
    raw = bytes(buf).split(b"\n", 1)[0].decode("utf-8", errors="replace")
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        print(f"ipc: invalid JSON response: {raw!r}", file=sys.stderr)
        return _EXIT_ERR
    if as_json:
        print(json.dumps(data))
    else:
        print(data.get("output", ""))
    if data.get("status") == "error":
        return _EXIT_ERR
    return _EXIT_OK


def cmd_plugin_install(settings: Settings, name: str, module_stem: str) -> int:
    """Enable a plugin row in extensions.toml if the module file exists."""
    plugin_dir = settings.data_dir / "plugins"
    path = plugin_dir / f"{module_stem}.py"
    if not path.is_file():
        print(
            f"plugin install: missing {path} — create the module first "
            f"(or pass --module STEM if the file uses another name).",
            file=sys.stderr,
        )
        return _EXIT_USAGE
    config_path = settings.config_dir / "extensions.toml"
    upsert_plugin_enabled(
        config_path,
        name=name,
        module_stem=module_stem,
        enabled=True,
        auto_start=True,
    )
    logger.info("Plugin %r enabled in %s", name, config_path)
    print(
        f"Plugin {name!r} enabled in {config_path}. "
        "Restart delta-daemon (systemctl --user restart delta-daemon) so on_init runs.",
    )
    return _EXIT_OK


def cmd_plugin_unload(settings: Settings, name: str) -> int:
    """Set enabled=false for a plugin; on_shutdown runs when the daemon stops."""
    config_path = settings.config_dir / "extensions.toml"
    if not set_plugin_enabled(config_path, name, enabled=False):
        print(
            f"plugin unload: no plugin named {name!r} in {config_path}.",
            file=sys.stderr,
        )
        return _EXIT_ERR
    logger.info("Plugin %r disabled in %s", name, config_path)
    print(
        f"Plugin {name!r} disabled in {config_path}. "
        "on_shutdown runs when delta-daemon stops; restart to unload it from memory.",
    )
    return _EXIT_OK


def cmd_reference(topic: str) -> int:
    print(render_reference(topic), end="")
    return _EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deltai",
        description="deltai — terminal access to the local delta-daemon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package version.")

    sub.add_parser("paths", help="Print resolved XDG and socket paths.")

    ref = sub.add_parser(
        "reference",
        help="Print terminal command reference (systemd, curl, arch-update-guard, REPL).",
    )
    ref.add_argument(
        "--topic",
        dest="reference_topic",
        choices=list(TOPIC_CHOICES),
        default="all",
        metavar="TOPIC",
        help="Section: %(choices)s (default: all).",
    )

    hp = sub.add_parser("health", help="GET /health on the daemon HTTP socket.")
    hp.add_argument(
        "--socket",
        dest="daemon_socket",
        metavar="PATH",
        help="Override DELTA_DAEMON_SOCKET (HTTP over UDS).",
    )

    st = sub.add_parser(
        "status",
        help="Fastfetch-style health: daemon, Ollama, IPC, SQLite, optional project /api/health.",
    )
    st.add_argument(
        "--socket",
        dest="daemon_socket",
        metavar="PATH",
        help="Override DELTA_DAEMON_SOCKET (HTTP over UDS).",
    )
    st.add_argument(
        "--project-url",
        dest="project_url",
        metavar="URL",
        default=None,
        help="Base URL for project dev app (default: env DELTAI_STATUS_PROJECT_URL or http://127.0.0.1:8000).",
    )
    st.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object (scripting / automation).",
    )

    ep = sub.add_parser("execute", help="POST /v1/execute with a query.")
    ep.add_argument(
        "--socket",
        dest="daemon_socket",
        metavar="PATH",
        help="Override DELTA_DAEMON_SOCKET (HTTP over UDS).",
    )
    ep.add_argument(
        "--session",
        dest="session_id",
        metavar="ID",
        help="Optional session id passed to the orchestrator.",
    )
    ep.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON response instead of output text only.",
    )
    ep.add_argument(
        "query_parts",
        nargs="*",
        help="Query text (words). If omitted, read stdin (non-TTY).",
    )

    ip = sub.add_parser(
        "ipc",
        help="Send one line-delimited JSON request on DELTA_IPC_SOCKET.",
    )
    ip.add_argument(
        "--ipc-socket",
        dest="ipc_socket",
        metavar="PATH",
        help="Override DELTA_IPC_SOCKET.",
    )
    ip.add_argument(
        "--session",
        dest="session_id",
        metavar="ID",
        help="Optional session id passed to the orchestrator.",
    )
    ip.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON response instead of output text only.",
    )
    ip.add_argument(
        "query_parts",
        nargs="*",
        help="Query text (words). If omitted, read stdin (non-TTY).",
    )

    plug = sub.add_parser(
        "plugin",
        help="Manage optional daemon plugins (extensions.toml + ~/.local/share/deltai/plugins/).",
    )
    plug_sub = plug.add_subparsers(dest="plugin_cmd", required=True)

    pin = plug_sub.add_parser(
        "install",
        help="Enable a plugin (requires plugins/<module>.py on disk).",
    )
    pin.add_argument("name", help="Registry name for this plugin.")
    pin.add_argument(
        "--module",
        dest="module_stem",
        metavar="STEM",
        default=None,
        help="File stem under plugins/ if different from name (default: same as name).",
    )

    pun = plug_sub.add_parser(
        "unload",
        help="Disable a plugin in config (daemon runs on_shutdown on stop).",
    )
    pun.add_argument("name", help="Registry name to disable.")

    return p


def run(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["status"]
    settings = load_settings()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return cmd_version()
    if args.command == "paths":
        return cmd_paths(settings)
    if args.command == "reference":
        return cmd_reference(args.reference_topic)
    if args.command == "health":
        path = args.daemon_socket or settings.daemon_socket_path
        return cmd_health(path)
    if args.command == "status":
        path = args.daemon_socket or settings.daemon_socket_path
        base = args.project_url or os.getenv(
            "DELTAI_STATUS_PROJECT_URL",
            "http://127.0.0.1:8000",
        )
        return run_status(settings, path, base, args.json)
    if args.command == "execute":
        path = args.daemon_socket or settings.daemon_socket_path
        return cmd_execute(
            path,
            list(args.query_parts),
            args.session_id,
            args.json,
        )
    if args.command == "ipc":
        ipc_path = args.ipc_socket or settings.ipc_socket_path
        return cmd_ipc(
            ipc_path,
            list(args.query_parts),
            args.session_id,
            args.json,
        )
    if args.command == "plugin":
        if args.plugin_cmd == "install":
            stem = args.module_stem if args.module_stem is not None else args.name
            return cmd_plugin_install(settings, args.name, stem)
        if args.plugin_cmd == "unload":
            return cmd_plugin_unload(settings, args.name)
    raise RuntimeError(f"unhandled command: {args.command!r}")


def main() -> None:
    """Console script entrypoint (setuptools)."""
    sys.exit(run())


if __name__ == "__main__":
    main()
