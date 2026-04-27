"""Plain-text terminal reference for `deltai reference` (copy-paste friendly)."""

from __future__ import annotations

# Topics exposed on the CLI (default: all).
TOPIC_CHOICES = (
    "all",
    "daemon",
    "deltai",
    "arch-cli",
    "arch-http",
    "repl",
    "tools",
)

_TWO_RUNTIMES = """\
================================================================================
Two runtimes (do not mix them)
================================================================================
  delta-daemon     Local daemon: HTTP over a Unix domain socket. Use:
                   deltai(1), delta(1), delta-daemon(1). Not the same as :8000.

  project FastAPI  Dev / full stack: uvicorn on TCP (default 127.0.0.1:8000).
                   Rich REPL: python cli.py from project/. curl(1) examples
                   below use this base only.
"""

_DAEMON = """\
================================================================================
delta-daemon — systemd user service (Linux)
================================================================================
  Install unit (once):
    cp systemd/user/delta-daemon.service ~/.config/systemd/user/
    systemctl --user daemon-reload

  systemctl --user enable delta-daemon
  systemctl --user start delta-daemon
  systemctl --user stop delta-daemon
  systemctl --user restart delta-daemon
  systemctl --user status delta-daemon

  journalctl --user -u delta-daemon -f

  Env (see also: deltai paths): DELTA_DAEMON_SOCKET, DELTA_IPC_SOCKET,
  DELTA_DATA_DIR, DELTA_CONFIG_DIR, DELTA_CACHE_DIR, DELTA_SQLITE_PATH
"""

_DELTAI = """\
================================================================================
deltai / delta — CLI (talks to delta-daemon over UDS)
================================================================================
  deltai --help

  deltai version
      Print package version.

  deltai paths
      Print resolved XDG paths, SQLite, daemon and IPC socket paths.

  deltai health [--socket PATH]
      GET /health on the daemon HTTP socket.

  deltai execute [--socket PATH] [--session ID] [--json] [WORDS ...]
      POST /v1/execute. With no words and stdin is not a TTY, query is read
      from stdin.
      Example:  deltai execute what is 2+2
      Example:  echo 'status' | deltai execute

  deltai ipc [--ipc-socket PATH] [--session ID] [--json] [WORDS ...]
      One line-delimited JSON request on DELTA_IPC_SOCKET (or override).

  deltai plugin install NAME [--module STEM]
      Enable a plugin in extensions.toml (module under plugins/).

  deltai plugin unload NAME
      Disable a plugin in extensions.toml.

  deltai reference [--topic TOPIC]
      This reference. Topics: all, daemon, deltai, arch-cli, arch-http,
      repl, tools.
"""

_ARCH_CLI = """\
================================================================================
arch-update-guard — host shell (needs project/ tree on disk)
================================================================================
  The console script resolves project/ via DELTAI_PROJECT_ROOT or
  DELTA_PROJECT_ROOT (parent of project/), cwd walk-up, or next to the
  installed package. Then it runs the same logic as project/core/.

  arch-update-guard --help

  --mode {auto,manual}       Persist scheduler mode in SQLite
  --interval SEC             Auto-check interval (min 60 seconds)
  --check                    Run pending update check
  --snapshot                 With --check, create a snapshot
  --label TEXT               Snapshot label
  --list-snapshots           List recent snapshots (JSON)
  --show-snapshot ID         Print snapshot metadata JSON
  --diff FROM TO             Compare two snapshot IDs
  --rollback ID              Plan (dry-run) or execute rollback (see below)
  --dry-run                  Rollback plan only (default)
  --no-dry-run               Execute rollback staging
  --apply-etc                With rollback execute, write /etc (root only)

  Examples:
    arch-update-guard --check
    arch-update-guard --check --snapshot --label pre-update
    arch-update-guard --list-snapshots
    arch-update-guard --diff SNAP_A SNAP_B
    arch-update-guard --rollback SNAP_ID
    arch-update-guard --rollback SNAP_ID --no-dry-run
"""

_ARCH_HTTP = """\
================================================================================
Arch update guard — HTTP (project FastAPI on TCP)
================================================================================
  Same routes are mounted at:
    /arch-guard/...           (preferred)
    /ext/arch_update_guard/... (legacy alias)

  Set base (default dev port):
    BASE=http://127.0.0.1:8000
    AG=$BASE/arch-guard

  Optional: pipe JSON through jq(1) for readability.

  curl -fsS "$AG/health"

  curl -fsS "$AG/settings"
  curl -fsS -X PUT "$AG/settings" -H 'Content-Type: application/json' \\
    -d '{"mode":"manual","enabled":true,"auto_interval_sec":3600}'

  curl -fsS "$AG/pending"
  curl -fsS "$AG/pending?include_reverse_deps=true&reverse_deps_limit=5"

  curl -fsS -X POST "$AG/check" -H 'Content-Type: application/json' \\
    -d '{"create_snapshot":false,"label":"","include_reverse_deps":false}'

  curl -fsS -X POST "$AG/snapshots" -H 'Content-Type: application/json' \\
    -d '{"kind":"manual","label":"","include_reverse_deps":false,"backup_sqlite":true}'

  curl -fsS "$AG/snapshots?limit=50&offset=0"
  curl -fsS "$AG/snapshots/SNAPSHOT_ID"

  curl -fsS -X POST "$AG/snapshots/compare" -H 'Content-Type: application/json' \\
    -d '{"from_snapshot_id":"FROM_ID","to_snapshot_id":"TO_ID"}'

  curl -fsS "$AG/diffs/FROM_ID/TO_ID"

  curl -fsS -X POST "$AG/rollback" -H 'Content-Type: application/json' \\
    -d '{"snapshot_id":"ID","dry_run":true,"apply_etc":false}'

  curl -fsS "$AG/rollback/jobs?limit=20"

  curl -fsS -X POST "$AG/refresh-news" -H 'Content-Type: application/json' \\
    -d '{"wiki_query":"","force":false}'
"""

_REPL = """\
================================================================================
project/cli.py — Rich REPL slash commands (TCP backend)
================================================================================
  From directory project/ with backend up, e.g.:
    uvicorn main:app --host 127.0.0.1 --port 8000
    python cli.py [--host HOST] [--port PORT] [--no-banner]

  /help                 Show this command list
  /health               Subsystem health status
  /stats                System stats (CPU, RAM, GPU, VRAM)
  /budget               Cloud budget: spent / limit / remaining
  /resources            VRAM pressure, circuit breaker, recovery log
  /memory               Knowledge base stats (chunks, files, disk)
  /history              Conversation history metadata
  /clear                Clear conversation history
  /backup               Emergency backup system status
  /heal                 Self-heal loop status
  /events [n]           Recent health events (default: last 10)
  /cls                  Clear terminal screen
  /quit, /exit          Exit the terminal

  Any other line        Sent as chat to POST /chat (streaming)
"""

_TOOLS = """\
================================================================================
Arch update guard — LLM tool names (orchestrator / POST /chat)
================================================================================
  These are not separate shell binaries; the model invokes them when relevant.
  From the host you can approximate with natural language via deltai execute:

    arch_pending_updates_report
    arch_refresh_news_digest
    arch_create_update_snapshot
    arch_compare_snapshots
    arch_rollback_plan

  Example:  deltai execute 'Give me the arch pending updates report'
"""

_SECTION_ORDER = (
    "two_runtimes",
    "daemon",
    "deltai",
    "arch_cli",
    "arch_http",
    "repl",
    "tools",
)

_SECTIONS: dict[str, str] = {
    "two_runtimes": _TWO_RUNTIMES,
    "daemon": _DAEMON,
    "deltai": _DELTAI,
    "arch_cli": _ARCH_CLI,
    "arch_http": _ARCH_HTTP,
    "repl": _REPL,
    "tools": _TOOLS,
}

_TOPIC_TO_KEYS: dict[str, tuple[str, ...]] = {
    "all": _SECTION_ORDER,
    "daemon": ("two_runtimes", "daemon"),
    "deltai": ("two_runtimes", "deltai"),
    "arch-cli": ("two_runtimes", "arch_cli"),
    "arch-http": ("two_runtimes", "arch_http"),
    "repl": ("two_runtimes", "repl"),
    "tools": ("two_runtimes", "tools"),
}


def render_reference(topic: str) -> str:
    """Return full reference text for topic name."""
    keys = _TOPIC_TO_KEYS.get(topic)
    if keys is None:
        raise ValueError(f"unknown topic {topic!r}")
    parts = [_SECTIONS[k] for k in keys]
    header = """\
================================================================================
deltai terminal reference
================================================================================
  deltai reference [--topic TOPIC]

  Topics:  all | daemon | deltai | arch-cli | arch-http | repl | tools

"""
    if topic != "all":
        header = ""
    return header + "\n\n".join(parts) + "\n"
