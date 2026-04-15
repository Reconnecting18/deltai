"""
deltai Terminal — standalone CLI client.
Connects to the deltai FastAPI backend via HTTP. Zero backend imports.

Usage:
    python cli.py [--host HOST] [--port PORT] [--no-banner]

Or directly:
    python cli.py
"""

import asyncio
import json
import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import httpx
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# ── CONFIG ─────────────────────────────────────────────────────────────

VERSION = "1.0.0"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
CONNECT_TIMEOUT = 3.0
STREAM_TIMEOUT = 120.0

# ── THEME ──────────────────────────────────────────────────────────────
# Matches deltai dashboard palette: militaristic, muted, tactical

DELTAI_THEME = Theme(
    {
        "deltai": "#5a8a7a bold",
        "deltai.bright": "bold white",
        "accent": "#5a8a7a",
        "cmd": "#5a8a7a",
        "meta": "#445060 italic",
        "meta.tool": "#5a8a7a italic",
        "warn": "#c49a3a",
        "danger": "#a03030 bold",
        "ok": "#5a8a7a",
        "dim": "#445060",
        "text": "#b8c4cc",
        "heading": "#5a8a7a bold",
    }
)

console = Console(theme=DELTAI_THEME, highlight=False, force_terminal=True)

# ── ASCII BANNER ───────────────────────────────────────────────────────

_TITLE = "MODULAR AI EXTENSION"
_SUB = "Local AI System // v{ver}"

_B = "\u2588"  # full block (letter body)
_LH = "\u258c"  # left half (right-edge shadow)
_UH = "\u2580"  # upper half (bottom shadow)
_QUL = "\u2598"  # quadrant upper-left (corner shadow)

# 5-row letters + shadow row. Claude Code 3D: ▌ right edge, ▀ bottom, ▘ corner
# N: 2-wide diagonal, perfectly uniform
# E=9 wide (8+shadow), 3=9 wide (8+shadow), N=11 wide (10+shadow)
_GAP = "  "
_E_ART = (
    f"{_B * 8}{_LH}",
    f"{_B * 3}{_LH}{_UH * 4}{_QUL}",
    f"{_B * 6}{_LH}  ",
    f"{_B * 3}{_LH}{_UH * 2}{_QUL}  ",
    f"{_B * 8}{_LH}",
    f"{_UH * 8}{_QUL}",
)
_3_ART = (
    f" {_B * 7}{_LH}",
    f" {_UH * 4}{_B * 3}{_LH}",
    f"  {_B * 6}{_LH}",
    f"  {_UH * 3}{_B * 3}{_LH}",
    f" {_B * 7}{_LH}",
    f" {_UH * 7}{_QUL}",
)
_N_ART = (
    f"{_B * 3}{_LH}   {_B * 3}{_LH}",
    f"{_B * 5}{_LH} {_B * 3}{_LH}",
    f"{_B * 3}{_UH}{_B * 2}{_LH}{_B * 3}{_LH}",
    f"{_B * 3}{_LH}{_UH}{_B * 5}{_LH}",
    f"{_B * 3}{_LH}   {_B * 3}{_LH}",
    f"{_UH * 3}{_QUL}   {_UH * 3}{_QUL}",
)
_DELTAI_LINES = tuple(f"{_E_ART[i]}{_GAP}{_3_ART[i]}{_GAP}{_N_ART[i]}" for i in range(6))


def _style_art(art: str, is_shadow: bool = False) -> str:
    """Color letter body chars as accent, shadow chars (▌▀▘) as dim."""
    shadow_chars = {_LH, _UH, _QUL}
    out = []
    cur_style = None
    for ch in art:
        if ch in shadow_chars:
            want = "dim"
        elif ch == _B:
            want = "accent"
        else:
            want = None
        if want != cur_style:
            if cur_style:
                out.append("[/]")
            if want:
                out.append(f"[{want}]")
            cur_style = want
        out.append(ch)
    if cur_style:
        out.append("[/]")
    return "".join(out)


def _build_banner() -> str:
    w = 72  # inner width
    lines = []
    border = f"[accent]  +{'=' * w}+[/]"
    empty = f"[accent]  |[/]{' ' * w}[accent]|[/]"
    lines.append("")
    lines.append(border)
    lines.append(empty)
    for i, art_line in enumerate(_DELTAI_LINES):
        right = ""
        if i == 2:
            right = _TITLE
        elif i == 3:
            right = _SUB.replace("{ver}", VERSION)
        left = f"  {_style_art(art_line)}"
        if right:
            mid = f"  [text]{right}[/]" if i == 2 else f"  [dim]{right}[/]"
        else:
            mid = ""
        raw_len = 2 + len(art_line) + (2 + len(right) if right else 0)
        pad = w - raw_len
        row = f"[accent]  |[/]{left}{mid}{' ' * max(pad, 1)}[accent]|[/]"
        lines.append(row)
    lines.append(empty)
    lines.append(border)
    lines.append("")
    return "\n".join(lines)


BANNER = _build_banner()


def print_banner():
    console.print(BANNER)


# ── HTTP HELPERS ───────────────────────────────────────────────────────


def _url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


async def api_get(client: httpx.AsyncClient, path: str) -> dict | None:
    """GET a JSON endpoint. Returns parsed dict or None on error."""
    try:
        resp = await client.get(path, timeout=CONNECT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def api_post(client: httpx.AsyncClient, path: str, data: dict = None) -> dict | None:
    """POST to a JSON endpoint. Returns parsed dict or None on error."""
    try:
        resp = await client.post(path, json=data or {}, timeout=CONNECT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


async def api_delete(client: httpx.AsyncClient, path: str) -> bool:
    """DELETE an endpoint. Returns True on success."""
    try:
        resp = await client.delete(path, timeout=CONNECT_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


# ── CONNECTION CHECK ───────────────────────────────────────────────────


async def check_connection(client: httpx.AsyncClient) -> dict | None:
    """Check backend connection. Returns health dict or None."""
    for attempt in range(3):
        health = await api_get(client, "/api/health")
        if health is not None:
            return health
        if attempt < 2:
            console.print(f"  [warn]Connection attempt {attempt + 1}/3 failed. Retrying...[/]")
            await asyncio.sleep(2)
    return None


def _status_style(status: str) -> str:
    """Map subsystem status to a Rich style."""
    s = status.lower()
    if s in ("online", "active", "ready", "local"):
        return "ok"
    elif s in ("standby", "disabled", "unavailable"):
        return "dim"
    elif s in ("degraded", "stopped"):
        return "warn"
    else:
        return "danger"


async def show_startup_status(client: httpx.AsyncClient, health: dict):
    """Display compact startup status from health and stats."""

    # Subsystem status line
    online = sum(1 for v in health.values() if v.lower() in ("online", "active", "ready", "local"))
    total = len(health)

    status_parts = []
    for name, status in health.items():
        style = _status_style(status)
        status_parts.append(f"[{style}]{name.upper()}[/]")

    color = "ok" if online == total else "warn" if online >= total - 2 else "danger"
    console.print(f"  [{color}]SYSTEMS  {online}/{total} ONLINE[/]  {' '.join(status_parts)}")

    # Stats snapshot
    stats = await api_get(client, "/stats")
    if stats:
        gpu = stats.get("gpu", {})
        vram_used = gpu.get("vram_used_mb", "?")
        vram_total = gpu.get("vram_total_mb", "?")
        temp = gpu.get("temp_c", "?")

        models = stats.get("models", [])
        model_str = models[0] if models else "none"

        budget = stats.get("budget", {})
        spent = budget.get("daily_spent", 0)
        limit = budget.get("daily_budget", 5)

        console.print(
            f"  [dim]MODEL[/] [text]{model_str}[/]  "
            f"[dim]VRAM[/] [text]{vram_used}/{vram_total} MB[/]  "
            f"[dim]GPU[/] [text]{temp}C[/]  "
            f"[dim]BUDGET[/] [text]${spent:.2f}/${limit:.2f}[/]"
        )

    console.print()


# ── CHAT STREAMING ─────────────────────────────────────────────────────


async def stream_chat(client: httpx.AsyncClient, message: str):
    """Send a chat message and stream the NDJSON response."""
    full_text = ""
    response_started = False

    try:
        async with client.stream(
            "POST",
            "/chat",
            json={"message": message},
            timeout=httpx.Timeout(STREAM_TIMEOUT, connect=CONNECT_TIMEOUT),
        ) as resp:
            if resp.status_code != 200:
                console.print(f"  [danger]ERROR: Backend returned HTTP {resp.status_code}[/]")
                return

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = ev.get("t", "")

                if t == "route":
                    model = ev.get("model", "?")
                    tier = ev.get("tier", "?")
                    split = " [SPLIT]" if ev.get("split") else ""
                    cat = f" ({ev['query_category']})" if ev.get("query_category") else ""
                    console.print(f"  [meta]routed -> {model} (tier {tier}){split}{cat}[/]")

                elif t == "rag":
                    n = ev.get("n", 0)
                    if n > 0:
                        console.print(
                            f"  [meta]{n} memory chunk{'s' if n != 1 else ''} injected[/]"
                        )

                elif t == "split_phase":
                    phase = ev.get("phase", "?")
                    msg = ev.get("c", "")
                    console.print(f"  [warn]SPLIT PHASE {phase}: {msg}[/]")

                elif t == "tool":
                    name = ev.get("n", "?")
                    args = ev.get("a", {})
                    args_str = ""
                    if args:
                        # Show first arg value, truncated
                        first_val = str(list(args.values())[0])[:60]
                        args_str = f" {first_val}"
                        if len(str(list(args.values())[0])) > 60:
                            args_str += "..."
                    console.print(f"  [meta.tool]>> tool: {name}{args_str}[/]")

                elif t == "result":
                    name = ev.get("n", "?")
                    summary = ev.get("s", "")[:80]
                    console.print(f"  [meta.tool]<< {name}: {summary}[/]")

                elif t == "retry":
                    name = ev.get("n", "?")
                    msg = ev.get("c", "")
                    console.print(f"  [warn]RETRY: {name} -- {msg}[/]")

                elif t == "text":
                    chunk = ev.get("c", "")
                    if chunk:
                        if not response_started:
                            # Print deltai prompt prefix before first text
                            console.print()
                            console.print("  [deltai]deltai>[/] ", end="")
                            response_started = True
                        # Print chunk live (raw, no markup interpretation)
                        console.out(chunk, end="", highlight=False)
                        full_text += chunk

                elif t == "done":
                    turns = ev.get("turns")
                    if response_started:
                        console.print()  # newline after streamed text
                        # Re-render as markdown if it contains formatting
                        if any(c in full_text for c in ["#", "```", "**", "- ", "1."]):
                            console.print()
                            md = Markdown(full_text)
                            console.print(
                                Panel(
                                    md,
                                    border_style="accent",
                                    padding=(0, 1),
                                    expand=False,
                                )
                            )
                    if turns is not None:
                        console.print(f"  [dim][{turns}T][/]")
                    console.print()

                elif t == "error":
                    msg = ev.get("c", "Unknown error")
                    console.print(f"\n  [danger]ERROR: {msg}[/]\n")

                elif t == "emergency":
                    msg = ev.get("c", "")
                    console.print(
                        Panel(
                            f"EMERGENCY: {msg}",
                            border_style="danger",
                            title="[danger]BACKUP ACTIVATED[/]",
                        )
                    )

                elif t == "session":
                    active = ev.get("active", False)
                    sid = ev.get("session_id", "")
                    state = "ACTIVE" if active else "ENDED"
                    console.print(f"  [warn]SESSION {state}: {sid}[/]")

    except httpx.ConnectError:
        console.print("\n  [danger]ERROR: Cannot reach deltai backend. Is it running?[/]\n")
    except httpx.ReadTimeout:
        console.print("\n  [warn]Response timed out after {STREAM_TIMEOUT}s[/]\n")
    except KeyboardInterrupt:
        if response_started:
            console.print("\n  [dim](cancelled)[/]\n")
    except Exception as e:
        console.print(f"\n  [danger]ERROR: {type(e).__name__}[/]\n")


# ── SLASH COMMANDS ─────────────────────────────────────────────────────


async def cmd_help(client, args):
    """Show available commands."""
    table = Table(
        title="deltai TERMINAL COMMANDS",
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        title_style="heading",
        show_header=True,
        header_style="deltai",
    )
    table.add_column("Command", style="cmd", min_width=18)
    table.add_column("Description", style="text")

    cmds = [
        ("/help", "Show this command list"),
        ("/health", "Subsystem health status"),
        ("/stats", "System stats (CPU, RAM, GPU, VRAM)"),
        ("/budget", "Cloud budget: spent / limit / remaining"),
        ("/resources", "VRAM pressure, circuit breaker, recovery log"),
        ("/memory", "Knowledge base stats (chunks, files, disk)"),
        ("/history", "Conversation history metadata"),
        ("/clear", "Clear conversation history"),
        ("/backup", "Emergency backup system status"),
        ("/heal", "Self-heal loop status"),
        ("/events [n]", "Recent health events (default: last 10)"),
        ("/cls", "Clear terminal screen"),
        ("/quit, /exit", "Exit the terminal"),
        ("", ""),
        ("[dim](any text)[/]", "[dim]Chat with deltai directly[/]"),
    ]
    for cmd, desc in cmds:
        table.add_row(cmd, desc)

    console.print()
    console.print(table)
    console.print()


async def cmd_health(client, args):
    """Show subsystem health status."""
    health = await api_get(client, "/api/health")
    if health is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    table = Table(
        title="SUBSYSTEM HEALTH",
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        title_style="heading",
    )
    table.add_column("Subsystem", style="text", min_width=14)
    table.add_column("Status", min_width=12)

    for name, status in health.items():
        style = _status_style(status)
        table.add_row(name.upper(), f"[{style}]{status.upper()}[/]")

    online = sum(1 for v in health.values() if v.lower() in ("online", "active", "ready", "local"))
    total = len(health)

    console.print()
    console.print(table)
    color = "ok" if online == total else "warn" if online >= total - 2 else "danger"
    console.print(f"  [{color}]{online}/{total} systems operational[/]")
    console.print()


async def cmd_stats(client, args):
    """Show full system statistics."""
    stats = await api_get(client, "/stats")
    if stats is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    table = Table(
        title="SYSTEM STATUS",
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        title_style="heading",
    )
    table.add_column("Component", style="deltai", min_width=14)
    table.add_column("Value", style="text", min_width=30)

    cpu = stats.get("cpu", {})
    table.add_row(
        "CPU",
        f"{cpu.get('percent', '?')}% @ {cpu.get('freq_mhz', '?')} MHz "
        f"({cpu.get('cores_physical', '?')}P/{cpu.get('cores_logical', '?')}T)",
    )

    ram = stats.get("ram", {})
    table.add_row(
        "RAM",
        f"{ram.get('used_gb', '?')}/{ram.get('total_gb', '?')} GB ({ram.get('percent', '?')}%)",
    )

    disk = stats.get("disk", {})
    table.add_row(
        "DISK",
        f"{disk.get('used_gb', '?')}/{disk.get('total_gb', '?')} GB ({disk.get('percent', '?')}%)",
    )

    gpu = stats.get("gpu", {})
    if "error" not in gpu:
        vram_free = gpu.get("vram_total_mb", 0) - gpu.get("vram_used_mb", 0)
        tier = "A" if vram_free > 9000 else "B" if vram_free > 3000 else "C"
        table.add_row("GPU", f"{gpu.get('name', '?')}")
        table.add_row(
            "VRAM",
            f"{gpu.get('vram_used_mb', '?')}/{gpu.get('vram_total_mb', '?')} MB "
            f"({vram_free:,} free, Tier {tier})",
        )
        table.add_row(
            "GPU UTIL",
            f"{gpu.get('gpu_percent', '?')}%  "
            f"{gpu.get('temp_c', '?')}C  "
            f"{gpu.get('power_w', '?')}/{gpu.get('power_limit_w', '?')}W",
        )
    else:
        table.add_row("GPU", f"[dim]{gpu.get('error', 'unavailable')}[/]")

    models = stats.get("models", [])
    table.add_row("MODELS", ", ".join(models) if models else "[dim]none loaded[/]")

    budget = stats.get("budget", {})
    if budget:
        spent = budget.get("daily_spent", 0)
        limit = budget.get("daily_budget", 5)
        remaining = budget.get("daily_remaining", limit)
        color = "ok" if remaining > limit * 0.3 else "warn" if remaining > limit * 0.1 else "danger"
        table.add_row(
            "BUDGET", f"[{color}]${spent:.2f} / ${limit:.2f}  (${remaining:.2f} remaining)[/]"
        )

    console.print()
    console.print(table)
    console.print()


async def cmd_budget(client, args):
    """Show cloud budget status."""
    data = await api_get(client, "/budget/status")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    spent = data.get("daily_spent", 0)
    limit = data.get("daily_budget", 5)
    remaining = data.get("daily_remaining", limit)
    ok = data.get("budget_ok", True)

    color = (
        "ok" if ok and remaining > limit * 0.3 else "warn" if remaining > limit * 0.1 else "danger"
    )

    # Budget bar
    bar_width = 30
    filled = int((spent / limit) * bar_width) if limit > 0 else 0
    filled = min(filled, bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    console.print()
    console.print("  [heading]CLOUD BUDGET[/]")
    console.print(f"  [{color}]{bar}[/]  [{color}]${spent:.2f} / ${limit:.2f}[/]")
    console.print(
        f"  [dim]Remaining: ${remaining:.2f}  |  Status: {'OK' if ok else 'EXHAUSTED'}[/]"
    )
    console.print()


async def cmd_resources(client, args):
    """Show resource manager status."""
    data = await api_get(client, "/resources/status")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    table = Table(
        title="RESOURCE STATUS",
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        title_style="heading",
    )
    table.add_column("Component", style="deltai", min_width=18)
    table.add_column("Value", style="text")

    rm = data.get("resource_manager", {})
    cb = data.get("circuit_breaker", {})

    table.add_row("VRAM Warnings", str(rm.get("vram_warnings", 0)))
    table.add_row("Ollama Failures", str(rm.get("ollama_failures", 0)))
    table.add_row("Watcher Restarts", str(rm.get("watcher_restarts", 0)))

    cb_state = cb.get("state", "?")
    cb_color = "ok" if cb_state == "closed" else "warn" if cb_state == "half-open" else "danger"
    table.add_row("Circuit Breaker", f"[{cb_color}]{cb_state.upper()}[/]")
    table.add_row("Backoff", f"{cb.get('backoff_sec', '?')}s")

    actions = rm.get("recent_actions", [])
    if actions:
        last_actions = actions[-3:]
        for a in last_actions:
            table.add_row("[dim]Action[/]", f"[dim]{a.get('action', '?')}[/]")

    console.print()
    console.print(table)
    console.print()


async def cmd_memory(client, args):
    """Show knowledge base stats."""
    data = await api_get(client, "/memory/stats")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    console.print()
    console.print("  [heading]KNOWLEDGE BASE[/]")
    console.print(f"  [dim]Chunks:[/]  [text]{data.get('total_chunks', '?')}[/]")
    console.print(f"  [dim]Files:[/]   [text]{data.get('total_files', '?')}[/]")
    console.print(f"  [dim]Disk:[/]    [text]{data.get('disk_mb', '?')} MB[/]")

    sources = data.get("sources", [])
    if sources:
        console.print("  [dim]Sources:[/]")
        for s in sources[:15]:
            console.print(f"    [dim]-[/] [text]{s}[/]")
    console.print()


async def cmd_history(client, args):
    """Show conversation history metadata."""
    data = await api_get(client, "/chat/history")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    turns = data.get("turns", 0)
    max_turns = data.get("max_turns", 10)
    history = data.get("history", [])

    console.print()
    console.print("  [heading]CONVERSATION HISTORY[/]")
    console.print(f"  [dim]Turns:[/] [text]{turns}/{max_turns}[/]")

    if history:
        console.print()
        for pair in history[-3:]:
            user_msg = pair.get("user", "")[:60]
            deltai_msg = pair.get("assistant", "")[:60]
            console.print(
                f"  [dim]>[/]    [text]{user_msg}{'...' if len(pair.get('user', '')) > 60 else ''}[/]"
            )
            console.print(
                f"  [deltai]deltai>[/] [text]{deltai_msg}{'...' if len(pair.get('assistant', '')) > 60 else ''}[/]"
            )
            console.print()
    console.print()


async def cmd_clear(client, args):
    """Clear conversation history."""
    ok = await api_delete(client, "/chat/history")
    if ok:
        console.print("  [ok]History cleared.[/]")
    else:
        console.print("  [danger]Failed to clear history.[/]")


async def cmd_backup(client, args):
    """Show backup system status."""
    data = await api_get(client, "/backup/status")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    console.print()
    console.print("  [heading]BACKUP SYSTEM[/]")

    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                status = val.get("status", val.get("healthy", "?"))
                style = "ok" if str(status).lower() in ("healthy", "true", "ok") else "warn"
                console.print(f"  [dim]{key}:[/] [{style}]{status}[/]")
            else:
                console.print(f"  [dim]{key}:[/] [text]{val}[/]")
    console.print()


async def cmd_heal(client, args):
    """Show self-heal loop status."""
    data = await api_get(client, "/self-heal/status")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    console.print()
    console.print("  [heading]SELF-HEAL STATUS[/]")
    console.print(f"  [dim]Enabled:[/]  [text]{data.get('enabled', '?')}[/]")
    console.print(f"  [dim]Interval:[/] [text]{data.get('interval_sec', '?')}s[/]")
    console.print(f"  [dim]Model:[/]    [text]{data.get('model', '?')}[/]")
    console.print(f"  [dim]Idle:[/]     [text]{data.get('idle', '?')}[/]")

    actions = data.get("recent_actions", [])
    if actions:
        console.print("  [dim]Recent:[/]")
        for a in actions[-5:]:
            console.print(f"    [dim]-[/] [text]{a}[/]")
    console.print()


async def cmd_events(client, args):
    """Show recent health events."""
    limit = 10
    if args:
        try:
            limit = int(args[0])
        except (ValueError, IndexError):
            pass

    data = await api_get(client, f"/health/events?limit={limit}")
    if data is None:
        console.print("  [danger]Cannot reach backend.[/]")
        return

    events = data if isinstance(data, list) else data.get("events", [])

    if not events:
        console.print("  [dim]No recent health events.[/]")
        return

    table = Table(
        title=f"HEALTH EVENTS (last {limit})",
        box=box.SIMPLE_HEAVY,
        border_style="accent",
        title_style="heading",
    )
    table.add_column("Type", style="deltai", min_width=20)
    table.add_column("Data", style="text", max_width=50)
    table.add_column("Time", style="dim", min_width=10)

    for ev in events[-limit:]:
        ev_type = ev.get("type", "?")
        ev_data = json.dumps(ev.get("data", {}))[:50]
        ev_time = ev.get("timestamp", "?")
        if isinstance(ev_time, (int, float)):
            ev_time = time.strftime("%H:%M:%S", time.localtime(ev_time))
        table.add_row(ev_type, ev_data, str(ev_time))

    console.print()
    console.print(table)
    console.print()


# ── COMMAND DISPATCH ───────────────────────────────────────────────────

COMMANDS = {
    "/help": cmd_help,
    "/health": cmd_health,
    "/stats": cmd_stats,
    "/budget": cmd_budget,
    "/resources": cmd_resources,
    "/memory": cmd_memory,
    "/history": cmd_history,
    "/clear": cmd_clear,
    "/backup": cmd_backup,
    "/heal": cmd_heal,
    "/events": cmd_events,
}


async def dispatch_command(client: httpx.AsyncClient, raw: str) -> bool:
    """
    Dispatch a slash command. Returns True if handled, False if not a command.
    """
    parts = raw.strip().split(None, 2)
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    if cmd in ("/quit", "/exit"):
        console.print("  [dim]deltai terminal closed.[/]")
        return "exit"

    if cmd == "/cls":
        console.clear()
        return True

    handler = COMMANDS.get(cmd)
    if handler:
        await handler(client, args)
        return True

    # Check for partial match
    matches = [c for c in COMMANDS if c.startswith(cmd)]
    if len(matches) == 1:
        await COMMANDS[matches[0]](client, args)
        return True
    elif len(matches) > 1:
        console.print(f"  [warn]Ambiguous command. Did you mean: {', '.join(matches)}?[/]")
        return True

    return False


# ── MAIN REPL ──────────────────────────────────────────────────────────


async def repl(client: httpx.AsyncClient):
    """Main read-eval-print loop."""
    ctrl_c_count = 0

    while True:
        try:
            # Prompt
            try:
                raw = console.input("[deltai]deltai >[/] ")
            except EOFError:
                break

            ctrl_c_count = 0  # reset on successful input
            raw = raw.strip()

            if not raw:
                continue

            # Slash command?
            if raw.startswith("/"):
                result = await dispatch_command(client, raw)
                if result == "exit":
                    break
                if result:
                    continue
                # Unknown command — treat as chat
                console.print("  [dim]Unknown command. Type /help for available commands.[/]")
                continue

            # Chat message
            await stream_chat(client, raw)

        except KeyboardInterrupt:
            ctrl_c_count += 1
            console.print()
            if ctrl_c_count >= 2:
                console.print("  [dim]deltai terminal closed.[/]")
                break
            console.print("  [dim]Press Ctrl+C again to exit, or type /quit[/]")


# ── ENTRY POINT ────────────────────────────────────────────────────────


async def main():
    # Parse simple CLI args
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    show_banner = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--no-banner":
            show_banner = False
            i += 1
        else:
            i += 1

    base_url = f"http://{host}:{port}"

    # Banner
    if show_banner:
        console.print()
        print_banner()

    # Connect
    console.print(f"  [dim]Connecting to deltai backend at {base_url}...[/]")

    async with httpx.AsyncClient(base_url=base_url) as client:
        health = await check_connection(client)

        if health is None:
            console.print()
            console.print(
                Panel(
                    f"Cannot reach deltai backend at {base_url}\n\n"
                    "Start the backend first:\n"
                    "  cd ~/deltai/project\n"
                    "  .\\venv\\Scripts\\activate\n"
                    "  uvicorn main:app --port 8000",
                    border_style="danger",
                    title="[danger]CONNECTION FAILED[/]",
                )
            )
            return

        # Startup status
        await show_startup_status(client, health)

        console.print("  [dim]Type /help for commands. Chat by typing a message.[/]")
        console.print()

        # REPL
        await repl(client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
