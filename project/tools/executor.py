"""
E3N Tool Executor — runs tools safely and returns results.
Each tool function returns a string result that gets fed back to the model.
"""

import os
import subprocess
import json
import psutil
import platform

# ── SAFETY ──────────────────────────────────────────────────────────────
PROTECTED_PATHS = [
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
]

BLOCKED_COMMANDS = [
    "format-volume", "format ", "remove-item -recurse c:\\",
    "stop-computer", "restart-computer",
    "del c:\\windows", "rd c:\\windows",
    "reg delete hklm",
    # PowerShell code execution / bypass
    "invoke-expression", "iex ", "iex(", "iex(",
    "-encodedcommand", "-enc ",
    "start-process", "invoke-webrequest", "invoke-restmethod",
    "downloadstring", "downloadfile",
    "new-object net.webclient", "bitstransfer",
    # cmd escape
    "cmd /c", "cmd.exe", "cmd /k",
    # User/group manipulation
    "net user", "net localgroup",
    # Registry via PowerShell
    "set-itemproperty hklm", "new-itemproperty hklm",
    # Script execution bypass
    "-executionpolicy bypass", "-ep bypass", "-exec bypass",
]

def _is_path_safe_write(path: str) -> bool:
    normalized = os.path.normpath(path).lower()
    for protected in PROTECTED_PATHS:
        if normalized.startswith(protected.lower()):
            return False
    return True

def _is_command_safe(cmd: str) -> bool:
    lower = cmd.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in lower:
            return False
    return True


# ── TYPE COERCION ───────────────────────────────────────────────────────

def _coerce_int(val, default=None):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def _coerce_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower().strip() in ("true", "1", "yes")
    return bool(val)


# ── TOOL IMPLEMENTATIONS ────────────────────────────────────────────────

def read_file(path: str, max_lines=200) -> str:
    try:
        max_lines = _coerce_int(max_lines, 200)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            return f"ERROR: File not found: {path}"
        if not os.path.isfile(path):
            return f"ERROR: Not a file: {path}"
        size = os.path.getsize(path)
        if size > 2_000_000:
            return f"ERROR: File too large ({size:,} bytes). Max 2MB."
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        truncated = lines[:max_lines]
        result = "".join(truncated)
        if total > max_lines:
            result += f"\n\n[TRUNCATED — showing {max_lines}/{total} lines]"
        return result
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str, append=False) -> str:
    try:
        append = _coerce_bool(append, False)
        path = os.path.normpath(path)
        if not _is_path_safe_write(path):
            return f"ERROR: Write blocked — protected path: {path}"
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Wrote"
        return f"OK: {action} {path} ({len(content)} chars)"
    except Exception as e:
        return f"ERROR: {e}"


def list_directory(path: str, recursive=False) -> str:
    try:
        recursive = _coerce_bool(recursive, False)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            return f"ERROR: Directory not found: {path}"
        if not os.path.isdir(path):
            return f"ERROR: Not a directory: {path}"
        entries = []
        if recursive:
            for root, dirs, files in os.walk(path):
                depth = root.replace(path, "").count(os.sep)
                if depth >= 3:
                    dirs.clear()
                    continue
                dirs[:] = [d for d in dirs if d not in (
                    "node_modules", ".git", "__pycache__", "venv",
                    ".venv", "dist", "build", ".next"
                )]
                indent = "  " * depth
                rel = os.path.relpath(root, path)
                if rel != ".":
                    entries.append(f"{indent}[DIR]  {rel}/")
                for f in sorted(files)[:50]:
                    fpath = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(fpath)
                        entries.append(f"{indent}  {f}  ({_fmt_size(sz)})")
                    except Exception:
                        entries.append(f"{indent}  {f}")
        else:
            items = sorted(os.listdir(path))
            for item in items[:100]:
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    entries.append(f"[DIR]  {item}/")
                else:
                    try:
                        sz = os.path.getsize(full)
                        entries.append(f"       {item}  ({_fmt_size(sz)})")
                    except Exception:
                        entries.append(f"       {item}")
        if not entries:
            return f"{path} is empty."
        return f"{path}\n" + "\n".join(entries)
    except Exception as e:
        return f"ERROR: {e}"


def _fmt_size(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1_048_576: return f"{b/1024:.1f} KB"
    return f"{b/1_048_576:.1f} MB"


def run_powershell(command: str, timeout=15) -> str:
    try:
        timeout = _coerce_int(timeout, 15)
        timeout = min(timeout, 30)
        if not _is_command_safe(command):
            return f"ERROR: Command blocked for safety: {command}"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="C:\\e3n"
        )
        output = ""
        if result.stdout.strip():
            output += result.stdout.strip()
        if result.stderr.strip():
            if output:
                output += "\n"
            output += f"[STDERR] {result.stderr.strip()}"
        if result.returncode != 0 and not output:
            output = f"Command exited with code {result.returncode}"
        if not output:
            output = "OK (no output)"
        if len(output) > 4000:
            output = output[:4000] + "\n\n[TRUNCATED — output exceeded 4000 chars]"
        return output
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def get_system_info(include_processes=False) -> str:
    try:
        include_processes = _coerce_bool(include_processes, False)
        info = []
        cpu_pct = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        info.append(f"CPU: {cpu_pct}% @ {round(freq.current) if freq else '?'} MHz")
        info.append(f"     {psutil.cpu_count(logical=False)}P/{psutil.cpu_count()} threads")
        ram = psutil.virtual_memory()
        info.append(f"RAM: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)")
        disk = psutil.disk_usage("C:\\")
        info.append(f"Disk C: {disk.used/1e9:.0f}/{disk.total/1e9:.0f} GB ({disk.percent}%)")
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): name = name.decode()
            info.append(f"GPU: {name}")
            info.append(f"     Utilization: {util.gpu}%")
            info.append(f"     VRAM: {mem.used/1e6:.0f}/{mem.total/1e6:.0f} MB")
            info.append(f"     Temperature: {temp}°C")
            info.append(f"     Power: {power:.0f}W")
        except Exception:
            info.append("GPU: unavailable")
        if include_processes:
            info.append("\nTop processes by memory:")
            procs = []
            for p in psutil.process_iter(["name", "memory_info", "cpu_percent"]):
                try:
                    mem_mb = p.info["memory_info"].rss / 1e6
                    procs.append((p.info["name"], mem_mb, p.info["cpu_percent"] or 0))
                except Exception:
                    pass
            procs.sort(key=lambda x: x[1], reverse=True)
            for name, mem_mb, cpu_pct in procs[:10]:
                info.append(f"  {name:<30} {mem_mb:>7.0f} MB  CPU {cpu_pct:.0f}%")
        return "\n".join(info)
    except Exception as e:
        return f"ERROR: {e}"


# ── KNOWLEDGE / RAG TOOLS ──────────────────────────────────────────────

def search_knowledge(query: str, n_results=5) -> str:
    """Search the ChromaDB knowledge base."""
    try:
        from memory import query_knowledge
        n_results = _coerce_int(n_results, 5)
        matches = query_knowledge(query, n_results=n_results)
        if not matches:
            return "No relevant results found in the knowledge base."
        lines = [f"Found {len(matches)} relevant chunk(s):\n"]
        for i, m in enumerate(matches, 1):
            lines.append(f"--- Result {i} (source: {m['source']}, distance: {m['distance']}) ---")
            lines.append(m["text"])
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "ERROR: Memory system not available (chromadb not installed)"
    except Exception as e:
        return f"ERROR: {e}"


def memory_stats() -> str:
    """Get knowledge base statistics."""
    try:
        from memory import get_memory_stats
        stats = get_memory_stats()
        lines = [
            f"Knowledge Base Stats:",
            f"  Total chunks: {stats['total_chunks']}",
            f"  Files ingested: {stats['total_files']}",
            f"  Disk usage: {stats['disk_mb']} MB",
        ]
        if stats["sources"]:
            lines.append(f"  Sources:")
            for s in stats["sources"]:
                lines.append(f"    - {s}")
        else:
            lines.append("  No files ingested yet.")
            lines.append(f"  Drop files in C:\\e3n\\data\\knowledge\\ to ingest.")
        return "\n".join(lines)
    except ImportError:
        return "ERROR: Memory system not available (chromadb not installed)"
    except Exception as e:
        return f"ERROR: {e}"


# ── TELEMETRY TOOLS (conditional) ─────────────────────────────────────

_TELEMETRY_API_URL = os.getenv("TELEMETRY_API_URL", "").strip()

def _telemetry_get(endpoint: str, params: dict = None) -> str:
    """Helper: GET from telemetry API with error handling."""
    if not _TELEMETRY_API_URL:
        return "ERROR: Telemetry API not configured (TELEMETRY_API_URL not set)"
    import httpx as _httpx
    try:
        with _httpx.Client(timeout=10) as client:
            resp = client.get(f"{_TELEMETRY_API_URL}{endpoint}", params=params)
            if resp.status_code != 200:
                return f"ERROR: Telemetry API returned HTTP {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            return json.dumps(data, indent=2)
    except Exception as e:
        return f"ERROR: Cannot reach telemetry API at {_TELEMETRY_API_URL}: {e}"


def get_session_status(**kwargs) -> str:
    """Get current racing session status."""
    return _telemetry_get("/api/session")


def get_lap_summary(lap_number=None, **kwargs) -> str:
    """Get lap time summary."""
    params = {}
    if lap_number is not None:
        params["lap"] = _coerce_int(lap_number)
    return _telemetry_get("/api/laps", params=params if params else None)


def get_tire_status(**kwargs) -> str:
    """Get current tire status."""
    return _telemetry_get("/api/tires")


def get_strategy_recommendation(remaining_laps=None, **kwargs) -> str:
    """Get pit strategy recommendation."""
    params = {}
    if remaining_laps is not None:
        params["remaining_laps"] = _coerce_int(remaining_laps)
    return _telemetry_get("/api/strategy", params=params if params else None)


# ── DISPATCH ────────────────────────────────────────────────────────────

EXECUTORS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_powershell": run_powershell,
    "get_system_info": get_system_info,
    "search_knowledge": search_knowledge,
    "memory_stats": memory_stats,
}

if _TELEMETRY_API_URL:
    EXECUTORS["get_session_status"] = get_session_status
    EXECUTORS["get_lap_summary"] = get_lap_summary
    EXECUTORS["get_tire_status"] = get_tire_status
    EXECUTORS["get_strategy_recommendation"] = get_strategy_recommendation

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns result string."""
    fn = EXECUTORS.get(name)
    if not fn:
        return f"ERROR: Unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"ERROR: Bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: Tool {name} failed: {e}"
