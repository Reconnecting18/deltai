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


# ── WEB SEARCH TOOL ──────────────────────────────────────────────────

def web_search(query: str, max_results=5) -> str:
    """Search the web via DuckDuckGo Lite. Blocked during active racing sessions."""
    try:
        # Session guard: block during active racing (performance protection)
        try:
            import sys
            if "." not in sys.path:
                sys.path.insert(0, ".")
            from router import _session_active
            if _session_active:
                return "ERROR: Web search disabled during active racing sessions (performance protection)"
        except (ImportError, AttributeError):
            pass

        max_results = min(_coerce_int(max_results, 5), 10)
        if not query or not query.strip():
            return "ERROR: Empty search query"

        import httpx as _httpx
        import re as _re

        # DuckDuckGo HTML Lite — simple table-based results, no JS required, no API key
        resp = _httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query.strip(), "kl": "us-en"},
            headers={"User-Agent": "E3N/1.0 (local AI assistant)"},
            timeout=10.0,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return f"ERROR: Search returned HTTP {resp.status_code}"

        html = resp.text

        # Parse DuckDuckGo Lite results
        # Structure: <td> cells in order: number, title (with <a> link), spacer, snippet, spacer, url
        results = []

        # Extract all <a> links with DuckDuckGo redirect URLs (contain uddg= param)
        link_pattern = _re.compile(
            r'<a[^>]*href="([^"]*uddg=[^"]+)"[^>]*>(.+?)</a>',
            _re.DOTALL
        )
        links = link_pattern.findall(html)

        # Extract all <td> content for snippets
        td_pattern = _re.compile(r'<td[^>]*>(.*?)</td>', _re.DOTALL)
        tds = td_pattern.findall(html)

        # Build snippet map: find td cells that contain substantial text (snippets)
        snippets = []
        for td in tds:
            clean = _re.sub(r'<[^>]+>', '', td).strip()
            clean = _re.sub(r'&\w+;', ' ', clean).strip()
            # Snippets are longer text blocks (not numbers, not URLs, not spacers)
            if len(clean) > 40 and not clean.startswith('http') and not clean.startswith('www.'):
                snippets.append(_re.sub(r'\s+', ' ', clean))

        from urllib.parse import unquote, urlparse, parse_qs
        for i, (raw_url, raw_title) in enumerate(links[:max_results]):
            clean_title = _re.sub(r'<[^>]+>', '', raw_title).strip()
            clean_title = _re.sub(r'&#x27;', "'", clean_title)
            clean_title = _re.sub(r'&amp;', '&', clean_title)
            # Extract actual URL from DuckDuckGo redirect
            try:
                parsed = parse_qs(urlparse(raw_url).query)
                actual_url = unquote(parsed.get('uddg', [raw_url])[0])
            except Exception:
                actual_url = raw_url
            snippet = snippets[i] if i < len(snippets) else ""
            if clean_title:
                results.append(f"[{i+1}] {clean_title}\n    {actual_url}\n    {snippet}")

        if not results:
            return f"No results found for: {query}"

        header = f"Web search results for: {query}\n{'=' * 40}\n"
        return header + "\n\n".join(results)

    except Exception as e:
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            return "ERROR: Search timed out (10s limit). Try a simpler query."
        return f"ERROR: Web search failed: {e}"


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


# ── SELF-DIAGNOSTIC TOOLS ──────────────────────────────────────────────

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_E3N_MODELS = {"e3n-qwen14b", "e3n-qwen3b", "e3n-nemo", "e3n"}
_CRITICAL_PATHS = [
    r"C:\e3n\data\knowledge",
    r"C:\e3n\data\chromadb",
    r"C:\e3n\project\.env",
    r"C:\e3n\modelfiles",
]


def _ollama_get(endpoint: str, timeout: int = 5):
    """Helper: GET from Ollama API. Returns parsed JSON or None."""
    import httpx as _hx
    try:
        resp = _hx.get(f"{_OLLAMA_URL}{endpoint}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _ollama_post(endpoint: str, payload: dict, timeout: int = 15):
    """Helper: POST to Ollama API. Returns parsed JSON or None."""
    import httpx as _hx
    try:
        resp = _hx.post(f"{_OLLAMA_URL}{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def self_diagnostics(subsystem=None) -> str:
    """Run E3N self-diagnostics across all subsystems or deep-check one."""
    try:
        if subsystem:
            return _diag_deep(subsystem.lower().strip())
        return _diag_full()
    except Exception as e:
        return f"ERROR: Diagnostics failed: {e}"


def _diag_full() -> str:
    """Full sweep across all subsystems."""
    lines = ["E3N SELF-DIAGNOSTICS", "=" * 40]

    # Ollama
    tags = _ollama_get("/api/tags")
    ps = _ollama_get("/api/ps")
    if tags is None:
        lines.append("Ollama:    DOWN — cannot reach Ollama API")
    else:
        all_models = [m["name"].split(":")[0] for m in tags.get("models", [])]
        e3n_found = [m for m in all_models if m in _E3N_MODELS]
        loaded = [m.get("name", "?") for m in (ps or {}).get("models", [])]
        loaded_str = ", ".join(loaded) if loaded else "none"
        lines.append(f"Ollama:    ONLINE ({len(e3n_found)}/4 E3N models, loaded: {loaded_str})")
        missing = _E3N_MODELS - set(e3n_found)
        if missing:
            lines.append(f"           MISSING: {', '.join(sorted(missing))}")

    # ChromaDB
    try:
        from memory import get_memory_stats
        stats = get_memory_stats()
        lines.append(f"ChromaDB:  ONLINE ({stats['total_chunks']} chunks, {stats['total_files']} files, {stats['disk_mb']} MB)")
    except Exception as e:
        lines.append(f"ChromaDB:  ERROR — {e}")

    # GPU / VRAM
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        used = round(mem.used / 1e6)
        total = round(mem.total / 1e6)
        free = total - used
        tier = "A" if free > 9000 else "B" if free > 3000 else "C"
        lines.append(f"GPU:       {name} — {used:,}/{total:,} MB ({free:,} free, Tier {tier}, {util.gpu}% util, {temp}C)")
        if temp > 83:
            lines.append("           WARNING: GPU temperature high")
    except Exception:
        lines.append("GPU:       UNAVAILABLE")

    # Voice
    try:
        from voice import get_voice_status, VOICE_ENABLED
        if VOICE_ENABLED:
            vs = get_voice_status()
            stt = vs.get("stt", {}).get("status", "?")
            tts = vs.get("tts", {}).get("status", "?")
            lines.append(f"Voice:     ONLINE (STT: {stt}, TTS: {tts})")
        else:
            lines.append("Voice:     DISABLED")
    except ImportError:
        lines.append("Voice:     NOT INSTALLED")

    # Watcher
    try:
        from watcher import watcher_running
        lines.append(f"Watcher:   {'ACTIVE' if watcher_running() else 'STOPPED'}")
    except ImportError:
        lines.append("Watcher:   NOT AVAILABLE")

    # Critical paths
    missing_paths = [p for p in _CRITICAL_PATHS if not os.path.exists(p)]
    if missing_paths:
        lines.append(f"Paths:     MISSING: {', '.join(missing_paths)}")
    else:
        lines.append("Paths:     ALL OK")

    # Backup models
    backup_ok = []
    backup_fail = []
    for model in ["e3n-nemo", "e3n"]:
        if tags:
            all_names = [m["name"].split(":")[0] for m in tags.get("models", [])]
            if model in all_names:
                backup_ok.append(model)
            else:
                backup_fail.append(model)
    if backup_fail:
        lines.append(f"Backup:    DEGRADED (missing: {', '.join(backup_fail)})")
    elif backup_ok:
        lines.append(f"Backup:    HEALTHY ({', '.join(backup_ok)} available)")
    else:
        lines.append("Backup:    UNKNOWN (Ollama unreachable)")

    # Issues summary
    issues = [l for l in lines if "DOWN" in l or "MISSING" in l or "ERROR" in l or "WARNING" in l or "STOPPED" in l or "DEGRADED" in l]
    lines.append("")
    if issues:
        lines.append(f"Issues found: {len(issues)}")
        lines.append("Run self_diagnostics with subsystem name for fix suggestions.")
    else:
        lines.append("No issues detected.")

    return "\n".join(lines)


def _diag_deep(subsystem: str) -> str:
    """Deep diagnostic for a specific subsystem with fix suggestions."""
    if subsystem == "ollama":
        lines = ["OLLAMA DEEP DIAGNOSTICS", "=" * 40]
        tags = _ollama_get("/api/tags")
        if tags is None:
            lines.append("STATUS: DOWN — cannot reach Ollama at " + _OLLAMA_URL)
            lines.append("")
            lines.append("FIX: Run `ollama serve` in a terminal, or use repair_subsystem('check_ollama')")
            return "\n".join(lines)
        all_models = {m["name"].split(":")[0]: m for m in tags.get("models", [])}
        ps = _ollama_get("/api/ps")
        loaded = ps.get("models", []) if ps else []
        lines.append("Available models:")
        for name, info in sorted(all_models.items()):
            size_gb = round(info.get("size", 0) / 1e9, 1)
            is_e3n = "  [E3N]" if name in _E3N_MODELS else ""
            lines.append(f"  {name:<20} {size_gb} GB{is_e3n}")
        lines.append(f"\nLoaded in VRAM ({len(loaded)}):")
        if loaded:
            for m in loaded:
                vram_mb = round(m.get("size_vram", m.get("size", 0)) / 1e6)
                lines.append(f"  {m.get('name', '?'):<20} ~{vram_mb} MB")
        else:
            lines.append("  (none)")
        missing = _E3N_MODELS - set(all_models.keys())
        if missing:
            lines.append(f"\nMISSING E3N MODELS: {', '.join(sorted(missing))}")
            for m in sorted(missing):
                mf = f"C:\\e3n\\modelfiles\\E3N-{m.replace('e3n-', '')}.modelfile"
                if m == "e3n":
                    mf = "C:\\e3n\\modelfiles\\E3N.modelfile"
                lines.append(f"  FIX: ollama create {m} -f {mf}")
        return "\n".join(lines)

    elif subsystem == "chromadb":
        lines = ["CHROMADB DEEP DIAGNOSTICS", "=" * 40]
        try:
            from memory import get_memory_stats, get_file_details, CHROMADB_PATH
            stats = get_memory_stats()
            lines.append(f"Path:    {CHROMADB_PATH}")
            lines.append(f"Chunks:  {stats['total_chunks']}")
            lines.append(f"Files:   {stats['total_files']}")
            lines.append(f"Disk:    {stats['disk_mb']} MB")
            if stats['total_chunks'] == 0:
                lines.append("\nWARNING: Empty knowledge base")
                lines.append("FIX: Drop files in C:\\e3n\\data\\knowledge\\ or use repair_subsystem('reindex_knowledge')")
            files = get_file_details()
            if files:
                lines.append(f"\nIngested files ({len(files)}):")
                for f in files[:20]:
                    lines.append(f"  {f['source']:<40} {f['chunks']} chunks")
        except Exception as e:
            lines.append(f"ERROR: {e}")
            lines.append("FIX: Check ChromaDB installation, verify C:\\e3n\\data\\chromadb exists")
        return "\n".join(lines)

    elif subsystem == "gpu":
        lines = ["GPU DEEP DIAGNOSTICS", "=" * 40]
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            used = round(mem.used / 1e6)
            total = round(mem.total / 1e6)
            free = total - used
            tier = "A (14B fits)" if free > 9000 else "B (3B on GPU)" if free > 3000 else "C (CPU only)"
            lines.append(f"Device:  {name}")
            lines.append(f"VRAM:    {used:,} / {total:,} MB ({free:,} MB free)")
            lines.append(f"Tier:    {tier}")
            lines.append(f"Util:    {util.gpu}%")
            lines.append(f"Temp:    {temp}C")
            lines.append(f"Power:   {power:.0f}W")
            if free < 3000:
                lines.append("\nWARNING: Low VRAM — running in CPU-only mode (Tier C)")
                lines.append("FIX: Use manage_ollama_models('unload', model='...') or repair_subsystem('clear_vram')")
            if temp > 83:
                lines.append("\nWARNING: High GPU temperature")
                lines.append("FIX: Check cooling, reduce load, or unload models")
        except Exception as e:
            lines.append(f"GPU unavailable: {e}")
        return "\n".join(lines)

    elif subsystem == "voice":
        lines = ["VOICE DEEP DIAGNOSTICS", "=" * 40]
        try:
            from voice import get_voice_status, VOICE_ENABLED
            vs = get_voice_status()
            lines.append(f"Enabled: {VOICE_ENABLED}")
            lines.append(f"STT:     {vs.get('stt', {})}")
            lines.append(f"TTS:     {vs.get('tts', {})}")
            if not VOICE_ENABLED:
                lines.append("\nFIX: Set VOICE_ENABLED=true in .env")
        except ImportError:
            lines.append("Voice module not installed")
            lines.append("FIX: pip install faster-whisper edge-tts")
        return "\n".join(lines)

    elif subsystem == "watcher":
        lines = ["WATCHER DEEP DIAGNOSTICS", "=" * 40]
        try:
            from watcher import watcher_running
            from memory import KNOWLEDGE_PATH
            running = watcher_running()
            lines.append(f"Status:  {'ACTIVE' if running else 'STOPPED'}")
            lines.append(f"Path:    {KNOWLEDGE_PATH}")
            lines.append(f"Exists:  {os.path.exists(KNOWLEDGE_PATH)}")
            if os.path.exists(KNOWLEDGE_PATH):
                files = [f for f in os.listdir(KNOWLEDGE_PATH) if not f.startswith(".")]
                lines.append(f"Files:   {len(files)}")
            if not running:
                lines.append("\nFIX: Use repair_subsystem('restart_watcher')")
        except ImportError as e:
            lines.append(f"Watcher not available: {e}")
        return "\n".join(lines)

    elif subsystem == "backup":
        lines = ["BACKUP SYSTEM DIAGNOSTICS", "=" * 40]
        tags = _ollama_get("/api/tags")
        chains = [
            ("e3n-qwen14b", "e3n-nemo", "e3n"),
            ("e3n-qwen3b", "e3n"),
        ]
        if tags:
            available = {m["name"].split(":")[0] for m in tags.get("models", [])}
            for chain in chains:
                primary = chain[0]
                backups = chain[1:]
                chain_str = " -> ".join(chain)
                statuses = ["OK" if m in available else "MISSING" for m in chain]
                lines.append(f"Chain: {chain_str}")
                for m, s in zip(chain, statuses):
                    lines.append(f"  {m:<16} {s}")
        else:
            lines.append("Cannot check — Ollama unreachable")
        return "\n".join(lines)

    elif subsystem == "paths":
        lines = ["CRITICAL PATH DIAGNOSTICS", "=" * 40]
        for p in _CRITICAL_PATHS:
            exists = os.path.exists(p)
            is_dir = os.path.isdir(p) if exists else False
            if exists and is_dir:
                count = len(os.listdir(p))
                lines.append(f"  OK    {p}  ({count} items)")
            elif exists:
                size = os.path.getsize(p)
                lines.append(f"  OK    {p}  ({size} bytes)")
            else:
                lines.append(f"  MISS  {p}")
                lines.append(f"        FIX: mkdir {p}" if "." not in os.path.basename(p) else f"        FIX: Check installation")
        return "\n".join(lines)

    else:
        return f"ERROR: Unknown subsystem '{subsystem}'. Options: ollama, chromadb, gpu, voice, watcher, backup, paths"


def manage_ollama_models(action: str, model: str = None) -> str:
    """Manage Ollama models in VRAM. Actions: status, unload, preload."""
    action = (action or "").lower().strip()

    if action == "status":
        tags = _ollama_get("/api/tags")
        ps = _ollama_get("/api/ps")
        if tags is None:
            return "ERROR: Cannot reach Ollama — is it running?"
        lines = ["OLLAMA MODEL STATUS", "=" * 40]
        loaded = (ps or {}).get("models", [])
        lines.append(f"Loaded in VRAM ({len(loaded)}):")
        if loaded:
            for m in loaded:
                vram_mb = round(m.get("size_vram", m.get("size", 0)) / 1e6)
                lines.append(f"  {m.get('name', '?'):<24} ~{vram_mb} MB")
        else:
            lines.append("  (none)")
        all_models = tags.get("models", [])
        loaded_names = {m.get("name", "") for m in loaded}
        avail = [m for m in all_models if m["name"] not in loaded_names]
        if avail:
            lines.append(f"\nAvailable (not loaded):")
            for m in avail:
                size_gb = round(m.get("size", 0) / 1e9, 1)
                e3n_tag = " [E3N]" if m["name"].split(":")[0] in _E3N_MODELS else ""
                lines.append(f"  {m['name']:<24} {size_gb} GB{e3n_tag}")
        # VRAM summary
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = round(mem.used / 1e6)
            total = round(mem.total / 1e6)
            free = total - used
            tier = "A" if free > 9000 else "B" if free > 3000 else "C"
            lines.append(f"\nVRAM: {used:,}/{total:,} MB ({free:,} free) — Tier {tier}")
        except Exception:
            pass
        return "\n".join(lines)

    elif action == "unload":
        if not model:
            return "ERROR: 'model' parameter required for unload"
        model_base = model.split(":")[0]
        if model_base not in _E3N_MODELS:
            return f"ERROR: Can only manage E3N models. '{model}' not in allowlist: {', '.join(sorted(_E3N_MODELS))}"
        import httpx as _hx
        try:
            resp = _hx.post(f"{_OLLAMA_URL}/api/generate",
                            json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
            if resp.status_code == 200:
                return f"OK: Unloaded {model} from VRAM. Use manage_ollama_models('status') to verify."
            return f"ERROR: Ollama returned HTTP {resp.status_code}"
        except Exception as e:
            return f"ERROR: Failed to unload {model}: {e}"

    elif action == "preload":
        if not model:
            return "ERROR: 'model' parameter required for preload"
        model_base = model.split(":")[0]
        if model_base not in _E3N_MODELS:
            return f"ERROR: Can only manage E3N models. '{model}' not in allowlist: {', '.join(sorted(_E3N_MODELS))}"
        # Sim guard: block preloading 14B while sim is running
        if "14b" in model.lower():
            try:
                from router import is_sim_running
                if is_sim_running():
                    return "BLOCKED: Cannot preload 14B model while sim is running — VRAM needed for game"
            except ImportError:
                pass
        # Check VRAM availability
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = round((mem.total - mem.used) / 1e6)
            model_sizes = {"e3n-qwen14b": 8500, "e3n-qwen3b": 2500, "e3n-nemo": 7200, "e3n": 4700}
            needed = model_sizes.get(model_base, 5000)
            if free_mb < needed + 500:
                return f"WARNING: Only {free_mb} MB VRAM free, {model} needs ~{needed} MB. Unload another model first."
        except Exception:
            pass
        import httpx as _hx
        try:
            resp = _hx.post(f"{_OLLAMA_URL}/api/generate",
                            json={"model": model, "prompt": "ping", "keep_alive": "5m",
                                  "options": {"num_predict": 1}}, timeout=60)
            if resp.status_code == 200:
                return f"OK: Preloaded {model} into VRAM (5min keep-alive)."
            return f"ERROR: Ollama returned HTTP {resp.status_code}"
        except Exception as e:
            return f"ERROR: Failed to preload {model}: {e}"

    else:
        return f"ERROR: Unknown action '{action}'. Use 'status', 'unload', or 'preload'."


def repair_subsystem(repair: str) -> str:
    """Attempt a safe, allowlisted repair on an E3N subsystem."""
    repair = (repair or "").lower().strip()

    if repair == "restart_watcher":
        try:
            from watcher import stop_watcher, start_watcher, watcher_running
            stop_watcher()
            start_watcher()
            running = watcher_running()
            return f"OK: Watcher {'restarted successfully' if running else 'restart attempted but not confirmed running'}"
        except Exception as e:
            return f"ERROR: Failed to restart watcher: {e}"

    elif repair == "clear_vram":
        try:
            import httpx as _hx
            ps = _ollama_get("/api/ps")
            if ps is None:
                return "ERROR: Cannot reach Ollama — is it running?"
            loaded = ps.get("models", [])
            if not loaded:
                return "OK: No models loaded in VRAM — nothing to clear."
            unloaded = []
            for m in loaded:
                model_name = m.get("name", "")
                if model_name:
                    try:
                        _hx.post(f"{_OLLAMA_URL}/api/generate",
                                 json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=10)
                        unloaded.append(model_name)
                    except Exception:
                        pass
            return f"OK: Unloaded {len(unloaded)} model(s) from VRAM: {', '.join(unloaded)}"
        except Exception as e:
            return f"ERROR: Failed to clear VRAM: {e}"

    elif repair == "reindex_knowledge":
        try:
            from memory import ingest_all
            results = ingest_all()
            ok = len([r for r in results if r["status"] == "ok"])
            skipped = len([r for r in results if r["status"] == "skipped"])
            errors = len([r for r in results if r["status"] == "error"])
            return f"OK: Re-indexed knowledge base — {ok} ingested, {skipped} skipped, {errors} errors"
        except Exception as e:
            return f"ERROR: Failed to reindex: {e}"

    elif repair == "check_ollama":
        tags = _ollama_get("/api/tags")
        if tags is not None:
            model_count = len(tags.get("models", []))
            return f"OK: Ollama is running ({model_count} models available)"
        # Try to start Ollama
        try:
            import subprocess
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            import time
            time.sleep(5)
            tags = _ollama_get("/api/tags")
            if tags is not None:
                return f"OK: Started Ollama successfully ({len(tags.get('models', []))} models available)"
            return "WARNING: Started Ollama process but it's not responding yet. Wait a few seconds and retry."
        except Exception as e:
            return f"ERROR: Failed to start Ollama: {e}"

    else:
        return f"ERROR: Unknown repair '{repair}'. Options: restart_watcher, clear_vram, reindex_knowledge, check_ollama"


def resource_status() -> str:
    """Get E3N resource self-manager status — VRAM pressure, circuit breaker, auto-recovery actions."""
    import httpx as _hx
    lines = ["E3N RESOURCE STATUS", "=" * 40]

    # VRAM
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        used = round(mem.used / 1e6)
        total = round(mem.total / 1e6)
        free = total - used
        tier = "A (14B)" if free > 9000 else "B (3B GPU)" if free > 3000 else "C (CPU)"
        lines.append(f"VRAM:     {used:,}/{total:,} MB ({free:,} free) — Tier {tier}")
        lines.append(f"GPU Util: {util.gpu}%")
        if free < 3000:
            lines.append("          WARNING: Low VRAM — Tier C active")
    except Exception:
        lines.append("VRAM:     Unavailable")

    # Loaded models
    ps = _ollama_get("/api/ps")
    if ps:
        loaded = ps.get("models", [])
        if loaded:
            lines.append(f"\nLoaded models ({len(loaded)}):")
            for m in loaded:
                vram_mb = round(m.get("size_vram", m.get("size", 0)) / 1e6)
                lines.append(f"  {m.get('name', '?'):<20} ~{vram_mb} MB")
        else:
            lines.append("\nNo models loaded in VRAM")

    # Resource manager state (from main.py endpoint)
    try:
        resp = _hx.get("http://localhost:8000/resources/status", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            rm = data.get("resource_manager", {})
            cb = data.get("circuit_breaker", {})
            lines.append(f"\nResource Manager:")
            lines.append(f"  VRAM warnings:    {rm.get('vram_warnings', 0)}")
            lines.append(f"  Ollama failures:  {rm.get('ollama_failures', 0)}")
            lines.append(f"  Watcher restarts: {rm.get('watcher_restarts', 0)}")
            actions = rm.get("recent_actions", [])
            if actions:
                lines.append(f"  Recent actions ({len(actions)}):")
                for a in actions[-5:]:
                    lines.append(f"    {a.get('action', '?')}")

            lines.append(f"\nCircuit Breaker:")
            lines.append(f"  State:   {cb.get('state', '?')}")
            lines.append(f"  Backoff: {cb.get('backoff_sec', '?')}s")
    except Exception:
        lines.append("\nResource manager: endpoint unreachable (server may not be running)")

    # Sim detection
    try:
        from router import is_sim_running, is_session_active
        lines.append(f"\nSim running: {is_sim_running()}")
        lines.append(f"Session active: {is_session_active()}")
    except ImportError:
        pass

    return "\n".join(lines)


# ── DISPATCH ────────────────────────────────────────────────────────────

EXECUTORS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_powershell": run_powershell,
    "get_system_info": get_system_info,
    "search_knowledge": search_knowledge,
    "memory_stats": memory_stats,
    "web_search": web_search,
    "self_diagnostics": self_diagnostics,
    "manage_ollama_models": manage_ollama_models,
    "repair_subsystem": repair_subsystem,
    "resource_status": resource_status,
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
