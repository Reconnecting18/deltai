"""
deltai Tool Executor — runs tools safely and returns results.
Each tool function returns a string result that gets fed back to the model.
"""

import json
import logging
import os
import subprocess

import path_guard
import psutil
import safe_errors

_LOG = logging.getLogger("deltai.executor")
_TOOL_ERR = "An unexpected error occurred"


def _tool_error(exc: BaseException, log_message: str) -> str:
    safe_errors.log_exception(_LOG, log_message, exc)
    return f"ERROR: {_TOOL_ERR}"


# ── SAFETY ──────────────────────────────────────────────────────────────
PROTECTED_PATHS = [
    "/etc",
    "/boot",
    "/proc",
    "/sys",
    "/root",
    "/usr/bin",
    "/usr/sbin",
    "/sbin",
    "/bin",
]

BLOCKED_COMMANDS = [
    # Filesystem destruction
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    # Privilege escalation
    "sudo ",
    "su -",
    "sudo -i",
    "chmod 777 /",
    "chown root",
    # User/group manipulation
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "passwd ",
    # System shutdown/reboot
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    # Dangerous network ops
    "wget ",
    "curl ",
    "nc -",
    # Fork bomb and shell escape patterns
    ":(){ :|:& };:",
    "> /dev/sda",
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
        try:
            path = path_guard.resolve_tool_path(path)
        except ValueError as e:
            return f"ERROR: {safe_errors.public_error_detail(e)}"
        if not os.path.exists(path):
            return f"ERROR: File not found: {path}"
        if not os.path.isfile(path):
            return f"ERROR: Not a file: {path}"
        size = os.path.getsize(path)
        if size > 2_000_000:
            return f"ERROR: File too large ({size:,} bytes). Max 2MB."
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        truncated = lines[:max_lines]
        result = "".join(truncated)
        if total > max_lines:
            result += f"\n\n[TRUNCATED — showing {max_lines}/{total} lines]"
        return result
    except Exception as e:
        return _tool_error(e, "read_file failed")


def write_file(path: str, content: str, append=False) -> str:
    try:
        append = _coerce_bool(append, False)
        try:
            path = path_guard.resolve_tool_path(path)
        except ValueError as e:
            return f"ERROR: {safe_errors.public_error_detail(e)}"
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
        return _tool_error(e, "write_file failed")


def list_directory(path: str, recursive=False) -> str:
    try:
        recursive = _coerce_bool(recursive, False)
        try:
            path = path_guard.resolve_tool_path(path)
        except ValueError as e:
            return f"ERROR: {safe_errors.public_error_detail(e)}"
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
                dirs[:] = [
                    d
                    for d in dirs
                    if d
                    not in (
                        "node_modules",
                        ".git",
                        "__pycache__",
                        "venv",
                        ".venv",
                        "dist",
                        "build",
                        ".next",
                    )
                ]
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
        return _tool_error(e, "list_directory failed")


def _fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1_048_576:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1_048_576:.1f} MB"


def run_shell(command: str, timeout=15) -> str:
    try:
        timeout = _coerce_int(timeout, 15)
        timeout = min(timeout, 30)
        if not _is_command_safe(command):
            return f"ERROR: Command blocked for safety: {command}"
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.expanduser("~/deltai"),
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
        return _tool_error(e, "run_shell failed")


def get_system_info(include_processes=False) -> str:
    try:
        include_processes = _coerce_bool(include_processes, False)
        info = []
        cpu_pct = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        info.append(f"CPU: {cpu_pct}% @ {round(freq.current) if freq else '?'} MHz")
        info.append(f"     {psutil.cpu_count(logical=False)}P/{psutil.cpu_count()} threads")
        ram = psutil.virtual_memory()
        info.append(f"RAM: {ram.used / 1e9:.1f}/{ram.total / 1e9:.1f} GB ({ram.percent}%)")
        disk = psutil.disk_usage("/")
        info.append(f"Disk /: {disk.used / 1e9:.0f}/{disk.total / 1e9:.0f} GB ({disk.percent}%)")
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
            info.append(f"GPU: {name}")
            info.append(f"     Utilization: {util.gpu}%")
            info.append(f"     VRAM: {mem.used / 1e6:.0f}/{mem.total / 1e6:.0f} MB")
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
        return _tool_error(e, "get_system_info failed")


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
        return _tool_error(e, "search_knowledge failed")


def memory_stats() -> str:
    """Get knowledge base statistics."""
    try:
        from memory import get_memory_stats

        stats = get_memory_stats()
        lines = [
            "Knowledge Base Stats:",
            f"  Total chunks: {stats['total_chunks']}",
            f"  Files ingested: {stats['total_files']}",
            f"  Disk usage: {stats['disk_mb']} MB",
        ]
        if stats["sources"]:
            lines.append("  Sources:")
            for s in stats["sources"]:
                lines.append(f"    - {s}")
        else:
            lines.append("  No files ingested yet.")
            lines.append("  Drop files in ~/deltai/data/knowledge/ to ingest.")
        return "\n".join(lines)
    except ImportError:
        return "ERROR: Memory system not available (chromadb not installed)"
    except Exception as e:
        return _tool_error(e, "memory_stats failed")


# ── WEB SEARCH TOOL ──────────────────────────────────────────────────


def web_search(query: str, max_results=5) -> str:
    """Search the web via DuckDuckGo Lite. Blocked during active GPU focus sessions."""
    try:
        # Session guard: block during focus workload (performance protection)
        try:
            import sys

            if "." not in sys.path:
                sys.path.insert(0, ".")
            from router import _session_active

            if _session_active:
                return "ERROR: Web search disabled during active GPU focus sessions (performance protection)"
        except (ImportError, AttributeError):
            pass

        max_results = min(_coerce_int(max_results, 5), 10)
        if not query or not query.strip():
            return "ERROR: Empty search query"

        import re as _re

        import httpx as _httpx

        # DuckDuckGo HTML Lite — simple table-based results, no JS required, no API key
        resp = _httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query.strip(), "kl": "us-en"},
            headers={"User-Agent": "deltai/1.0 (local AI assistant)"},
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
        link_pattern = _re.compile(r'<a[^>]*href="([^"]*uddg=[^"]+)"[^>]*>(.+?)</a>', _re.DOTALL)
        links = link_pattern.findall(html)

        # Extract all <td> content for snippets
        td_pattern = _re.compile(r"<td[^>]*>(.*?)</td>", _re.DOTALL)
        tds = td_pattern.findall(html)

        # Build snippet map: find td cells that contain substantial text (snippets)
        snippets = []
        for td in tds:
            clean = _re.sub(r"<[^>]+>", "", td).strip()
            clean = _re.sub(r"&\w+;", " ", clean).strip()
            # Snippets are longer text blocks (not numbers, not URLs, not spacers)
            if len(clean) > 40 and not clean.startswith("http") and not clean.startswith("www."):
                snippets.append(_re.sub(r"\s+", " ", clean))

        from urllib.parse import parse_qs, unquote, urlparse

        for i, (raw_url, raw_title) in enumerate(links[:max_results]):
            clean_title = _re.sub(r"<[^>]+>", "", raw_title).strip()
            clean_title = _re.sub(r"&#x27;", "'", clean_title)
            clean_title = _re.sub(r"&amp;", "&", clean_title)
            # Extract actual URL from DuckDuckGo redirect
            try:
                parsed = parse_qs(urlparse(raw_url).query)
                actual_url = unquote(parsed.get("uddg", [raw_url])[0])
            except Exception:
                actual_url = raw_url
            snippet = snippets[i] if i < len(snippets) else ""
            if clean_title:
                results.append(f"[{i + 1}] {clean_title}\n    {actual_url}\n    {snippet}")

        if not results:
            return f"No results found for: {query}"

        header = f"Web search results for: {query}\n{'=' * 40}\n"
        return header + "\n\n".join(results)

    except Exception as e:
        safe_errors.log_exception(_LOG, "web_search failed", e)
        import httpx as _httpx_err

        if isinstance(e, (_httpx_err.TimeoutException, TimeoutError)):
            return "ERROR: Search timed out (10s limit). Try a simpler query."
        return f"ERROR: {_TOOL_ERR}"


def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return clean article text. Blocked during active GPU focus sessions."""
    try:
        # Session guard: block during focus workload
        try:
            import sys as _sys

            if "." not in _sys.path:
                _sys.path.insert(0, ".")
            from router import _session_active

            if _session_active:
                return "ERROR: fetch_url disabled during active GPU focus sessions (performance protection)"
        except (ImportError, AttributeError):
            pass

        url = (url or "").strip()
        if not url or not url.startswith("http"):
            return "ERROR: Invalid URL — must start with http:// or https://"
        if len(url) > 2000:
            return "ERROR: URL too long"

        max_chars = max(500, min(_coerce_int(max_chars, 8000), 20000))

        try:
            import trafilatura as _trafilatura  # type: ignore

            downloaded = _trafilatura.fetch_url(url)
            if not downloaded:
                return f"ERROR: Could not download content from {url}"
            text = _trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            if not text or not text.strip():
                return f"ERROR: No extractable text content found at {url}"
            text = text.strip()[:max_chars]
            char_note = f" [truncated to {max_chars} chars]" if len(text) == max_chars else ""
            return f"Content from {url}{char_note}:\n\n{text}"
        except ImportError:
            # Fallback: plain httpx fetch + strip HTML tags
            import re as _re

            import httpx as _httpx

            resp = _httpx.get(
                url,
                headers={"User-Agent": "deltai/1.0 (local AI assistant)"},
                timeout=15.0,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return f"ERROR: HTTP {resp.status_code} from {url}"
            html = resp.text
            text = _re.sub(r"<[^>]+>", " ", html)
            text = _re.sub(r"&\w+;", " ", text)
            text = _re.sub(r"\s+", " ", text).strip()
            if not text:
                return f"ERROR: No text content found at {url}"
            text = text[:max_chars]
            return f"Content from {url} (HTML stripped — install trafilatura for better extraction):\n\n{text}"

    except Exception as e:
        safe_errors.log_exception(_LOG, "fetch_url failed", e)
        import httpx as _httpx_err2

        if isinstance(e, (_httpx_err2.TimeoutException, TimeoutError)):
            return f"ERROR: Request timed out fetching {url}"
        return f"ERROR: {_TOOL_ERR}"


# ── COMPUTATION DELEGATION TOOLS ──────────────────────────────────────

# Sandboxed builtins for calculate tool — math + statistics only, no I/O
_CALC_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "pow": pow,
    "int": int,
    "float": float,
    "bool": bool,
    "range": range,
    "list": list,
    "tuple": tuple,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "True": True,
    "False": False,
    "None": None,
}


def calculate(expression: str, description: str = None) -> str:
    """Evaluate a mathematical expression in a sandboxed Python environment."""
    try:
        if not expression or not expression.strip():
            return "ERROR: Empty expression"
        expression = expression.strip()
        if len(expression) > 500:
            return "ERROR: Expression too long (max 500 chars)"

        # Block dangerous patterns
        _blocked = [
            "import ",
            "exec(",
            "eval(",
            "open(",
            "__",
            "os.",
            "sys.",
            "subprocess",
            "compile(",
            "globals(",
            "locals(",
            "getattr(",
            "setattr(",
            "delattr(",
            "dir(",
            "breakpoint(",
            "input(",
            "print(",
        ]
        expr_lower = expression.lower()
        for b in _blocked:
            if b in expr_lower:
                return f"ERROR: Blocked operation in expression: {b.strip()}"

        import math
        import statistics

        safe_globals = {"__builtins__": _CALC_SAFE_BUILTINS, "math": math, "statistics": statistics}
        result = eval(expression, safe_globals)

        desc = f" ({description})" if description else ""
        if isinstance(result, float):
            # Smart formatting: avoid excessive decimals
            if result == int(result) and abs(result) < 1e15:
                return f"= {int(result)}{desc}"
            return f"= {result:.6g}{desc}"
        return f"= {result}{desc}"
    except ZeroDivisionError:
        return "ERROR: Division by zero"
    except Exception as e:
        return _tool_error(e, "calculate failed")


def solve_math(operation: str, expression: str, variable: str = "x", point: str = None) -> str:
    """Symbolic mathematics engine powered by SymPy. Handles calculus, algebra, linear algebra, and more."""
    try:
        import sympy
        from sympy import (
            Abs,
            E,
            Eq,
            Function,
            I,
            Matrix,
            Rational,
            acos,
            asin,
            atan,
            cos,
            cosh,
            diff,
            exp,
            expand,
            factor,
            integrate,
            limit,
            log,
            oo,
            pi,
            series,
            sign,
            simplify,
            sin,
            sinh,
            solve,
            sqrt,
            symbols,
            sympify,
            tan,
            tanh,
        )
        from sympy.integrals.transforms import laplace_transform

        if not operation or not expression:
            return "ERROR: operation and expression are required"
        if len(expression) > 500:
            return "ERROR: Expression too long (max 500 chars)"

        operation = operation.strip().lower()
        expression = expression.strip()

        # Block dangerous patterns (same as calculate)
        _blocked = [
            "import ",
            "exec(",
            "eval(",
            "open(",
            "__",
            "os.",
            "sys.",
            "subprocess",
            "compile(",
            "globals(",
            "locals(",
            "getattr(",
            "setattr(",
            "delattr(",
            "breakpoint(",
            "input(",
        ]
        expr_lower = expression.lower()
        for b in _blocked:
            if b in expr_lower:
                return f"ERROR: Blocked operation: {b.strip()}"

        # Safe symbol namespace
        x, y, z, t, s = symbols("x y z t s")
        a, b_sym, c, n, r = symbols("a b c n r")
        theta, phi, omega, k, m, g = symbols("theta phi omega k m g")
        f = Function("f")

        safe_locals = {
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "s": s,
            "a": a,
            "b": b_sym,
            "c": c,
            "n": n,
            "r": r,
            "theta": theta,
            "phi": phi,
            "omega": omega,
            "k": k,
            "m": m,
            "g": g,
            "f": f,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "asin": asin,
            "acos": acos,
            "atan": atan,
            "sinh": sinh,
            "cosh": cosh,
            "tanh": tanh,
            "exp": exp,
            "log": log,
            "ln": log,
            "sqrt": sqrt,
            "abs": Abs,
            "sign": sign,
            "Abs": Abs,
            "pi": pi,
            "e": E,
            "E": E,
            "I": I,
            "oo": oo,
            "Rational": Rational,
            "Matrix": Matrix,
            "Eq": Eq,
        }

        var = symbols(variable) if variable else x

        # Parse expression safely
        if operation == "matrix" or operation == "eigenvalues":
            # Matrix operations need eval with controlled namespace
            safe_eval_ns = {
                "__builtins__": {},
                "Matrix": Matrix,
                "Rational": Rational,
                "symbols": symbols,
                "sqrt": sqrt,
            }
            safe_eval_ns.update(safe_locals)
            expr = eval(expression, safe_eval_ns)
        else:
            expr = sympify(expression, locals=safe_locals)

        # Execute operation
        if operation == "solve":
            result = solve(expr, var)
            if isinstance(result, list):
                solutions = [str(s) for s in result]
                approx = []
                for s in result:
                    try:
                        val = complex(s.evalf())
                        if val.imag == 0:
                            approx.append(f"{val.real:.6g}")
                        else:
                            approx.append(f"{val.real:.4g} + {val.imag:.4g}i")
                    except Exception:
                        approx.append("?")
                return f"{variable} = {solutions}\nApprox: {approx}"
            return f"{variable} = {result}"

        elif operation in ("differentiate", "diff", "derivative"):
            result = diff(expr, var)
            return f"d/d{variable}({expression}) = {result}"

        elif operation in ("integrate", "integral"):
            if point:
                # Definite integral: point should be "a,b"
                try:
                    bounds = point.split(",")
                    lo = sympify(bounds[0].strip(), locals=safe_locals)
                    hi = sympify(bounds[1].strip(), locals=safe_locals)
                    result = integrate(expr, (var, lo, hi))
                    try:
                        approx = float(result.evalf())
                        return f"integral({expression}, {variable}={lo}..{hi}) = {result} (approx {approx:.6g})"
                    except Exception:
                        return f"integral({expression}, {variable}={lo}..{hi}) = {result}"
                except (IndexError, ValueError):
                    return (
                        "ERROR: For definite integral, point should be 'lower,upper' (e.g., '0,pi')"
                    )
            result = integrate(expr, var)
            return f"integral({expression}) d{variable} = {result} + C"

        elif operation == "limit":
            if not point:
                return "ERROR: 'point' is required for limit (e.g., '0', 'oo', '-oo')"
            pt = sympify(point, locals=safe_locals)
            result = limit(expr, var, pt)
            return f"lim({variable}->{point}) {expression} = {result}"

        elif operation == "simplify":
            result = simplify(expr)
            return f"simplify({expression}) = {result}"

        elif operation == "expand":
            result = expand(expr)
            return f"expand({expression}) = {result}"

        elif operation == "factor":
            result = factor(expr)
            return f"factor({expression}) = {result}"

        elif operation == "series":
            pt = sympify(point, locals=safe_locals) if point else 0
            result = series(expr, var, pt, n=6)
            return f"series({expression}) around {variable}={pt}:\n{result}"

        elif operation == "laplace":
            t_sym = symbols("t", positive=True)
            s_sym = symbols("s")
            expr_t = sympify(expression, locals={**safe_locals, "t": t_sym})
            result, cond1, cond2 = laplace_transform(expr_t, t_sym, s_sym)
            return f"L{{{expression}}} = {result}  (convergence: {cond2})"

        elif operation == "eigenvalues":
            if not isinstance(expr, sympy.Matrix):
                return "ERROR: Expression must be a Matrix for eigenvalues (e.g., 'Matrix([[1,2],[3,4]])')"
            eigenvals = expr.eigenvals()
            expr.eigenvects()
            lines = [f"Eigenvalues of {expression}:"]
            for val, mult in eigenvals.items():
                try:
                    approx = f" (approx {complex(val.evalf()).real:.6g})"
                except Exception:
                    approx = ""
                lines.append(f"  lambda = {val} (multiplicity {mult}){approx}")
            return "\n".join(lines)

        elif operation == "matrix":
            # General matrix operation — expression should be a full statement
            if isinstance(expr, sympy.Matrix):
                lines = ["Matrix result:"]
                lines.append(str(expr))
                lines.append(f"det = {expr.det()}")
                try:
                    lines.append(f"rank = {expr.rank()}")
                except Exception:
                    pass
                return "\n".join(lines)
            return f"Result: {expr}"

        else:
            valid = "solve, differentiate, integrate, limit, simplify, expand, factor, matrix, series, laplace, eigenvalues"
            return f"ERROR: Unknown operation '{operation}'. Valid: {valid}"

    except sympy.SympifyError:
        return "ERROR: Could not parse expression"
    except Exception as e:
        return _tool_error(e, "solve_math failed")


def summarize_data(data: str, focus: str = "all") -> str:
    """Summarize structured data into key statistics."""
    try:
        if not data or not data.strip():
            return "ERROR: Empty data"
        if len(data) > 50000:
            return "ERROR: Data too large (max 50000 chars)"

        focus = (focus or "all").lower().strip()
        import re
        import statistics

        # Try to extract numbers from the data
        numbers = []

        # Try JSON first
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, (int, float)):
                        numbers.append(float(item))
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, (int, float)):
                                numbers.append(float(v))
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, (int, float)):
                        numbers.append(float(v))
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, (int, float)):
                                numbers.append(float(item))
        except (json.JSONDecodeError, TypeError):
            # Fall back to regex number extraction
            number_matches = re.findall(r"-?\d+\.?\d*", data)
            numbers = [float(n) for n in number_matches if abs(float(n)) < 1e15]

        if not numbers:
            # Text summary: word/line count
            lines = data.strip().split("\n")
            words = data.split()
            return f"Text data: {len(lines)} lines, {len(words)} words, {len(data)} chars. No numeric data found."

        lines = [f"DATA SUMMARY ({len(numbers)} values)", "=" * 35]

        # Basic stats (always included)
        lines.append(f"Count:  {len(numbers)}")
        lines.append(f"Min:    {min(numbers):.4g}")
        lines.append(f"Max:    {max(numbers):.4g}")
        lines.append(f"Range:  {max(numbers) - min(numbers):.4g}")

        if len(numbers) >= 2:
            mean = statistics.mean(numbers)
            median = statistics.median(numbers)
            stdev = statistics.stdev(numbers) if len(numbers) >= 2 else 0
            lines.append(f"Mean:   {mean:.4g}")
            lines.append(f"Median: {median:.4g}")
            lines.append(f"Stdev:  {stdev:.4g}")

            if focus in ("all", "distribution"):
                q1 = statistics.quantiles(numbers, n=4)[0] if len(numbers) >= 4 else min(numbers)
                q3 = statistics.quantiles(numbers, n=4)[2] if len(numbers) >= 4 else max(numbers)
                lines.append(f"Q1:     {q1:.4g}")
                lines.append(f"Q3:     {q3:.4g}")

            if focus in ("all", "outliers") and stdev > 0:
                outliers = [n for n in numbers if abs(n - mean) > 2 * stdev]
                if outliers:
                    lines.append(
                        f"\nOutliers (>2σ): {len(outliers)} — {[round(o, 4) for o in outliers[:5]]}"
                    )
                else:
                    lines.append("\nOutliers: none (all within 2σ)")

            if focus in ("all", "trends") and len(numbers) >= 3:
                # Simple trend: compare first third vs last third
                third = max(1, len(numbers) // 3)
                first_avg = statistics.mean(numbers[:third])
                last_avg = statistics.mean(numbers[-third:])
                if first_avg != 0:
                    change_pct = ((last_avg - first_avg) / abs(first_avg)) * 100
                    direction = (
                        "increasing"
                        if change_pct > 5
                        else "decreasing"
                        if change_pct < -5
                        else "stable"
                    )
                    lines.append(f"\nTrend: {direction} ({change_pct:+.1f}% first→last third)")
                else:
                    lines.append(f"\nTrend: first={first_avg:.4g}, last={last_avg:.4g}")

        return "\n".join(lines)
    except Exception as e:
        return _tool_error(e, "summarize_data failed")


def lookup_reference(query: str) -> str:
    """Quick formula/constant lookup from knowledge base — top 1-2 most relevant chunks."""
    try:
        if not query or not query.strip():
            return "ERROR: Empty query"
        from memory import query_knowledge

        # Use lower threshold (0.6) for more targeted results, fewer results
        matches = query_knowledge(query.strip(), n_results=2)
        if not matches:
            return f"No reference found for: {query}. Try search_knowledge for broader results."
        lines = [f"REFERENCE: {query}", "-" * 35]
        for i, m in enumerate(matches, 1):
            lines.append(f"[{m['source']}] (relevance: {1 - m['distance']:.0%})")
            lines.append(m["text"])
            if i < len(matches):
                lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "ERROR: Memory system not available (chromadb not installed)"
    except Exception as e:
        return _tool_error(e, "lookup_reference failed")


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
        return _tool_error(
            e,
            f"telemetry GET failed ({_TELEMETRY_API_URL})",
        )


def get_session_status(**kwargs) -> str:
    """Get current session snapshot from the optional telemetry HTTP API."""
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
_DELTAI_MODELS = {"deltai-qwen14b", "deltai-qwen3b", "deltai-nemo", "deltai-fallback", "deltai"}
_CRITICAL_PATHS = [
    r"~/deltai/data\knowledge",
    r"~/deltai/data\chromadb",
    r"~/deltai/project\.env",
    r"~/deltai/modelfiles",
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
    """Run deltai self-diagnostics across all subsystems or deep-check one."""
    try:
        if subsystem:
            return _diag_deep(subsystem.lower().strip())
        return _diag_full()
    except Exception as e:
        return _tool_error(e, "self_diagnostics")


def _diag_full() -> str:
    """Full sweep across all subsystems."""
    lines = ["deltai SELF-DIAGNOSTICS", "=" * 40]

    # Ollama
    tags = _ollama_get("/api/tags")
    ps = _ollama_get("/api/ps")
    if tags is None:
        lines.append("Ollama:    DOWN — cannot reach Ollama API")
    else:
        all_models = [m["name"].split(":")[0] for m in tags.get("models", [])]
        deltai_found = [m for m in all_models if m in _DELTAI_MODELS]
        loaded = [m.get("name", "?") for m in (ps or {}).get("models", [])]
        loaded_str = ", ".join(loaded) if loaded else "none"
        lines.append(
            f"Ollama:    ONLINE ({len(deltai_found)}/4 deltai models, loaded: {loaded_str})"
        )
        missing = _DELTAI_MODELS - set(deltai_found)
        if missing:
            lines.append(f"           MISSING: {', '.join(sorted(missing))}")

    # ChromaDB
    try:
        from memory import get_memory_stats

        stats = get_memory_stats()
        lines.append(
            f"ChromaDB:  ONLINE ({stats['total_chunks']} chunks, {stats['total_files']} files, {stats['disk_mb']} MB)"
        )
    except Exception as e:
        safe_errors.log_exception(_LOG, "self_diagnostics chromadb", e)
        lines.append(f"ChromaDB:  ERROR — {_TOOL_ERR}")

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
        lines.append(
            f"GPU:       {name} — {used:,}/{total:,} MB ({free:,} free, Tier {tier}, {util.gpu}% util, {temp}C)"
        )
        if temp > 83:
            lines.append("           WARNING: GPU temperature high")
    except Exception:
        lines.append("GPU:       UNAVAILABLE")

    # Voice
    try:
        from voice import VOICE_ENABLED, get_voice_status

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
    for model in ["deltai-nemo", "deltai-fallback", "deltai"]:
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
    issues = [
        line
        for line in lines
        if "DOWN" in line
        or "MISSING" in line
        or "ERROR" in line
        or "WARNING" in line
        or "STOPPED" in line
        or "DEGRADED" in line
    ]
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
            lines.append(
                "FIX: Run `ollama serve` in a terminal, or use repair_subsystem('check_ollama')"
            )
            return "\n".join(lines)
        all_models = {m["name"].split(":")[0]: m for m in tags.get("models", [])}
        ps = _ollama_get("/api/ps")
        loaded = ps.get("models", []) if ps else []
        lines.append("Available models:")
        for name, info in sorted(all_models.items()):
            size_gb = round(info.get("size", 0) / 1e9, 1)
            is_deltai = "  [deltai]" if name in _DELTAI_MODELS else ""
            lines.append(f"  {name:<20} {size_gb} GB{is_deltai}")
        lines.append(f"\nLoaded in VRAM ({len(loaded)}):")
        if loaded:
            for m in loaded:
                vram_mb = round(m.get("size_vram", m.get("size", 0)) / 1e6)
                lines.append(f"  {m.get('name', '?'):<20} ~{vram_mb} MB")
        else:
            lines.append("  (none)")
        missing = _DELTAI_MODELS - set(all_models.keys())
        if missing:
            lines.append(f"\nMISSING deltai MODELS: {', '.join(sorted(missing))}")
            for m in sorted(missing):
                mf = f"~/deltai/modelfiles/deltai-{m.replace('deltai-', '')}.modelfile"
                if m == "deltai":
                    mf = "~/deltai/modelfiles/deltai.modelfile"
                lines.append(f"  FIX: ollama create {m} -f {mf}")
        return "\n".join(lines)

    elif subsystem == "chromadb":
        lines = ["CHROMADB DEEP DIAGNOSTICS", "=" * 40]
        try:
            from memory import CHROMADB_PATH, get_file_details, get_memory_stats

            stats = get_memory_stats()
            lines.append(f"Path:    {CHROMADB_PATH}")
            lines.append(f"Chunks:  {stats['total_chunks']}")
            lines.append(f"Files:   {stats['total_files']}")
            lines.append(f"Disk:    {stats['disk_mb']} MB")
            if stats["total_chunks"] == 0:
                lines.append("\nWARNING: Empty knowledge base")
                lines.append(
                    "FIX: Drop files in ~/deltai/data\\knowledge\\ or use repair_subsystem('reindex_knowledge')"
                )
            files = get_file_details()
            if files:
                lines.append(f"\nIngested files ({len(files)}):")
                for f in files[:20]:
                    lines.append(f"  {f['source']:<40} {f['chunks']} chunks")
        except Exception as e:
            safe_errors.log_exception(_LOG, "_diag_deep chromadb", e)
            lines.append(f"ERROR: {_TOOL_ERR}")
            lines.append("FIX: Check ChromaDB installation, verify ~/deltai/data\\chromadb exists")
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
            tier = (
                "A (14B fits)"
                if free > 9000
                else "B (3B on GPU)"
                if free > 3000
                else "C (CPU only)"
            )
            lines.append(f"Device:  {name}")
            lines.append(f"VRAM:    {used:,} / {total:,} MB ({free:,} MB free)")
            lines.append(f"Tier:    {tier}")
            lines.append(f"Util:    {util.gpu}%")
            lines.append(f"Temp:    {temp}C")
            lines.append(f"Power:   {power:.0f}W")
            if free < 3000:
                lines.append("\nWARNING: Low VRAM — running in CPU-only mode (Tier C)")
                lines.append(
                    "FIX: Use manage_ollama_models('unload', model='...') or repair_subsystem('clear_vram')"
                )
            if temp > 83:
                lines.append("\nWARNING: High GPU temperature")
                lines.append("FIX: Check cooling, reduce load, or unload models")
        except Exception as e:
            safe_errors.log_exception(_LOG, "_diag_deep gpu", e)
            lines.append(f"GPU unavailable: {_TOOL_ERR}")
        return "\n".join(lines)

    elif subsystem == "voice":
        lines = ["VOICE DEEP DIAGNOSTICS", "=" * 40]
        try:
            from voice import VOICE_ENABLED, get_voice_status

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
            from memory import KNOWLEDGE_PATH
            from watcher import watcher_running

            running = watcher_running()
            lines.append(f"Status:  {'ACTIVE' if running else 'STOPPED'}")
            lines.append(f"Path:    {KNOWLEDGE_PATH}")
            lines.append(f"Exists:  {os.path.exists(KNOWLEDGE_PATH)}")
            if os.path.exists(KNOWLEDGE_PATH):
                files = [f for f in os.listdir(KNOWLEDGE_PATH) if not f.startswith(".")]
                lines.append(f"Files:   {len(files)}")
            if not running:
                lines.append("\nFIX: Use repair_subsystem('restart_watcher')")
        except ImportError:
            lines.append("Watcher not available")
        return "\n".join(lines)

    elif subsystem == "backup":
        lines = ["BACKUP SYSTEM DIAGNOSTICS", "=" * 40]
        tags = _ollama_get("/api/tags")
        chains = [
            ("deltai-qwen14b", "deltai-nemo", "deltai-fallback"),
            ("deltai-qwen3b", "deltai"),
        ]
        if tags:
            available = {m["name"].split(":")[0] for m in tags.get("models", [])}
            for chain in chains:
                chain[0]
                chain[1:]
                chain_str = " -> ".join(chain)
                statuses = ["OK" if m in available else "MISSING" for m in chain]
                lines.append(f"Chain: {chain_str}")
                for m, s in zip(chain, statuses, strict=False):
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
                lines.append(
                    f"        FIX: mkdir {p}"
                    if "." not in os.path.basename(p)
                    else "        FIX: Check installation"
                )
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
            lines.append("\nAvailable (not loaded):")
            for m in avail:
                size_gb = round(m.get("size", 0) / 1e9, 1)
                deltai_tag = " [deltai]" if m["name"].split(":")[0] in _DELTAI_MODELS else ""
                lines.append(f"  {m['name']:<24} {size_gb} GB{deltai_tag}")
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
        if model_base not in _DELTAI_MODELS:
            return f"ERROR: Can only manage deltai models. '{model}' not in allowlist: {', '.join(sorted(_DELTAI_MODELS))}"
        import httpx as _hx

        try:
            resp = _hx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=10,
            )
            if resp.status_code == 200:
                return (
                    f"OK: Unloaded {model} from VRAM. Use manage_ollama_models('status') to verify."
                )
            return f"ERROR: Ollama returned HTTP {resp.status_code}"
        except Exception as e:
            return _tool_error(e, "manage_ollama_models unload")

    elif action == "preload":
        if not model:
            return "ERROR: 'model' parameter required for preload"
        model_base = model.split(":")[0]
        if model_base not in _DELTAI_MODELS:
            return f"ERROR: Can only manage deltai models. '{model}' not in allowlist: {', '.join(sorted(_DELTAI_MODELS))}"
        # Focus workload guard: block preloading 14B while foreground app contests VRAM
        if "14b" in model.lower():
            try:
                from router import is_sim_running

                if is_sim_running():
                    return "BLOCKED: Cannot preload 14B model during GPU focus workload — VRAM reserved for foreground use"
            except ImportError:
                pass
        # Check VRAM availability
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = round((mem.total - mem.used) / 1e6)
            model_sizes = {
                "deltai-qwen14b": 8500,
                "deltai-qwen3b": 2500,
                "deltai-nemo": 7200,
                "deltai-fallback": 4700,
                "deltai": 4700,
            }
            needed = model_sizes.get(model_base, 5000)
            if free_mb < needed + 500:
                return f"WARNING: Only {free_mb} MB VRAM free, {model} needs ~{needed} MB. Unload another model first."
        except Exception:
            pass
        import httpx as _hx

        try:
            resp = _hx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": "ping",
                    "keep_alive": "5m",
                    "options": {"num_predict": 1},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return f"OK: Preloaded {model} into VRAM (5min keep-alive)."
            return f"ERROR: Ollama returned HTTP {resp.status_code}"
        except Exception as e:
            return _tool_error(e, "manage_ollama_models preload")

    else:
        return f"ERROR: Unknown action '{action}'. Use 'status', 'unload', or 'preload'."


def repair_subsystem(repair: str) -> str:
    """Attempt a safe, allowlisted repair on an deltai subsystem."""
    repair = (repair or "").lower().strip()

    if repair == "restart_watcher":
        try:
            from watcher import start_watcher, stop_watcher, watcher_running

            stop_watcher()
            start_watcher()
            running = watcher_running()
            return f"OK: Watcher {'restarted successfully' if running else 'restart attempted but not confirmed running'}"
        except Exception as e:
            return _tool_error(e, "repair_subsystem restart_watcher")

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
                        _hx.post(
                            f"{_OLLAMA_URL}/api/generate",
                            json={"model": model_name, "prompt": "", "keep_alive": 0},
                            timeout=10,
                        )
                        unloaded.append(model_name)
                    except Exception:
                        pass
            return f"OK: Unloaded {len(unloaded)} model(s) from VRAM: {', '.join(unloaded)}"
        except Exception as e:
            return _tool_error(e, "repair_subsystem clear_vram")

    elif repair == "reindex_knowledge":
        try:
            from memory import ingest_all

            results = ingest_all()
            ok = len([r for r in results if r["status"] == "ok"])
            skipped = len([r for r in results if r["status"] == "skipped"])
            errors = len([r for r in results if r["status"] == "error"])
            return (
                f"OK: Re-indexed knowledge base — {ok} ingested, {skipped} skipped, {errors} errors"
            )
        except Exception as e:
            return _tool_error(e, "repair_subsystem reindex_knowledge")

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
            )
            import time

            time.sleep(5)
            tags = _ollama_get("/api/tags")
            if tags is not None:
                return f"OK: Started Ollama successfully ({len(tags.get('models', []))} models available)"
            return "WARNING: Started Ollama process but it's not responding yet. Wait a few seconds and retry."
        except Exception as e:
            return _tool_error(e, "repair_subsystem check_ollama")

    else:
        return f"ERROR: Unknown repair '{repair}'. Options: restart_watcher, clear_vram, reindex_knowledge, check_ollama"


def resource_status() -> str:
    """Get deltai resource self-manager status — VRAM pressure, circuit breaker, auto-recovery actions."""
    import httpx as _hx

    lines = ["deltai RESOURCE STATUS", "=" * 40]

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
            lines.append("\nResource Manager:")
            lines.append(f"  VRAM warnings:    {rm.get('vram_warnings', 0)}")
            lines.append(f"  Ollama failures:  {rm.get('ollama_failures', 0)}")
            lines.append(f"  Watcher restarts: {rm.get('watcher_restarts', 0)}")
            actions = rm.get("recent_actions", [])
            if actions:
                lines.append(f"  Recent actions ({len(actions)}):")
                for a in actions[-5:]:
                    lines.append(f"    {a.get('action', '?')}")

            lines.append("\nCircuit Breaker:")
            lines.append(f"  State:   {cb.get('state', '?')}")
            lines.append(f"  Backoff: {cb.get('backoff_sec', '?')}s")
    except Exception:
        lines.append("\nResource manager: endpoint unreachable (server may not be running)")

    # Sim detection
    try:
        from router import is_session_active, is_sim_running

        lines.append(f"\nFocus workload (sim/heavy app): {is_sim_running()}")
        lines.append(f"Session active: {is_session_active()}")
    except ImportError:
        pass

    return "\n".join(lines)


# ── ADAPTER SURGERY TOOL ────────────────────────────────────────────────


def manage_adapters(
    action: str, domain: str = None, adapter_name: str = None, dataset: str = None
) -> str:
    """Manage deltai's augmentation slot adapters."""
    try:
        from training import (
            ADAPTER_DOMAINS,
            get_active_adapters,
            get_adapter,
            list_adapters,
            merge_adapters,
            rollback_adapter,
            set_active_adapter,
            start_domain_training,
            update_adapter,
        )
    except ImportError:
        return "ERROR: Training module not available"

    if action == "status":
        lines = ["ADAPTER AUGMENTATION SLOTS", "=" * 40]
        active = get_active_adapters()
        for d in ADAPTER_DOMAINS:
            active_name = active.get(d, None)
            adapters = list_adapters(domain=d)
            lines.append(f"\n[{d.upper()}] Active: {active_name or 'none'}")
            if adapters:
                for a in adapters:
                    status_mark = " ★" if a["name"] == active_name else ""
                    score = f" (score: {a['eval_score']})" if a.get("eval_score") else ""
                    lines.append(
                        f"  {a['name']} v{a['version']} — "
                        f"{a['examples_used']} examples, "
                        f"r={a.get('lora_r', '?')}, "
                        f"freeze={a.get('frozen_layers', 0)}"
                        f"{score}{status_mark}"
                    )
            else:
                lines.append("  (no adapters trained)")
        return "\n".join(lines)

    elif action == "train":
        if not domain:
            return f"ERROR: domain required for training. Valid: {ADAPTER_DOMAINS}"
        result = start_domain_training(domain=domain, dataset_name=dataset)
        if result.get("status") == "error":
            return f"ERROR: {result['reason']}"
        return (
            f"Domain training started: {result['adapter_name']}\n"
            f"Domain: {result['domain']} v{result['version']}\n"
            f"Dataset: {result['dataset']}\n"
            f"Frozen layers: {result['frozen_layers']}\n"
            f"LoRA rank: {result['lora_r']}"
        )

    elif action == "merge":
        result = merge_adapters()
        if result.get("status") == "error":
            return f"ERROR: {result['reason']}"
        return (
            f"Merge complete: {result['output_model']}\n"
            f"Method: {result['method']} (density={result['density']})\n"
            f"Adapters merged: {', '.join(result['adapters_merged'])}"
        )

    elif action == "promote":
        if not adapter_name:
            return "ERROR: adapter_name required for promote"
        info = get_adapter(adapter_name)
        if not info:
            return f"ERROR: Adapter not found: {adapter_name}"
        domain = info.get("domain")
        result = set_active_adapter(domain, adapter_name)
        if result.get("status") == "ok":
            update_adapter(adapter_name, promoted=True)
            return f"Promoted {adapter_name} to active slot for domain: {domain}"
        return f"ERROR: {result.get('reason', 'unknown')}"

    elif action == "rollback":
        if not domain:
            return f"ERROR: domain required for rollback. Valid: {ADAPTER_DOMAINS}"
        result = rollback_adapter(domain)
        if result.get("status") == "error":
            return f"ERROR: {result['reason']}"
        return (
            f"Rolled back {domain}: {result.get('previous', 'none')} → {result['rolled_back_to']}"
        )

    else:
        return f"ERROR: Unknown action '{action}'. Valid: status, train, merge, promote, rollback"


# ── DISPATCH ────────────────────────────────────────────────────────────

EXECUTORS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_shell": run_shell,
    "get_system_info": get_system_info,
    "search_knowledge": search_knowledge,
    "memory_stats": memory_stats,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "calculate": calculate,
    "solve_math": solve_math,
    "summarize_data": summarize_data,
    "lookup_reference": lookup_reference,
    "self_diagnostics": self_diagnostics,
    "manage_ollama_models": manage_ollama_models,
    "repair_subsystem": repair_subsystem,
    "resource_status": resource_status,
    "manage_adapters": manage_adapters,
}

if _TELEMETRY_API_URL:
    EXECUTORS["get_session_status"] = get_session_status
    EXECUTORS["get_lap_summary"] = get_lap_summary
    EXECUTORS["get_tire_status"] = get_tire_status
    EXECUTORS["get_strategy_recommendation"] = get_strategy_recommendation


def register_handler(name: str, fn) -> None:
    """
    Register a tool executor from an extension.
    ``fn`` must be a callable that accepts keyword arguments matching the tool's
    parameter schema and returns a plain string result.

    Example (inside an extension's setup function)::

        from tools.executor import register_handler

        def _my_tool(input: str) -> str:
            return f"processed: {input}"

        register_handler("my_tool", _my_tool)
    """
    EXECUTORS[name] = fn


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns result string."""
    fn = EXECUTORS.get(name)
    if not fn:
        return f"ERROR: Unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"ERROR: Bad arguments for {name}: {safe_errors.public_error_detail(e)}"
    except Exception as e:
        return _tool_error(e, f"execute_tool {name}")
