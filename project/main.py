from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import json
import re
import os
import psutil
import platform
import logging
from dotenv import load_dotenv

from tools.definitions import TOOLS
from tools.executor import execute_tool
from router import route, is_cloud_available, get_gpu_utilization, classify_complexity
from anthropic_client import stream_chat as anthropic_stream

load_dotenv()
psutil.cpu_percent(interval=0.1)

logger = logging.getLogger("e3n")

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ── RAG IMPORTS ─────────────────────────────────────────────────────────
RAG_AVAILABLE = False
try:
    from memory import query_knowledge, ingest_all, get_memory_stats
    from watcher import start_watcher
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG system unavailable: {e}")

app = FastAPI(title="E3N")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
E3N_MODEL  = os.getenv("E3N_MODEL", "e3n")

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


# ── STARTUP ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    if RAG_AVAILABLE:
        try:
            results = ingest_all()
            ingested = [r for r in results if r["status"] == "ok"]
            if ingested:
                logger.info(f"Initial ingestion: {len(ingested)} file(s)")
            start_watcher()
            logger.info("File watcher started")
        except Exception as e:
            logger.error(f"RAG startup failed: {e}")

    cloud = is_cloud_available()
    logger.info(f"Cloud available: {cloud}")
    logger.info(f"GPU utilization: {get_gpu_utilization()}%")


# ── TEXT-AS-TOOL FALLBACK PARSER ────────────────────────────────────────

def _sanitize_python_json(text: str) -> str:
    import re as _re
    text = _re.sub(r'(?<=[\s:,\[])True(?=[\s,}\]])', 'true', text)
    text = _re.sub(r'(?<=[\s:,\[])False(?=[\s,}\]])', 'false', text)
    text = _re.sub(r'(?<=[\s:,\[])None(?=[\s,}\]])', 'null', text)
    return text

def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_sanitize_python_json(text))

def try_parse_text_tool_call(content: str):
    if not content:
        return None
    text = content.strip()
    match = re.search(
        r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"parameters"\s*:\s*(\{[^{}]*\})',
        text, re.DOTALL
    )
    if match:
        try:
            return (match.group(1), _safe_json_loads(match.group(2)))
        except (json.JSONDecodeError, KeyError):
            pass
    md_match = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', text, re.DOTALL)
    if md_match:
        try:
            obj = _safe_json_loads(md_match.group(1))
            if "name" in obj:
                return (obj["name"], obj.get("parameters", obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            pass
    try:
        obj = _safe_json_loads(text)
        if isinstance(obj, dict) and "name" in obj:
            return (obj["name"], obj.get("parameters", obj.get("arguments", {})))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ── RAG CONTEXT BUILDER ────────────────────────────────────────────────

def build_rag_context(user_message: str) -> str:
    if not RAG_AVAILABLE:
        return ""
    try:
        matches = query_knowledge(user_message, n_results=5, threshold=0.75)
        if not matches:
            return ""
        context_parts = ["[KNOWLEDGE CONTEXT — from your ingested documents]"]
        for m in matches:
            context_parts.append(
                f"[Source: {m['source']} | Relevance: {1 - m['distance']:.0%}]\n{m['text']}"
            )
        context_parts.append("[END CONTEXT]\n")
        return "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return ""


# ── CHAT (with smart routing) ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    deep: bool = False
    force_local: bool = False

MAX_TOOL_ROUNDS = 6

@app.get("/")
def root():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint with smart routing + tool calling + RAG.

    Stream protocol (JSON lines):
      {"t":"route","backend":"...","model":"...","tier":N,"reason":"..."}
      {"t":"rag","n":N}
      {"t":"tool","n":"tool_name","a":{args}}
      {"t":"result","n":"tool_name","s":"summary"}
      {"t":"text","c":"chunk"}
      {"t":"done"}
      {"t":"error","c":"message"}
    """
    # ── ROUTE DECISION ──────────────────────────────────────
    decision = route(
        message=req.message,
        force_cloud=req.deep,
        force_local=req.force_local,
    )
    logger.info(f"Route: {decision}")

    # ── RAG CONTEXT ─────────────────────────────────────────
    rag_context = build_rag_context(req.message)

    # ── ANTHROPIC PATH ──────────────────────────────────────
    if decision.backend == "anthropic":
        async def stream_cloud():
            yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

            if rag_context:
                chunk_count = rag_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

            async for line in anthropic_stream(
                message=req.message,
                model=decision.model,
                rag_context=rag_context,
            ):
                yield line

        return StreamingResponse(stream_cloud(), media_type="application/x-ndjson")

    # ── OLLAMA PATH (with tool calling) ─────────────────────
    if rag_context:
        user_content = f"{rag_context}\n{req.message}"
    else:
        user_content = req.message

    messages = [{"role": "user", "content": user_content}]

    async def stream_local():
        nonlocal messages

        yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

        if rag_context:
            chunk_count = rag_context.count("[Source:")
            yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

        rounds = 0
        async with httpx.AsyncClient(timeout=120) as client:
            while rounds < MAX_TOOL_ROUNDS:
                rounds += 1
                payload = {
                    "model": decision.model,
                    "messages": messages,
                    "tools": TOOLS,
                    "stream": False,
                }
                try:
                    resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                    data = resp.json()
                except Exception as e:
                    yield json.dumps({"t": "error", "c": f"Backend error: {e}"}) + "\n"
                    yield json.dumps({"t": "done"}) + "\n"
                    return

                msg = data.get("message", {})
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "")

                if not tool_calls and content:
                    parsed = try_parse_text_tool_call(content)
                    if parsed:
                        name, args = parsed
                        tool_calls = [{"function": {"name": name, "arguments": args}}]
                        msg["tool_calls"] = tool_calls
                        msg["content"] = ""

                if not tool_calls:
                    if content:
                        chunk_size = 4
                        for i in range(0, len(content), chunk_size):
                            yield json.dumps({"t": "text", "c": content[i:i+chunk_size]}) + "\n"
                    yield json.dumps({"t": "done"}) + "\n"
                    return

                messages.append(msg)
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", {})
                    yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"
                    result = execute_tool(name, args)
                    summary = result[:300].replace("\n", " ").replace("\r", "")
                    if len(result) > 300:
                        summary += "..."
                    yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"
                    messages.append({"role": "tool", "content": result})

            payload_final = {"model": decision.model, "messages": messages, "stream": False}
            try:
                resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload_final)
                data = resp.json()
                content = data.get("message", {}).get("content", "Max tool rounds reached.")
                for i in range(0, len(content), 4):
                    yield json.dumps({"t": "text", "c": content[i:i+4]}) + "\n"
            except Exception as e:
                yield json.dumps({"t": "error", "c": str(e)}) + "\n"
            yield json.dumps({"t": "done"}) + "\n"

    return StreamingResponse(stream_local(), media_type="application/x-ndjson")


# ── ROUTER CONFIG ENDPOINTS ────────────────────────────────────────────

@app.get("/router/status")
def router_status():
    """Get current routing configuration and status."""
    return {
        "cloud_enabled": os.getenv("CLOUD_ENABLED", "true").lower() == "true",
        "cloud_available": is_cloud_available(),
        "gpu_utilization": get_gpu_utilization(),
        "gpu_threshold": int(os.getenv("GPU_THRESHOLD", "70")),
        "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "sonnet_model": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514"),
        "opus_model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-20250514"),
        "local_model": os.getenv("E3N_MODEL", "e3n"),
        "strong_model": os.getenv("E3N_STRONG_MODEL", ""),
    }

class CloudToggle(BaseModel):
    enabled: bool

@app.post("/router/cloud")
def toggle_cloud(req: CloudToggle):
    """Toggle cloud on/off at runtime."""
    os.environ["CLOUD_ENABLED"] = str(req.enabled).lower()
    return {"cloud_enabled": req.enabled}


# ── MEMORY ENDPOINTS ────────────────────────────────────────────────────

@app.get("/memory/stats")
def memory_stats_endpoint():
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return get_memory_stats()
    except Exception as e:
        return {"error": str(e)}

@app.post("/memory/ingest")
def memory_ingest_endpoint():
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        results = ingest_all()
        return {
            "results": results,
            "ingested": len([r for r in results if r["status"] == "ok"]),
            "skipped": len([r for r in results if r["status"] == "skipped"]),
            "errors": len([r for r in results if r["status"] == "error"]),
        }
    except Exception as e:
        return {"error": str(e)}


# ── STATS ───────────────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    result = {}
    result["cpu"] = {
        "name": platform.processor() or "Intel i7-12700K",
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "percent": psutil.cpu_percent(interval=None),
        "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
    }
    ram = psutil.virtual_memory()
    result["ram"] = {
        "total_gb": round(ram.total / 1e9, 1),
        "used_gb": round(ram.used / 1e9, 1),
        "percent": ram.percent,
    }
    disk = psutil.disk_usage("C:\\")
    result["disk"] = {
        "total_gb": round(disk.total / 1e9, 1),
        "used_gb": round(disk.used / 1e9, 1),
        "percent": disk.percent,
    }
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name = pynvml.nvmlDeviceGetName(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            except:
                power_limit = 170
            result["gpu"] = {
                "name": name if isinstance(name, str) else name.decode(),
                "vram_total_mb": round(mem.total / 1e6),
                "vram_used_mb": round(mem.used / 1e6),
                "vram_percent": round(mem.used / mem.total * 100, 1),
                "gpu_percent": util.gpu,
                "power_w": round(power, 1),
                "power_limit_w": round(power_limit, 1),
                "temp_c": temp,
            }
        except Exception as e:
            result["gpu"] = {"error": str(e)}
    else:
        result["gpu"] = {"error": "NVML unavailable"}
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            tags = json.loads(r.read())
            result["models"] = [m["name"] for m in tags.get("models", [])]
    except:
        result["models"] = []
    if RAG_AVAILABLE:
        try:
            ms = get_memory_stats()
            result["memory_mb"] = ms["disk_mb"]
            result["memory_chunks"] = ms["total_chunks"]
            result["memory_files"] = ms["total_files"]
        except:
            result["memory_mb"] = 0
            result["memory_chunks"] = 0
            result["memory_files"] = 0
    else:
        try:
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(os.getenv("CHROMADB_PATH", "C:\\e3n\\data\\chromadb"))
                for f in fn
            )
            result["memory_mb"] = round(total / 1e6, 1)
        except:
            result["memory_mb"] = 0
    result["model"] = E3N_MODEL
    result["platform"] = platform.system() + " " + platform.release()
    result["rag_available"] = RAG_AVAILABLE
    result["cloud_available"] = is_cloud_available()
    return result


# ── MODELFILE ───────────────────────────────────────────────────────────

MODELFILE_PATH = r"C:\e3n\modelfiles\E3N.modelfile"
MODULES_DIR = os.path.join(_HERE, '..', 'modelfiles')
MODULE_FILES = {
    "modelfile": MODELFILE_PATH,
    "protocols": os.path.join(MODULES_DIR, "protocols.md"),
    "personality": os.path.join(MODULES_DIR, "personality.md"),
    "pilot": os.path.join(MODULES_DIR, "pilot.md"),
}

@app.get("/modelfile")
def get_modelfile():
    try:
        with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        return PlainTextResponse(f"# Error: {e}")

class ModelfileUpdate(BaseModel):
    content: str

@app.post("/modelfile")
def save_modelfile(update: ModelfileUpdate):
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/module/{name}")
def get_module(name: str):
    path = MODULE_FILES.get(name)
    if not path:
        return PlainTextResponse(f"# Unknown module: {name}", status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        return PlainTextResponse(f"# Error reading {name}: {e}")

@app.post("/module/{name}")
def save_module(name: str, update: ModelfileUpdate):
    path = MODULE_FILES.get(name)
    if not path:
        return {"ok": False, "error": f"Unknown module: {name}"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
