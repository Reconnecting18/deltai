from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import json
import os
import psutil
import platform
from dotenv import load_dotenv

load_dotenv()
psutil.cpu_percent(interval=0.1)  # prime the cpu measurement on startup

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

app = FastAPI(title="deltai")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DELTAI_MODEL  = os.getenv("DELTAI_MODEL", "deltai")

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")

class ChatRequest(BaseModel):
    message: str
    deep: bool = False

@app.get("/")
def root():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))

@app.post("/chat")
async def chat(req: ChatRequest):
    model = os.getenv("DELTAI_DEEP_MODEL", DELTAI_MODEL) if req.deep else DELTAI_MODEL
    payload = {
        "model": model,
        "prompt": req.message,
        "stream": True,
    }
    async def stream_response():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST",
                f"{OLLAMA_URL}/api/generate", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if token := data.get("response"):
                            yield token
                        if data.get("done"):
                            break
    return StreamingResponse(stream_response(), media_type="text/plain")

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
        "used_gb":  round(ram.used  / 1e9, 1),
        "percent":  ram.percent,
    }

    disk = psutil.disk_usage("C:\\")
    result["disk"] = {
        "total_gb": round(disk.total / 1e9, 1),
        "used_gb":  round(disk.used  / 1e9, 1),
        "percent":  disk.percent,
    }

    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name   = pynvml.nvmlDeviceGetName(handle)
            temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            except:
                power_limit = 170
            result["gpu"] = {
                "name":          name if isinstance(name, str) else name.decode(),
                "vram_total_mb": round(mem.total / 1e6),
                "vram_used_mb":  round(mem.used  / 1e6),
                "vram_percent":  round(mem.used / mem.total * 100, 1),
                "gpu_percent":   util.gpu,
                "power_w":       round(power, 1),
                "power_limit_w": round(power_limit, 1),
                "temp_c":        temp,
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

    chroma_path = os.getenv("CHROMADB_PATH", "~/deltai/data\\chromadb")
    try:
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(chroma_path)
            for f in fn
        )
        result["memory_mb"] = round(total / 1e6, 1)
    except:
        result["memory_mb"] = 0

    result["model"] = DELTAI_MODEL
    result["platform"] = platform.system() + " " + platform.release()

    return result


MODELFILE_PATH = r"~/deltai/modelfiles\deltai.modelfile"
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
        return PlainTextResponse(f"# Error reading modelfile: {e}")

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
