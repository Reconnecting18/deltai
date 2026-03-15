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

app = FastAPI(title="E3N")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
E3N_MODEL  = os.getenv("E3N_MODEL", "e3n")

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    deep: bool = False

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    model = os.getenv("E3N_DEEP_MODEL", E3N_MODEL) if req.deep else E3N_MODEL
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

    chroma_path = os.getenv("CHROMADB_PATH", "C:\\e3n\\data\\chromadb")
    try:
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(chroma_path)
            for f in fn
        )
        result["memory_mb"] = round(total / 1e6, 1)
    except:
        result["memory_mb"] = 0

    result["model"] = E3N_MODEL
    result["platform"] = platform.system() + " " + platform.release()

    return result


MODELFILE_PATH = r"C:\e3n\modelfiles\E3N.modelfile"

@app.get("/modelfile")
def get_modelfile():
    try:
        with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return PlainTextResponse(content)
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
