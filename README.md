# E3N — Personal AI Assistant

A local JARVIS-style AI desktop application built on Ollama + llama3.1:8b, with a real-time system dashboard and COD Infinite Warfare aesthetic.

---

## Stack

| Layer | Tech |
|---|---|
| UI | Electron (frameless) + vanilla JS |
| Backend | FastAPI (Python) |
| AI inference | Ollama → llama3.1:8b (`e3n` modelfile) |
| Monitoring | pynvml + psutil |
| Memory (Phase 2) | ChromaDB + RAG |
| Racing coach (Phase 3) | UDP telemetry — Le Mans Ultimate port 20777 |

---

## Hardware

- GPU: RTX 3060 12GB
- CPU: i7-12700K (12P / 20 threads)
- RAM: 34.1 GB
- OS: Windows 11

---

## Project Structure

```
C:\e3n\
├── app\                    ← Electron desktop app
│   ├── main.js             ← Window + IPC + backend spawn
│   ├── preload.js
│   └── package.json
├── project\                ← FastAPI backend
│   ├── main.py             ← /chat /stats /modelfile endpoints
│   ├── .env                ← NOT committed (OLLAMA_URL, E3N_MODEL)
│   ├── launch.py
│   └── static\
│       └── index.html      ← Full dashboard UI (single file)
├── modelfiles\
│   ├── E3N.modelfile       ← Active Ollama personality
│   └── E3N_personality.txt ← Human-readable character reference
├── data\
│   ├── chromadb\           ← Phase 2 (not committed)
│   ├── sqlite\             ← Phase 3 (not committed)
│   └── knowledge\          ← Drop PDFs/docs here for RAG ingestion
├── E3N_STATUS.md           ← Living project doc
└── .gitignore
```

**Ollama models** are stored at `C:\Users\ethan\.ollama\models\` — not in this repo.

---

## Running E3N

```powershell
cd C:\e3n\app
npm start
```

Or double-click `C:\e3n\Launch E3N.bat`

**Backend only (browser):**
```powershell
cd C:\e3n\project
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

---

## First-Time Setup (new machine)

```powershell
# 1. Install Ollama, then pull model
ollama pull llama3.1:8b
ollama create e3n -f C:\e3n\modelfiles\E3N.modelfile

# 2. Python backend
cd C:\e3n\project
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn httpx python-dotenv psutil pynvml

# 3. Electron app
cd C:\e3n\app
npm install
npm start
```

Create `C:\e3n\project\.env`:
```
OLLAMA_URL=http://localhost:11434
E3N_MODEL=e3n
```

---

## Dashboard Features

- **3D particle sphere** — 280-node rotating globe, cursor repulsion, health-reactive brightness
- **Live stats** — GPU util/VRAM/power/temp, CPU%, RAM, disk
- **Performance graphs** — 60s sparklines for GPU, CPU, RAM
- **Temperature panel** — GPU temp with NORMAL/WARM/HOT status
- **Floating windows** — Terminal and Memory as draggable/resizable overlays
- **Widget drag** — Alt+drag to rearrange blocks between left/right panels
- **Idle/Active states** — sphere brightens when AI is responding

---

## E3N Personality

Character is a blend of:
- **E3N "Ethan"** (COD: Infinite Warfare) — dry wit, loyal, mission-focused
- **BT-7274** (Titanfall 2) — precise, literal, never softens facts

Rules: lead with the answer, no filler phrases, never start with "I", humor only when relaxed.

Edit personality: open Memory widget → edit modelfile → Save & Rebuild → run `ollama create e3n -f C:\e3n\modelfiles\E3N.modelfile`

---

## Roadmap

### ✅ Phase 1 — Foundation (complete)
- Ollama + custom modelfile
- FastAPI backend with streaming chat
- Electron frameless desktop app
- Live system monitoring dashboard
- 3D particle entity

### 🔲 Phase 2 — Memory
- ChromaDB vector store
- File watcher on `data/knowledge/`
- RAG injection into every chat prompt
- E3N references your documents and notes

### 🔲 Phase 3 — Racing Coach
- UDP listener port 20777 (Le Mans Ultimate)
- SQLite lap logger
- Post-lap AI coaching via E3N

### 🔲 Phase 4 — Remote Access
- Tailscale for phone/remote access
- Nightly S3 backup of data/

---

## Context for Claude Sessions

This project is built and iterated entirely through Claude (Anthropic) using the Claude Projects feature. Each session should reference this README and `E3N_STATUS.md` for current state. The full build log is in the Claude Project transcript.

**Key decisions made:**
- Python for backend (not C++) — C++ reserved for future low-level inference work
- Electron over pywebview — Python 3.14 breaks pythonnet
- Ollama ignores `OLLAMA_MODELS` env var on this machine — models stay at default path
- Single `index.html` for entire UI — no bundler/framework

---

*Private project — Ethan, 2025*
