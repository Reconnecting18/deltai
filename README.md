<p align="center">
  <img src="https://img.shields.io/badge/status-operational-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/platform-Windows%2011-0078D4?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/python-3.14-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
</p>

# E3N

**Local AI system with VRAM-aware routing, RAG memory, and a training pipeline — built to serve as a race engineer brain for Le Mans Ultimate.**

E3N is the intelligence layer: reasoning, memory, tool execution, and model routing. External services (like a future Telemetry API) connect via the `/ingest` endpoint to push context into its RAG memory. Named after E3N from COD: Infinite Warfare — dry wit, loyal, mission-focused — blended with BT-7274's precision.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Electron (frameless)          app/main.js                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Dashboard UI               static/index.html (~60KB) │  │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │ Health   │ │ Terminal │ │ Widgets  │ │ WS Alert │  │  │
│  │  │ Monitor  │ │ + Chat   │ │ + Graphs │ │ Toasts   │  │  │
│  │  └─────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │ NDJSON streaming
┌──────────────────────────┼──────────────────────────────────┐
│  FastAPI Backend          main.py :8000                     │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │ Router │ │ RAG      │ │ Tools    │ │ Training        │  │
│  │ VRAM-  │ │ ChromaDB │ │ 7 core + │ │ QLoRA + GGUF    │  │
│  │ aware  │ │ + nomic  │ │ 4 telem  │ │ + A/B eval      │  │
│  └───┬────┘ └──────────┘ └──────────┘ └─────────────────┘  │
│      │      ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│      │      │ Persist  │ │ Voice    │ │ Session Mode    │  │
│      │      │ SQLite   │ │ STT/TTS  │ │ GPU Protection  │  │
│      │      └──────────┘ └──────────┘ └─────────────────┘  │
└──────┼──────────────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────────────┐
│  Ollama                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ e3n-qwen14b  │  │ e3n-qwen3b   │  │ e3n-nemo / e3n    │  │
│  │ Tier A (14B) │  │ Tier B/C (3B)│  │ Emergency backup  │  │
│  │ >9GB VRAM    │  │ 3-9GB / CPU  │  │ Last resort only  │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### External Services → E3N

```
Telemetry API ──POST /ingest──►  E3N ChromaDB (RAG)
(future)        POST /ingest/batch
                tags: ["alert"] ──► WebSocket → Dashboard toast
```

E3N does **not** process telemetry, UDP packets, or game data directly. External services push summarized context in, and RAG pulls it when the user asks a question.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3060 12GB |
| CPU | Intel i7-12700K |
| RAM | 34GB DDR4 |
| OS | Windows 11 |

---

## Quick Start

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- [Node.js](https://nodejs.org) (for Electron)
- Python 3.12+ with venv

### Setup

```powershell
# 1. Pull models and create E3N variants
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama create e3n-qwen14b -f C:\e3n\modelfiles\E3N-qwen14b.modelfile
ollama create e3n-qwen3b -f C:\e3n\modelfiles\E3N-qwen3b.modelfile

# 2. Python backend
cd C:\e3n\project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt    # or install manually — see below

# 3. Electron app
cd C:\e3n\app
npm install
```

### Run

```powershell
# Terminal 1 — Backend
cd C:\e3n\project
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Desktop app
cd C:\e3n\app
npm start
```

Or open a browser to `http://localhost:8000` for the dashboard without Electron.

### Configuration

Copy and edit `C:\e3n\project\.env`:

```env
# Models
OLLAMA_URL=http://localhost:11434
E3N_MODEL=e3n-qwen14b
E3N_SMALL_MODEL=e3n-qwen3b

# Cloud (optional — dormant until key is set)
ANTHROPIC_API_KEY=
CLOUD_BUDGET_DAILY=5.00

# Session / Racing
SESSION_GPU_PROTECT=true
SESSION_FORCE_CLOUD=true
SESSION_SOURCE_PATTERN=lmu
SESSION_TIMEOUT_SEC=60

# Voice
VOICE_ENABLED=true
TTS_VOICE=en-US-GuyNeural

# Training
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LLAMA_CPP_PATH=C:\e3n\tools\llama.cpp
```

---

## Key Features

### VRAM-Aware Model Routing
The router detects available VRAM in real-time and selects the best model:

| Tier | VRAM Free | Model | Use Case |
|------|-----------|-------|----------|
| A | > 9GB | Qwen2.5-14B (Q4_K_M) | Full power — no sim running |
| B | 3–9GB | Qwen2.5-3B on GPU | Sim running, GPU shared |
| C | < 3GB | Qwen2.5-3B on CPU | VRAM critical |

Sim process detection via psutil. On LMU launch → unload large model, preload 3B. On sim close → swap back to 14B within 30 seconds.

### Session Mode & GPU Protection
When actively racing, **zero GPU inference** — protects frame rate at 250+ km/h:
- Cloud available → routes to Anthropic API
- No cloud → CPU-only local (slower but zero GPU impact)
- LoRA training blocked during sessions
- Auto-activates from `/ingest` source pattern matching

### RAG Memory
ChromaDB with nomic-embed-text embeddings. Multi-query expansion generates 2–3 search variants, results are deduplicated and reranked with source grouping and recency bias.

- **Knowledge base**: Drop files in `data/knowledge/` — watchdog auto-ingests
- **Ingest connector**: External services push context via `POST /ingest` with TTL and source tags
- **Live data**: Source and age filtering for real-time telemetry queries
- **Batch ingest**: Single embedding call for up to 100 items

### Telemetry Query Classification
During active sessions, queries are classified into 4 categories with tailored prompt templates:

| Category | Example | Behavior |
|----------|---------|----------|
| `telemetry_lookup` | "fuel remaining" | Short-circuit from RAG — no LLM |
| `telemetry_coaching` | "why am I slow in S3" | Coach template (observation → impact → action) |
| `telemetry_strategy` | "should I pit now" | Engineer template (recommendation + numbers) |
| `telemetry_debrief` | "full race analysis" | Deep analysis, no time pressure |

### Training Pipeline
Progressive improvement loop: **use E3N → auto-capture → accumulate dataset → QLoRA fine-tune → deploy → repeat.**

- **QLoRA**: 4-bit training on Qwen2.5-3B (~6-7GB VRAM) via PEFT/TRL
- **GGUF export**: Merge adapter → convert to GGUF Q4_K_M → register in Ollama
- **A/B evaluation**: Compare two models on a dataset with latency + quality metrics
- **Auto-capture**: Good exchanges automatically saved; racing exchanges routed to `e3n-racing` dataset with RAG context
- **Safety**: Blocked during racing sessions or when sim is running; cancellable mid-training

### Voice Module
- **STT**: faster-whisper (CTranslate2) — GPU or CPU, VRAM-aware
- **TTS**: edge-tts (Microsoft Edge API) with Windows SAPI fallback
- **Full loop**: `POST /voice/chat` — audio in → transcribe → chat → synthesize → audio out

### Emergency Backup
Last-resort failover chain if the primary model fails:
- `e3n-qwen14b` → `e3n-nemo` (Mistral Nemo 12B) → `e3n` (LLaMA 3.1 8B)
- Background health pings every hour; invisible during normal operation

---

## Project Structure

```
C:\e3n\
├── app\                          Electron desktop app
│   └── main.js                   Frameless window + IPC
├── project\                      FastAPI backend
│   ├── main.py                   Chat, ingest, session, training, voice endpoints
│   ├── router.py                 VRAM detection, sim detection, tier routing, session mode
│   ├── memory.py                 ChromaDB RAG — embed, query, ingest, batch, TTL cleanup
│   ├── persistence.py            SQLite — conversation history, budget, session export
│   ├── anthropic_client.py       Cloud inference with tool use (dormant until API key)
│   ├── training.py               QLoRA fine-tuning, dataset CRUD, A/B eval, auto-capture
│   ├── voice.py                  STT (faster-whisper) + TTS (edge-tts)
│   ├── watcher.py                Watchdog file watcher for knowledge dir
│   ├── tools/
│   │   ├── definitions.py        7 core + 4 conditional telemetry tool schemas
│   │   └── executor.py           Tool execution with retry + telemetry API calls
│   ├── static/
│   │   └── index.html            Full dashboard UI (single file — CSS + HTML + JS)
│   ├── tests/
│   │   └── verify_full.py        33-test verification suite
│   └── .env                      Configuration (not committed)
├── modelfiles\                   Ollama modelfiles (identical prompts, different FROM)
│   ├── E3N-qwen14b.modelfile
│   ├── E3N-qwen3b.modelfile
│   ├── E3N-nemo.modelfile
│   └── E3N.modelfile
├── data\
│   ├── chromadb\                 Vector store (not committed)
│   ├── sqlite\                   Persistent state (not committed)
│   ├── knowledge\                Drop files here for RAG ingestion
│   └── training\                 Datasets, adapters, GGUF exports, eval results
├── tools\
│   └── llama.cpp\                GGUF conversion toolchain
└── CLAUDE.md                     Full project context for AI sessions
```

---

## API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Streaming chat (NDJSON) — routes automatically |
| GET | `/chat/history` | Conversation history metadata |
| DELETE | `/chat/history` | Clear conversation history |

### Ingest
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Push context into RAG (source, context, TTL, tags) |
| POST | `/ingest/batch` | Batch ingest up to 100 items |

### Session
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/session/start` | Manually activate session mode |
| POST | `/session/end` | Deactivate + export session history |
| GET | `/session/status` | Current session state |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/training/datasets` | List all datasets |
| POST | `/training/datasets` | Create dataset |
| POST | `/training/datasets/{name}/add` | Add training example |
| POST | `/training/datasets/{name}/export` | Export (alpaca/sharegpt/chatml) |
| POST | `/training/start` | Start fine-tuning (lora/fewshot/auto) |
| POST | `/training/stop` | Cancel training |
| GET | `/training/status` | Progress, loss, step count |
| POST | `/training/eval/ab` | A/B model comparison |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | 8–9 subsystem health check |
| GET | `/stats` | System stats + model info + budget |
| GET | `/budget/status` | Cloud spend tracking |
| GET | `/backup/status` | Emergency backup diagnostics |
| WS | `/ws/alerts` | Real-time alert notifications |

### Voice
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/voice/stt` | Audio → transcription |
| POST | `/voice/tts` | Text → audio |
| POST | `/voice/chat` | Full voice loop (audio in → audio out) |
| GET | `/voice/status` | STT/TTS subsystem health |

---

## Build Phases

| Phase | Status | Scope |
|-------|--------|-------|
| 1 | **Complete** | Dashboard, RAG, tools, router, cloud client |
| 2 | **Complete** | VRAM-aware routing, Qwen migration, sim detection, /ingest, emergency backup |
| 3 | **Complete** | Cloud tool-use, split workload, cost budget, conversation history, SQLite persistence |
| 4 | **Complete** | Smarter RAG, training pipeline, text-as-tool hardening, error recovery, voice module |
| 5 | **Complete** | Telemetry prep — session mode, GPU protection, query classification, racing prompts, WebSocket alerts, batch ingest, racing auto-capture |
| — | **Separate project** | Telemetry API for Le Mans Ultimate — connects to E3N via /ingest |

---

## Dashboard

Tactical operations center aesthetic with a muted grey-green palette. Features include:

- **Header**: Subsystem health monitor (X/8 ONLINE) + cloud budget display
- **3D particle sphere**: Network node map with labeled subsystems
- **Terminal**: Streaming chat with conversation history (CLR + turn counter)
- **Live widgets**: GPU/CPU/RAM stats with 60s sparklines
- **WebSocket toasts**: Real-time alert notifications with priority coloring
- **Drag-and-drop**: Alt+drag widgets between panels

---

## Verification

Run the full test suite (33 tests across all subsystems):

```powershell
cd C:\e3n\project
.\venv\Scripts\activate
python tests/verify_full.py
```

---

## License

Private project.

*Built by Ethan, 2025–2026.*
