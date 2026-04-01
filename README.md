<p align="center">
  <img src="https://img.shields.io/badge/status-operational-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/platform-Windows%2011-0078D4?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/python-3.14-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Ollama-Qwen2.5-FF6F00?style=flat-square" alt="Ollama">
</p>

# E3N

**Local AI system with VRAM-aware routing, RAG memory, structured reasoning, and a training pipeline — built to serve as a race engineer brain for Le Mans Ultimate.**

E3N is the intelligence layer: reasoning, memory, tool execution, and model routing. External services (like a future Telemetry API) connect via the `/ingest` endpoint to push context into its RAG memory. Named after E3N from COD: Infinite Warfare and BT-7274 from Titanfall 2 — warm, loyal, confident, with a distinct voice. Not a chatbot. A teammate.

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
│  │ VRAM + │ │ ChromaDB │ │ 7 core + │ │ QLoRA + GGUF    │  │
│  │ Quant  │ │ + Cold   │ │ 4 telem  │ │ + Adapters      │  │
│  └───┬────┘ └──────────┘ └──────────┘ └─────────────────┘  │
│      │      ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│      │      │ Persist  │ │ Voice    │ │ ReAct + Ingest  │  │
│      │      │ SQLite   │ │ STT/TTS  │ │ Pipeline        │  │
│      │      └──────────┘ └──────────┘ └─────────────────┘  │
└──────┼──────────────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────────────┐
│  Ollama (dynamic quantization + partial GPU offloading)     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ e3n-qwen14b  │  │ e3n-qwen3b   │  │ e3n-nemo / e3n    │  │
│  │ Tier A (14B) │  │ Tier B/C (3B)│  │ Emergency backup  │  │
│  │ Q6/Q4/Q3     │  │ Q4/Q2        │  │ Last resort only  │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### External Services → E3N

```
Telemetry API ──POST /ingest──►  Async Queue → Batch Embed → ChromaDB (RAG)
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

## AI-assisted development

This repo is set up for **Cursor** and **Claude Code** (and any tool that reads markdown context):

| Artifact | Role |
|----------|------|
| [AGENTS.md](AGENTS.md) | Short onboarding for agents (boundaries, paths, verify commands); entry point for Cursor |
| [CLAUDE.md](CLAUDE.md) | Full architecture, stream protocol, file map, status, and workflow |
| [`.cursor/rules/`](.cursor/rules/) | Small Cursor rules (`.mdc`) — scope, UI, and backend test reminders |
| [`.vscode/launch.json`](.vscode/launch.json) | Debug/run FastAPI (debugpy) and Electron from the **workspace folder** (Cursor / VS Code) |
| [`.vscode/tasks.json`](.vscode/tasks.json) | Shell tasks for uvicorn and `npm start` (Windows and Unix venv paths) |
| [`.claude/launch.json`](.claude/launch.json) | Claude Code launch definitions using `${workspaceFolder}` (venv path targets **Windows** `Scripts\python.exe`; use the “python on PATH” entry or edit paths on Linux/macOS) |

When you change behavior that affects onboarding or verification, update **AGENTS.md** and **CLAUDE.md** as needed. Contributor expectations: [CONTRIBUTING.md](CONTRIBUTING.md).

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
pip install -r requirements.txt

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

# Voice
VOICE_ENABLED=true
TTS_VOICE=en-US-AndrewNeural

# Training
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LLAMA_CPP_PATH=C:\e3n\tools\llama.cpp

# Phase 8 — Advanced Intelligence & Resource Optimization
REACT_ENABLED=true              # ReAct reasoning loop for complex queries
REACT_MAX_ITERATIONS=3          # Max think-act-observe cycles
RAR_ENABLED=true                # Iterative RAG (two-round retrieval)
DEDUP_THRESHOLD=0.15            # Semantic dedup cosine distance threshold
MIXTURE_LORA_ENABLED=false      # Dynamic LoRA adapter routing (needs trained adapters)
VRAM_TIER_AB_MIN_MB=5000        # Partial GPU offload threshold
WARM_TO_COLD_AGE_SEC=86400      # Age before demoting to cold storage (24h)
INGEST_QUEUE_MAX=500            # Async ingest pipeline queue size
INGEST_FLUSH_INTERVAL=2.0       # Seconds between batch flushes

# Phase 9 — Progressive Intelligence
REASONING_TRACE_ENABLED=true    # Persist ReAct reasoning chains for reuse
REASONING_TRACE_MAX=500         # Max stored traces
QUALITY_CAPTURE_THRESHOLD=0.6   # Min quality score for auto-capture
REACT_ALLOW_CLARIFY=true        # Allow ReAct to ask clarifying questions
SMART_CAPTURE_ENABLED=true      # Quality-tiered capture with dedup
SMART_HISTORY_ENABLED=true      # Intelligent context window compression
KNOWLEDGE_GAP_DETECTION=true    # Log unanswered queries as knowledge gaps

# Phase 10 — Continuous Intelligence
DAILY_TRAIN_ENABLED=true        # Run daily training cycle at 2 AM
DAILY_TRAIN_MIN_VRAM_MB=7000    # Min free VRAM to proceed with training
DAILY_TRAIN_AUTO_PROMOTE=false  # Auto-promote better adapters after eval
DAILY_TRAIN_AUTO_MERGE=false    # Auto-TIES merge after promotion
SESSION_SYNTHESIS_ENABLED=true  # Synthesize session knowledge at session end
SESSION_SYNTHESIS_MODEL=local14b # Model for synthesis (local14b or local3b)
DPO_ENABLED=false               # DPO training (enable when negatives accumulate)
```

---

## Key Features

### Intelligent Model Routing

The router detects available VRAM in real-time and selects the optimal model, quantization level, and GPU layer count:

| Tier | VRAM Free | Model | Strategy |
|------|-----------|-------|----------|
| A | > 9GB | Qwen2.5-14B Q6_K/Q4_K_M | Full GPU — best quality |
| AB | 5–9GB | Qwen2.5-14B (partial offload) | N layers on GPU, rest in RAM |
| B | 3–5GB | Qwen2.5-3B Q4_K_M on GPU | Sim running, GPU shared |
| C | < 3GB | Qwen2.5-3B Q2_K/CPU | VRAM critical — emergency |

**Dynamic quantization**: The router selects from pre-registered quantization variants (Q6_K → Q4_K_M → Q3_K_M → Q2_K) based on available VRAM, providing smooth quality degradation instead of cliff edges.

**Partial GPU offloading**: For the Tier AB zone (5–9GB), the router calculates the optimal `num_gpu` — putting as many transformer layers on GPU as fit while spilling the rest to RAM. This gives 50–70% GPU speed at ~4GB VRAM instead of 8.5GB.

**Dynamic context window**: `num_ctx` scales automatically with available VRAM (4096 → 2048 → 1024) to reduce KV cache memory usage under pressure.

### ReAct Structured Reasoning

For complex queries (Tier 2/3 complexity), E3N enters a structured **Think → Act → Observe → Confidence** reasoning loop:

1. **THINK**: Identify what information is needed
2. **ACT**: Call tools, query RAG, run calculations
3. **OBSERVE**: Analyze results, check for gaps
4. **CONFIDENCE**: Rate HIGH / MEDIUM / LOW — drives next action
5. **FINAL**: Synthesize complete answer

Up to 3 iterations per query. Dramatically improves multi-step problem solving on local models without any model changes — pure prompt engineering.

**Reasoning trace memory**: Past ReAct chains are persisted in SQLite and retrieved by embedding similarity. When E3N encounters a problem similar to one it solved before, it injects the prior reasoning as context — avoiding redundant work.

**Confidence protocol**: HIGH confidence → immediate answer. MEDIUM → answer with caveat about what's missing. LOW → asks the user a clarifying question (`CLARIFY:` event) before proceeding.

### Progressive Intelligence

E3N gets smarter over time through feedback loops:

- **Quality scoring**: Every response scored 0.0–1.0 on 6 signals (specificity, tool success, structural match, etc.)
- **Adaptive routing**: Router learns from quality scores which model/tier works best per domain — soft ±1 tier adjustment
- **Smart auto-capture**: Quality-tiered training data collection with dedup and negative examples for DPO
- **Tool filtering**: 19 tools pre-filtered to 5–8 relevant tools per query based on domain — saves ~1000 prompt tokens
- **Knowledge gap detection**: Failed queries logged as gaps, surfaced at `/knowledge/gaps` for review
- **Iterative distillation**: Identify weak domains → generate targeted teacher data → retrain → evaluate → promote
- **Smart history**: Context window compression (full/compressed/summary) based on token budget

### RAG Memory

ChromaDB with nomic-embed-text embeddings. Three-tier hierarchical storage with intelligent retrieval:

- **Iterative retrieval**: Two-round RAG — first round retrieves, second round fills knowledge gaps with model-directed sub-queries
- **Semantic deduplication**: Near-duplicate chunks detected before ingest (cosine < 0.15) — metadata freshened instead of duplicating
- **Hierarchical storage**: Hot (ChromaDB in-memory) → Warm (ChromaDB persistent, <24h) → Cold (SQLite + zlib compression, >24h)
- **Cold tier search**: Brute-force cosine fallback when hot/warm returns insufficient results
- **Knowledge base**: Drop files in `data/knowledge/` — watchdog auto-ingests
- **Async ingest pipeline**: Queue-based non-blocking ingestion with batch embedding (2s flush interval)

### Adaptive Resource Management

The resource self-manager runs every 30 seconds with predictive intelligence:

- **Predictive VRAM**: 60-second sliding window tracks VRAM trend — preemptive model unload when declining >100MB/s
- **Thermal throttling**: GPU temperature monitoring via pynvml — auto-unloads large models above 80°C before GPU self-throttles
- **OS process priority**: Automatically lowers E3N + Ollama to `BELOW_NORMAL_PRIORITY_CLASS` during racing or VRAM pressure, restores when idle
- **Proactive model lifecycle**: Auto-unload 14B on sim start, preload 3B; reverse on sim stop after 30s cooldown
- **Circuit breaker**: 3-failure threshold on Ollama with exponential backoff (5s → 60s), half-open recovery
- **Memory compaction**: Background warm→cold tier demotion every ~10 minutes

### Mixture-of-LoRA Adapter Routing

Instead of statically merging all domain adapters (TIES), E3N can dynamically route queries to the best single LoRA adapter:

| Domain | Adapter | Purpose |
|--------|---------|---------|
| Racing | r=16, 50% frozen | Tire strategy, telemetry, driving technique |
| Engineering | r=16, 33% frozen | Physics, calculus, statics, thermo |
| Personality | r=8, 67% frozen | E3N voice and response style |
| Reasoning | r=32, 22% frozen | Chain-of-thought, analysis |

Query domain classification routes to the best adapter at full fidelity — no merge quality loss.

### Streaming Ingest Pipeline

High-throughput async ingest with backpressure:

```
POST /ingest → Queue (500 max) → Batch Worker (every 2s or 10 items) → Embed → Dedup → ChromaDB
                                                                              ↓
                                                                     Alert → WebSocket
```

- **Non-blocking**: `/ingest` returns immediately after queuing
- **Backpressure**: Returns `429` when queue is full
- **Metrics**: Queue depth, processed count, average latency via `/ingest/pipeline/status`

### Training Pipeline

Two improvement paths: **RAG knowledge enrichment** (preferred for 3B) and **QLoRA fine-tuning** (for larger datasets).

- **Knowledge enrichment**: 8 reference documents (143 expert examples) — ingested into ChromaDB, retrieved via iterative RAG at query time
- **QLoRA**: 4-bit training on Qwen2.5-3B (~6-7GB VRAM) via PEFT/TRL — full pipeline verified end-to-end
- **Adapter surgery**: 4 domain slots (racing/engineering/personality/reasoning) with selective layer freezing and TIES merge
- **A/B evaluation**: Compare two models on a dataset with latency + quality metrics
- **Auto-capture**: Good exchanges automatically saved to training datasets

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
<repo-root>\
├── AGENTS.md                     Short context for Cursor / agents
├── CLAUDE.md                     Full project context for AI sessions
├── CONTRIBUTING.md               How to file issues and PRs
├── .cursor\rules\                Cursor project rules (.mdc)
├── .github\ISSUE_TEMPLATE\       GitHub issue forms
├── .vscode\                      launch.json + tasks.json (workspace-relative)
├── .claude\                      Claude Code launch config
├── app\                          Electron desktop app
│   └── main.js                   Frameless window + IPC
├── project\                      FastAPI backend
│   ├── main.py                   Chat, ingest pipeline, ReAct loop, resource manager
│   ├── router.py                 VRAM detection, dynamic quant, partial offload, adaptive routing
│   ├── memory.py                 Hierarchical RAG — hot/warm/cold, iterative retrieval, dedup
│   ├── quality.py                Response quality scoring engine (6-signal heuristic)
│   ├── persistence.py            SQLite — history, budget, traces, quality, routing, gaps
│   ├── anthropic_client.py       Cloud inference with tool use (dormant until API key)
│   ├── training.py               QLoRA, adapter surgery, TIES merge, dataset CRUD, A/B eval
│   ├── voice/                    STT (faster-whisper) + TTS (edge-tts) package
│   ├── watcher.py                Watchdog file watcher for knowledge dir
│   ├── tools/
│   │   ├── definitions.py        Tool JSON schemas (core + telemetry)
│   │   └── executor.py           Tool execution with retry + safety checks
│   ├── static/
│   │   └── index.html            Full dashboard UI (single file — CSS + HTML + JS)
│   ├── tests/
│   │   ├── verify_full.py        Core system tests
│   │   ├── verify_stress.py      Stress simulation tests
│   │   ├── verify_resource_mgmt.py  Resource management tests
│   │   └── verify_distill.py     Distillation pipeline tests
│   └── .env                      Configuration (not committed)
├── modelfiles\                   Ollama modelfiles (identical prompts, different FROM)
├── data\
│   ├── chromadb\                 Vector store — hot/warm tiers (not committed)
│   ├── cold_memory.db            Cold tier compressed archive (not committed)
│   ├── sqlite\                   Persistent state (not committed)
│   ├── knowledge\                Drop files here for RAG ingestion
│   └── training\                 Datasets, adapters, GGUF exports, eval results
├── scripts\
│   ├── backup_s3.py                  S3 backup — full/incremental/restore
│   ├── setup_backup_task.ps1         Task Scheduler — 3 AM S3 backup
│   ├── daily_training.py             Daily autonomous training orchestrator (2 AM)
│   └── setup_daily_training_task.ps1 Task Scheduler — 2 AM training cycle
└── tools\
    └── llama.cpp\                GGUF conversion toolchain (gitignored unless cloned)
```

---

## API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Streaming chat (NDJSON) — auto-routes, ReAct for complex queries |
| GET | `/chat/history` | Conversation history metadata |
| DELETE | `/chat/history` | Clear conversation history |

### Ingest
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Push context into RAG (async queue, non-blocking) |
| POST | `/ingest/batch` | Batch ingest up to 100 items |
| GET | `/ingest/pipeline/status` | Queue depth, throughput, latency metrics |

### Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memory/compact` | Trigger warm→cold tier compaction |
| GET | `/memory/cold/stats` | Cold tier storage statistics |
| GET | `/knowledge/gaps` | Unresolved knowledge gaps |
| POST | `/knowledge/gaps/{id}/resolve` | Mark gap as resolved |

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
| GET | `/training/weaknesses` | Identify weak domains from quality feedback |
| POST | `/training/improve/{domain}` | Trigger iterative improvement cycle |

### Adapters
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/adapters` | List all adapters (optional domain filter) |
| GET | `/adapters/active` | Current active adapter map |
| POST | `/adapters/train` | Start domain-specific LoRA training |
| POST | `/adapters/merge` | TIES merge active adapters → production GGUF |
| POST | `/adapters/eval/{name}` | Evaluate adapter against baseline |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | 8–9 subsystem health check |
| GET | `/stats` | System stats + model info + budget |
| GET | `/budget/status` | Cloud spend tracking |
| GET | `/backup/status` | Emergency backup diagnostics |
| GET | `/resources/status` | VRAM prediction, GPU temp, process priority, circuit breaker |
| GET | `/self-heal/status` | AI self-heal loop status + recent actions |
| GET | `/health/events` | Timestamped health event log |
| WS | `/ws/alerts` | Real-time alert notifications |
| WS | `/ws/health` | Real-time health state events |

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
| 5 | **Complete** | Telemetry prep — WebSocket alerts, batch ingest, conditional telemetry tools |
| 6 | **Complete** | Dashboard UI — training pipeline, diagnostics, live voice chat, settings, self-heal |
| 7 | **Complete** | Adapter surgery — modular domain LoRA, TIES merge, computation tools, distillation |
| 8 | **Complete** | Advanced intelligence — ReAct reasoning, iterative RAG, dynamic quant/offload, hierarchical memory, async ingest, MoLoRA, predictive VRAM, process priority |
| 9 | **Complete** | Progressive intelligence — reasoning trace memory, quality scoring, tool filtering, confidence protocol, adaptive routing, smart capture, iterative distillation, cross-domain transfer, smart history, knowledge gaps |
| 10 | **Complete** | Continuous intelligence — daily autonomous training (2 AM scheduler), telemetry + audio adapter domains (6 total), DPO training, session knowledge synthesis, 5 new training datasets, CoT Protocol 7 |
| — | **Separate** | Telemetry API for Le Mans Ultimate — connects to E3N via /ingest |

---

## Dashboard

Tactical operations center aesthetic with a muted grey-green palette. Features include:

- **Header**: Subsystem health monitor (X/8 ONLINE) + cloud budget display
- **3D particle sphere**: Network node map with speech waveform animation during voice chat
- **Terminal**: Streaming chat with conversation history (CLR + turn counter)
- **Live voice chat**: Always-listening JARVIS/E3N style — Whisper pre-warms, VAD auto-detects speech
- **Web search**: DuckDuckGo-powered autonomous web search tool
- **Electronic voice filter**: Subtle AI texture via Web Audio API — configurable intensity
- **Settings panel**: Audio devices, TTS config, voice filter, system status
- **Training pipeline**: Start/stop training, mode selector, live progress + loss
- **Diagnostics panel**: Circuit breaker, VRAM prediction, self-heal status, resource action log
- **Live widgets**: GPU/CPU/RAM stats with 60s sparklines
- **WebSocket toasts**: Real-time alert notifications with priority coloring

---

## Backup & Restore

Automated nightly backup of E3N's persistent data to AWS S3.

### Setup

```powershell
# 1. Install boto3
cd C:\e3n\project
.\venv\Scripts\activate
pip install boto3

# 2. Configure AWS credentials
aws configure
# Or set environment variables:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...

# 3. Set bucket name (system-wide)
setx E3N_S3_BUCKET your-bucket-name /M

# 4. Register nightly task (run as Administrator)
powershell -ExecutionPolicy Bypass -File C:\e3n\scripts\setup_backup_task.ps1
```

### Manual Backup

```powershell
python scripts/backup_s3.py              # full backup
python scripts/backup_s3.py --dry-run    # preview what would be uploaded
python scripts/backup_s3.py --list       # list available backups
```

### Restore

```powershell
# List available backup dates
python scripts/backup_s3.py --list

# Restore from a specific date
python scripts/backup_s3.py --restore 2026-03-24
```

This restores all files to `C:\e3n\data\` — ChromaDB, SQLite, knowledge base, training data, cold memory, and adapter registry. After restoring, restart the E3N backend to pick up the recovered data.

### What Gets Backed Up

| Directory/File | Contents |
|----------------|----------|
| `data/chromadb/` | Vector store (hot/warm RAG tiers) |
| `data/sqlite/` | Conversation history, cloud budget |
| `data/knowledge/` | Knowledge base documents |
| `data/training/` | Datasets, adapters, GGUF exports, eval results |
| `data/cold_memory.db` | Cold tier compressed archive |

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `E3N_S3_BUCKET` | *(required)* | S3 bucket name |
| `E3N_S3_PREFIX` | `e3n-backups` | Prefix inside bucket |
| `E3N_S3_REGION` | `us-east-1` | AWS region |
| `E3N_S3_RETENTION` | `30` | Days to keep old backups (0 = forever) |
| `E3N_DATA_PATH` | `C:\e3n\data` | Path to data directory |

---

## Daily Autonomous Training

E3N improves itself every night without human intervention via a scheduled training cycle.

### Setup (run as Administrator)

```powershell
# Register daily training task at 2:00 AM (before the 3 AM S3 backup)
powershell -ExecutionPolicy Bypass -File C:\e3n\scripts\setup_daily_training_task.ps1
```

### How It Works

| Phase | Description |
|-------|-------------|
| Guards | Sim not running, VRAM ≥ 7GB, no active session |
| Weakness Analysis | Identifies domains with avg quality < 0.6 from routing feedback |
| Targeted Distillation | 14B teacher generates examples for up to 2 weak domains |
| Curriculum | Daily topic rotation (Mon=telemetry, Tue=strategy, Wed=engineering, Thu=simulations, Fri=audio, Sat=cross-domain, Sun=personality) |
| QLoRA Training | Trains one domain adapter on blended new + existing data |
| Gap Review | Counts unresolved knowledge gaps |
| Report | JSON report written to `data/training/daily_reports/YYYY-MM-DD.json` |

### Manual Usage

```powershell
# Dry run — all phases, no actual training
python scripts/daily_training.py --dry-run

# Force a specific day's curriculum (0=Mon, 1=Tue...)
python scripts/daily_training.py --day 0

# Skip QLoRA, distillation + report only
python scripts/daily_training.py --no-train

# View last daily report
python scripts/daily_training.py --report-only
```

### Adapter Domains (6 total)

| Domain | Purpose | LoRA Rank | Freeze Depth |
|--------|---------|-----------|--------------|
| Racing | Tire strategy, lap analysis, driving technique | r=16 | 18/36 (50%) |
| Engineering | Physics, FEA/CFD, thermodynamics, statics | r=16 | 12/36 (33%) |
| Personality | E3N voice, response style | r=8 | 24/36 (67%) |
| Reasoning | Chain-of-thought, multi-step analysis | r=32 | 8/36 (22%) |
| **Telemetry** | Multi-sensor correlation, anomaly detection, degradation curves | r=16 | 14/36 (39%) |
| **Audio** | Engine knock, bearing noise, brake squeal, turbo surge diagnosis | r=16 | 16/36 (44%) |

---

## Verification

**From a git clone** (replace `<repo-root>` with your checkout path):

```powershell
cd <repo-root>\project
.\venv\Scripts\activate
python tests/verify_full.py          # Core system tests
python tests/verify_stress.py        # Stress simulation tests
python tests/verify_resource_mgmt.py # Resource management tests
```

**Author layout** (`C:\e3n`):

```powershell
cd C:\e3n\project
.\venv\Scripts\activate
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
```

For distillation-related changes, also run `python tests/verify_distill.py` (see [CLAUDE.md](CLAUDE.md)).

---

## License

Private project.

*Built by Ethan, 2025–2026.*
