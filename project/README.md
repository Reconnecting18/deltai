# deltai

Local AI intelligence layer running on Windows 11. Named after deltai from COD: Infinite Warfare, with personality blending deltai (dry wit, loyal) and BT-7274 (precise, protocol-driven).

deltai is the AI brain — reasoning, memory, tool execution, and model routing. External services connect via the `/ingest` endpoint to push context into its RAG memory.

## Architecture

- **Backend:** FastAPI (Python) on `localhost:8000`
- **Frontend:** Single-file dashboard (HTML/CSS/JS) with tactical ops center aesthetic
- **Models:** Ollama — Qwen2.5-14B (primary), Qwen2.5-3B (sim racing fallback), with emergency backup chain
- **RAG:** ChromaDB with multi-query expansion, source-grouped reranking, and recency bias
- **Voice:** STT (faster-whisper) + TTS (Piper/edge-tts) + RVC voice conversion
- **Training:** QLoRA fine-tuning with adapter surgery — 4 domain-specific augmentation slots (racing, engineering, personality, reasoning), TIES merge, selective layer freezing
- **Routing:** VRAM-aware tier system (A/B/C) with sim process detection, cloud budget enforcement, and adapter domain classification
- **Tools:** 19 tools — file I/O, PowerShell, RAG search, computation delegation, self-diagnostics, adapter management, conditional telemetry

## Quick Start

```powershell
# Terminal 1 — Backend
cd ~/deltai/project
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Electron
cd ~/deltai/app
npm start
```

## Key Features

- **VRAM-aware routing** — automatically selects model tier based on GPU availability and sim process detection
- **Adapter surgery** — modular LoRA augmentation slots with independent training, versioning, evaluation, and TIES merge into production GGUFs
- **Resource self-management** — background VRAM lifecycle, Ollama auto-restart, circuit breaker, AI-driven self-heal loop
- **Knowledge base** — drop `.md` files in `data/knowledge/`, watchdog auto-ingests into ChromaDB
- **Ingest connector** — external services push structured context via `POST /ingest` with TTL and source tags
- **Session mode** — GPU protection during racing, routes inference to cloud or CPU fallback
- **Emergency backup chain** — automatic failover: qwen14b -> nemo -> llama 8B
- **Cloud integration** — Anthropic API support with split workload (local tools + cloud reasoning), daily budget cap
- **Voice** — full STT/TTS loop with RVC voice conversion
- **Training pipeline** — QLoRA fine-tuning, knowledge distillation, A/B evaluation, auto-capture of good exchanges

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3060 12GB |
| CPU | Intel i7-12700K |
| RAM | 34GB |
| OS | Windows 11 |

## Project Structure

```
~/deltai/
  project/           # FastAPI backend
    main.py           # App, endpoints, resource manager
    router.py         # VRAM-aware model routing
    memory.py         # ChromaDB RAG
    training.py       # QLoRA + adapter surgery
    tools/            # Tool definitions + executor
    voice/            # STT/TTS/RVC package
    static/           # Dashboard UI
    tests/            # 4 test suites (139 tests)
  app/                # Electron wrapper
  modelfiles/         # Ollama modelfiles (4)
  data/               # ChromaDB, knowledge, training datasets
```

## Status

Personal project. All 7 build phases complete. 139/139 tests passing.
