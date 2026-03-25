# E3N Project Status

## Hardware
- GPU: RTX 3060 12GB · i7-12700K · Windows 11 · 34.1 GB RAM

## Folder Structure
```
C:\e3n\
  ├── E3N_STATUS.md
  ├── modelfiles\
  │   ├── E3N.modelfile          ← active personality
  │   └── E3N_personality.txt   ← character reference doc
  ├── project\                   ← FastAPI backend
  │   ├── .env
  │   ├── main.py
  │   ├── static\index.html     ← Electron UI
  │   └── venv\
  └── data\
      ├── chromadb\             ← Phase 2 memory (inactive)
      ├── sqlite\               ← Phase 3 telemetry
      └── knowledge\            ← drop PDFs here

C:\Users\ethan\.ollama\         ← Ollama models (do not move)
C:\e3n\app\                     ← Electron desktop app
```

## Completed
- [x] Ollama + llama3.1:8b installed
- [x] E3N modelfile — Rajdhani personality, operator profile
- [x] FastAPI backend — /chat, /stats, /modelfile endpoints
- [x] Electron desktop app — transparent, frameless, IPC window controls
- [x] Dashboard UI — GPU/CPU/RAM/Disk/Memory live stats, sparklines
- [x] 3D sphere particle entity — 280 nodes, connection lines, health-reactive
- [x] Idle/active brightness states on sphere
- [x] Cursor repulsion on sphere particles
- [x] Memory panel — editable modelfile
- [x] Terminal — monospace AI chat

## To Start E3N
```
cd C:\e3n\app
npm start
```

## Next Steps
### Phase 2 — Memory (ChromaDB + RAG)
```
pip install chromadb watchdog
```
Add /chat context injection from ChromaDB
File watcher on C:\e3n\data\knowledge\

### Phase 4 — Remote Access
Tailscale install (free, 10 min setup)
S3 nightly backup of C:\e3n\data\

## SSD Migration
1. Buy Samsung T7 500GB (~$60)
2. Copy C:\e3n\ to SSD root
3. Set OLLAMA_MODELS env var BEFORE installing Ollama on new machine
4. Install Ollama, pull llama3.1:8b
5. Run: ollama create e3n -f E3N.modelfile
6. Update paths in .env to new drive letter
7. npm start from SSD — full memory intact

## Draggable Widgets
Hold Alt + drag any panel to reposition it on the dashboard.
