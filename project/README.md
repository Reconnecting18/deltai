# deltai — FastAPI backend (`project/`)

This directory is the **main HTTP daemon**: FastAPI app (`main.py`), router, RAG, tools, training hooks, and the single-file dashboard under `static/`. User-facing product docs and philosophy live in the repository root [README.md](../README.md). Deep assistant context: [CLAUDE.md](../CLAUDE.md).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) for local models (optional models per your `modelfiles/`)
- Repository root: `pip install -e .[dev]` (installs the `delta` package and dev tools)

## Run locally (Linux)

From the **repository root**:

```bash
cd project
python -m venv ../venv && source ../venv/bin/activate   # or use your own venv layout
pip install -e ..[dev]
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open the dashboard at `http://127.0.0.1:8000`. API details are in [CLAUDE.md](../CLAUDE.md).

## systemd (user service)

The in-repo unit is [../systemd/user/delta-daemon.service](../systemd/user/delta-daemon.service). After installing the package so `delta-daemon` is on `PATH`:

```bash
cp ../systemd/user/delta-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now delta-daemon
journalctl --user -u delta-daemon -f
```

SQLite and XDG paths are set in the unit file; override with `Environment=` drop-ins if needed.

## Terminal health

With `delta-daemon` running, from the repo (after `pip install -e ..[dev]`): run **`deltai`** or **`deltai status`** for a fastfetch-style view (daemon, Ollama, IPC, SQLite, optional `http://127.0.0.1:8000/api/health` when the project server is up). Use **`deltai status --json`** for scripts.

## Optional: Windows development

If you develop on Windows, use PowerShell equivalents for paths and venv activation. Core documentation and agent rules target **Linux paths** in examples; keep production config portable.

## Verify after backend changes

```bash
cd project && source ../venv/bin/activate   # adjust venv path
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
# If you touched training/distillation:
python tests/verify_distill.py
```

## Layout (this tree)

| Path | Role |
|------|------|
| `main.py` | FastAPI app, endpoints, streaming, integrations |
| `router.py` | VRAM-aware model routing |
| `memory.py` | ChromaDB RAG |
| `extensions/training/pipeline.py` / `build.py` | Fine-tuning and adapter workflows (import as `training` via shim) |
| `tools/` | Tool definitions and executor |
| `static/` | Single-file dashboard |
| `tests/` | Verification scripts |

Runtime data (`data/`, SQLite, Chroma) is gitignored; see root README for XDG layout.
