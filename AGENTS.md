# deltai — Agent Onboarding

Short context for Cursor, Copilot, and other coding agents. **Full architecture, endpoints, stream protocol, and development workflow live in [CLAUDE.md](CLAUDE.md).** Update that file when behavior or structure changes.

## What this is

deltai is a modular AI extension system for Linux. It runs as a **systemd user service** (no root required), exposes an HTTP/WebSocket API on `localhost:8000`, and provides: local LLM routing (Ollama), RAG memory (ChromaDB), a structured tool system, task automation, and a plugin/ingest API for external services.

It is the open, user-controlled Linux answer to Copilot+Windows — built around Linux philosophies: user-space first, user choice, modularity, transparency, no lock-in.

## Non-negotiable boundaries

- **Never require root.** deltai runs entirely in user space. No `sudo`, no system-wide writes.
- **No domain-specific ingest.** Never add telemetry parsers, game data readers, or domain pipelines to the core. External services push context via `POST /ingest`.
- **Linux paths only.** No `C:\...` Windows paths anywhere in code or config.
- **User choice.** Every feature is opt-in. Defaults must be safe and minimal.
- **No lock-in.** Local models are the default. Cloud is dormant until explicitly configured.

## Repository paths

| Area | Path |
|------|------|
| Backend (FastAPI daemon) | `project/` |
| Desktop shell (optional) | `app/` |
| Ollama modelfiles | `modelfiles/` |
| systemd user unit | `systemd/user/` |
| Runtime data (gitignored) | `data/` |
| Scripts | `scripts/` |

## How to run (development)

```bash
# Backend
cd deltai
python -m venv venv && source venv/bin/activate
pip install -e .[dev]
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Desktop shell (optional)
cd app && npm install && npm start
```

The main package metadata now includes ChromaDB, so `pip install .` covers
runtime installs and `pip install -e .[dev]` covers local development.

Or as a systemd user service:
```bash
cp systemd/user/deltai.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user start deltai
journalctl --user -u deltai -f
```

## Verify after substantive backend changes

```bash
cd project && source venv/bin/activate
python tests/verify_full.py
python tests/verify_stress.py
python tests/verify_resource_mgmt.py
# If you touched training/distillation:
python tests/verify_distill.py
```

## Doc maintenance

When you change user-visible behavior, configuration, or architecture:

1. Update [CLAUDE.md](CLAUDE.md) — the primary context for all AI sessions.
2. Update [README.md](README.md) — user-facing changes, endpoints, config.
3. Keep [AGENTS.md](AGENTS.md) accurate — onboarding, boundaries, verify commands.
4. If you change RAG, ingest, models, or adapters, update [docs/local-model-workflow.md](docs/local-model-workflow.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues and PR expectations.
