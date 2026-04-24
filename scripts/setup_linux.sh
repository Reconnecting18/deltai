#!/usr/bin/env bash
# One-shot Linux setup: venv, editable install, optional .env copy.
# Run from the repository root: bash scripts/setup_linux.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "setup_linux: python3 is required." >&2
  exit 1
fi

if [[ ! -d project ]]; then
  echo "setup_linux: run this from the deltai repo root (missing project/)." >&2
  exit 1
fi

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
python -m pip install -U pip wheel
pip install -e ".[dev]"

if [[ ! -f project/.env ]] && [[ -f project/.env.example ]]; then
  cp project/.env.example project/.env
  echo "Created project/.env from .env.example — edit models and URLs as needed."
fi

echo ""
echo "Setup complete. Activate the venv and use:"
echo "  source venv/bin/activate"
echo "  deltai-server              # HTTP API on http://127.0.0.1:8000"
echo "  deltai                     # terminal client (needs server running)"
echo "Or: cd project && uvicorn main:app --host 127.0.0.1 --port 8000"
