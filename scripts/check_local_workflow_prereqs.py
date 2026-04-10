"""
Read-only checklist for the local Qwen improvement workflow prerequisites.

Does not start the API or mutate data. Loads project/.env when present (same
pattern as daily_training.py) so paths match your machine.

Usage (repo root):
  python scripts/check_local_workflow_prereqs.py

Exit code 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import sys

import httpx

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.join(_REPO_ROOT, "project")


def _load_env() -> None:
    env_path = os.path.join(_PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = val


def _check(name: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return ok


def main() -> int:
    _load_env()

    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

    print("deltai local workflow prereqs (read-only)\n")

    all_ok = True

    knowledge = os.getenv("KNOWLEDGE_PATH", r"~/deltai/data\knowledge")
    chroma = os.getenv("CHROMADB_PATH", r"~/deltai/data\chromadb")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

    all_ok &= _check(
        "KNOWLEDGE_PATH",
        os.path.isdir(knowledge),
        knowledge if os.path.isdir(knowledge) else f"missing or not a directory: {knowledge}",
    )
    chroma_ready = os.path.isdir(chroma) or os.path.isdir(os.path.dirname(chroma))
    all_ok &= _check(
        "CHROMADB_PATH",
        chroma_ready,
        chroma if chroma_ready else f"neither dir nor parent exists: {chroma}",
    )

    try:
        from memory import KNOWLEDGE_PATH as kp  # noqa: E402

        all_ok &= _check("memory import", True, f"resolved KNOWLEDGE_PATH={kp}")
    except Exception as e:
        all_ok &= _check("memory import", False, str(e))

    try:
        import watcher  # noqa: F401, E402

        all_ok &= _check("watcher import", True, "watchdog + memory ingest hooks load")
    except Exception as e:
        all_ok &= _check("watcher import", False, str(e))

    try:
        import training  # noqa: F401, E402

        all_ok &= _check("training import", True, "training module loadable")
    except Exception as e:
        all_ok &= _check("training import", False, str(e))

    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        ok = r.status_code == 200
        all_ok &= _check(
            "Ollama",
            ok,
            f"{ollama_url} tags -> HTTP {r.status_code}" if ok else f"{ollama_url} -> HTTP {r.status_code}",
        )
    except Exception as e:
        all_ok &= _check("Ollama", False, f"{ollama_url} ({e})")

    modelfiles = os.path.join(_REPO_ROOT, "modelfiles")
    m14 = os.path.join(modelfiles, "deltai-qwen14b.modelfile")
    m3 = os.path.join(modelfiles, "deltai-qwen3b.modelfile")
    mf_ok = os.path.isfile(m14) and os.path.isfile(m3)
    all_ok &= _check(
        "modelfiles",
        mf_ok,
        "deltai-qwen14b + deltai-qwen3b present" if mf_ok else f"expected under {modelfiles}",
    )

    workflow_doc = os.path.join(_REPO_ROOT, "docs", "local-model-workflow.md")
    all_ok &= _check(
        "workflow doc",
        os.path.isfile(workflow_doc),
        workflow_doc if os.path.isfile(workflow_doc) else f"missing: {workflow_doc}",
    )

    print()
    if all_ok:
        print("All checks passed.")
        return 0
    print("One or more checks failed. Fix paths, venv deps, or start Ollama.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
