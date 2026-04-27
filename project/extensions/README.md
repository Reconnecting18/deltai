# Extensions

This directory is your personal space for features that don't belong in deltai's core.
Drop a package (a folder with `__init__.py`) here and it is auto-discovered at startup —
no edits to core files required.

---

## Quick start

1. Copy `example_extension/` to `my_feature/`.
2. Edit `my_feature/__init__.py`.
3. Restart deltai.

That's it.

---

## What an extension can do

| Hook | How | Notes |
|------|-----|-------|
| Add API routes | Mount an `APIRouter` in `setup(app)` | Prefix recommended, e.g. `/ext/my_feature/…` |
| Add tools | Define `TOOLS = [...]` at module level | Same schema as `project/tools/definitions.py` |
| Register tool executor | Call `tools.executor.register_handler(name, fn)` in `setup` | Handler receives `(args: dict) -> str` |
| Run code at startup | Put logic inside `setup(app)` | Runs synchronously; schedule coroutines with `asyncio.create_task` |
| Run code at shutdown | Define `shutdown()` (sync **or** async) | Called during app lifespan cleanup |

---

## Extension package layout

```
extensions/
└── my_feature/
    ├── __init__.py   ← required; contains setup(), TOOLS, shutdown()
    ├── routes.py     ← optional; APIRouter definition
    ├── logic.py      ← optional; your business logic
    └── README.md     ← optional; notes for yourself
```

Only `__init__.py` is required.

---

## Minimal template

```python
# extensions/my_feature/__init__.py

def setup(app):
    from fastapi import APIRouter
    router = APIRouter(prefix="/ext/my_feature", tags=["my_feature"])

    @router.get("/hello")
    def hello():
        return {"message": "Hello from my_feature!"}

    app.include_router(router)
```

---

## Adding a custom tool

```python
# extensions/my_feature/__init__.py
from tools.executor import register_handler

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "Does something useful.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input text"}
                },
                "required": ["input"]
            }
        }
    }
]

def _my_tool_handler(args: dict) -> str:
    return f"Processed: {args['input']}"

def setup(app):
    register_handler("my_tool", _my_tool_handler)
```

---

## Keeping personal extensions out of git

Add a line to the project-level `.gitignore` (or a local `.git/info/exclude`) for any extension you don't want to commit:

```
project/extensions/my_feature/
```

The `example_extension/` ships as a reference; your own packages are yours to manage. Arch update guard is integrated in `project/core/arch_update_guard/` (core), not under `extensions/`.

---

## Other extension mechanisms

deltai has more than one way to extend behavior; pick the surface that matches **which process** you run.

| Mechanism | Process | Use when |
|-----------|---------|----------|
| **`project/extensions/`** (this dir) | Full backend `project/main.py` | In-repo packages: routes, tools, startup hooks; no core edits. |
| **`project/core/arch_update_guard/`** | Same full backend | Arch Linux maintenance assistant (API, scheduler, tools). Toggle with **`ARCH_GUARD_*`** env vars (see `project/core/arch_update_guard/settings.py`). |
| **`delta.core.PluginManager`** | **`delta-daemon`** only (`delta-daemon` systemd service) | End-user `.py` plugins under XDG + `extensions.toml`; managed with **`deltai plugin install|unload`**. |

See [`src/delta/core/README.md`](../../src/delta/core/README.md) for daemon plugins. Arch Update Guard’s CLI entrypoint (`arch-update-guard`) needs the **`project/`** tree on disk; set **`DELTAI_PROJECT_ROOT`** if you installed the wheel outside the clone.

---

## Rules

- Extensions run in the **same process** as deltai and have full access to its internals.
  Write defensively — catch exceptions so a broken extension doesn't crash the daemon.
- Never write outside XDG app dirs or the project tree without explicit user action.
- No root, no `sudo`.
- Follow the [non-negotiable boundaries](../../AGENTS.md#non-negotiable-boundaries) in AGENTS.md.
