"""
deltai Extensions Loader

Discovers and loads user extensions from subdirectories of this folder.
Each extension is a package (subdirectory with an __init__.py) that may define:

  setup(app)   — called with the FastAPI app instance after creation.
                 Register routers, middleware, event handlers, etc. here.
                 Optional but typical.

  TOOLS        — list of tool-definition dicts (same schema as project/tools/definitions.py).
                 Automatically merged into the global tool list at startup.
                 Optional.

  shutdown()   — called during app shutdown (sync or async).
                 Optional.

Extensions whose __init__.py raises any exception during import or setup are
skipped with a warning so a broken extension never prevents deltai from starting.

Directories that start with '_' (e.g. __pycache__) are ignored.
"""

import asyncio
import importlib
import logging
import os

logger = logging.getLogger("deltai.extensions")

_HERE = os.path.dirname(os.path.abspath(__file__))

# List of (name, module) for every successfully loaded extension.
_loaded: list[tuple[str, object]] = []


def load_extensions(app) -> None:
    """
    Scan this directory for extension packages and load each one.
    Called once by main.py after the FastAPI app is created.
    """
    candidates = sorted(
        name for name in os.listdir(_HERE)
        if not name.startswith("_")
        and os.path.isdir(os.path.join(_HERE, name))
        and os.path.isfile(os.path.join(_HERE, name, "__init__.py"))
    )

    for name in candidates:
        try:
            module = importlib.import_module(f"extensions.{name}")
            _loaded.append((name, module))
            if hasattr(module, "setup"):
                module.setup(app)
                logger.info(f"Extension loaded: {name}")
            else:
                logger.info(f"Extension registered (no setup hook): {name}")
        except Exception as exc:
            logger.warning(f"Extension '{name}' failed to load — skipping: {exc}")


def get_extension_tools() -> list[dict]:
    """
    Return the combined list of tool definitions contributed by all extensions.
    Called by tools/definitions.py after extensions are loaded.
    """
    tools: list[dict] = []
    for _name, module in _loaded:
        if hasattr(module, "TOOLS"):
            tools.extend(module.TOOLS)
    return tools


async def shutdown_extensions() -> None:
    """
    Run shutdown hooks for all loaded extensions.
    Called by main.py lifespan cleanup.
    """
    for name, module in _loaded:
        if hasattr(module, "shutdown"):
            try:
                result = module.shutdown()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning(f"Extension '{name}' shutdown failed: {exc}")
