"""
Example Extension — deltai extension template.

Copy this directory, rename it, and edit to taste.
This example:
  - adds a GET /ext/example/hello endpoint
  - registers a 'greet' tool the LLM can call
  - logs a message at startup and shutdown

Delete or disable this extension once you've made your own.
"""

import logging

logger = logging.getLogger("deltai.extensions.example_extension")

# ── Optional: tool definitions ──────────────────────────────────────────────
# Define any tools you want the LLM to be able to call.
# Schema must match project/tools/definitions.py entries.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "greet",
            "description": (
                "Return a friendly greeting for a given name. "
                "Use this when the operator asks to greet someone."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The name to greet."}},
                "required": ["name"],
            },
        },
    }
]


def _greet_handler(args: dict) -> str:
    """Tool executor for the 'greet' tool."""
    name = args.get("name", "world")
    return f"Hello, {name}! (from example_extension)"


# ── Required: setup(app) ─────────────────────────────────────────────────────
def setup(app) -> None:
    """
    Called once with the FastAPI app after it is created.
    Register routers, middleware, startup tasks, and tool handlers here.
    """
    from fastapi import APIRouter
    from tools.executor import register_handler

    # Register the tool handler so the executor can call it.
    register_handler("greet", _greet_handler)

    # Add a simple route under /ext/example_extension/
    router = APIRouter(
        prefix="/ext/example_extension",
        tags=["example_extension"],
    )

    @router.get("/hello")
    def hello(name: str = "world"):
        """Simple greeting endpoint — useful for smoke-testing your extension."""
        return {"message": f"Hello, {name}!", "source": "example_extension"}

    app.include_router(router)
    logger.info("example_extension: routes registered at /ext/example_extension/")


# ── Optional: shutdown() ─────────────────────────────────────────────────────
def shutdown() -> None:
    """
    Called during deltai shutdown.
    Clean up background tasks, open files, etc.
    Can also be 'async def shutdown()' if you need to await something.
    """
    logger.info("example_extension: shutdown")
