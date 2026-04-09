"""Ollama integration adapter.

This module should absorb reusable local-model invocation logic from E3N
(router/model call paths) while exposing a DELTA-focused interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class OllamaClient:
    """Minimal async Ollama client for DELTA model execution."""

    base_url: str

    async def generate(self, model: str, prompt: str) -> str:
        """Run a non-streaming generation call against Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
