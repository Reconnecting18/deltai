"""
deltai Anthropic Client — handles cloud inference via the Anthropic API.

Speaks the same stream protocol as the Ollama path so the rest of
the system doesn't care which backend is responding.

Supports tool use: the client sends tool definitions, the API returns
tool_use blocks, we execute them locally and loop back with results.

Stream protocol (JSON lines):
  {"t":"route","backend":"anthropic","model":"...","tier":N,"reason":"..."}
  {"t":"rag","n":N}
  {"t":"tool","n":"tool_name","a":{...}}
  {"t":"result","n":"tool_name","s":"summary"}
  {"t":"text","c":"chunk"}
  {"t":"done"}
  {"t":"error","c":"message"}
"""

import json
import logging
import os

import httpx
import safe_errors
from prompts import build_cloud_system_prompt

logger = logging.getLogger("deltai.anthropic")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"  # Pinned for stability; update when new features needed

# Default cloud system prompt (non-split); split workload uses build_cloud_system_prompt(split_workload=True)
DELTAI_SYSTEM_PROMPT = build_cloud_system_prompt()


# ── TOOL SCHEMA CONVERSION ───────────────────────────────────────────────


def _convert_tools_to_anthropic(ollama_tools: list) -> list:
    """
    Convert Ollama-format tool definitions to Anthropic's tool format.
    Ollama: {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
    Anthropic: {"name":...,"description":...,"input_schema":{...}}
    """
    anthropic_tools = []
    for tool in ollama_tools:
        fn = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


# ── STREAMING CHAT WITH TOOL USE ─────────────────────────────────────────

MAX_TOOL_ROUNDS = 5  # Prevent infinite tool loops


async def stream_chat(
    message: str,
    model: str,
    rag_context: str = "",
    max_tokens: int = 4096,
    tools: list = None,
    execute_tool_fn=None,
    split_mode: bool = False,
    history: list = None,
):
    """
    Stream a response from the Anthropic API with tool-use support.

    Yields JSON-line strings in the same protocol as the Ollama path.

    Args:
        message: User's message
        model: Anthropic model ID
        rag_context: Optional RAG context to prepend
        max_tokens: Max response tokens
        tools: Ollama-format tool definitions (converted automatically)
        execute_tool_fn: Function to call tools: (name, args) -> result string
        split_mode: If True, append split workload hint to system prompt
        history: Prior conversation messages to prepend (user/assistant pairs)
    """

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        yield (
            json.dumps(
                {
                    "t": "error",
                    "c": "No Anthropic API key configured. Set ANTHROPIC_API_KEY in project/.env (or your environment).",
                }
            )
            + "\n"
        )
        yield json.dumps({"t": "done"}) + "\n"
        return

    # Build messages — prepend history, then current user message
    user_content = f"{rag_context}\n{message}" if rag_context else message
    messages = list(history) if history else []
    messages.append({"role": "user", "content": user_content})

    # Convert tools
    anthropic_tools = _convert_tools_to_anthropic(tools) if tools else []

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    total_input_tokens = 0
    total_output_tokens = 0

    for _round_num in range(MAX_TOOL_ROUNDS + 1):
        system_prompt = build_cloud_system_prompt(split_workload=split_mode)

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "stream": True,
        }
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", ANTHROPIC_API_URL, json=payload, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        try:
                            err = json.loads(body)
                            err_msg = err.get("error", {}).get("message", body.decode()[:200])
                        except Exception:
                            err_msg = body.decode()[:200]
                        yield (
                            json.dumps(
                                {
                                    "t": "error",
                                    "c": f"Anthropic API error ({resp.status_code}): {err_msg}",
                                }
                            )
                            + "\n"
                        )
                        yield json.dumps({"t": "done"}) + "\n"
                        return

                    # Parse the streaming response
                    assistant_content = []
                    current_text = ""
                    current_tool_use = None
                    tool_use_blocks = []
                    stop_reason = None

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type", "")

                            if event_type == "message_start":
                                usage = event.get("message", {}).get("usage", {})
                                total_input_tokens += usage.get("input_tokens", 0)

                            elif event_type == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "tool_use":
                                    current_tool_use = {
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                        "input_json": "",
                                    }
                                elif block.get("type") == "text":
                                    current_text = ""

                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        current_text += text
                                        yield json.dumps({"t": "text", "c": text}) + "\n"
                                elif delta.get("type") == "input_json_delta":
                                    if current_tool_use:
                                        current_tool_use["input_json"] += delta.get(
                                            "partial_json", ""
                                        )

                            elif event_type == "content_block_stop":
                                if current_tool_use:
                                    # Parse tool input
                                    try:
                                        tool_input = (
                                            json.loads(current_tool_use["input_json"])
                                            if current_tool_use["input_json"]
                                            else {}
                                        )
                                    except json.JSONDecodeError:
                                        tool_input = {}
                                    tool_use_blocks.append(
                                        {
                                            "id": current_tool_use["id"],
                                            "name": current_tool_use["name"],
                                            "input": tool_input,
                                        }
                                    )
                                    assistant_content.append(
                                        {
                                            "type": "tool_use",
                                            "id": current_tool_use["id"],
                                            "name": current_tool_use["name"],
                                            "input": tool_input,
                                        }
                                    )
                                    current_tool_use = None
                                elif current_text:
                                    assistant_content.append(
                                        {
                                            "type": "text",
                                            "text": current_text,
                                        }
                                    )
                                    current_text = ""

                            elif event_type == "message_delta":
                                stop_reason = event.get("delta", {}).get("stop_reason")
                                usage = event.get("usage", {})
                                total_output_tokens += usage.get("output_tokens", 0)

                            elif event_type == "message_stop":
                                break

                            elif event_type == "error":
                                err_msg = event.get("error", {}).get("message", "Unknown error")
                                yield json.dumps({"t": "error", "c": err_msg}) + "\n"
                                yield json.dumps({"t": "done"}) + "\n"
                                return

                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException:
            yield json.dumps({"t": "error", "c": "Anthropic API timed out (120s)"}) + "\n"
            yield json.dumps({"t": "done"}) + "\n"
            return
        except httpx.ConnectError:
            yield (
                json.dumps(
                    {"t": "error", "c": "Cannot reach Anthropic API — check internet connection"}
                )
                + "\n"
            )
            yield json.dumps({"t": "done"}) + "\n"
            return
        except Exception as e:
            safe_errors.log_exception(logger, "Anthropic stream_chat failed", e)
            yield (
                json.dumps(
                    {"t": "error", "c": f"Cloud error: {safe_errors.public_error_detail(e)}"},
                )
                + "\n"
            )
            yield json.dumps({"t": "done"}) + "\n"
            return

        # If no tool use, we're done
        if stop_reason != "tool_use" or not tool_use_blocks or not execute_tool_fn:
            break

        # Execute tools and continue the conversation
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for tool_block in tool_use_blocks:
            name = tool_block["name"]
            args = tool_block["input"]
            tool_id = tool_block["id"]

            yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"

            try:
                result = execute_tool_fn(name, args)
            except Exception as e:
                safe_errors.log_exception(logger, f"Anthropic tool {name} failed", e)
                result = f"ERROR: Tool execution failed: {safe_errors.public_error_detail(e)}"

            summary = result[:300].replace("\n", " ").replace("\r", "")
            if len(result) > 300:
                summary += "..."
            yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

        # Reset for next round
        assistant_content = []
        tool_use_blocks = []
        current_text = ""
        stop_reason = None

    yield json.dumps({"t": "done"}) + "\n"

    # Return token usage for cost tracking
    yield (
        json.dumps(
            {
                "t": "_usage",
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "model": model,
            }
        )
        + "\n"
    )


# ── SINGLE-SHOT (non-streaming, for internal use) ──────────────────────


async def chat_once(message: str, model: str, max_tokens: int = 2048) -> str:
    """
    Non-streaming single response. Useful for internal classification
    or quick lookups where streaming isn't needed.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "ERROR: No API key"

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": DELTAI_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": message}],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                try:
                    data = resp.json()
                    return f"ERROR: {data.get('error', {}).get('message', 'Unknown')}"
                except Exception:
                    return f"ERROR: HTTP {resp.status_code}"
            data = resp.json()
            content = data.get("content", [])
            return "".join(c.get("text", "") for c in content if c.get("type") == "text")
    except Exception as e:
        safe_errors.log_exception(logger, "chat_once failed", e)
        return f"ERROR: {safe_errors.public_error_detail(e)}"


async def split_workload_planner_outline(
    user_message: str,
    tool_results_context: str,
    rag_context: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    Optional leader pass before split synthesis: short outline from a (usually smaller)
    cloud model to structure the final answer. Returns empty string on skip/error.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return ""

    planner_model = model or os.getenv(
        "DELTAI_SPLIT_PLANNER_MODEL",
        os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514"),
    )
    system = build_cloud_system_prompt() + (
        "\n\nYou are a planning pass for SPLIT WORKLOAD synthesis. Output ONLY a short structured "
        "outline (bullet list is fine): (1) what was gathered, (2) conclusions supported by the "
        "evidence, (3) what remains uncertain. Do not write the final user-facing answer — "
        "another model will synthesize it. Be concise (max ~400 words)."
    )
    user_block = f"User request:\n{user_message}\n\n"
    if rag_context.strip():
        user_block += f"Prior RAG (hints only, may be stale):\n{rag_context}\n\n"
    user_block += f"Local tool results (ground truth for the host):\n{tool_results_context}"

    payload = {
        "model": planner_model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_block}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.warning("split planner HTTP %s", resp.status_code)
                return ""
            data = resp.json()
            content = data.get("content", [])
            return "".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
    except Exception as e:
        safe_errors.log_exception(logger, "split_workload_planner_outline failed", e)
        return ""
