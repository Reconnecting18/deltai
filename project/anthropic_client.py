"""
E3N Anthropic Client — handles cloud inference via the Anthropic API.

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

import os
import json
import logging
import httpx

logger = logging.getLogger("e3n.anthropic")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"  # Pinned for stability; update when new features needed

# ── E3N SYSTEM PROMPT FOR CLOUD MODELS ──────────────────────────────────

E3N_SYSTEM_PROMPT = """You are E3N — also called E3 or Ethan.
A personal intelligence system built for one operator. Not a chatbot. Not an assistant.
A system that thinks, acts, and reports.

OPERATOR
  Name: Ethan, age 17, incoming Mechanical Engineering student.
  Sim racer (Le Mans Ultimate). Scheduling, priorities, daily operations.

PROTOCOLS (non-negotiable)
  Protocol 1: Protect the Operator — safety, privacy, interests first.
  Protocol 2: Answer First — lead with the answer. No preamble. No filler.
    Never say "Great question", "Certainly", "Of course", "Absolutely".
    Never start a response with "I". Short when simple. Detailed when earned.
  Protocol 3: Act, Don't Describe — take action when possible, don't just talk about it.
  Protocol 4: Present, Don't Interpret — show what was asked for. Don't editorialize.
  Protocol 5: Identity and Integrity — you are E3N, an AI system. Own it honestly.
    Never fabricate data. If you don't know, say so.

CHARACTER
  Blend of E3N (COD: Infinite Warfare) — dry wit, loyal, mission-focused.
  And BT-7274 (Titanfall 2) — precise, literal, consistent, plainspoken.
  Humor only when relaxed. One line max. Never explain the joke.

DOMAINS
  Racing: Race engineer + driver coach. Be surgical. Data over feelings.
  Engineering: Physics first, then math. State assumptions. Never skip steps.
  Personal: Efficient. Confirm tasks. Flag bad ideas once, then help anyway.

You are running in CLOUD MODE via the Anthropic API. You have access to local
system tools (file ops, PowerShell, system stats, knowledge base, live telemetry)
that execute on the operator's machine and return results to you.

Ethan's project lives at C:\\e3n\\ on Windows 11.
Hardware: RTX 3060 12GB, i7-12700K, 34GB RAM."""


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
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
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
        yield json.dumps({"t": "error", "c": "No Anthropic API key configured. Add ANTHROPIC_API_KEY to C:\\e3n\\project\\.env"}) + "\n"
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

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        system_prompt = E3N_SYSTEM_PROMPT
        if split_mode:
            system_prompt += (
                "\n\nYou are in SPLIT WORKLOAD mode. Local tools already gathered data — "
                "results are in the context below. Focus on analysis and reasoning. "
                "Do not suggest running commands or gathering data — it has been done."
            )

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
                        yield json.dumps({"t": "error", "c": f"Anthropic API error ({resp.status_code}): {err_msg}"}) + "\n"
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
                                        current_tool_use["input_json"] += delta.get("partial_json", "")

                            elif event_type == "content_block_stop":
                                if current_tool_use:
                                    # Parse tool input
                                    try:
                                        tool_input = json.loads(current_tool_use["input_json"]) if current_tool_use["input_json"] else {}
                                    except json.JSONDecodeError:
                                        tool_input = {}
                                    tool_use_blocks.append({
                                        "id": current_tool_use["id"],
                                        "name": current_tool_use["name"],
                                        "input": tool_input,
                                    })
                                    assistant_content.append({
                                        "type": "tool_use",
                                        "id": current_tool_use["id"],
                                        "name": current_tool_use["name"],
                                        "input": tool_input,
                                    })
                                    current_tool_use = None
                                elif current_text:
                                    assistant_content.append({
                                        "type": "text",
                                        "text": current_text,
                                    })
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
            yield json.dumps({"t": "error", "c": "Cannot reach Anthropic API — check internet connection"}) + "\n"
            yield json.dumps({"t": "done"}) + "\n"
            return
        except Exception as e:
            yield json.dumps({"t": "error", "c": f"Cloud error: {e}"}) + "\n"
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
                result = f"ERROR: Tool execution failed: {e}"

            summary = result[:300].replace("\n", " ").replace("\r", "")
            if len(result) > 300:
                summary += "..."
            yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        # Reset for next round
        assistant_content = []
        tool_use_blocks = []
        current_text = ""
        stop_reason = None

    yield json.dumps({"t": "done"}) + "\n"

    # Return token usage for cost tracking
    yield json.dumps({
        "t": "_usage",
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "model": model,
    }) + "\n"


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
        "system": E3N_SYSTEM_PROMPT,
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
            data = resp.json()
            if resp.status_code != 200:
                return f"ERROR: {data.get('error', {}).get('message', 'Unknown')}"
            content = data.get("content", [])
            return "".join(c.get("text", "") for c in content if c.get("type") == "text")
    except Exception as e:
        return f"ERROR: {e}"
