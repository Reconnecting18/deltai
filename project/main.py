"""deltai FastAPI application (development server on TCP :8000)."""
import os
import sys
import types

from dotenv import load_dotenv

load_dotenv()

from deltai_api.logging_setup import RequestIdMiddleware, configure_logging

configure_logging()

import deltai_api.core as core

_this = sys.modules[__name__]
for _name, _value in core.__dict__.items():
    if _name.startswith("__"):
        continue
    if isinstance(_value, types.ModuleType):
        continue
    setattr(_this, _name, _value)

import asyncio
import json
import platform
import time as _time

import httpx
import safe_errors
from path_guard import realpath_under

try:
    import pynvml
except Exception:
    pynvml = None

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = core._HERE

CHAT_DEPENDENCIES = [Depends(require_chat_api_key)] if core.DELTAI_CHAT_API_KEY else []

app = FastAPI(title="deltai", lifespan=core.lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=core._cors_allow_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)

app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")

if core.EXTENSIONS_AVAILABLE:
    try:
        core.load_extensions(app)
        from extensions import get_extension_tools
        from tools.definitions import _merge_extension_tools

        _merge_extension_tools(get_extension_tools())
    except Exception as _ext_err:
        core.logger.warning("Extensions failed to initialise: %s", _ext_err)

try:
    from mcp_http import maybe_mount_mcp_http

    maybe_mount_mcp_http(app)
except ImportError:
    pass


@app.get("/")
def root():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))


@app.post("/chat", dependencies=CHAT_DEPENDENCIES)
async def chat(req: ChatRequest):
    """
    Chat endpoint with smart routing + tool calling + RAG.

    Stream protocol (JSON lines):
      {"t":"route","backend":"...","model":"...","tier":N,"reason":"..."}
      {"t":"rag","n":N}
      {"t":"tool","n":"tool_name","a":{args}}
      {"t":"result","n":"tool_name","s":"summary"}
      {"t":"text","c":"chunk"}
      {"t":"done"}
      {"t":"error","c":"message"}
    """
    global _last_chat_start
    _last_chat_start = _time.time()
    # ── ROUTE DECISION ──────────────────────────────────────
    decision = await route(
        message=req.message,
        force_cloud=req.deep,
        force_local=req.force_local,
    )
    logger.info(f"Route: {decision}")

    # ── GREETING SHORT-CIRCUIT ───────────────────────────────
    greeting_response = _check_greeting(req.message)
    if greeting_response:

        async def stream_greeting():
            route_data = decision.to_dict()
            route_data["reason"] = "greeting — short-circuit"
            yield json.dumps({"t": "route", **route_data}) + "\n"
            yield json.dumps({"t": "text", "c": greeting_response}) + "\n"
            yield json.dumps({"t": "done"}) + "\n"

        logger.info(
            "Greeting short-circuit (message_len=%s reply_len=%s)",
            len(req.message or ""),
            len(greeting_response or ""),
        )
        return StreamingResponse(stream_greeting(), media_type="application/x-ndjson")

    # ── RAG CONTEXT ─────────────────────────────────────────
    rag_context = build_rag_context(req.message)

    # ── SPLIT WORKLOAD PATH ─────────────────────────────────
    # Phase 1: Local 3B model gathers data via tools (free, fast)
    # Phase 2: Cloud model reasons over the enriched context (premium quality)
    if decision.backend == "anthropic" and decision.split:

        async def stream_split():
            # ── Route event with split flag ──
            route_data = decision.to_dict()
            route_data["split"] = True
            yield json.dumps({"t": "route", **route_data}) + "\n"

            if rag_context:
                chunk_count = rag_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

            # ── Phase 1: Local tool gathering ──
            yield (
                json.dumps({"t": "split_phase", "phase": 1, "c": "Gathering data locally..."})
                + "\n"
            )

            local_model, cpu_only, backup = _pick_local_model()
            if not local_model:
                # No local model — fall back to standard cloud with tools
                yield (
                    json.dumps(
                        {
                            "t": "split_phase",
                            "phase": 0,
                            "c": "No local model — sending to cloud...",
                        }
                    )
                    + "\n"
                )
                full_response = ""
                async for line in anthropic_stream(
                    message=req.message,
                    model=decision.model,
                    rag_context=rag_context,
                    tools=TOOLS,
                    execute_tool_fn=execute_tool,
                    history=_get_history(),
                ):
                    if line.strip().startswith('{"t":"_usage"'):
                        try:
                            usage = json.loads(line.strip())
                            record_cloud_usage(
                                usage.get("input_tokens", 0),
                                usage.get("output_tokens", 0),
                                usage.get("model", decision.model),
                            )
                        except Exception:
                            pass
                        continue
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("t") == "text":
                            full_response += ev.get("c", "")
                        elif ev.get("t") == "done":
                            _append_to_history(req.message, full_response)
                            yield (
                                json.dumps({"t": "done", "turns": len(_conversation_history) // 2})
                                + "\n"
                            )
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    yield line
                return

            # Build messages for local model
            if rag_context:
                local_content = f"{rag_context}\n{req.message}"
            else:
                local_content = req.message
            local_messages = (
                [{"role": "system", "content": build_local_system_prompt()}]
                + _get_history()
                + [{"role": "user", "content": local_content}]
            )

            gathered = []  # list of {"tool": name, "args": args, "result": text}

            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    for _round_num in range(MAX_TOOL_ROUNDS):
                        (
                            data,
                            err,
                            used_model,
                            is_emergency,
                        ) = await _inference_with_emergency_fallback(
                            client,
                            local_model,
                            local_messages,
                            TOOLS,
                            backup,
                            num_gpu=decision.num_gpu,
                            num_ctx=decision.num_ctx,
                        )

                        if data is None:
                            logger.warning(f"Split Phase 1 failed: {err}")
                            break

                        msg = data.get("message", {})
                        tool_calls = msg.get("tool_calls")
                        content = msg.get("content", "")

                        # Text-as-tool fallback
                        if not tool_calls and content:
                            parsed = try_parse_text_tool_call(content)
                            if parsed:
                                name, args = parsed
                                tool_calls = [{"function": {"name": name, "arguments": args}}]
                                msg["tool_calls"] = tool_calls
                                msg["content"] = ""

                        if not tool_calls:
                            break  # No more tools to call

                        local_messages.append(msg)
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name", "unknown")
                            args = fn.get("arguments", {})
                            yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"
                            result = await asyncio.to_thread(execute_tool, name, args)
                            summary = result[:300].replace("\n", " ").replace("\r", "")
                            if len(result) > 300:
                                summary += "..."
                            yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"
                            # Truncate large results before packaging for cloud
                            truncated = result[:8000]
                            if len(result) > 8000:
                                truncated += f"\n... [truncated, {len(result)} chars total]"
                            gathered.append({"tool": name, "args": args, "result": truncated})
                            local_messages.append({"role": "tool", "content": result})

            except Exception as e:
                safe_errors.log_exception(logger, "Split Phase 1 exception", e)
                yield (
                    json.dumps(
                        {
                            "t": "retry",
                            "n": "split",
                            "c": "Phase 1 encountered an error, falling back to cloud...",
                        }
                    )
                    + "\n"
                )

            # ── Fallback: no tools called → standard cloud with tools ──
            if not gathered:
                yield (
                    json.dumps(
                        {
                            "t": "split_phase",
                            "phase": 0,
                            "c": "No data to gather — sending to cloud...",
                        }
                    )
                    + "\n"
                )
                full_response = ""
                async for line in anthropic_stream(
                    message=req.message,
                    model=decision.model,
                    rag_context=rag_context,
                    tools=TOOLS,
                    execute_tool_fn=execute_tool,
                    history=_get_history(),
                ):
                    if line.strip().startswith('{"t":"_usage"'):
                        try:
                            usage = json.loads(line.strip())
                            record_cloud_usage(
                                usage.get("input_tokens", 0),
                                usage.get("output_tokens", 0),
                                usage.get("model", decision.model),
                            )
                        except Exception:
                            pass
                        continue
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("t") == "text":
                            full_response += ev.get("c", "")
                        elif ev.get("t") == "done":
                            _append_to_history(req.message, full_response)
                            yield (
                                json.dumps({"t": "done", "turns": len(_conversation_history) // 2})
                                + "\n"
                            )
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    yield line
                return

            # ── Phase 2: Cloud reasoning over gathered data ──
            yield (
                json.dumps(
                    {"t": "split_phase", "phase": 2, "c": "Data gathered, reasoning in cloud..."}
                )
                + "\n"
            )

            # Format gathered results into context
            split_context_parts = ["[SPLIT WORKLOAD — Local tool results]"]
            for g in gathered:
                split_context_parts.append(
                    f"[Tool: {g['tool']} | Args: {json.dumps(g['args'])}]\n{g['result']}"
                )
            split_context_parts.append("[END SPLIT CONTEXT]")
            split_context = "\n\n".join(split_context_parts)

            planner_prefix = ""
            if os.getenv("DELTAI_SPLIT_PLANNER_ENABLED", "false").lower() in ("true", "1", "yes"):
                outline = await split_workload_planner_outline(
                    req.message,
                    split_context,
                    rag_context=rag_context or "",
                )
                if outline.strip():
                    planner_prefix = (
                        "[PLANNER OUTLINE — structured synthesis prep]\n"
                        f"{outline}\n[END PLANNER OUTLINE]\n\n"
                    )

            # Combine with RAG context if present
            if rag_context:
                combined_context = f"{rag_context}\n\n{planner_prefix}{split_context}"
            else:
                combined_context = f"{planner_prefix}{split_context}"

            split_synth_model = os.getenv("DELTAI_SPLIT_SYNTH_MODEL", "").strip() or decision.model

            full_response = ""
            async for line in anthropic_stream(
                message=req.message,
                model=split_synth_model,
                rag_context=combined_context,
                tools=None,
                split_mode=True,
                history=_get_history(),
            ):
                if line.strip().startswith('{"t":"_usage"'):
                    try:
                        usage = json.loads(line.strip())
                        record_cloud_usage(
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                            usage.get("model", decision.model),
                        )
                    except Exception:
                        pass
                    continue
                try:
                    ev = json.loads(line.strip())
                    if ev.get("t") == "text":
                        full_response += ev.get("c", "")
                    elif ev.get("t") == "done":
                        _append_to_history(req.message, full_response)
                        yield (
                            json.dumps({"t": "done", "turns": len(_conversation_history) // 2})
                            + "\n"
                        )
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                yield line

        return StreamingResponse(stream_split(), media_type="application/x-ndjson")

    # ── ANTHROPIC PATH ──────────────────────────────────────
    if decision.backend == "anthropic":

        async def stream_cloud():
            yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

            if rag_context:
                chunk_count = rag_context.count("[Source:")
                yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

            full_response = ""

            async for line in anthropic_stream(
                message=req.message,
                model=decision.model,
                rag_context=rag_context,
                tools=TOOLS,
                execute_tool_fn=execute_tool,
                history=_get_history(),
            ):
                # Intercept _usage events for cost tracking
                if line.strip().startswith('{"t":"_usage"'):
                    try:
                        usage = json.loads(line.strip())
                        record_cloud_usage(
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                            usage.get("model", decision.model),
                        )
                    except Exception:
                        pass
                    continue  # Don't send _usage to frontend
                # Accumulate text for history
                try:
                    ev = json.loads(line.strip())
                    if ev.get("t") == "text":
                        full_response += ev.get("c", "")
                    elif ev.get("t") == "done":
                        _append_to_history(req.message, full_response)
                        yield (
                            json.dumps({"t": "done", "turns": len(_conversation_history) // 2})
                            + "\n"
                        )
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                yield line

        return StreamingResponse(stream_cloud(), media_type="application/x-ndjson")

    # ── OLLAMA PATH (with tool calling) ─────────────────────
    if rag_context:
        user_content = f"{rag_context}\n{req.message}"
    else:
        user_content = req.message

    messages = (
        [{"role": "system", "content": build_local_system_prompt()}]
        + _get_smart_history(max_tokens=decision.num_ctx)
        + [{"role": "user", "content": user_content}]
    )

    # Filter tools based on query domain and complexity
    relevant_tools = filter_tools(
        TOOLS,
        domain=decision.adapter_domain,
        tier=decision.tier,
        category=decision.query_category,
        query=req.message,
    )

    # Track metadata for quality scoring and routing feedback
    _chat_metadata = {
        "tier": decision.tier,
        "domain": decision.adapter_domain or "general",
        "model": decision.model,
        "tool_calls": [],
        "tool_results": [],
        "react_used": False,
    }
    _chat_start_time = _time.time()

    async def stream_local():
        nonlocal messages

        yield json.dumps({"t": "route", **decision.to_dict()}) + "\n"

        if rag_context:
            chunk_count = rag_context.count("[Source:")
            yield json.dumps({"t": "rag", "n": chunk_count}) + "\n"

        # ── ReAct reasoning path for complex local queries ──
        if _is_react_eligible(decision):
            _chat_metadata["react_used"] = True
            yield json.dumps({"t": "react", "c": "Entering structured reasoning mode..."}) + "\n"
            async with httpx.AsyncClient(timeout=120) as react_client:
                final_text, tool_events = await _react_reasoning_loop(
                    react_client,
                    decision.model,
                    req.message,
                    rag_context,
                    _get_smart_history(max_tokens=decision.num_ctx),
                    relevant_tools,
                    execute_tool,
                    num_gpu=decision.num_gpu,
                    num_ctx=decision.num_ctx,
                )
                for ev in tool_events:
                    yield json.dumps(ev) + "\n"
                    if ev.get("t") == "tool":
                        _chat_metadata["tool_calls"].append(ev.get("n", ""))
                    if ev.get("t") == "result":
                        _chat_metadata["tool_results"].append(
                            {"name": ev.get("n", ""), "success": True}
                        )
                yield json.dumps({"t": "text", "c": final_text}) + "\n"
                _chat_metadata["latency_ms"] = (_time.time() - _chat_start_time) * 1000
                _append_to_history(req.message, final_text, chat_metadata=_chat_metadata)
                yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
            return

        # Track which model is active (may change if emergency backup engages)
        active_model = decision.model
        backup = decision.backup_model
        emergency_active = False

        rounds = 0
        async with httpx.AsyncClient(timeout=120) as client:
            while rounds < MAX_TOOL_ROUNDS:
                rounds += 1

                # ── Inference with retry-first, backup-last ──
                data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                    client,
                    active_model,
                    messages,
                    TOOLS,
                    backup if not emergency_active else None,
                    num_gpu=decision.num_gpu,
                    num_ctx=decision.num_ctx,
                )

                if is_emergency and not emergency_active:
                    emergency_active = True
                    active_model = used_model
                    backup = None  # don't cascade further
                    yield (
                        json.dumps(
                            {
                                "t": "emergency",
                                "c": f"PRIMARY MODEL OFFLINE — Running on backup: {used_model}",
                            }
                        )
                        + "\n"
                    )

                if data is None:
                    # User-friendly error messages
                    err_l = (err or "").lower()
                    if "timed out" in err_l:
                        friendly = "Model took too long to respond. Try a shorter question or check if Ollama is running."
                    elif "connect" in err_l or "refused" in err_l:
                        friendly = "Cannot reach Ollama. Make sure it's running (ollama serve)."
                    elif "all models failed" in err_l:
                        friendly = "All models are offline. Check Ollama and model availability."
                    else:
                        friendly = "Backend error. Please try again."
                    try:
                        from delta.storage.reports import write_chat_turn_report

                        _chat_metadata["latency_ms"] = (_time.time() - _chat_start_time) * 1000
                        write_chat_turn_report(
                            user_message=req.message,
                            assistant_response="",
                            chat_metadata={**_chat_metadata, "route": decision.to_dict()},
                            status="error",
                            user_visible_error=friendly,
                            internal_detail=err or "",
                        )
                    except ImportError:
                        pass
                    yield json.dumps({"t": "error", "c": friendly}) + "\n"
                    yield json.dumps({"t": "done"}) + "\n"
                    return

                msg = data.get("message", {})
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "")

                if not tool_calls and content:
                    parsed = try_parse_text_tool_call(content)
                    if parsed:
                        name, args = parsed
                        tool_calls = [{"function": {"name": name, "arguments": args}}]
                        msg["tool_calls"] = tool_calls
                        msg["content"] = ""

                if not tool_calls:
                    if content:
                        chunk_size = 4
                        for i in range(0, len(content), chunk_size):
                            yield json.dumps({"t": "text", "c": content[i : i + chunk_size]}) + "\n"
                    _append_to_history(req.message, content)
                    yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"
                    return

                messages.append(msg)
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", {})
                    yield json.dumps({"t": "tool", "n": name, "a": args}) + "\n"
                    try:
                        result = await asyncio.to_thread(execute_tool, name, args)
                    except Exception as tool_err:
                        safe_errors.log_exception(logger, "Tool execution crashed", tool_err)
                        result = "ERROR: Tool execution crashed"
                    # If tool errored, retry once with sanitized args
                    if result.startswith("ERROR:") and name in (
                        "run_shell",
                        "read_file",
                        "write_file",
                    ):
                        yield (
                            json.dumps(
                                {
                                    "t": "retry",
                                    "n": name,
                                    "c": "Retrying with adjusted parameters...",
                                }
                            )
                            + "\n"
                        )
                        sanitized_args = dict(args)
                        if "path" in sanitized_args:
                            sanitized_args["path"] = os.path.normpath(sanitized_args["path"])
                        if "timeout" in sanitized_args and name == "run_shell":
                            sanitized_args["timeout"] = min(
                                int(sanitized_args.get("timeout", 15)), 30
                            )
                        try:
                            result2 = await asyncio.to_thread(execute_tool, name, sanitized_args)
                            if not result2.startswith("ERROR:"):
                                result = result2
                        except Exception:
                            pass  # keep original error
                    summary = result[:300].replace("\n", " ").replace("\r", "")
                    if len(result) > 300:
                        summary += "..."
                    yield json.dumps({"t": "result", "n": name, "s": summary}) + "\n"
                    messages.append({"role": "tool", "content": result})

            # ── Max tool rounds — final response ──
            data, err, used_model, is_emergency = await _inference_with_emergency_fallback(
                client,
                active_model,
                messages,
                None,
                backup if not emergency_active else None,
                num_gpu=decision.num_gpu,
                num_ctx=decision.num_ctx,
            )
            if is_emergency and not emergency_active:
                yield (
                    json.dumps(
                        {
                            "t": "emergency",
                            "c": f"PRIMARY MODEL OFFLINE — Running on backup: {used_model}",
                        }
                    )
                    + "\n"
                )
            if data is not None:
                content = data.get("message", {}).get("content", "Max tool rounds reached.")
                for i in range(0, len(content), 4):
                    yield json.dumps({"t": "text", "c": content[i : i + 4]}) + "\n"
                _append_to_history(req.message, content)
            else:
                err_l = (err or "").lower()
                if "timed out" in err_l:
                    friendly = "Model took too long to respond. Try a shorter question or check if Ollama is running."
                elif "connect" in err_l or "refused" in err_l:
                    friendly = "Cannot reach Ollama. Make sure it's running (ollama serve)."
                elif "all models failed" in err_l:
                    friendly = "All models are offline. Check Ollama and model availability."
                else:
                    friendly = "Backend error. Please try again."
                try:
                    from delta.storage.reports import write_chat_turn_report

                    _chat_metadata["latency_ms"] = (_time.time() - _chat_start_time) * 1000
                    write_chat_turn_report(
                        user_message=req.message,
                        assistant_response="",
                        chat_metadata={**_chat_metadata, "route": decision.to_dict()},
                        status="error",
                        user_visible_error=friendly,
                        internal_detail=err or "",
                    )
                except ImportError:
                    pass
                yield json.dumps({"t": "error", "c": friendly}) + "\n"
            yield json.dumps({"t": "done", "turns": len(_conversation_history) // 2}) + "\n"

    return StreamingResponse(stream_local(), media_type="application/x-ndjson")


# ── CONVERSATION HISTORY ENDPOINTS ─────────────────────────────────────


@app.delete("/chat/history")
def clear_chat_history():
    """Clear conversation history."""
    _conversation_history.clear()
    try:
        db_clear_history()
    except Exception as e:
        logger.warning("Failed to clear DB history [%s]", type(e).__name__)
    return {"ok": True, "message": "History cleared"}


@app.get("/chat/history")
def get_chat_history():
    """Get conversation history metadata."""
    turns = len(_conversation_history) // 2
    return {"turns": turns, "max_turns": HISTORY_MAX_TURNS}


# ── ROUTER CONFIG ENDPOINTS ────────────────────────────────────────────


@app.get("/router/status")
def router_status():
    """Get current routing configuration and status."""
    return {
        "cloud_enabled": os.getenv("CLOUD_ENABLED", "true").lower() == "true",
        "cloud_available": is_cloud_available_sync(),
        "gpu_utilization": get_gpu_utilization(),
        "gpu_threshold": int(os.getenv("GPU_THRESHOLD", "70")),
        "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "sonnet_model": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-20250514"),
        "opus_model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-20250514"),
        "local_model": os.getenv("DELTAI_MODEL", "deltai"),
        "strong_model": os.getenv("DELTAI_STRONG_MODEL", ""),
    }


class CloudToggle(BaseModel):
    enabled: bool


@app.post("/router/cloud")
def toggle_cloud(req: CloudToggle):
    """Toggle cloud on/off at runtime."""
    os.environ["CLOUD_ENABLED"] = str(req.enabled).lower()
    return {"cloud_enabled": req.enabled}


# ── CLOUD BUDGET ENDPOINTS ────────────────────────────────────────────


@app.get("/budget/status")
def budget_endpoint():
    """Cloud cost budget status — daily spend, limit, remaining."""
    return get_budget_status()


# ── MEMORY ENDPOINTS ────────────────────────────────────────────────────


@app.get("/memory/stats")
def memory_stats_endpoint():
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        stats = get_memory_stats()
        stats["watcher_active"] = watcher_running()
        return stats
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.post("/memory/ingest")
def memory_ingest_endpoint(_ingest_auth: None = Depends(require_ingest_api_key)):
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        results = ingest_all()
        return {
            "results": results,
            "ingested": len([r for r in results if r["status"] == "ok"]),
            "skipped": len([r for r in results if r["status"] == "skipped"]),
            "errors": len([r for r in results if r["status"] == "error"]),
        }
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.get("/memory/files")
def memory_files_endpoint():
    """Get detailed per-file info from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available", "files": []}
    try:
        files = get_file_details()
        return {"files": files}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e), "files": []}


@app.delete("/memory/files/{filepath:path}")
def memory_delete_file(filepath: str):
    """Remove a specific file's chunks from ChromaDB."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        full_path = os.path.realpath(os.path.join(KNOWLEDGE_PATH, filepath))
        if not full_path.startswith(os.path.realpath(KNOWLEDGE_PATH)):
            return {"status": "error", "reason": "Invalid path — outside knowledge directory"}
        result = remove_file(full_path)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── STREAMING INGEST PIPELINE ─────────────────────────────────────────
# Async queue-based ingest: /ingest returns immediately, background worker
# batches and processes embeddings efficiently.

_INGEST_QUEUE_MAX = int(os.getenv("INGEST_QUEUE_MAX", "500"))
_INGEST_FLUSH_INTERVAL = float(os.getenv("INGEST_FLUSH_INTERVAL", "2.0"))  # seconds
_INGEST_FLUSH_BATCH_SIZE = int(os.getenv("INGEST_FLUSH_BATCH_SIZE", "10"))

_ingest_queue: asyncio.Queue | None = None
_ingest_metrics = {
    "queued": 0,
    "processed": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "last_flush": 0.0,
}


async def _ingest_pipeline_worker():
    """
    Background worker: drain ingest queue, batch-embed, write to ChromaDB.
    Flushes every FLUSH_INTERVAL seconds or when FLUSH_BATCH_SIZE items accumulate.
    """
    global _ingest_queue
    _ingest_queue = asyncio.Queue(maxsize=_INGEST_QUEUE_MAX)
    await asyncio.sleep(5)  # let startup finish
    logger.info("Ingest pipeline worker started")

    while True:
        batch = []
        try:
            # Collect items up to batch size or flush interval
            deadline = _time.time() + _INGEST_FLUSH_INTERVAL
            while len(batch) < _INGEST_FLUSH_BATCH_SIZE:
                timeout = max(0.1, deadline - _time.time())
                try:
                    item = await asyncio.wait_for(_ingest_queue.get(), timeout=timeout)
                    batch.append(item)
                except TimeoutError:
                    break

            if not batch or not RAG_AVAILABLE:
                continue

            # Process batch
            start = _time.time()
            items_for_batch = [
                {
                    "source": it["source"],
                    "context": it["context"],
                    "ttl": it["ttl"],
                    "tags": it["tags"],
                }
                for it in batch
            ]
            try:
                await asyncio.to_thread(ingest_context_batch, items_for_batch)
                _ingest_metrics["processed"] += len(batch)
                elapsed = (_time.time() - start) * 1000
                _ingest_metrics["avg_latency_ms"] = (
                    _ingest_metrics["avg_latency_ms"] * 0.8 + elapsed * 0.2
                )
            except Exception as e:
                _ingest_metrics["errors"] += len(batch)
                logger.warning("Ingest pipeline batch failed [%s]", type(e).__name__)

            _ingest_metrics["last_flush"] = _time.time()

            # Forward alerts
            for it in batch:
                if "alert" in it.get("tags", []):
                    alert = {
                        "source": it["source"],
                        "context": it["context"],
                        "tags": it["tags"],
                        "ts": _time.time(),
                        "priority": "critical"
                        if "critical" in it["tags"]
                        else "high"
                        if "high" in it["tags"]
                        else "normal",
                    }
                    _recent_alerts.append(alert)
                    await _broadcast_alert(alert)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            safe_errors.log_exception(logger, "Ingest pipeline error", e)
            await asyncio.sleep(1)


# ── INGEST CONNECTOR ──────────────────────────────────────────────────
# External services push structured context into deltai's RAG memory.


class IngestRequest(BaseModel):
    source: str  # identifies the sender (e.g., "lmu-telemetry")
    context: str  # human-readable content to embed
    ttl: int = 0  # seconds until auto-expiry (0 = permanent)
    tags: list[str] = []  # optional filtering tags


@app.post("/ingest")
async def ingest_endpoint(
    req: IngestRequest, _ingest_auth: None = Depends(require_ingest_api_key)
):
    """
    Ingest structured context from an external service into ChromaDB.
    This is deltai's connector — any service can push context here.
    When the async pipeline is active, returns immediately after queuing.
    """
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    if not req.source or not req.source.strip():
        return {"status": "error", "reason": "source is required"}
    if not req.context or not req.context.strip():
        return {"status": "error", "reason": "context is required"}

    # Try async pipeline first (non-blocking)
    if _ingest_queue is not None:
        try:
            _ingest_queue.put_nowait(
                {
                    "source": req.source.strip(),
                    "context": req.context.strip(),
                    "ttl": req.ttl,
                    "tags": req.tags,
                }
            )
            _ingest_metrics["queued"] += 1
            # Immediate alert forwarding (don't wait for pipeline)
            if "alert" in req.tags:
                alert = {
                    "source": req.source,
                    "context": req.context,
                    "tags": req.tags,
                    "ts": _time.time(),
                    "priority": "critical"
                    if "critical" in req.tags
                    else "high"
                    if "high" in req.tags
                    else "normal",
                }
                _recent_alerts.append(alert)
                await _broadcast_alert(alert)
            return {"status": "queued", "queue_depth": _ingest_queue.qsize()}
        except asyncio.QueueFull:
            return JSONResponse(
                status_code=429,
                content={"status": "error", "reason": "Ingest queue full — backpressure"},
            )

    # Fallback: synchronous ingest
    try:
        result = await asyncio.to_thread(
            ingest_context,
            source=req.source.strip(),
            context=req.context.strip(),
            ttl=req.ttl,
            tags=req.tags,
        )
        if "alert" in req.tags:
            alert = {
                "source": req.source,
                "context": req.context,
                "tags": req.tags,
                "ts": _time.time(),
                "priority": "critical"
                if "critical" in req.tags
                else "high"
                if "high" in req.tags
                else "normal",
            }
            _recent_alerts.append(alert)
            await _broadcast_alert(alert)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


class IngestBatchRequest(BaseModel):
    items: list[dict]  # each: {"source": str, "context": str, "ttl": int, "tags": list[str]}


@app.post("/ingest/batch")
async def ingest_batch_endpoint(
    req: IngestBatchRequest, _ingest_auth: None = Depends(require_ingest_api_key)
):
    """Batch ingest multiple context items. Max 100 per batch."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        result = await asyncio.to_thread(ingest_context_batch, req.items)
        # Forward any alerts
        for item in req.items:
            tags = item.get("tags", [])
            if "alert" in tags:
                alert = {
                    "source": item.get("source", ""),
                    "context": item.get("context", ""),
                    "tags": tags,
                    "ts": _time.time(),
                    "priority": "critical"
                    if "critical" in tags
                    else "high"
                    if "high" in tags
                    else "normal",
                }
                _recent_alerts.append(alert)
                await _broadcast_alert(alert)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


@app.get("/ingest/pipeline/status")
def ingest_pipeline_status(_ingest_auth: None = Depends(require_ingest_api_key)):
    """Get ingest pipeline queue metrics."""
    return {
        "pipeline_active": _ingest_queue is not None,
        "queue_depth": _ingest_queue.qsize() if _ingest_queue else 0,
        "queue_max": _INGEST_QUEUE_MAX,
        **_ingest_metrics,
    }


@app.post("/ingest/cleanup")
def ingest_cleanup(_ingest_auth: None = Depends(require_ingest_api_key)):
    """Manually trigger TTL cleanup of expired ingested entries."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return cleanup_expired()
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── HIERARCHICAL MEMORY ENDPOINTS ─────────────────────────────────────


@app.post("/memory/compact")
async def memory_compact():
    """Manually trigger warm→cold memory compaction."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        result = await asyncio.to_thread(compact_warm_to_cold)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


@app.get("/memory/cold/stats")
def cold_memory_stats():
    """Get cold tier storage statistics."""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available"}
    try:
        return get_cold_stats()
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


# ── KNOWLEDGE GAP ENDPOINTS ───────────────────────────────────────────


@app.get("/knowledge/gaps")
def knowledge_gaps():
    """Get unresolved knowledge gaps."""
    try:
        gaps = get_unresolved_gaps(limit=50)
        return {"gaps": gaps, "count": len(gaps)}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.post("/knowledge/gaps/{gap_id}/resolve")
def resolve_gap(gap_id: int):
    """Mark a knowledge gap as resolved."""
    try:
        resolve_knowledge_gap(gap_id)
        return {"status": "ok", "resolved": gap_id}
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── TRAINING INTELLIGENCE ENDPOINTS ──────────────────────────────────


@app.get("/training/weaknesses")
def training_weaknesses():
    """Identify domains where deltai performs poorly."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training not available"}
    try:
        from training import identify_weak_domains

        return {"weaknesses": identify_weak_domains()}
    except Exception as e:
        return {"error": safe_errors.public_error_detail(e)}


@app.post("/training/improve/{domain}")
async def training_improve(domain: str):
    """Trigger an improvement cycle for a weak domain."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training not available"}
    try:
        from training import run_improvement_cycle

        result = await asyncio.to_thread(run_improvement_cycle, domain)
        return result
    except Exception as e:
        return {"status": "error", "reason": safe_errors.public_error_detail(e)}


# ── WEBSOCKET ALERTS ─────────────────────────────────────────────────


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for proactive alerts (e.g. tagged ingest notifications)."""
    await websocket.accept()
    _alert_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _alert_clients.remove(websocket)
        except ValueError:
            pass


@app.get("/alerts/recent")
def recent_alerts():
    """Get the last 20 alerts."""
    return {"alerts": list(_recent_alerts)}


# ── HEALTH EVENT BUS ENDPOINTS ────────────────────────────────────────


@app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    """WebSocket endpoint for real-time health state events."""
    await websocket.accept()
    _health_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _health_clients.remove(websocket)
        except ValueError:
            pass


@app.get("/health/events")
def health_events(limit: int = 50):
    """Get recent health events."""
    return {"events": list(_health_events)[-limit:]}


# ── SELF-HEAL STATUS ─────────────────────────────────────────────────


@app.get("/self-heal/status")
def self_heal_status():
    """AI self-heal loop status."""
    heal_actions = [
        a for a in _resource_state["actions_taken"] if "Self-heal" in a.get("action", "")
    ]
    return {
        "enabled": _SELF_HEAL_ENABLED,
        "interval_sec": _SELF_HEAL_INTERVAL,
        "model": _SELF_HEAL_MODEL,
        "idle": _is_idle(),
        "circuit_breaker": _circuit_breaker["state"],
        "recent_actions": heal_actions[-10:],
    }


# ── RESOURCE MANAGEMENT STATUS ────────────────────────────────────────


@app.get("/resources/status")
def resource_status():
    """Resource self-manager status — VRAM, circuit breaker, auto-recovery."""
    vram_free = get_vram_free_mb()
    decline_rate = _predict_vram_decline()
    gpu_temp = _get_gpu_temp()
    return {
        "resource_manager": {
            "vram_free_mb": vram_free,
            "vram_warnings": _resource_state["vram_warnings"],
            "vram_decline_rate_mb_s": round(decline_rate, 1),
            "vram_prediction": "declining"
            if decline_rate > 50
            else "stable"
            if abs(decline_rate) < 20
            else "recovering",
            "gpu_temp_c": gpu_temp,
            "process_priority": "below_normal" if _resource_state["priority_lowered"] else "normal",
            "ollama_failures": _resource_state["ollama_failures"],
            "watcher_restarts": _resource_state["watcher_restarts"],
            "last_vram_action": _resource_state["last_vram_action"] or None,
            "last_recovery": _resource_state["last_recovery"] or None,
            "recent_actions": _resource_state["actions_taken"][-10:],
        },
        "circuit_breaker": {
            "state": _circuit_breaker["state"],
            "failures": _circuit_breaker["failures"],
            "backoff_sec": _circuit_breaker["backoff_sec"],
        },
    }


# ── SUBSYSTEM HEALTH ──────────────────────────────────────────────────


@app.get("/api/health")
async def system_health():
    """Returns status of all deltai subsystems for the header health monitor."""
    results = {}

    # Ollama
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            results["ollama"] = "online" if r.status_code == 200 else "down"
    except Exception:
        results["ollama"] = "down"

    # FastAPI — always online if this endpoint responds
    results["fastapi"] = "online"

    # ChromaDB / RAG
    if RAG_AVAILABLE:
        try:
            stats = get_memory_stats()
            results["chromadb"] = "online" if stats else "degraded"
        except Exception:
            results["chromadb"] = "degraded"
    else:
        results["chromadb"] = "down"

    # Smart Router
    try:
        model, cpu_only, backup = _pick_local_model()
        results["router"] = "local" if model else "down"
    except Exception:
        results["router"] = "down"

    # File Watcher
    if RAG_AVAILABLE:
        try:
            results["watcher"] = "active" if watcher_running() else "stopped"
        except Exception:
            results["watcher"] = "stopped"
    else:
        results["watcher"] = "stopped"

    # Tool System
    results["tools"] = "online" if len(TOOLS) > 0 else "down"

    # Anthropic API
    results["anthropic"] = "ready" if is_cloud_available_sync() else "standby"

    # Telemetry API (conditional — only when configured)
    if TELEMETRY_API_URL:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{TELEMETRY_API_URL}/health")
                results["telemetry"] = "online" if r.status_code == 200 else "down"
        except Exception:
            results["telemetry"] = "down"

    return results


# ── EMERGENCY BACKUP STATUS ────────────────────────────────────────────


@app.get("/backup/status")
def backup_status():
    """Emergency backup system status."""
    backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() in ("true", "1", "yes")
    strong = os.getenv("DELTAI_STRONG_MODEL", "deltai-qwen14b")
    default = os.getenv("DELTAI_MODEL", "deltai-qwen3b")
    return {
        "enabled": backup_enabled,
        "mapping": {
            strong: os.getenv("DELTAI_BACKUP_STRONG_MODEL", "deltai-nemo"),
            default: os.getenv("DELTAI_BACKUP_MODEL", "deltai-fallback"),
        },
        "models": _backup_health_status,
        "last_activated": _backup_last_activated if _backup_last_activated > 0 else None,
        "health_interval_sec": int(os.getenv("BACKUP_HEALTH_INTERVAL_SEC", "3600")),
        "max_retries_before_backup": BACKUP_MAX_RETRIES,
    }


# ── TRAINING ENDPOINTS ─────────────────────────────────────────────────


class DatasetCreate(BaseModel):
    name: str


class ExampleAdd(BaseModel):
    input: str | None = ""
    output: str
    instruction: str | None = ""
    category: str = "general"


class ExportRequest(BaseModel):
    format: str = "alpaca"


@app.get("/training/datasets")
def training_list_datasets():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable", "datasets": []}
    return {"datasets": list_datasets()}


@app.post("/training/datasets")
def training_create_dataset(req: DatasetCreate):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return create_dataset(req.name)


@app.delete("/training/datasets/{name}")
def training_delete_dataset(name: str):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return delete_dataset(name)


@app.get("/training/datasets/{name}")
def training_get_dataset(name: str):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable", "examples": []}
    return get_dataset(name)


@app.post("/training/datasets/{name}/add")
def training_add_example(name: str, req: ExampleAdd):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    # Combine instruction + input for Alpaca-style training
    input_text = req.input or ""
    if req.instruction:
        input_text = req.instruction + ("\n" + input_text if input_text else "")
    return add_example(name, input_text, req.output, req.category)


@app.delete("/training/datasets/{name}/{idx}")
def training_remove_example(name: str, idx: int):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return remove_example(name, idx)


@app.post("/training/datasets/{name}/export")
def training_export_dataset(name: str, req: ExportRequest):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return export_dataset(name, req.format)


@app.get("/training/status")
def training_status():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return get_training_status()


class TrainingStart(BaseModel):
    dataset: str
    base_model: str = "deltai-qwen3b"
    output_model: str | None = None
    mode: str = "auto"  # "lora", "fewshot", "auto", or "distill"
    teacher_dataset: str | None = None  # for distill mode
    replay_datasets: list[str] | None = None  # for distill mode


@app.post("/training/start")
def training_start(req: TrainingStart):
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return start_training(
        dataset_name=req.dataset,
        base_model=req.base_model,
        output_model=req.output_model,
        mode=req.mode,
        teacher_dataset=req.teacher_dataset,
        replay_datasets=req.replay_datasets,
    )


@app.post("/training/stop")
def training_stop():
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    return stop_training()


@app.get("/training/lora/status")
def training_lora_status():
    """Check if LoRA training dependencies are installed."""
    if not TRAINING_AVAILABLE:
        return {"available": False, "reason": "Training system unavailable"}
    from training import check_lora_deps

    ok, reason = check_lora_deps()
    return {"available": ok, "reason": reason if not ok else "ready"}


class ABEvalRequest(BaseModel):
    model_a: str
    model_b: str
    dataset: str
    max_examples: int = 20


@app.post("/training/eval/ab")
def training_ab_eval(req: ABEvalRequest):
    """Run A/B evaluation comparing two models on a dataset."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import run_ab_eval

    return run_ab_eval(req.model_a, req.model_b, req.dataset, req.max_examples)


# ── ADAPTER SURGERY ENDPOINTS ────────────────────────────────────────
# Modular LoRA augmentation slots: train, merge, evaluate, promote.


@app.get("/adapters")
def adapters_list(domain: str = None, status: str = None):
    """List all registered adapters, optionally filtered."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import list_adapters

    return {"adapters": list_adapters(domain=domain, status=status)}


@app.get("/adapters/active")
def adapters_active():
    """Get the current active adapter map."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_active_adapters

    return {"active": get_active_adapters()}


@app.get("/adapters/{name}")
def adapters_get(name: str):
    """Get details for a specific adapter."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_adapter

    result = get_adapter(name)
    if not result:
        return JSONResponse({"error": f"Adapter not found: {name}"}, status_code=404)
    return result


class AdapterTrainRequest(BaseModel):
    domain: str
    dataset: str = None
    freeze_layers: int = None
    lr: float = None
    epochs: int = None


@app.post("/adapters/train")
def adapters_train(req: AdapterTrainRequest):
    """Start domain-targeted adapter training for an augmentation slot."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import start_domain_training

    return start_domain_training(
        domain=req.domain,
        dataset_name=req.dataset,
        freeze_layers=req.freeze_layers,
        lr_override=req.lr,
        epochs_override=req.epochs,
    )


class AdapterMergeRequest(BaseModel):
    adapters: list[str] = None
    method: str = "ties"
    density: float = 0.5
    output_model: str = None


@app.post("/adapters/merge")
def adapters_merge(req: AdapterMergeRequest):
    """Merge active adapters into a single production GGUF model via TIES."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import merge_adapters

    return merge_adapters(
        adapter_names=req.adapters,
        method=req.method,
        density=req.density,
        output_model=req.output_model,
    )


@app.post("/adapters/eval/{name}")
def adapters_eval(name: str, dataset: str = None, max_examples: int = 20):
    """Evaluate an adapter against the baseline model."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import eval_adapter

    return eval_adapter(name, eval_dataset=dataset, max_examples=max_examples)


@app.post("/adapters/promote/{name}")
def adapters_promote(name: str):
    """Promote an adapter to the active slot for its domain."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import get_adapter, set_active_adapter, update_adapter

    info = get_adapter(name)
    if not info:
        return JSONResponse({"error": f"Adapter not found: {name}"}, status_code=404)
    domain = info.get("domain")
    result = set_active_adapter(domain, name)
    if result.get("status") == "ok":
        update_adapter(name, promoted=True)
    return result


class AdapterRollbackRequest(BaseModel):
    domain: str
    version: int = None


@app.post("/adapters/rollback")
def adapters_rollback(req: AdapterRollbackRequest):
    """Roll back a domain to a previous adapter version."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import rollback_adapter

    return rollback_adapter(req.domain, target_version=req.version)


@app.delete("/adapters/{name}")
def adapters_delete(name: str, delete_files: bool = False):
    """Remove an adapter from the registry."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import remove_adapter

    return remove_adapter(name, delete_files=delete_files)


# ── KNOWLEDGE DISTILLATION ENDPOINTS ─────────────────────────────────


class TeacherGenerateRequest(BaseModel):
    queries: list[str]
    teacher: str = "local14b"  # "local14b" or "anthropic"
    dataset: str = "deltai-teacher"
    category: str = "distilled"


@app.post("/training/generate-teacher")
def training_generate_teacher(req: TeacherGenerateRequest):
    """Generate training data using a teacher model (14B local or Anthropic API)."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    if req.teacher not in ("local14b", "anthropic"):
        return {"error": f"Unknown teacher: {req.teacher}. Use 'local14b' or 'anthropic'."}
    if not req.queries:
        return {"error": "No queries provided"}
    from training import generate_teacher_data

    return generate_teacher_data(
        queries=req.queries,
        teacher=req.teacher,
        dataset_name=req.dataset,
        category=req.category,
    )


class BlendRequest(BaseModel):
    sources: list[dict]  # [{"dataset":"name", "weight":0.3, "max_examples":100}]
    output: str


@app.post("/training/blend")
def training_blend(req: BlendRequest):
    """Blend multiple datasets with weighted sampling for distillation training."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    if not req.sources:
        return {"error": "No source datasets provided"}
    if not req.output:
        return {"error": "No output dataset name provided"}
    from training import blend_datasets

    return blend_datasets(sources=req.sources, output_name=req.output)


class RetentionRequest(BaseModel):
    model: str
    baseline: str = "deltai-qwen3b"
    min_pass_rate: float = 0.7


@app.post("/training/verify-retention")
def training_verify_retention(req: RetentionRequest):
    """Verify a trained model hasn't lost base capabilities."""
    if not TRAINING_AVAILABLE:
        return {"error": "Training system unavailable"}
    from training import verify_retention

    return verify_retention(
        model_name=req.model,
        baseline_model=req.baseline,
        min_pass_rate=req.min_pass_rate,
    )


# ── STATS ───────────────────────────────────────────────────────────────


@app.get("/stats")
def stats():
    result = {}
    result["cpu"] = {
        "name": platform.processor() or "Intel i7-12700K",
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "percent": psutil.cpu_percent(interval=None),
        "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
    }
    ram = psutil.virtual_memory()
    result["ram"] = {
        "total_gb": round(ram.total / 1e9, 1),
        "used_gb": round(ram.used / 1e9, 1),
        "percent": ram.percent,
    }
    disk = psutil.disk_usage("/")
    result["disk"] = {
        "total_gb": round(disk.total / 1e9, 1),
        "used_gb": round(disk.used / 1e9, 1),
        "percent": disk.percent,
    }
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            name = pynvml.nvmlDeviceGetName(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            except Exception:
                power_limit = 170
            result["gpu"] = {
                "name": name if isinstance(name, str) else name.decode(),
                "vram_total_mb": round(mem.total / 1e6),
                "vram_used_mb": round(mem.used / 1e6),
                "vram_percent": round(mem.used / mem.total * 100, 1),
                "gpu_percent": util.gpu,
                "power_w": round(power, 1),
                "power_limit_w": round(power_limit, 1),
                "temp_c": temp,
            }
        except Exception as e:
            safe_errors.log_exception(logger, "stats GPU probe failed", e)
            result["gpu"] = {"error": safe_errors.public_error_detail(e)}
    else:
        result["gpu"] = {"error": "NVML unavailable"}
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        tags = resp.json()
        DELTAI_MODELS = {
            "deltai-qwen14b",
            "deltai-qwen3b",
            "deltai-nemo",
            "deltai-fallback",
            "deltai",
        }
        result["models"] = [
            m["name"] for m in tags.get("models", []) if m["name"].split(":")[0] in DELTAI_MODELS
        ]
    except Exception:
        result["models"] = []
    if RAG_AVAILABLE:
        try:
            ms = get_memory_stats()
            result["memory_mb"] = ms["disk_mb"]
            result["memory_chunks"] = ms["total_chunks"]
            result["memory_files"] = ms["total_files"]
        except Exception:
            result["memory_mb"] = 0
            result["memory_chunks"] = 0
            result["memory_files"] = 0
    else:
        try:
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(
                    os.path.expanduser(os.getenv("CHROMADB_PATH", "~/.local/share/deltai/chromadb"))
                )
                for f in fn
            )
            result["memory_mb"] = round(total / 1e6, 1)
        except Exception:
            result["memory_mb"] = 0
    try:
        active_model, _, _ = _pick_local_model()
        result["model"] = active_model
    except Exception:
        result["model"] = DELTAI_MODEL
    result["platform"] = platform.system() + " " + platform.release()
    result["rag_available"] = RAG_AVAILABLE
    result["cloud_available"] = is_cloud_available_sync()
    result["watcher_active"] = watcher_running() if RAG_AVAILABLE else False
    # Backup status — minimal, invisible infrastructure
    backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() in ("true", "1", "yes")
    all_healthy = (
        all(s.get("healthy", False) for s in _backup_health_status.values())
        if _backup_health_status
        else True
    )
    result["backup"] = {"enabled": backup_enabled, "healthy": all_healthy}
    result["budget"] = get_budget_status()
    return result


# ── MODELFILE ───────────────────────────────────────────────────────────

MODULES_DIR = os.path.realpath(os.path.join(_HERE, "..", "modelfiles"))
MODELFILE_PATH = os.path.join(MODULES_DIR, "deltai.modelfile")
MODULE_FILES = {
    "modelfile": MODELFILE_PATH,
    "protocols": os.path.join(MODULES_DIR, "protocols.md"),
    "personality": os.path.join(MODULES_DIR, "personality.md"),
    "pilot": os.path.join(MODULES_DIR, "pilot.md"),
}
_ALLOWED_MODULE_PATHS = frozenset(realpath_under(MODULES_DIR, p) for p in MODULE_FILES.values())


def _resolve_module_path(name: str) -> str | None:
    candidate = MODULE_FILES.get(name)
    if not candidate:
        return None
    try:
        resolved = realpath_under(MODULES_DIR, candidate)
    except ValueError:
        return None
    return resolved if resolved in _ALLOWED_MODULE_PATHS else None


@app.get("/modelfile")
def get_modelfile():
    path = _resolve_module_path("modelfile")
    if not path:
        return PlainTextResponse("# Modelfile path unavailable", status_code=500)
    try:
        with open(path, encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        safe_errors.log_exception(logger, "get_modelfile failed", e)
        return PlainTextResponse("# Error reading modelfile", status_code=500)


class ModelfileUpdate(BaseModel):
    content: str


@app.post("/modelfile")
def save_modelfile(update: ModelfileUpdate):
    path = _resolve_module_path("modelfile")
    if not path:
        return {"ok": False, "error": "Modelfile path unavailable"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e)}


@app.get("/module/{name}")
def get_module(name: str):
    path = _resolve_module_path(name)
    if not path:
        return PlainTextResponse(f"# Unknown module: {name}", status_code=404)
    try:
        with open(path, encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        safe_errors.log_exception(logger, f"get_module {name} failed", e)
        return PlainTextResponse(f"# Error reading {name}", status_code=500)


@app.post("/module/{name}")
def save_module(name: str, update: ModelfileUpdate):
    path = _resolve_module_path(name)
    if not path:
        return {"ok": False, "error": f"Unknown module: {name}"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": safe_errors.public_error_detail(e)}
