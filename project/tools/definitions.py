"""
E3N Tool Definitions — JSON schemas for Ollama tool calling.
llama3.1 uses these to decide when and how to invoke tools.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file on the local filesystem. Use this when the operator asks about a file, wants to review code, or needs file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file (e.g., C:\\e3n\\project\\main.py)"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Default 200. Use for large files."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file on the local filesystem. Use when the operator asks to create, update, or save a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path for the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing file instead of overwriting. Default false."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory. Use when the operator asks what's in a folder or needs to find files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list subdirectories recursively (max 3 levels). Default false."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": "Execute a PowerShell command on the local Windows system. Use for system tasks, process management, app launching, or any shell operation. Commands run with normal user privileges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default 15."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get current system stats: GPU utilization, VRAM, CPU%, RAM, disk, temperature, running processes. Use when the operator asks about system status or performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_processes": {
                        "type": "boolean",
                        "description": "If true, include top 10 processes by memory usage. Default false."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search E3N's knowledge base (ChromaDB vector store) for information from ingested documents. Use when the operator asks about topics that might be in their knowledge files, references past notes, or asks you to recall something from their documents. Files in C:\\e3n\\data\\knowledge\\ are auto-ingested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what to find"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return. Default 5."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_stats",
            "description": "Get stats about E3N's knowledge base: number of chunks, files ingested, disk usage. Use when the operator asks about memory status or what's been ingested.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use when the operator asks about current events, recent information, specifications, or topics not in the knowledge base. Returns titles, URLs, and snippets. NOT available during active racing sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'Ferrari 296 GT3 specs', 'Python FastAPI tutorial')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return. Default 5, max 10."
                    }
                },
                "required": ["query"]
            }
        }
    },
]

# ── COMPUTATION DELEGATION TOOLS ─────────────────────────────────────

COMPUTATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression or run a short Python calculation. Use for any math beyond basic arithmetic: unit conversions, engineering formulas, fuel/tire calculations, statistics. Never do multi-step math in your response — use this tool instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python expression to evaluate (e.g., 'math.sqrt(144)', '9.81 * 75 * math.sin(math.radians(30))', 'statistics.mean([1.32, 1.34, 1.31])')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this calculation represents (e.g., 'Normal force on 30-degree incline for 75kg mass')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_data",
            "description": "Summarize structured data (numbers, lists, tables) into key statistics. Use when tool results return large datasets or when you need to identify trends, outliers, or averages before analyzing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data to summarize (JSON string, CSV, or text with numbers)"
                    },
                    "focus": {
                        "type": "string",
                        "description": "What aspect to focus on: 'trends', 'outliers', 'averages', 'distribution', or 'all'. Default 'all'."
                    }
                },
                "required": ["data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_reference",
            "description": "Look up a specific formula, constant, or reference value from the knowledge base. Faster and more targeted than search_knowledge — returns top 1-2 most relevant chunks. Use for quick factual lookups like 'Young's modulus of steel', 'Bernoulli equation', 'tire pressure formula'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up (e.g., 'Reynolds number formula', 'Pacejka magic formula coefficients', 'moment of inertia for cylinder')"
                    }
                },
                "required": ["query"]
            }
        }
    },
]

TOOLS.extend(COMPUTATION_TOOLS)

# ── SELF-DIAGNOSTIC TOOLS ────────────────────────────────────────────

DIAGNOSTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "self_diagnostics",
            "description": "Run E3N self-diagnostics. With no arguments, checks all subsystems (Ollama, ChromaDB, GPU/VRAM, voice, watcher, backup models, critical paths). With a subsystem specified, runs a deep check with fix suggestions. Use when something seems broken, slow, or after tool failures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "description": "Optional: deep-check one subsystem. One of: ollama, chromadb, gpu, voice, watcher, backup, paths. Omit for full sweep."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_ollama_models",
            "description": "Manage Ollama models in VRAM. Actions: 'status' (list loaded + available models with VRAM usage), 'unload' (remove a model from VRAM to free memory), 'preload' (load a model into VRAM). Use to manage VRAM pressure or switch models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'status', 'unload', or 'preload'"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name for unload/preload (e.g., 'e3n-qwen14b'). Required for unload/preload."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "repair_subsystem",
            "description": "Attempt a safe repair on an E3N subsystem. Available repairs: 'restart_watcher' (restart file watcher), 'clear_vram' (unload all models from VRAM), 'reindex_knowledge' (re-ingest all knowledge files), 'check_ollama' (verify/start Ollama). Use after self_diagnostics identifies an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repair": {
                        "type": "string",
                        "description": "Repair action: 'restart_watcher', 'clear_vram', 'reindex_knowledge', or 'check_ollama'"
                    }
                },
                "required": ["repair"]
            }
        }
    },
]

DIAGNOSTIC_TOOLS.append({
    "type": "function",
    "function": {
        "name": "resource_status",
        "description": "Get E3N resource self-manager status: current VRAM pressure and tier, loaded models, circuit breaker state, auto-recovery actions taken, sim/session detection. Use to understand system resource health and recent automatic interventions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
})

TOOLS.extend(DIAGNOSTIC_TOOLS)

# ── ADAPTER SURGERY TOOLS ────────────────────────────────────────────

ADAPTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "manage_adapters",
            "description": "Manage E3N's augmentation slot adapters: list trained adapters and their domains, train new domain-specific adapters (racing, engineering, personality, reasoning), merge adapters into production models via TIES, promote/rollback adapters, or check active adapter map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "train", "merge", "promote", "rollback"],
                        "description": "Action to perform. status: list all adapters and active slots. train: start domain training. merge: combine active adapters into production GGUF. promote: set adapter as active for its domain. rollback: revert domain to previous adapter.",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["racing", "engineering", "personality", "reasoning"],
                        "description": "Domain for train/rollback actions.",
                    },
                    "adapter_name": {
                        "type": "string",
                        "description": "Adapter name for promote action (e.g., 'racing-v2').",
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name for training. Auto-selected from domain if omitted.",
                    },
                },
                "required": ["action"],
            },
        },
    },
]

TOOLS.extend(ADAPTER_TOOLS)

# Quick lookup by name
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

# ── CONDITIONAL TELEMETRY TOOLS ───────────────────────────────────────
# Only available when TELEMETRY_API_URL is configured

import os as _os

_TELEMETRY_API_URL = _os.getenv("TELEMETRY_API_URL", "").strip()

TELEMETRY_TOOLS = []
if _TELEMETRY_API_URL:
    TELEMETRY_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_session_status",
                "description": "Get the current racing session status from the telemetry API: session type, elapsed time, flag status, weather, track conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_lap_summary",
                "description": "Get lap time summary from telemetry: last lap, best lap, sector times, delta to best, position. Optionally specify a lap number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lap_number": {
                            "type": "integer",
                            "description": "Specific lap number to query. Default: latest lap."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_tire_status",
                "description": "Get current tire status: temperatures (inner/middle/outer), pressures, wear percentage, compound for all four tires.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_strategy_recommendation",
                "description": "Get pit strategy recommendation from telemetry data: optimal pit window, tire compound suggestion, fuel target, estimated time loss.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "remaining_laps": {
                            "type": "integer",
                            "description": "Estimated remaining laps. If not provided, calculated from fuel and pace."
                        }
                    },
                    "required": []
                }
            }
        },
    ]
    # Add telemetry tools to main TOOLS list
    TOOLS.extend(TELEMETRY_TOOLS)
    # Update TOOL_MAP
    for t in TELEMETRY_TOOLS:
        TOOL_MAP[t["function"]["name"]] = t
