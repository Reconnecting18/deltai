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
]

# Quick lookup by name
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}
