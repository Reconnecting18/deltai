"""Tool definitions for arch update guard (merged into tools.definitions)."""

ARCH_GUARD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arch_pending_updates_report",
            "description": (
                "Arch Linux: list pending package upgrades using checkupdates or pacman -Qu, "
                "with optional reverse-dependency hints (pactree). Returns JSON evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_reverse_deps": {
                        "type": "boolean",
                        "description": "If true, run pactree -ru for the first few pending packages (slower).",
                        "default": False,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arch_refresh_news_digest",
            "description": (
                "Fetch official Arch Linux news RSS (optional ArchWiki snippet), "
                "POST digest to deltai RAG via /ingest (source arch_news)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "wiki_query": {
                        "type": "string",
                        "description": "Optional ArchWiki search phrase (e.g. pacnew, mkinitcpio).",
                        "default": "",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Bypass short in-process rate limit between refreshes.",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arch_create_update_snapshot",
            "description": (
                "Create a stored Arch/deltai pre-update snapshot (pacman -Q, key /etc configs, "
                "Ollama tags, optional SQLite file copy). Returns JSON with snapshot_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Optional human-readable label.", "default": ""},
                    "include_reverse_deps": {
                        "type": "boolean",
                        "description": "Include pactree hints in pending summary inside snapshot.",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arch_compare_snapshots",
            "description": (
                "Compare two arch_guard snapshot IDs: package add/remove/upgrade and /etc file changes. "
                "Returns JSON diff with severity hints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_snapshot_id": {"type": "string", "description": "Earlier snapshot UUID."},
                    "to_snapshot_id": {"type": "string", "description": "Later snapshot UUID."},
                },
                "required": ["from_snapshot_id", "to_snapshot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arch_rollback_plan",
            "description": (
                "Plan rollback from a snapshot: list restore steps for /etc files and staged SQLite copy. "
                "With dry_run=false, stages SQLite restore; apply_etc restores captured /etc only if "
                "DELTAI_TOOL_AUTO_APPROVE=1 (opt-in). Otherwise use CLI or curl for explicit apply_etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "snapshot_id": {"type": "string", "description": "Target snapshot UUID."},
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, only return the plan JSON (default true).",
                        "default": True,
                    },
                    "apply_etc": {
                        "type": "boolean",
                        "description": (
                            "If dry_run is false: restore captured /etc files from the snapshot "
                            "(requires root for in-place writes). Only honored when "
                            "DELTAI_TOOL_AUTO_APPROVE=1 on the server."
                        ),
                        "default": False,
                    },
                },
                "required": ["snapshot_id"],
            },
        },
    },
]
