# Security Policy

## Scope

This policy covers the deltai repository: the project FastAPI app under `project/`, the packaged `delta-daemon` under `src/delta/`, scripts in `scripts/`, and tracked extensions under `project/extensions/`.

## Default security posture

- The **project FastAPI app** binds to **loopback** (`127.0.0.1:8000`) and has **no app-level auth** on `POST /chat`, ingest, training, or tool execution by default.
- The **packaged daemon** (`delta-daemon`) listens on a **Unix domain socket** scoped to the running user.
- The `run_shell` tool executes `bash -c` as the daemon's Unix user. Its keyword blocklist is best-effort, **not** a sandbox — assume arbitrary code execution for that user if an attacker can drive tool calls.
- Optional shared secrets harden HTTP exposure when set in `project/.env`:
  - **`DELTAI_CHAT_API_KEY`** — locks `POST /chat`.
  - **`DELTAI_INGEST_API_KEY`** — locks `POST /ingest`, `/ingest/batch`, `/ingest/cleanup`, `/memory/ingest`, and `/ingest/pipeline/status`.
  - **`DELTAI_MCP_HTTP_KEY`** — locks the optional MCP Streamable HTTP mount.
  - **`DELTAI_CORS_ORIGINS`** — explicit browser allow-list (default is `*`).

For the full posture (binding guidance, MCP, Arch rollback gate, request correlation), see the **Security posture (operators)** section in [CLAUDE.md](CLAUDE.md).

## Supported versions

deltai is in early development and does not yet ship versioned releases. Security fixes are applied on `main` (the lightweight upstream branch) and propagate to `personal` overlays via merge. Pin to a specific commit SHA if you require stability.

## Reporting a vulnerability

- **Preferred:** use [GitHub Security Advisories](https://github.com/Reconnecting18/deltai/security/advisories/new) to report privately.
- **Alternative:** if advisories are not available, open an issue on `Reconnecting18/deltai` **without** sensitive proof-of-concept details and request a private channel.

When reporting, please include:
- deltai branch / commit SHA (or release tag).
- OS / distro and Python version.
- Whether the issue affects the project FastAPI app, `delta-daemon`, an extension, or a CLI tool.
- A minimal reproduction and impact summary.

You should expect an initial acknowledgement within a few days. Fixes for confirmed issues land on `main` first.

## Out of scope

- Misconfigurations that bind the project app to a non-loopback address without auth (`DELTAI_CHAT_API_KEY` / a reverse proxy).
- Reports against third-party dependencies — please file those upstream and link the advisory here.
- Behaviour that requires write access to the user's filesystem prior to exploitation (e.g. modifying `project/.env`).
