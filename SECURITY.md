# Security Policy

## deltai (this repository)

The FastAPI **project app** defaults to loopback binding and no auth on `POST /chat` or tools. For any non-localhost exposure, use a reverse proxy, firewall, and optional **`DELTAI_CHAT_API_KEY`** / **`DELTAI_INGEST_API_KEY`** (see [CLAUDE.md](CLAUDE.md) Security posture). Report sensitive issues privately to the maintainers (use GitHub Security Advisories if enabled for the repo).

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.
