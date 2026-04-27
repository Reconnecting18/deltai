"""
Workstation sync manifest generator

Walks declared directory roots and emits a JSON manifest (paths, sizes, mtimes, optional SHA-256).
Use for reconciliation after offline work or before/after cloud sync.

Usage:
    python scripts/workstation_sync_manifest.py --output /tmp/manifest.json
    python scripts/workstation_sync_manifest.py --dry-run
    python scripts/workstation_sync_manifest.py --write-state --output /tmp/manifest.json

Environment variables:
    WORKSTATION_MANIFEST_ROOTS   Comma-separated roots (expanduser applied), e.g. ~/work,~/docs
    WORKSTATION_MANIFEST_CONFIG  Optional JSON file: {"roots": ["/abs/path", ...]} or
                                 [{"path": "/abs", "label": "work"}]

    DELTA_DATA_DIR             Used with --write-state (default ~/.local/share/deltai)

Flags:
    --hash-max-mb N   Hash files only up to N MiB (default 128); larger files omit sha256
    --no-hash         Skip hashing entirely
    --format jsonl    One JSON object per line instead of a single array
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deltai.workstation_manifest")

MI = 1024 * 1024


def _data_dir() -> Path:
    raw = (os.getenv("DELTA_DATA_DIR") or "").strip()
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return Path(os.path.expanduser("~/.local/share/deltai")).resolve()


def _load_roots() -> list[tuple[str, str | None]]:
    """Return list of (resolved_path, label)."""
    out: list[tuple[str, str | None]] = []
    cfg_path = (os.getenv("WORKSTATION_MANIFEST_CONFIG") or "").strip()
    if cfg_path:
        p = Path(os.path.expanduser(cfg_path))
        if not p.is_file():
            raise SystemExit(f"WORKSTATION_MANIFEST_CONFIG not a file: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "roots" in data:
            for item in data["roots"]:
                if isinstance(item, str):
                    out.append((str(Path(os.path.expanduser(item)).resolve()), None))
                elif isinstance(item, dict) and "path" in item:
                    label = item.get("label")
                    out.append((str(Path(os.path.expanduser(str(item["path"]))).resolve()), str(label) if label else None))
        else:
            raise SystemExit("WORKSTATION_MANIFEST_CONFIG JSON must have a 'roots' array")

    env_roots = (os.getenv("WORKSTATION_MANIFEST_ROOTS") or "").strip()
    if env_roots:
        for part in env_roots.split(","):
            part = part.strip()
            if part:
                out.append((str(Path(os.path.expanduser(part)).resolve()), None))

    if not out:
        raise SystemExit("Set WORKSTATION_MANIFEST_ROOTS and/or WORKSTATION_MANIFEST_CONFIG")

    seen: set[str] = set()
    unique: list[tuple[str, str | None]] = []
    for path, label in out:
        if path not in seen:
            seen.add(path)
            unique.append((path, label))
    return unique


def _file_sha256(path: Path, max_bytes: int) -> str | None:
    if path.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_state(patch: dict) -> None:
    dd = _data_dir()
    dd.mkdir(parents=True, exist_ok=True)
    path = dd / "workstation_cloud_state.json"
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    for k, v in patch.items():
        existing[k] = v
    existing["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Wrote state %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate workstation sync manifest")
    ap.add_argument("--dry-run", action="store_true", help="List roots only; no manifest body")
    ap.add_argument("--output", "-o", help="Write manifest here (default stdout)")
    ap.add_argument("--format", choices=("json", "jsonl"), default="json")
    ap.add_argument("--hash-max-mb", type=int, default=128, help="Max file size to hash (MiB)")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument(
        "--write-state",
        action="store_true",
        help="Update DELTA_DATA_DIR/workstation_cloud_state.json manifest summary",
    )
    args = ap.parse_args()

    try:
        roots = _load_roots()
    except SystemExit as e:
        logger.error("%s", e)
        return 2

    max_bytes = max(0, args.hash_max_mb) * MI
    entries: list[dict] = []
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    if args.dry_run:
        for root, label in roots:
            logger.info("[DRY RUN] root %s label=%s", root, label)
        return 0

    for root, label in roots:
        base = Path(root)
        if not base.is_dir():
            logger.warning("Skip missing directory: %s", base)
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            rel = str(f.resolve().relative_to(base.resolve()))
            rec: dict = {
                "root": str(base),
                "root_label": label,
                "rel_path": rel.replace("\\", "/"),
                "size": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            }
            if not args.no_hash and max_bytes > 0:
                try:
                    digest = _file_sha256(f, max_bytes)
                    if digest:
                        rec["sha256"] = digest
                except OSError:
                    rec["sha256_error"] = True
            entries.append(rec)

    payload: dict = {
        "schema": "workstation_sync_manifest",
        "schema_version": 1,
        "generated_at": generated_at,
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }

    text: str
    if args.format == "jsonl":
        lines = [json.dumps(e, separators=(",", ":")) for e in entries]
        text = "\n".join(lines) + ("\n" if lines else "")
    else:
        text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        logger.info("Wrote %s files to %s", len(entries), args.output)
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")

    if args.write_state:
        out_path = str(Path(args.output).resolve()) if args.output else None
        _merge_state(
            {
                "manifest": {
                    "last_output_path": out_path,
                    "file_count": len(entries),
                    "total_bytes": payload["total_bytes"],
                    "generated_at": generated_at,
                }
            }
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
