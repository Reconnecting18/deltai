"""
Workstation cloud preflight checks

Verifies optional tooling and reachability targets before backup/sync jobs.
Suitable for systemd user timers or CI.

Usage:
    python scripts/workstation_preflight.py
    python scripts/workstation_preflight.py --dry-run
    python scripts/workstation_preflight.py --require-restic
    python scripts/workstation_preflight.py --write-state

Environment variables:
    DELTA_DATA_DIR          deltai data root (default ~/.local/share/deltai); state file lives here
    APPWRITE_ENDPOINT       e.g. https://cloud.appwrite.io/v1 — optional GET .../health check
    WORKSTATION_S3_BUCKET   optional; if set with boto3 + AWS creds, tries head_bucket
    AWS_REGION / region     passed to boto3 client (default us-east-1)

State (optional --write-state):
    Merges into $DELTA_DATA_DIR/workstation_cloud_state.json under key "preflight".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deltai.workstation_preflight")


def _data_dir() -> Path:
    raw = (os.getenv("DELTA_DATA_DIR") or "").strip()
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return Path(os.path.expanduser("~/.local/share/deltai")).resolve()


def _ping_appwrite(endpoint: str, timeout: float = 5.0) -> tuple[bool, str]:
    base = endpoint.rstrip("/")
    url = f"{base}/health" if not base.endswith("/health") else base
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            if 200 <= code < 300:
                return True, f"ok http {code}"
            return False, f"http {code}"
    except urllib.error.HTTPError as e:
        return False, f"http_error {e.code}"
    except urllib.error.URLError as e:
        return False, f"url_error {e.reason!r}"
    except TimeoutError:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error {e}"


def _check_s3_bucket(bucket: str, region: str, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, "[DRY RUN] skip S3 head_bucket"
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return False, "boto3 not installed (pip install boto3)"
    try:
        c = boto3.client("s3", region_name=region)
        c.head_bucket(Bucket=bucket)
        return True, "head_bucket ok"
    except ClientError as e:
        return False, f"s3 error {e}"
    except OSError as e:
        return False, f"os error {e}"


def _merge_state(data_dir: Path, patch: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "workstation_cloud_state.json"
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
    ap = argparse.ArgumentParser(description="Workstation cloud preflight checks")
    ap.add_argument("--dry-run", action="store_true", help="Log checks only where applicable")
    ap.add_argument("--require-restic", action="store_true", help="Fail if restic is not in PATH")
    ap.add_argument(
        "--write-state",
        action="store_true",
        help="Merge results into DELTA_DATA_DIR/workstation_cloud_state.json",
    )
    args = ap.parse_args()

    failures: list[str] = []
    details: dict = {"checks": {}}

    # rclone (optional but recommended)
    rclone = shutil.which("rclone")
    details["checks"]["rclone"] = {"present": bool(rclone), "path": rclone}
    if not rclone:
        failures.append("rclone not in PATH")

    restic = shutil.which("restic")
    details["checks"]["restic"] = {"present": bool(restic), "path": restic}
    if args.require_restic and not restic:
        failures.append("restic not in PATH (--require-restic)")

    endpoint = (os.getenv("APPWRITE_ENDPOINT") or "").strip()
    if endpoint:
        ok, msg = _ping_appwrite(endpoint)
        details["checks"]["appwrite"] = {"ok": ok, "detail": msg}
        if not ok:
            failures.append(f"appwrite: {msg}")
    else:
        details["checks"]["appwrite"] = {"ok": None, "detail": "APPWRITE_ENDPOINT unset"}

    bucket = (os.getenv("WORKSTATION_S3_BUCKET") or "").strip()
    region = (os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1").strip()
    if bucket:
        ok, msg = _check_s3_bucket(bucket, region, args.dry_run)
        details["checks"]["s3_bucket"] = {"ok": ok, "detail": msg, "bucket": bucket}
        if not ok:
            failures.append(f"s3: {msg}")
    else:
        details["checks"]["s3_bucket"] = {"ok": None, "detail": "WORKSTATION_S3_BUCKET unset"}

    ok_all = len(failures) == 0
    details["ok"] = ok_all
    details["failures"] = failures
    details["checked_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()

    if args.write_state:
        _merge_state(_data_dir(), {"preflight": details})

    if args.dry_run:
        for f in failures:
            logger.warning("[DRY RUN] would fail: %s", f)
        logger.info("[DRY RUN] exit 0 (no hard failure in dry-run mode)")
        return 0

    if failures:
        for f in failures:
            logger.error("%s", f)
        return 1
    logger.info("Preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
