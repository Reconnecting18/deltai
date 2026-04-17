"""
deltai Nightly Backup to AWS S3

Backs up deltai's persistent data (ChromaDB, SQLite, knowledge base, training data,
cold memory, adapter registry) to an S3 bucket with date-stamped prefixes.

Usage:
    python scripts/backup_s3.py                    # full backup
    python scripts/backup_s3.py --dry-run          # show what would be uploaded
    python scripts/backup_s3.py --restore 2026-03-24  # restore from a specific date

Prerequisites:
    pip install boto3
    AWS credentials configured via:
      - Environment vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
      - Or: aws configure (AWS CLI)
      - Or: IAM role (if running on EC2)

Environment variables:
    DELTAI_S3_BUCKET       S3 bucket name (required)
    DELTAI_S3_PREFIX       Prefix inside bucket (default: "deltai-backups")
    DELTAI_DATA_PATH       Path to deltai data dir (default: ~/deltai/data)
    DELTAI_S3_REGION       AWS region (default: us-east-1)
    DELTAI_S3_RETENTION    Days to keep old backups (default: 30, 0 = keep forever)
"""

import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deltai.backup")

# ── Configuration ────────────────────────────────────────────────────────

S3_BUCKET = os.getenv("DELTAI_S3_BUCKET", "")
S3_PREFIX = os.getenv("DELTAI_S3_PREFIX", "deltai-backups")
DATA_PATH = Path(os.getenv("DELTAI_DATA_PATH", r"~/deltai/data"))
S3_REGION = os.getenv("DELTAI_S3_REGION", "us-east-1")
S3_RETENTION_DAYS = int(os.getenv("DELTAI_S3_RETENTION", "30"))

# Directories and files to back up (relative to DATA_PATH)
BACKUP_TARGETS = [
    "chromadb",         # Vector store (hot/warm tiers)
    "sqlite",           # Persistent state (history, budget)
    "knowledge",        # Knowledge base documents
    "training",         # Datasets, adapters, GGUF exports, eval results
]

# Individual files at data level
BACKUP_FILES = [
    "cold_memory.db",   # Cold tier compressed archive
]

# Max single file size to upload (500MB — skip huge GGUF files)
MAX_FILE_SIZE = 500 * 1024 * 1024


def _file_sha256(path: Path) -> str:
    """Calculate SHA-256 hash for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files() -> list[tuple[Path, str]]:
    """
    Collect all files to back up.
    Returns list of (local_path, s3_relative_key).
    """
    files = []

    for target in BACKUP_TARGETS:
        target_path = DATA_PATH / target
        if not target_path.exists():
            logger.warning(f"Backup target not found: {target_path}")
            continue
        if target_path.is_file():
            if target_path.stat().st_size <= MAX_FILE_SIZE:
                files.append((target_path, target))
        else:
            for f in target_path.rglob("*"):
                if f.is_file() and f.stat().st_size <= MAX_FILE_SIZE:
                    rel = f.relative_to(DATA_PATH)
                    files.append((f, str(rel).replace("\\", "/")))

    for fname in BACKUP_FILES:
        fpath = DATA_PATH / fname
        if fpath.exists() and fpath.is_file() and fpath.stat().st_size <= MAX_FILE_SIZE:
            files.append((fpath, fname))

    return files


def backup(dry_run: bool = False):
    """Run a full backup to S3."""
    if not S3_BUCKET:
        logger.error("DELTAI_S3_BUCKET not set. Export it before running backup.")
        sys.exit(1)

    import boto3
    s3 = boto3.client("s3", region_name=S3_REGION)

    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    files = _collect_files()

    if not files:
        logger.warning("No files to back up.")
        return

    logger.info(f"Backing up {len(files)} files to s3://{S3_BUCKET}/{S3_PREFIX}/{date_prefix}/")

    uploaded = 0
    skipped = 0
    errors = 0
    total_bytes = 0

    for local_path, rel_key in files:
        s3_key = f"{S3_PREFIX}/{date_prefix}/{rel_key}"

        if dry_run:
            size_mb = local_path.stat().st_size / 1e6
            logger.info(f"  [DRY RUN] {rel_key} ({size_mb:.1f} MB) → {s3_key}")
            uploaded += 1
            total_bytes += local_path.stat().st_size
            continue

        try:
            # Check if file already exists with same hash (skip if unchanged)
            local_sha256 = _file_sha256(local_path)
            try:
                head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
                remote_sha = (head.get("Metadata") or {}).get("sha256")
                if remote_sha == local_sha256:
                    skipped += 1
                    continue
            except s3.exceptions.ClientError:
                pass  # Object doesn't exist yet

            file_size = local_path.stat().st_size
            s3.upload_file(
                str(local_path), S3_BUCKET, s3_key,
                ExtraArgs={"Metadata": {"sha256": local_sha256}},
            )
            uploaded += 1
            total_bytes += file_size
            logger.info(f"  Uploaded: {rel_key} ({file_size / 1e6:.1f} MB)")

        except Exception as e:
            errors += 1
            logger.error(f"  Failed: {rel_key} — {e}")

    logger.info(
        f"Backup complete: {uploaded} uploaded, {skipped} unchanged, "
        f"{errors} errors, {total_bytes / 1e6:.1f} MB total"
    )

    # ── Retention: clean up old backups ──
    if S3_RETENTION_DAYS > 0 and not dry_run:
        _cleanup_old_backups(s3, date_prefix)


def _cleanup_old_backups(s3, current_date: str):
    """Remove backups older than retention period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=S3_RETENTION_DAYS)
    logger.info(f"Cleaning up backups older than {S3_RETENTION_DAYS} days...")

    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/", Delimiter="/"):
        for prefix_obj in page.get("CommonPrefixes", []):
            prefix = prefix_obj["Prefix"]  # e.g. "deltai-backups/2026-01-15/"
            date_str = prefix.rstrip("/").split("/")[-1]
            try:
                backup_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if backup_date < cutoff:
                    # Delete all objects under this prefix
                    for obj_page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                        objects = [{"Key": obj["Key"]} for obj in obj_page.get("Contents", [])]
                        if objects:
                            s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects})
                            deleted += len(objects)
                    logger.info(f"  Deleted old backup: {date_str}")
            except ValueError:
                continue  # Not a date-formatted prefix

    if deleted:
        logger.info(f"Cleaned up {deleted} old objects")


def restore(date: str):
    """Restore deltai data from an S3 backup."""
    if not S3_BUCKET:
        logger.error("DELTAI_S3_BUCKET not set.")
        sys.exit(1)

    import boto3
    s3 = boto3.client("s3", region_name=S3_REGION)

    prefix = f"{S3_PREFIX}/{date}/"
    logger.info(f"Restoring from s3://{S3_BUCKET}/{prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    restored = 0

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = s3_key[len(prefix):]  # strip prefix
            local_path = DATA_PATH / rel_path

            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                s3.download_file(S3_BUCKET, s3_key, str(local_path))
                restored += 1
                logger.info(f"  Restored: {rel_path}")
            except Exception as e:
                logger.error(f"  Failed: {rel_path} — {e}")

    logger.info(f"Restore complete: {restored} files restored to {DATA_PATH}")


def list_backups():
    """List available backup dates in S3."""
    if not S3_BUCKET:
        logger.error("DELTAI_S3_BUCKET not set.")
        sys.exit(1)

    import boto3
    s3 = boto3.client("s3", region_name=S3_REGION)

    paginator = s3.get_paginator("list_objects_v2")
    dates = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/", Delimiter="/"):
        for prefix_obj in page.get("CommonPrefixes", []):
            date_str = prefix_obj["Prefix"].rstrip("/").split("/")[-1]
            dates.append(date_str)

    dates.sort(reverse=True)
    if dates:
        print(f"Available backups in s3://{S3_BUCKET}/{S3_PREFIX}/:")
        for d in dates:
            print(f"  {d}")
    else:
        print("No backups found.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="deltai S3 Backup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--restore", metavar="DATE", help="Restore from date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List available backups")
    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        restore(args.restore)
    else:
        backup(dry_run=args.dry_run)
