"""
deltai Web Training Data Collector — Standalone Runner
Collects training examples from Wikipedia, arXiv, OpenF1, Semantic Scholar,
and motorsport web sources. Runs standalone (no FastAPI required).

Usage:
  python scripts/collect_training_data.py [options]

  --dry-run           Log all actions without writing any data
  --source SOURCE     Run only a specific source: wikipedia, arxiv, openf1,
                      papers, motorsport, all (default: all enabled in .env)
  --batch N           Override wikipedia batch size (default: WEB_COLLECT_WIKIPEDIA_BATCH)
  --max-per-source N  Override max items per non-Wikipedia source
  --verbose           Enable debug logging
  --report            Print a summary report after collection

Examples:
  python scripts/collect_training_data.py
  python scripts/collect_training_data.py --dry-run
  python scripts/collect_training_data.py --source wikipedia --batch 5000
  python scripts/collect_training_data.py --source arxiv --verbose
  python scripts/collect_training_data.py --source openf1 --report

Configuration (.env):
  WEB_COLLECT_ENABLED=true
  WEB_COLLECT_WIKIPEDIA=true
  WEB_COLLECT_ARXIV=true
  WEB_COLLECT_OPENF1=true
  WEB_COLLECT_MOTORSPORT=true
  WEB_COLLECT_PAPERS=true
  WEB_COLLECT_WIKIPEDIA_BATCH=2000
  WEB_COLLECT_MAX_PER_SOURCE=200
"""

import sys
import os
import json
import time
import logging
import argparse
import datetime

# ── Path setup ──────────────────────────────────────────────────────────────
# This file lives at project/extensions/training/collect_training_data.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Logging ─────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(_REPO_ROOT, "data", "training", "collect_logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_file = os.path.join(_LOG_DIR, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("deltai.collect_runner")

_VALID_SOURCES = {"wikipedia", "arxiv", "openf1", "papers", "motorsport", "all"}


def _load_env():
    env_path = os.path.join(_PROJECT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = val


def _print_report(report: dict):
    print("\n" + "=" * 60)
    print(f"COLLECTION REPORT — {report.get('started_at', '?')[:19]}")
    print("=" * 60)
    print(f"Status:        {report.get('status', '?').upper()}")
    if report.get("dry_run"):
        print("Mode:          DRY RUN")
    print(f"Total written: {report.get('total_written', 0)}")
    print(f"Total skipped: {report.get('total_skipped', 0)}")
    print(f"Total errors:  {report.get('total_errors', 0)}")

    sources = report.get("sources", {})
    if sources:
        print(f"\nPer-source breakdown:")
        for src, r in sources.items():
            status = r.get("status", "?")
            written = r.get("written", 0)
            skipped = r.get("skipped", 0)
            errors = r.get("errors", 0)
            extra = ""
            if src == "wikipedia":
                extra = f" [offset {r.get('offset_start', '?')} → {r.get('offset_end', '?')}]"
            print(f"  {src:<15} status={status:<8} written={written:<5} skipped={skipped:<5} errors={errors}{extra}")

    finished = report.get("finished_at", "")
    started = report.get("started_at", "")
    if finished and started:
        try:
            elapsed = (
                datetime.datetime.fromisoformat(finished) -
                datetime.datetime.fromisoformat(started)
            ).total_seconds()
            print(f"\nElapsed: {elapsed:.1f}s")
        except Exception:
            pass

    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="deltai Web Training Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_training_data.py                        # All sources (wrapper)
  python project/extensions/training/collect_training_data.py --dry-run
  python project/extensions/training/collect_training_data.py --source wikipedia
  python project/extensions/training/collect_training_data.py --source arxiv --report
  python project/extensions/training/collect_training_data.py --batch 5000
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without writing any data")
    parser.add_argument("--source", default="all", choices=sorted(_VALID_SOURCES),
                        help="Source to collect from (default: all enabled in .env)")
    parser.add_argument("--batch", type=int, default=None, metavar="N",
                        help="Wikipedia batch size override")
    parser.add_argument("--max-per-source", type=int, default=None, metavar="N",
                        help="Max items per non-Wikipedia source")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--report", action="store_true",
                        help="Print full report after collection")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _load_env()

    logger.info("=" * 50)
    logger.info("deltai Web Training Data Collector starting")
    logger.info(f"  Date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Source:     {args.source}")
    logger.info(f"  Dry run:    {args.dry_run}")
    if args.batch:
        logger.info(f"  Batch:      {args.batch}")
    logger.info("=" * 50)

    try:
        from collector import run_collection_cycle  # noqa: PLC0415
    except ImportError as e:
        logger.error(f"Failed to import collector module: {e}")
        logger.error(f"Ensure {_PROJECT_DIR} is in PYTHONPATH")
        return 1

    # Resolve source list
    if args.source == "all":
        sources_arg = None  # collector will use .env flags
    else:
        sources_arg = [args.source]

    start = time.time()
    try:
        report = run_collection_cycle(
            dry_run=args.dry_run,
            sources=sources_arg,
            wikipedia_batch=args.batch,
            max_per_source=args.max_per_source,
        )
    except Exception as e:
        logger.exception(f"Collection raised unhandled exception: {e}")
        return 1

    elapsed = round(time.time() - start, 1)
    logger.info(
        f"Collection complete in {elapsed}s — "
        f"written={report.get('total_written', 0)}, "
        f"status={report.get('status', '?')}"
    )

    # Save report JSON alongside logs
    try:
        report_file = os.path.join(
            _LOG_DIR,
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}_report.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved: {report_file}")
    except Exception as e:
        logger.warning(f"Could not save report: {e}")

    if args.report:
        _print_report(report)
    else:
        # Always print brief summary
        print(
            f"\nCollection: written={report.get('total_written', 0)}, "
            f"skipped={report.get('total_skipped', 0)}, "
            f"errors={report.get('total_errors', 0)}, "
            f"status={report.get('status', '?')}"
        )

    return 0 if report.get("status") in ("ok", "partial", "dry_run") else 1


if __name__ == "__main__":
    sys.exit(main())
