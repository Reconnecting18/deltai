"""
deltai Daily Autonomous Training Scheduler
Orchestrates the full daily self-improvement cycle for deltai's Qwen models.

Designed to run at 2:00 AM via a Linux scheduler (systemd user timer or cron).
Runs offline — does not require the FastAPI server to be running.

Phases:
  0. Guard checks (sim, session, VRAM, existing training)
  1. Weakness analysis (identify domains with avg quality < 0.6)
  2. Targeted distillation (teacher generates examples for weak domains)
  3. Daily curriculum (topic of the day, rotating weekly)
  4. QLoRA domain adapter training
  5. Eval + optional auto-promote
  6. Memory consolidation (warm→cold, gap review)
  7. Report (written to data/training/daily_reports/)

Usage:
  python scripts/daily_training.py [--dry-run] [--day N] [--no-train]

  --dry-run         Run all phases without actually training (log only)
  --day N           Override weekday (0=Mon, 1=Tue ... 6=Sun)
  --no-train        Skip QLoRA training phase (distillation only)
  --verbose         Extra logging
  --report-only     Print last report and exit
  --collect         Force-enable web collection phase before training
  --collect-only    Run web data collection only, skip all training

Configuration (.env):
  DAILY_TRAIN_ENABLED=true
  DAILY_TRAIN_MIN_VRAM_MB=7000
  DAILY_TRAIN_AUTO_PROMOTE=false
  DAILY_TRAIN_AUTO_MERGE=false
  SESSION_SYNTHESIS_ENABLED=true
  WEB_COLLECT_ENABLED=true
  WEB_COLLECT_WIKIPEDIA_BATCH=2000
  WEB_COLLECT_MAX_PER_SOURCE=200
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time

# ── Path setup: ensure project/ is importable ──────────────────────────────
# This file lives at project/extensions/training/daily_training.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Logging setup ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(_REPO_ROOT, "data", "training", "daily_reports")
os.makedirs(LOG_DIR, exist_ok=True)

_log_file = os.path.join(LOG_DIR, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("deltai.daily_training")


def _load_env():
    """Load .env file from project directory."""
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


def _check_enabled() -> bool:
    return os.getenv("DAILY_TRAIN_ENABLED", "true").lower() in ("true", "1", "yes")


def _get_vram_free_mb() -> int:
    """Get free VRAM in MB. Returns 0 on error."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free / 1024 / 1024)
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


def _is_sim_running() -> bool:
    """True if a configured foreground sim / heavy app process is running (VRAM guard)."""
    try:
        import psutil
        SIM_PROCESSES = os.getenv("SIM_PROCESS_NAMES",
                                  "LeMansUltimate_,LeMansUltimate.exe,LMU.exe").split(",")
        for proc in psutil.process_iter(["name"]):
            pname = (proc.info.get("name") or "").lower()
            if any(s.lower().rstrip("_") in pname for s in SIM_PROCESSES):
                return True
    except Exception:
        pass
    return False


def _check_session_active() -> bool:
    """Check if a GPU focus session is active via deltai /session/status API."""
    try:
        import httpx
        base_url = os.getenv("DELTAI_API_URL", "http://localhost:8000")
        resp = httpx.get(f"{base_url}/session/status", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("active", False)
    except Exception:
        pass
    return False


def _print_report(report_path: str):
    """Pretty-print the most recent daily report."""
    if not os.path.exists(report_path):
        print(f"No report found at {report_path}")
        return
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    print("\n" + "=" * 60)
    print(f"DAILY TRAINING REPORT — {report.get('date', 'unknown')}")
    print("=" * 60)
    print(f"Status:  {report.get('status', '?').upper()}")
    if report.get("skip_reason"):
        print(f"Skipped: {report['skip_reason']}")
    if report.get("dry_run"):
        print("Mode:    DRY RUN")

    phases = report.get("phases", {})

    guards = phases.get("guards", {})
    print("\n[Guards]")
    print(f"  VRAM free: {guards.get('vram_free_mb', '?')} MB")
    print(f"  Focus workload: {guards.get('focus_workload_active', guards.get('sim_running', '?'))}")
    print(f"  VRAM OK: {guards.get('vram_ok', '?')}")

    web = phases.get("web_collection", {})
    web_status = web.get("status", "disabled")
    if web_status != "disabled":
        print(f"\n[Web Collection] — status={web_status}")
        print(f"  Written: {web.get('total_written', 0)}, Skipped: {web.get('total_skipped', 0)}, Errors: {web.get('total_errors', 0)}")
        for src, r in web.get("sources", {}).items():
            extra = f" [offset->{r.get('offset_end', '?')}]" if src == "wikipedia" else ""
            print(f"  {src}: written={r.get('written', 0)}, status={r.get('status', '?')}{extra}")

    weak = phases.get("weakness_analysis", {}).get("weak_domains", [])
    print(f"\n[Weakness Analysis] — {len(weak)} weak domain(s)")
    for w in weak:
        print(f"  {w['domain']}: avg={w['avg_score']:.3f} ({w['samples']} samples)")

    distill = phases.get("targeted_distillation", [])
    print(f"\n[Targeted Distillation] — {len(distill)} domain(s)")
    for d in distill:
        print(f"  {d.get('domain')}: generated={d.get('generated', 0)}, status={d.get('status')}")

    curriculum = phases.get("curriculum", {})
    print(f"\n[Curriculum] — weekday {curriculum.get('weekday', '?')}")
    for ds in curriculum.get("datasets", []):
        print(f"  {ds['dataset']} ({ds['domain']}): {ds['examples']} examples")

    train = phases.get("training", {})
    print("\n[Training]")
    print(f"  Domain:  {train.get('domain', '?')}")
    print(f"  Status:  {train.get('status', '?')}")
    if train.get("blend"):
        print(f"  Blend:   {train['blend']} examples")
    if train.get("output"):
        print(f"  Output:  {train['output']}")
    if train.get("reason"):
        print(f"  Reason:  {train['reason']}")

    gaps = phases.get("knowledge_gaps", {})
    print(f"\n[Knowledge Gaps] — {gaps.get('unresolved_gaps', '?')} unresolved")

    errors = report.get("errors", [])
    if errors:
        print(f"\n[Errors] — {len(errors)}")
        for e in errors:
            print(f"  {e}")

    print("=" * 60)
    if report.get("report_file"):
        print(f"Report: {report['report_file']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="deltai Daily Autonomous Training Cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/daily_training.py               # Full cycle (thin wrapper)
  python project/extensions/training/daily_training.py --dry-run
  python project/extensions/training/daily_training.py --day 0
  python project/extensions/training/daily_training.py --no-train
  python project/extensions/training/daily_training.py --report-only
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all phases without training")
    parser.add_argument("--day", type=int, default=None, metavar="N",
                        help="Override weekday (0=Mon...6=Sun)")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip QLoRA training phase")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--report-only", action="store_true",
                        help="Print last report and exit")
    parser.add_argument("--collect", action="store_true",
                        help="Run web data collection phase before training (default: controlled by WEB_COLLECT_ENABLED in .env)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Run web data collection only, skip all training phases")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env
    _load_env()

    # --report-only: print last report and exit
    if args.report_only:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(LOG_DIR, f"{today}.json")
        # Try yesterday if today's not written yet
        if not os.path.exists(report_path):
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            report_path = os.path.join(LOG_DIR, f"{yesterday}.json")
        _print_report(report_path)
        return 0

    collect_only = args.collect_only
    force_collect = args.collect or collect_only

    logger.info("=" * 50)
    logger.info("deltai Daily Training Cycle starting")
    logger.info(f"  Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Dry run:      {args.dry_run}")
    logger.info(f"  Day:          {args.day if args.day is not None else 'auto'}")
    logger.info(f"  Auto train:   {not args.no_train}")
    logger.info(f"  Collect only: {collect_only}")
    logger.info("=" * 50)

    # ── --collect-only: run just the web collection phase ──────────────────
    if collect_only:
        logger.info("--collect-only: running web data collection, skipping training")
        if _is_sim_running():
            logger.warning("GPU focus workload detected — collection deferred.")
            return 0
        try:
            from collector import run_collection_cycle  # noqa: PLC0415
        except ImportError as e:
            logger.error(f"Failed to import collector module: {e}")
            logger.error(f"Ensure PYTHONPATH includes {_PROJECT_DIR}")
            return 1
        start_time = time.time()
        try:
            collect_report = run_collection_cycle(dry_run=args.dry_run)
        except Exception as e:
            logger.exception(f"Collection failed: {e}")
            return 1
        elapsed = round(time.time() - start_time, 1)
        logger.info(
            f"Collection complete in {elapsed}s — "
            f"written={collect_report.get('total_written', 0)}, "
            f"status={collect_report.get('status', '?')}"
        )
        return 0

    # Check if daily training is enabled
    if not _check_enabled():
        logger.info("DAILY_TRAIN_ENABLED=false — skipping cycle")
        return 0

    # Quick pre-flight guard checks (before importing heavy modules)
    min_vram = int(os.getenv("DAILY_TRAIN_MIN_VRAM_MB", "7000"))

    if _is_sim_running():
        logger.warning("GPU focus workload detected — daily training deferred. Will retry tomorrow.")
        return 0

    if _check_session_active():
        logger.warning("Active GPU focus session detected — daily training deferred.")
        return 0

    vram_free = _get_vram_free_mb()
    logger.info(f"VRAM free: {vram_free} MB (minimum: {min_vram} MB)")

    if vram_free < min_vram and not args.dry_run:
        logger.warning(
            f"Insufficient VRAM ({vram_free}MB < {min_vram}MB). "
            "Daily training deferred. Unload Ollama models or increase DAILY_TRAIN_MIN_VRAM_MB."
        )
        return 0

    # Force-enable collection if --collect was passed
    if force_collect:
        os.environ["WEB_COLLECT_ENABLED"] = "true"

    # Import training module (after guards pass, to avoid loading torch early)
    try:
        logger.info("Importing training module...")
        from training import run_daily_cycle  # noqa: PLC0415
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error(f"Ensure PYTHONPATH includes {_PROJECT_DIR}")
        return 1

    # Run the cycle
    logger.info("Starting daily cycle via training.run_daily_cycle()...")
    start_time = time.time()

    try:
        report = run_daily_cycle(
            force_day_override=args.day,
            dry_run=args.dry_run,
            auto_train=not args.no_train,
        )
    except Exception as e:
        logger.exception(f"Daily cycle raised an unhandled exception: {e}")
        return 1

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"Daily cycle complete in {elapsed}s — status: {report.get('status')}")

    if report.get("errors"):
        for err in report["errors"]:
            logger.warning(f"Cycle error: {err}")

    # Print summary
    _print_report(report.get("report_file", ""))

    # Exit code: 0 = success/skipped, 1 = error
    return 0 if report.get("status") in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
