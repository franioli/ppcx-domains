"""Run `ppcx_identify_domains.py` over multiple dates (local runner).

This script executes `ppcx_identify_domains.py` for many reference dates by
invoking it in subprocesses and managing concurrency with a
`ThreadPoolExecutor`. It is intended for interactive or single-node runs where
the launcher should manage parallelism. For large-scale batch processing and
cluster submission, use `ppcx_prepare_job_file.py` to generate a job list and
run it with GNU Parallel.

Any additional arguments passed to this script (that are not recognized flags)
will be forwarded directly to the underlying clustering script.

------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------

1. Run a range of dates locally with parallel threads:
    python ppcx_run_batch.py --date-range 2020-06-01 2020-06-05 --jobs 4

2. Run a single date:
    python ppcx_run_batch.py --dates 2020-07-01

3. Generate a job file with `ppcx_prepare_job_file.py` and run with GNU Parallel:
    python ppcx_prepare_job_file.py --date-range 2020-06-01 2020-08-01 > jobs.txt
    parallel -j 4 --bar --joblog run.log --resume < jobs.txt
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from ppcluster.utils.logger import setup_logger

logger = setup_logger(
    level=logging.INFO,
    name="ppcx",
    force=True,
    redirect_to_stdout=True,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--script-path",
        default="ppcx_identify_domains.py",
        help="Path to the clustering script to run (default: ppcx_identify_domains.py).",
    )

    # --- Date input options (not mutually exclusive: ranges + explicit dates can coexist) ---
    parser.add_argument(
        "--dates",
        help="Comma separated list of reference dates (YYYY-MM-DD). Example: 2020-01-01,2020-02-02",
    )
    parser.add_argument(
        "--dates-file",
        help="File with one date (YYYY-MM-DD) per line.",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        action="append",
        dest="date_ranges",
        help=(
            "A date range START END (YYYY-MM-DD). "
            "Can be repeated for multiple ranges. "
            "Example: --date-range 2016-06-01 2016-10-30 --date-range 2017-06-01 2017-10-30"
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes to run simultaneously with joblib (default 1).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current interpreter).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout (seconds) for each run subprocess (default none).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-to-file",
        default=True,
        action="store_true",
        help="Whether to log each subprocess output to a separate file (default: True).",
    )
    parser.add_argument(
        "--log-folder",
        default="logs",
        help="Folder to store subprocess logs when running in parallel (default: logs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands to stdout one per line and exit. Useful for piping to GNU Parallel.",
    )
    return parser.parse_known_args()


# === Date building ===


def _expand_date_range(start: str, end: str) -> list[str]:
    """Return all dates (inclusive) between start and end as YYYY-MM-DD strings."""
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    if ed < sd:
        raise SystemExit(f"Date range error: end '{end}' must be >= start '{start}'")
    return [
        (sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((ed - sd).days + 1)
    ]


def _load_dates_from_file(path: Path) -> list[str]:
    dates = []
    with open(path) as fh:
        for r in fh:
            s = r.strip()
            if not s:
                continue
            dates.append(s)
    return dates


def build_dates_list(args) -> list[str]:
    """
    Build a deduplicated, sorted list of dates from all CLI date sources.

    Sources (all optional, can be combined):
      - args.dates       : comma-separated dates
      - args.dates_file  : one date per line in a file
      - args.date_ranges : list of [start, end] pairs (from repeated --date-range)
    """
    if not any([args.dates, args.dates_file, args.date_ranges]):
        raise SystemExit(
            "No dates provided. Use --dates, --dates-file, or --date-range."
        )

    dates: set[str] = set()

    if args.dates:
        for d in args.dates.split(","):
            d = d.strip()
            if d:
                dates.add(d)

    if args.dates_file:
        dates.update(_load_dates_from_file(Path(args.dates_file)))

    if args.date_ranges:
        for start, end in args.date_ranges:
            dates.update(_expand_date_range(start, end))

    return sorted(dates)


# === Helper functions ===


def run_subprocess_task(
    cmd: list[str],
    identifier: str,
    timeout=None,
    cleanup_on_failure=False,
    capture_output=True,
    log_to_file=False,
    log_folder="logs",
) -> bool:
    """
    Generic function to run a command in a subprocess with logging and error handling.

    Args:
        cmd: List of command arguments to execute.
        identifier: Unique identifier for this task (e.g. date string), used for logs.
        timeout: Max execution time in seconds.
        cleanup_on_failure: Whether to trigger cleanup routine on failure.
        capture_output: (Unused in current logic, kept for signature compat)
        log_to_file: Whether to write stdout/stderr to a file.
        log_folder: Where to save log files.
    """
    # Run subprocess
    log_file = None
    if log_to_file:
        log_path = Path(log_folder)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"run_{identifier}.log"
        logger.info(f"START {identifier} (logging to {log_file})")
    else:
        logger.info(f"START {identifier}")

    # Ensure JAX in sub-processes doesn't hog all VRAM
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Limit each process to ~45% of GPU memory (safe for 2 concurrent jobs)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

    try:
        if log_to_file and log_file is not None:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
        else:
            # Sequential mode: stream directly to terminal
            proc = subprocess.run(
                cmd, capture_output=False, text=True, timeout=timeout, env=env
            )

        if proc.returncode == 0:
            logger.info(f"SUCCESS {identifier}")
            return True

        # Log minimal info on error, detailed info can be inspected if needed
        logger.error(
            f"FAILURE {identifier} (exit code {proc.returncode})\n"
            f"STDERR snippet: {proc.stderr[-500:] if proc.stderr else 'Empty'}"
        )
        if cleanup_on_failure:
            _cleanup_partial_results(identifier)
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT {identifier} (> {timeout}s)")
        if cleanup_on_failure:
            _cleanup_partial_results(identifier)
        return False

    except Exception as e:
        logger.error(f"EXCEPTION {identifier}: {e}")
        if cleanup_on_failure:
            _cleanup_partial_results(identifier)
        return False


def _cleanup_partial_results(date: str):
    """Remove partial output directories for a failed date."""
    import shutil

    # Common patterns for output directories
    patterns = [
        f"*_{date}_*",
        f"*{date.replace('-', '_')}*",
        f"*{date}*",
    ]

    # Search in common output locations
    base_dirs = [Path("."), Path("output"), Path("output_" + date[:4])]

    removed = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            for folder in base_dir.glob(pattern):
                if folder.is_dir():
                    try:
                        shutil.rmtree(folder)
                        removed.append(str(folder))
                        logger.info(f"[CLEANUP] Removed {folder}")
                    except Exception as exc:
                        logger.warning(f"[CLEANUP] Failed to remove {folder}: {exc}")

    if not removed:
        logger.debug(f"[CLEANUP] No partial results found for {date}")


if __name__ == "__main__":
    # parse arguments, including extra args for the clustering script
    args, extra_args = parse_args()

    # Update logger level based on arguments
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logger.setLevel(numeric_level)

    # Build deduplicated, sorted list of dates from all CLI sources
    dates = build_dates_list(args)

    # Build per-date commands
    tasks = []
    for d in dates:
        cmd = [args.python, args.script_path, "--date", d]
        if extra_args:
            cmd.extend(extra_args)
        tasks.append((d, cmd))

    # === Execute tasks ===
    logger.info(f"Total dates to process: {len(dates)}")
    if args.jobs > 1:
        logger.info(f"Running {len(tasks)} tasks with {args.jobs} parallel threads...")

        # We use ThreadPoolExecutor because the actual work happens in the
        # subprocess, not in the Python thread. This avoids the overhead
        # of spawning Python processes (multiprocessing/joblib).
        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_date = {
                executor.submit(
                    run_subprocess_task,
                    cmd=cmd,
                    identifier=d,
                    timeout=args.timeout,
                    log_to_file=args.log_to_file,
                    log_folder=args.log_folder,
                ): d
                for d, cmd in tasks
            }

            for future in tqdm(as_completed(future_to_date), total=len(tasks)):
                d = future_to_date[future]
                try:
                    results.append((d, future.result()))
                except Exception as exc:
                    logger.error(f"Task for {d} generated an exception: {exc}")
                    results.append((d, False))

    else:
        logger.info("Running in sequential mode.")
        results = []
        for d, cmd in tqdm(tasks):
            ok = run_subprocess_task(
                cmd=cmd,
                identifier=d,
                timeout=args.timeout,
                log_to_file=args.log_to_file,
                log_folder=args.log_folder,
            )
            results.append((d, ok))

    # === Summary ===
    success_count = sum(1 for _, ok in results if ok)
    total = len(results)
    logger.info("-" * 40)
    logger.info(f"SUMMARY: {success_count}/{total} succeeded.")
    if success_count < total:
        failed = [d for d, ok in results if not ok]
        logger.error(f"Failed dates ({len(failed)}): {', '.join(failed)}")
    logger.info("-" * 40)
