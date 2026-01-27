import argparse
import logging
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from ppcluster.utils.logger import setup_logger

logger = setup_logger(
    level=logging.INFO,
    name="ppcx",
    force=True,
    # log_to_file=True,
    # log_folder="logs",
    redirect_to_stdout=True,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch launcher for ppcx_mcmc_clustering.py -- run many reference dates."
    )
    parser.add_argument(
        "--script-path",
        default="ppcx_mcmc_clustering.py",
        help="Path to the clustering script to run (default: ppcx_mcmc_clustering.py).",
    )
    dates_group = parser.add_mutually_exclusive_group(required=True)
    dates_group.add_argument(
        "--dates",
        help="Comma separated list of reference dates (YYYY-MM-DD). Example: 2020-01-01,2020-02-02",
    )
    dates_group.add_argument(
        "--date-range",
        action="store_true",
        help="Run all consecutive dates between --start and --end (inclusive).",
    )
    dates_group.add_argument(
        "--dates-file",
        help="File with one date (YYYY-MM-DD) per line.",
    )
    dates_group.add_argument(
        "--random",
        type=int,
        help="Pick N random dates between --start and --end (inclusive).",
    )
    parser.add_argument(
        "--start", help="Start date for random sampling or date range (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", help="End date for random sampling or date range (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--season",
        help="Optional season to restrict random sampling, format 'M-M' (months numeric 1-12 inclusive). Example: --season 6-10 for Jun-Oct.",
    )
    parser.add_argument(
        "--exclude-months",
        help="Optional months to exclude from random sampling. Format: comma separated months/ranges, e.g. '1-5,11-12' to exclude Jan-May and Nov-Dec.",
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
    return parser.parse_known_args()


def run_one_date(
    python: str,
    script: str,
    date: str,
    timeout=None,
    cleanup_on_failure=False,
    extra_args=None,
) -> bool:
    # Build command str
    cmd = [python, script, "--date", date]
    if extra_args:
        cmd.extend(extra_args)

    # Run subprocess
    logger.info(f"START {date}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if proc.returncode == 0:
            logger.info(f"SUCCESS {date}")
            return True

        # Log minimal info on error, detailed info can be inspected if needed
        logger.error(
            f"FAILURE {date} (exit code {proc.returncode})\n"
            f"STDERR snippet: {proc.stderr[-500:] if proc.stderr else 'Empty'}"
        )
        if cleanup_on_failure:
            _cleanup_partial_results(date)
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT {date} (> {timeout}s)")
        if cleanup_on_failure:
            _cleanup_partial_results(date)
        return False

    except Exception as e:
        logger.error(f"EXCEPTION {date}: {e}")
        if cleanup_on_failure:
            _cleanup_partial_results(date)
        return False


# === Helper functions ===


def _load_dates_from_file(path: Path):
    dates = []
    with open(path) as fh:
        for r in fh:
            s = r.strip()
            if not s:
                continue
            dates.append(s)
    return dates


def _parse_months_spec(spec: str):
    """
    Parse a months spec like '1-5,11-12,7' into a set of ints {1,2,3,4,5,11,12,7}
    """
    out = set()
    if not spec:
        return out
    parts = spec.split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            if a_i <= b_i:
                out.update(range(a_i, b_i + 1))
            else:
                # wrap-around e.g. 11-2 -> 11,12,1,2
                out.update(list(range(a_i, 13)) + list(range(1, b_i + 1)))
        else:
            out.add(int(p))
    # keep only valid months
    return {m for m in out if 1 <= m <= 12}


def _random_dates_between(
    start: str, end: str, n: int, include_months=None, exclude_months=None
):
    """
    Return n unique random dates between start and end (inclusive).
    include_months / exclude_months: sets of month ints (1..12). If include_months is provided it is used;
    otherwise exclude_months (if provided) is applied.
    """
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    if ed < sd:
        raise ValueError("end must be >= start")
    days = (ed - sd).days + 1
    # build candidate list of dates respecting month filters
    candidates = []
    for i in range(days):
        d = sd + timedelta(days=i)
        m = d.month
        if include_months is not None:
            if m in include_months:
                candidates.append(d)
        elif exclude_months is not None:
            if m not in exclude_months:
                candidates.append(d)
        else:
            candidates.append(d)
    if not candidates:
        raise ValueError("No candidate dates available after applying month filters.")
    if n > len(candidates):
        raise ValueError(
            f"Requested {n} dates but only {len(candidates)} candidates available."
        )
    picked = sorted({random.choice(candidates) for _ in range(n)})
    # format as YYYY-MM-DD strings
    return [d.strftime("%Y-%m-%d") for d in picked]


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

    # build dates list
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    elif args.dates_file:
        dates = _load_dates_from_file(Path(args.dates_file))
    elif args.date_range:
        if not (args.start and args.end):
            raise SystemExit(
                "When using --date-range you must provide --start and --end"
            )
        sd = datetime.strptime(args.start, "%Y-%m-%d")
        ed = datetime.strptime(args.end, "%Y-%m-%d")
        if ed < sd:
            raise SystemExit("end must be >= start")
        days = (ed - sd).days + 1
        dates = [(sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    else:
        if not (args.start and args.end):
            raise SystemExit("When using --random you must provide --start and --end")
        # prepare month filters
        include_months = None
        exclude_months = None
        if args.season:
            # season provided as "M-M"
            try:
                include_months = _parse_months_spec(args.season)
            except Exception:
                raise SystemExit(
                    "Invalid --season format. Use M-M (e.g. 6-10)."
                ) from None
        elif args.exclude_months:
            exclude_months = _parse_months_spec(args.exclude_months)
        try:
            dates = _random_dates_between(
                args.start,
                args.end,
                args.random,
                include_months=include_months,
                exclude_months=exclude_months,
            )
        except Exception as exc:
            raise SystemExit(f"Could not sample dates: {exc}") from None

    if args.jobs > 1:
        # use prefer='processes' (default) if you want stronger isolation
        # prefer='threads' is efficient if the tasks are I/O bound (waiting for subprocess).
        logger.info(f"Running {len(dates)} jobs with {args.jobs} jobs...")
        results = Parallel(n_jobs=args.jobs, prefer="processes")(
            delayed(run_one_date)(
                args.python,
                args.script_path,
                d,
                timeout=args.timeout,
                extra_args=extra_args,
            )
            for d in tqdm(dates)
        )

    else:
        logger.info("Running in sequential mode.")
        results = []
        for d in tqdm(dates):
            ok = run_one_date(
                args.python,
                args.script_path,
                d,
                timeout=args.timeout,
                extra_args=extra_args,
            )
            results.append(ok)

    # summary
    success_count = sum(1 for ok in results if bool(ok))
    total = len(dates)
    logger.info("-" * 40)
    logger.info(f"SUMMARY: {success_count}/{total} succeeded.")
    if success_count < total:
        failed = [d for d, ok in zip(dates, results, strict=True) if not ok]
        logger.error(f"Failed dates ({len(failed)}): {', '.join(failed)}")
    logger.info("-" * 40)
