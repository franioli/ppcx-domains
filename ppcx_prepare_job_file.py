"""Generate a job file for `ppcx_identify_domains.py` (job-generator only).

This script only builds and prints one-line commands (one per date) suitable
for external batch tools such as GNU Parallel. It does NOT execute the
commands itself. For local, single-node execution with a thread pool use
`ppcx_run_batch.py`.

Any additional arguments passed to this script (that are not recognized flags)
will be forwarded directly to the underlying clustering script.

------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------

1. Generate a job list for GNU Parallel (recommended for large batches):
    python ppcx_prepare_job_file.py --date-range 2020-06-01 2020-08-01 > jobs.txt

2. Generate from explicit dates:
    python ppcx_prepare_job_file.py --dates 2020-07-01,2020-07-02 > jobs.txt

3. Run the generated job file with GNU Parallel:
    parallel -j 4 --bar --joblog run.log --resume < jobs.txt

4. If you prefer a local runner, use the runner script:
    python ppcx_run_batch.py --date-range 2020-06-01 2020-06-05 --jobs 4
"""

import argparse
import shlex
import sys
from datetime import datetime, timedelta
from pathlib import Path


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
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current interpreter).",
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


if __name__ == "__main__":
    # parse arguments, including extra args for the clustering script
    args, extra_args = parse_args()

    # Build deduplicated, sorted list of dates from all CLI sources
    dates = build_dates_list(args)

    # Build per-date commands
    tasks = []
    for d in dates:
        cmd = [args.python, args.script_path, "--date", d]
        if extra_args:
            cmd.extend(extra_args)
        tasks.append((d, cmd))

    # If dry-run, just print commands and exit (useful for GNU Parallel)
    for _, cmd in tasks:
        print(shlex.join(cmd))
