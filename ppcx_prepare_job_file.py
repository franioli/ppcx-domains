"""Generate a job file for any script that accepts a ``--date`` argument.

This script builds one-line shell commands (one per date) and writes them to
a job file (default: ``jobs.txt``) or prints them to stdout with ``--stdout``.
The target script is the first positional argument; it defaults to
``ppcx_identify_domains.py`` but any script with a compatible ``--date``
flag works (e.g. ``ppcx_detect_anomaly.py``).

This script does NOT execute the commands itself.

Extra arguments that are not recognised flags are forwarded to every
generated command line. This is the recommended way to pass configuration
overrides (OmegaConf dot-list syntax) or flags common to all dates:

    python ppcx_prepare_job_file.py ppcx_identify_domains.py \\
        --date-range 2024-08-23 2024-10-30 \\
        data.subset_name=2024_24mp mcmc.force_cpu=true

------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------

1. Write a job file (default behaviour):
    python ppcx_prepare_job_file.py ppcx_identify_domains.py \\
        --date-range 2020-06-01 2020-08-01 --output jobs.txt

2. Write for anomaly detection with extra overrides:
    python ppcx_prepare_job_file.py ppcx_detect_anomaly.py \\
        --date-range 2024-06-01 2024-10-30 --output jobs_anomaly.txt \\
        data.subset_name=2024_24mp mcmc.force_cpu=true

3. Write from a list of explicit dates:
    python ppcx_prepare_job_file.py ppcx_identify_domains.py \\
        --dates 2020-07-01,2020-07-02 --output jobs.txt

4. Print to stdout and pipe into GNU Parallel (parallel execution):
    python ppcx_prepare_job_file.py ppcx_identify_domains.py \\
        --date-range 2020-06-01 2020-08-01 --stdout | parallel -j 4

5. Run the job file with GNU Parallel (with logging and resume support):
    parallel -j 4 --bar --joblog run.log --resume < jobs.txt

6. Sequential execution using xargs (one job at a time):
    xargs -L 1 -a jobs.txt
"""

import argparse
import shlex
import sys
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_SCRIPT_PATH = "ppcx_identify_domains.py"


class Parser(argparse.ArgumentParser):
    """ArgumentParser that prints the full help message on any usage error."""

    def error(self, message: str) -> None:
        self.exit(2, f"\nerror: {message}\n")
        self.print_help(sys.stderr)


def build_parser() -> Parser:

    parser = Parser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "script",
        nargs="?",
        default=DEFAULT_SCRIPT_PATH,
        help=(
            f"Path to the target script to run (default: {DEFAULT_SCRIPT_PATH}). "
            "Any script that accepts a --date argument works here, e.g. "
            "ppcx_identify_domains.py or ppcx_detect_anomaly.py."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="jobs.txt",
        help=(
            "Path of the job file to write (default: jobs.txt). "
            "Ignored when --stdout is set."
        ),
    )

    # --- Date input options ---
    # --dates and --date-range can be freely combined to build a merged date list.
    # --dates-file provides the complete date list from a file and any --dates / --date-range arguments are ignored.
    parser.add_argument(
        "--dates",
        help=(
            "Comma-separated list of reference dates (YYYY-MM-DD). "
            "Can be combined with --date-range to add individual dates to a range. "
            "Mutually exclusive with --dates-file. "
            "Example: --dates 2020-01-01,2020-02-02"
        ),
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        action="append",
        dest="date_ranges",
        help=(
            "Inclusive date range START END (YYYY-MM-DD). "
            "Can be repeated to specify multiple ranges, and combined with --dates. "
            "Mutually exclusive with --dates-file. "
            "Example: --date-range 2016-06-01 2016-10-30 --date-range 2017-06-01 2017-10-30"
        ),
    )
    parser.add_argument(
        "--dates-file",
        help=(
            "Path to a file with one date (YYYY-MM-DD) per line. "
            "Mutually exclusive with --dates and --date-range: when this option is given, those two flags are ignored."
        ),
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current interpreter).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print commands to stdout instead of writing a job file.",
    )
    return parser


# === Date building ===


def _expand_date_range(start: str, end: str) -> list[str]:
    """Return all dates (inclusive) between start and end as YYYY-MM-DD strings."""
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    if ed < sd:
        raise argparse.ArgumentTypeError(
            f"Date range error: end '{end}' must be >= start '{start}'"
        )
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

    ``--dates-file`` takes priority: when provided, the file is the sole
    source and ``--dates`` / ``--date-range`` are ignored.
    Otherwise ``--dates`` (comma-separated) and ``--date-range`` (repeatable)
    are merged together.
    """
    if not any([args.dates, args.dates_file, args.date_ranges]):
        raise argparse.ArgumentTypeError(
            "No dates provided. Use --dates, --date-range, or --dates-file."
        )

    if args.dates_file:
        return sorted(_load_dates_from_file(Path(args.dates_file)))

    dates: set[str] = set()

    if args.dates:
        for d in args.dates.split(","):
            d = d.strip()
            if d:
                dates.add(d)

    if args.date_ranges:
        for start, end in args.date_ranges:
            dates.update(_expand_date_range(start, end))

    return sorted(dates)


if __name__ == "__main__":
    # parse arguments; unrecognised tokens are forwarded to the target script
    parser = build_parser()

    args, extra_args = parser.parse_known_args()

    # Build deduplicated, sorted list of dates from all CLI sources
    dates = build_dates_list(args)

    # Build per-date commands
    tasks = []
    for d in dates:
        cmd = [args.python, args.script, "--date", d]
        if extra_args:
            cmd.extend(extra_args)
        tasks.append((d, cmd))

    lines = [shlex.join(cmd) for _, cmd in tasks]

    if args.stdout:
        for line in lines:
            print(line)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(
            f"Job file written to: {output_path}  ({len(lines)} commands)",
            file=sys.stderr,
        )
