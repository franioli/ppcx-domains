"""
Analyze time series of clustering results across multiple dates.

Reads clustering outputs from multiple date folders and creates:
1. Time series plots of velocity statistics per sector
2. Multi-panel visualization of sector evolution over time
"""

import argparse
import logging
from pathlib import Path

from ppcluster import load_config, setup_logger

logger = setup_logger(logging.INFO, name="ppcx")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze time series of clustering results"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        required=True,
        help="Base directory containing result folders (e.g., output)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output directory for time series plots (default: input_dir/kinematic_sectors_time_series)",
    )
    parser.add_argument(
        "--folder-pattern",
        default=r".*_\d{4}-\d{2}-\d{2}$",
        help="Regex pattern for result folder names",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=31,
        help="Maximum number of dates to show in evolution figure",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, config_path: Path | None = None):
    config = load_config(config_path)

    input_dir = args.dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = args.out or (input_dir / "kinematic_sectors_time_series")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find result folders
    logger.info(f"Searching for result folders in {input_dir}...")
    folders = find_result_folders(input_dir, pattern=args.folder_pattern)
    logger.info(f"Found {len(folders)} result folders")
    if not folders:
        logger.error("No result folders found!")
        return

    logger.info(f"Time series analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
