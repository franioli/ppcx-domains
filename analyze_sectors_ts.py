"""
Analyze time series of clustering results across multiple dates.

Reads clustering outputs from multiple date folders and creates:
1. Time series plots of velocity statistics per sector
2. Multi-panel visualization of sector evolution over time
"""

import argparse
import logging
import re
from pathlib import Path

import joblib
import pandas as pd

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




def export_statistics_table(results_list: list, output_path: Path):
    """
    Export combined statistics table for all dates.

    Args:
        results_list: List of result dictionaries
        output_path: Path to save CSV file
    """
    all_stats = []

    for result in results_list:
        if result is None or result["stats"] is None:
            continue

        stats_df = result["stats"].copy()
        stats_df["date"] = result["date"]
        all_stats.append(stats_df)

    if not all_stats:
        logger.warning("No statistics to export")
        return

    combined_df = pd.concat(all_stats, ignore_index=True)

    # Reorder columns to have date first
    cols = ["date"] + [c for c in combined_df.columns if c != "date"]
    combined_df = combined_df[cols]

    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined statistics to {output_path}")


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

    # Load results from all dates
    logger.info("Loading results from all dates...")
    results_list = []
    for date, folder in folders:
        logger.info(f"Loading {date}...")
        result = load_sector_results(folder)
        if result is not None:
            results_list.append(result)
    logger.info(f"Successfully loaded {len(results_list)} / {len(folders)} results")

    if not results_list:
        logger.error("No valid results loaded!")
        return

    # Export combined statistics table
    logger.info("Exporting combined statistics table...")
    export_statistics_table(results_list, output_dir / "combined_statistics.csv")

    logger.info(f"Time series analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
